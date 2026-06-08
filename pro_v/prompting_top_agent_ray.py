#!/usr/bin/env python3
"""
Pro-V Top Agent with Ray Support (Simplified Architecture)

This is the main orchestration script that coordinates all agents.
Architecture: Each agent only has __init__ and run methods.
"""

import argparse
import json
import os
import sys
import time
import ray
import subprocess
import shutil
import re
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pro_v.agent.gen_tb import GenTBAgent
from pro_v.agent.pychecker import PyCheckerAgent
from pro_v.utils.llm_client import (
    create_llm_client_from_config,
    create_pychecker_llm_client_from_config
)


def get_circuit_type(rtl_code: str) -> str:
    """
    Determine circuit type by analyzing RTL code.
    Sequential circuits use clock edges (posedge/negedge).
    Combinational circuits do not.

    Args:
        rtl_code: RTL code to analyze

    Returns:
        "seq" if sequential (contains posedge/negedge), "cmb" if combinational
    """
    rtl_lower = rtl_code.lower()
    if "posedge" in rtl_lower or "negedge" in rtl_lower:
        return "seq"
    else:
        return "cmb"


def _stable_json(value: Any) -> str:
    """Stable representation for majority voting over generated outputs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _load_json_file(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _has_real_expected_output(value: Any) -> bool:
    """True when expected_outputs contains at least one non-empty binary output dict."""
    if isinstance(value, dict):
        if value and all(isinstance(v, str) and v and set(v) <= {"0", "1"} for v in value.values()):
            return True
        return any(_has_real_expected_output(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_real_expected_output(item) for item in value)
    return False


def _testbench_shape_score(testbench_data: Any, stimulus_data: Any, circuit_type: str) -> Dict[str, Any]:
    """Score one generated testbench before expensive simulation."""
    score = 0.0
    reasons = []

    if not isinstance(testbench_data, list) or not testbench_data:
        return {"score": -1000.0, "reasons": ["testbench is not a non-empty list"]}

    score += 10.0
    tb_len = len(testbench_data)
    stimulus_len = len(stimulus_data) if isinstance(stimulus_data, list) else 0

    if stimulus_len and tb_len == stimulus_len:
        score += 20.0
        reasons.append("length matches stimulus")
    elif stimulus_len:
        delta = abs(tb_len - stimulus_len)
        score -= min(50.0, float(delta))
        reasons.append(f"length mismatch: testbench={tb_len}, stimulus={stimulus_len}")

    expected_cap = 100 if circuit_type.lower() == "cmb" else 24
    if tb_len <= expected_cap:
        score += 10.0
    else:
        score -= min(100.0, float(tb_len - expected_cap))
        reasons.append(f"over cap: {tb_len}>{expected_cap}")

    valid_entries = 0
    entries_with_outputs = 0
    total_cycles = 0
    seq_mode = circuit_type.lower() == "seq"
    for entry in testbench_data:
        if not isinstance(entry, dict):
            continue
        outputs = entry.get("expected_outputs")
        if seq_mode:
            inputs = {
                k: v for k, v in entry.items()
                if k not in ("expected_outputs", "clock_cycles")
            }
            input_ok = bool(inputs) and "clock_cycles" in entry
        else:
            inputs = entry.get("inputs")
            input_ok = isinstance(inputs, dict)

        if input_ok and outputs is not None:
            valid_entries += 1
            if _has_real_expected_output(outputs):
                entries_with_outputs += 1
            if seq_mode:
                try:
                    total_cycles += int(entry.get("clock_cycles", 1) or 1)
                except Exception:
                    total_cycles += 1

    score += 20.0 * (valid_entries / max(tb_len, 1))
    if entries_with_outputs:
        score += 20.0 * (entries_with_outputs / max(tb_len, 1))
    else:
        score -= 25.0
        reasons.append("expected_outputs empty for every entry")

    if seq_mode and total_cycles > 512:
        score -= min(100.0, float(total_cycles - 512) / 4.0)
        reasons.append(f"total cycles over cap: {total_cycles}>512")

    return {"score": score, "reasons": reasons, "length": tb_len, "total_cycles": total_cycles}


def judge_pychecker_samples(
    pychecker_results: List[Dict[str, Any]],
    stimulus_json_path: str,
    circuit_type: str
) -> Dict[str, Any]:
    """
    Select the best PyChecker sample using lightweight judge signals:
    JSON validity, alignment with stimulus, bounded size, non-empty outputs,
    and majority agreement of expected outputs across samples.
    """
    try:
        stimulus_data = _load_json_file(stimulus_json_path)
    except Exception as exc:
        stimulus_data = []
        stimulus_error = str(exc)
    else:
        stimulus_error = None

    judged = []
    outputs_by_index = {}

    for sample in pychecker_results:
        sample_idx = sample.get("sample_idx")
        tb_path = sample.get("testbench_json_path")
        try:
            tb_data = _load_json_file(tb_path)
        except Exception as exc:
            judged.append({
                "sample_idx": sample_idx,
                "score": -1000.0,
                "reasons": [f"failed to load testbench: {exc}"],
                "testbench_json_path": tb_path,
            })
            continue

        shape = _testbench_shape_score(tb_data, stimulus_data, circuit_type)
        judged_item = {
            "sample_idx": sample_idx,
            "score": shape["score"],
            "reasons": shape["reasons"],
            "length": shape.get("length"),
            "total_cycles": shape.get("total_cycles"),
            "testbench_json_path": tb_path,
        }
        judged.append(judged_item)

        if isinstance(tb_data, list):
            for entry_idx, entry in enumerate(tb_data):
                if isinstance(entry, dict):
                    outputs_by_index.setdefault(entry_idx, []).append(
                        (sample_idx, _stable_json(entry.get("expected_outputs", {})))
                    )

    majority_by_index = {}
    for entry_idx, values in outputs_by_index.items():
        counts = {}
        for _, output_repr in values:
            counts[output_repr] = counts.get(output_repr, 0) + 1
        if counts:
            majority_by_index[entry_idx] = max(counts.items(), key=lambda kv: kv[1])[0]

    for judged_item in judged:
        tb_path = judged_item.get("testbench_json_path")
        if judged_item["score"] <= -1000:
            continue
        try:
            tb_data = _load_json_file(tb_path)
        except Exception:
            continue
        matches = 0
        comparable = 0
        for entry_idx, entry in enumerate(tb_data if isinstance(tb_data, list) else []):
            if entry_idx not in majority_by_index or not isinstance(entry, dict):
                continue
            comparable += 1
            if _stable_json(entry.get("expected_outputs", {})) == majority_by_index[entry_idx]:
                matches += 1
        if comparable:
            ratio = matches / comparable
            judged_item["majority_agreement"] = ratio
            judged_item["score"] += 30.0 * ratio
            judged_item["reasons"].append(f"majority agreement {ratio:.2f}")

    if stimulus_error:
        for judged_item in judged:
            judged_item["reasons"].append(f"stimulus load warning: {stimulus_error}")

    best = max(judged, key=lambda item: item["score"]) if judged else None
    if best is None:
        return {"selected_sample_idx": None, "samples": judged, "reason": "no samples to judge"}

    return {
        "selected_sample_idx": best["sample_idx"],
        "selected_score": best["score"],
        "selected_reasons": best["reasons"],
        "samples": judged,
    }


def load_benchmark_data(benchmark_path: str) -> Dict[int, Dict[str, Any]]:
    """Load benchmark data from test_benchmark_new.json

    Args:
        benchmark_path: Path to test_benchmark_new.json

    Returns:
        Dictionary mapping task_number to task data containing:
        - task_id: Task identifier
        - task_number: Task number
        - description: Problem description
        - header: Module header
        - module_code: Full RTL module code
        - mutants: List of mutant codes (optional)
    """
    print(f"Loading benchmark data from: {benchmark_path}")

    if not os.path.exists(benchmark_path):
        print(f"ERROR: Benchmark file not found: {benchmark_path}")
        return {}

    try:
        with open(benchmark_path, 'r') as f:
            benchmark_list = json.load(f)

        # CHANGED: handle both dict (single task) and list formats
        task_map = {}
        if isinstance(benchmark_list, dict):
            benchmark_entries = [benchmark_list]
        elif isinstance(benchmark_list, list):
            benchmark_entries = benchmark_list
        else:
            print(f"ERROR: Unexpected benchmark format: {type(benchmark_list)}")
            return {}

        for task in benchmark_entries:
            task_number = task.get("task_number") if isinstance(task, dict) else None
            if task_number is not None:
                task_map[task_number] = task

        print(f"Loaded {len(task_map)} tasks from benchmark")
        return task_map

    except Exception as e:
        print(f"ERROR: Failed to load benchmark data: {e}")
        import traceback
        traceback.print_exc()
        return {}


@ray.remote(num_cpus=1)
class TaskWorker:
    """
    Ray worker for processing individual tasks with 1 CPU
    Each worker has its own instances of the three agents
    """

    def __init__(self, llm_client_config: Dict[str, Any]):
        """Initialize task worker with agents

        Args:
            llm_client_config: Configuration for LLM client containing:
                - model: Model name
                - vllm_endpoints: Comma-separated vLLM endpoints
        """
        # Create LLM clients from configuration
        # Standard LLM client for GenTB and Verifier (uses TEMPERATURE)
        self.llm_client = create_llm_client_from_config(
            endpoints_csv=llm_client_config["vllm_endpoints"],
            model_name=llm_client_config["model"]
        )

        # PyChecker-specific LLM client (uses TEMPERATURE_SAMPLE for diversity)
        self.pychecker_llm_client = create_pychecker_llm_client_from_config(
            endpoints_csv=llm_client_config["vllm_endpoints"],
            model_name=llm_client_config["model"]
        )

        # Create a dedicated PyChecker worker for this TaskWorker
        # This worker will be shared by GenTBAgent and PyCheckerAgent
        from pro_v.tools.pychecker_worker import get_ray_pychecker_worker_cls
        PyCheckerWorkerCls = get_ray_pychecker_worker_cls()
        if PyCheckerWorkerCls:
            # Create one worker per TaskWorker for Python execution
            self.pychecker_worker = PyCheckerWorkerCls.remote(worker_id=0)
            print(f"TaskWorker: Created dedicated PyChecker worker")
        else:
            self.pychecker_worker = None
            print(f"TaskWorker: No PyChecker worker (Ray not available)")

        # Initialize the three agents (once per worker)
        # Pass the pychecker_worker to agents that need it
        self.gen_tb_agent = GenTBAgent(
            llm_client=self.llm_client,
            max_retries=3,
            worker=self.pychecker_worker
        )
        self.pychecker_agent = PyCheckerAgent(
            llm_client=self.pychecker_llm_client,
            max_retries=3,
            worker=self.pychecker_worker
        )

        print(f"TaskWorker initialized with 2 agents + PyChecker worker")
        print(f"  - GenTB Agent: LLM={llm_client_config['model']} (T={self.llm_client.temperature}), Worker={'Yes' if self.pychecker_worker else 'No'}")
        print(f"  - PyChecker Agent: LLM={llm_client_config['model']} (T={self.pychecker_llm_client.temperature}), Worker={'Yes' if self.pychecker_worker else 'No'}")

    def process_task(
        self,
        task_number: int,
        rtl_code: str,
        description: str,
        output_base_dir: str,
        sampling_size: int = 3,
        enable_verification: bool = False,
        task_id: str = None,
        header: str = None,
        mutants: List[str] = None
    ) -> Dict[str, Any]:
        """Process a single task through the complete pipeline

        New file structure:
            task_{number}/
            ├── task_info.json       # Task metadata
            ├── module_code.v        # RTL code
            ├── description.txt      # Specification
            ├── header.v             # Module header
            ├── stimulus.json        # Generated by GenTBAgent
            ├── golden_dut_0.py      # Sample 0
            ├── testbench_0.json
            ├── golden_dut_1.py      # Sample 1
            ├── testbench_1.json
            ├── golden_dut_2.py      # Sample 2
            ├── testbench_2.json
            └── sim_cmb/ or sim_seq/ # Simulation files
                ├── Makefile
                ├── input.vc
                ├── sim-main.cpp
                └── rfuzz-harness.h

        Args:
            task_number: Task number
            rtl_code: RTL code to process
            description: Specification description
            output_base_dir: Base output directory
            sampling_size: Number of pychecker samples to generate
            enable_verification: Whether to run verification
            task_id: Task identifier (e.g., "2012_q1g")
            header: Module header
            mutants: List of mutant RTL codes

        Returns:
            Task processing result
        """
        import shutil

        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Worker processing Task {task_number}")
        print(f"{'='*60}")

        # Determine circuit type by analyzing RTL code
        circuit_type = get_circuit_type(rtl_code)

        # Create output directory for this task
        # Convert to absolute path to avoid issues with different working directories
        output_dir = os.path.abspath(os.path.join(output_base_dir, f"task_{task_number}"))
        os.makedirs(output_dir, exist_ok=True)

        # Save task metadata and files
        task_info = {
            "task_number": task_number,
            "task_id": task_id or f"task_{task_number}",
            "circuit_type": circuit_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save task info
        with open(os.path.join(output_dir, "task_info.json"), "w") as f:
            json.dump(task_info, f, indent=2)

        # Save RTL code
        if rtl_code:
            with open(os.path.join(output_dir, "module_code.v"), "w") as f:
                f.write(rtl_code)

        # Save specification
        if description:
            with open(os.path.join(output_dir, "description.txt"), "w") as f:
                f.write(description)

        # Save header
        if header:
            with open(os.path.join(output_dir, "header.v"), "w") as f:
                f.write(header)

        # Save mutants
        if mutants:
            mutants_dir = os.path.join(output_dir, "mutants")
            os.makedirs(mutants_dir, exist_ok=True)
            for i, mutant_code in enumerate(mutants):
                with open(os.path.join(mutants_dir, f"mutant_{i}.v"), "w") as f:
                    f.write(mutant_code)

        result = {
            "task_number": task_number,
            "circuit_type": circuit_type,
            "success": False,
            "error": None
        }

        # Step 1: Run GenTBAgent to generate stimulus.json at task level
        print(f"\nTask {task_number} - Step 1: Running GenTBAgent (circuit_type={circuit_type})")
        gen_tb_result = self.gen_tb_agent.run(
            description=description,
            header=header,
            circuit_type=circuit_type,
            output_dir=output_dir
        )

        if not gen_tb_result["success"]:
            result["error"] = f"GenTB failed: {gen_tb_result.get('error', 'Unknown')}"
            result["gen_tb_result"] = gen_tb_result
            print(f"Task {task_number} - GenTBAgent FAILED")
            return self._finalize_result(result, start_time, output_dir)

        print(f"Task {task_number} - GenTBAgent succeeded (attempt {gen_tb_result['attempt']})")
        result["gen_tb_result"] = gen_tb_result
        stimulus_json_path = gen_tb_result["stimulus_json_path"]

        # Step 2: Run PyCheckerAgent multiple times (sampling)
        # Each sample generates golden_dut_{sample}.py and testbench_{sample}.json at task level
        print(f"\nTask {task_number} - Step 2: Running PyCheckerAgent {sampling_size} times")
        pychecker_results = []
        for sample_idx in range(sampling_size):
            # CHANGED: added header= parameter to match simple.py edits
            pychecker_result = self.pychecker_agent.run(
                description=description,
                header=header,
                circuit_type=circuit_type,
                stimulus_json_path=stimulus_json_path,
                output_dir=output_dir
            )

            if pychecker_result["success"]:
                # Rename generated files to include sample index
                old_golden_path = pychecker_result["golden_dut_path"]
                old_testbench_path = pychecker_result["testbench_json_path"]
                new_golden_path = os.path.join(output_dir, f"golden_dut_{sample_idx}.py")
                new_testbench_path = os.path.join(output_dir, f"testbench_{sample_idx}.json")

                # Rename files
                if os.path.exists(old_golden_path):
                    shutil.move(old_golden_path, new_golden_path)
                if os.path.exists(old_testbench_path):
                    shutil.move(old_testbench_path, new_testbench_path)

                print(f"  Sample {sample_idx}: SUCCESS (attempt {pychecker_result['attempt']})")
                pychecker_results.append({
                    "sample_idx": sample_idx,
                    "golden_dut_path": new_golden_path,
                    "testbench_json_path": new_testbench_path,
                    "result": pychecker_result
                })
            else:
                print(f"  Sample {sample_idx}: FAILED - {pychecker_result.get('error', 'Unknown')}")

        direct_success, direct_detail, direct_golden_path, direct_testbench_path = self._create_direct_oracle_seed_testbench(
            stimulus_json_path=stimulus_json_path,
            rtl_code=rtl_code,
            output_dir=output_dir,
            circuit_type=circuit_type,
            sample_idx=sampling_size,
        )
        if direct_success:
            print(f"  Sample {sampling_size}: DIRECT stimulus skeleton ({direct_detail})")
            pychecker_results.append({
                "sample_idx": sampling_size,
                "golden_dut_path": direct_golden_path,
                "testbench_json_path": direct_testbench_path,
                "result": {
                    "success": True,
                    "source": "direct_stimulus_oracle_seed",
                    "attempt": 0,
                    "detail": direct_detail,
                }
            })
        else:
            print(f"  Direct stimulus skeleton FAILED - {direct_detail}")

        if not pychecker_results:
            result["error"] = "All PyChecker samples failed"
            result["pychecker_results"] = []
            print(f"Task {task_number} - All PyCheckerAgent samples FAILED")
            return self._finalize_result(result, start_time, output_dir)

        print(f"Task {task_number} - {len(pychecker_results)}/{sampling_size} PyChecker samples succeeded")
        result["pychecker_results"] = pychecker_results

        # Step 3: Copy simulation files to sim_cmb or sim_seq directory
        print(f"\nTask {task_number} - Step 3: Copying simulation files")
        sim_dir_name = f"sim_{circuit_type}"
        sim_dest_dir = os.path.join(output_dir, sim_dir_name)
        os.makedirs(sim_dest_dir, exist_ok=True)

        # Determine source directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_source_dir = os.path.join(project_root, "pro_v", sim_dir_name)

        # Files to copy
        files_to_copy = [
            "Makefile", "input.vc",
            "sim-main.cpp", "rfuzz-harness.h", "rfuzz-harness.cpp", "harness-generator.py"
        ]
        for filename in files_to_copy:
            src_path = os.path.join(sim_source_dir, filename)
            dest_path = os.path.join(sim_dest_dir, filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"  Copied: {filename}")
            else:
                print(f"  WARNING: {filename} not found in {sim_source_dir}")

        module_code_path = os.path.join(output_dir, "module_code.v")
        top_module_path = os.path.join(sim_dest_dir, "top_module.v")
        if os.path.exists(module_code_path):
            shutil.copy(module_code_path, top_module_path)
            print(f"  Created: top_module.v from module_code.v")
        else:
            result["error"] = f"Missing module_code.v; cannot create {top_module_path}"
            print(f"Task {task_number} - Missing module_code.v for simulation setup")
            return self._finalize_result(result, start_time, output_dir)

        result["sim_dir"] = sim_dest_dir

        # Replace LLM-computed expected_outputs with outputs observed from the
        # benchmark RTL. The LLM proposes stimulus; RTL is the oracle.
        print(f"\nTask {task_number} - Step 3.5: Refreshing expected outputs from golden RTL")
        for sample in pychecker_results:
            refreshed, detail = self._refresh_expected_outputs_from_rtl(
                rtl_code=rtl_code,
                testbench_json_path=sample["testbench_json_path"],
                output_dir=output_dir,
                circuit_type=circuit_type,
                sample_idx=sample["sample_idx"],
            )
            sample["rtl_oracle_refresh"] = {"success": refreshed, "detail": detail}
            status = "OK" if refreshed else "FAILED"
            print(f"  Sample {sample['sample_idx']}: oracle refresh {status} ({detail})")

        # Step 4: Judge and select best testbench/golden DUT sample
        judge_result = judge_pychecker_samples(
            pychecker_results=pychecker_results,
            stimulus_json_path=stimulus_json_path,
            circuit_type=circuit_type,
        )

        sample_simulation_results = []
        for sample in pychecker_results:
            sample_idx = sample["sample_idx"]
            passed, detail = self._sample_passes_golden(
                rtl_code=rtl_code,
                testbench_json_path=sample["testbench_json_path"],
                output_dir=output_dir,
                circuit_type=circuit_type,
                sample_idx=sample_idx,
            )
            sample_simulation_results.append({
                "sample_idx": sample_idx,
                "passed_golden_sim": passed,
                "detail": detail,
            })

        selected_idx = judge_result.get("selected_sample_idx")
        passing_samples = [item for item in sample_simulation_results if item["passed_golden_sim"]]
        if passing_samples:
            selected_idx = passing_samples[0]["sample_idx"]
            judge_result["simulation_override"] = {
                "selected_sample_idx": selected_idx,
                "reason": "sample passed golden RTL simulation",
            }
        else:
            result["judge_result"] = judge_result
            result["sample_simulation_results"] = sample_simulation_results
            result["error"] = "No PyChecker sample passed golden RTL simulation"
            print(f"\nTask {task_number} - Step 4: No sample passed golden RTL simulation; skipping eval/mutants")
            for sim_item in sample_simulation_results:
                print(f"  Sample {sim_item['sample_idx']} golden sim: {sim_item['passed_golden_sim']} ({sim_item['detail']})")
            return self._finalize_result(result, start_time, output_dir)

        selected_sample = next(
            (sample for sample in pychecker_results if sample.get("sample_idx") == selected_idx),
            pychecker_results[0],
        )
        selected_sample_idx = selected_sample["sample_idx"]
        selected_testbench_path = selected_sample["testbench_json_path"]
        selected_golden_path = selected_sample["golden_dut_path"]

        print(f"\nTask {task_number} - Step 4: Judge selected sample {selected_sample_idx}")
        print(f"  Judge score: {judge_result.get('selected_score', 'N/A')}")
        for reason in judge_result.get("selected_reasons", []):
            print(f"  - {reason}")
        for sim_item in sample_simulation_results:
            print(f"  Sample {sim_item['sample_idx']} golden sim: {sim_item['passed_golden_sim']} ({sim_item['detail']})")
        if judge_result.get("simulation_override"):
            print(f"  Simulation override selected sample {selected_sample_idx}")
        result["judge_result"] = judge_result
        result["sample_simulation_results"] = sample_simulation_results
        result["selected_sample_idx"] = selected_sample_idx
        result["selected_testbench_path"] = selected_testbench_path
        result["selected_golden_path"] = selected_golden_path

        # Step 5: Run VerifierAgent with verification loop (if enabled)
        if enable_verification:
            print(f"\nTask {task_number} - Step 5: Running VerifierAgent with loop")
            verification_iterations = []
            max_verification_loops = 3  # Maximum number of verification iterations

            for loop_idx in range(max_verification_loops):
                print(f"\n  Verification Loop {loop_idx + 1}/{max_verification_loops}")

                # Run verification
                verification_result = self.verifier_agent.run(
                    rtl_code=rtl_code,
                    specification=description,
                    circuit_type=circuit_type,
                    stimulus_json_path=stimulus_json_path,
                    testbench_json_path=selected_testbench_path,
                    pychecker_code_path=selected_golden_path
                )

                verification_iterations.append({
                    "loop": loop_idx + 1,
                    "result": verification_result
                })

                decision = verification_result.get("decision", "UNKNOWN")
                print(f"  Decision: {decision}")
                print(f"  Reason: {verification_result.get('reason', 'N/A')}")

                # Handle verification decisions
                if decision == "COMPLETE":
                    print(f"  Verification COMPLETE - testbench is correct!")
                    result["verification_passed"] = True
                    break

                elif decision == "MODIFY_PYCHECKER":
                    print(f"  Need to modify PyChecker code")
                    print(f"  Error: {verification_result.get('details', {}).get('error_locations', 'N/A')}")
                    print(f"  Re-running PyChecker with corrections...")

                    # CHANGED: added header= parameter
                    pychecker_result = self.pychecker_agent.run(
                        description=description,
                        header=header,
                        circuit_type=circuit_type,
                        stimulus_json_path=stimulus_json_path,
                        output_dir=output_dir
                    )

                    if pychecker_result["success"]:
                        new_sample_idx = len(pychecker_results)
                        old_golden_path = pychecker_result["golden_dut_path"]
                        old_testbench_path = pychecker_result["testbench_json_path"]
                        new_golden_path = os.path.join(output_dir, f"golden_dut_{new_sample_idx}.py")
                        new_testbench_path = os.path.join(output_dir, f"testbench_{new_sample_idx}.json")

                        if os.path.exists(old_golden_path):
                            shutil.move(old_golden_path, new_golden_path)
                        if os.path.exists(old_testbench_path):
                            shutil.move(old_testbench_path, new_testbench_path)

                        selected_testbench_path = new_testbench_path
                        selected_golden_path = new_golden_path
                        selected_sample_idx = new_sample_idx
                        pychecker_results.append({
                            "sample_idx": new_sample_idx,
                            "golden_dut_path": new_golden_path,
                            "testbench_json_path": new_testbench_path,
                            "result": pychecker_result
                        })
                        print(f"  Generated new sample {new_sample_idx}")
                    else:
                        print(f"  PyChecker regeneration failed: {pychecker_result.get('error', 'Unknown')}")
                        result["verification_passed"] = False
                        break

                elif decision == "MODIFY_TESTBENCH":
                    print(f"  Need to modify testbench outputs")
                    print(f"  Incorrect fields: {verification_result.get('details', {}).get('incorrect_fields', 'N/A')}")
                    print(f"  Treating as PyChecker issue - regenerating...")

                    # CHANGED: added header= parameter
                    pychecker_result = self.pychecker_agent.run(
                        description=description,
                        header=header,
                        circuit_type=circuit_type,
                        stimulus_json_path=stimulus_json_path,
                        output_dir=output_dir
                    )

                    if pychecker_result["success"]:
                        new_sample_idx = len(pychecker_results)
                        old_golden_path = pychecker_result["golden_dut_path"]
                        old_testbench_path = pychecker_result["testbench_json_path"]
                        new_golden_path = os.path.join(output_dir, f"golden_dut_{new_sample_idx}.py")
                        new_testbench_path = os.path.join(output_dir, f"testbench_{new_sample_idx}.json")

                        if os.path.exists(old_golden_path):
                            shutil.move(old_golden_path, new_golden_path)
                        if os.path.exists(old_testbench_path):
                            shutil.move(old_testbench_path, new_testbench_path)

                        selected_testbench_path = new_testbench_path
                        selected_golden_path = new_golden_path
                        selected_sample_idx = new_sample_idx
                        pychecker_results.append({
                            "sample_idx": new_sample_idx,
                            "golden_dut_path": new_golden_path,
                            "testbench_json_path": new_testbench_path,
                            "result": pychecker_result
                        })
                        print(f"  Generated new sample {new_sample_idx}")
                    else:
                        print(f"  Testbench regeneration failed: {pychecker_result.get('error', 'Unknown')}")
                        result["verification_passed"] = False
                        break

                else:  # ERROR or UNKNOWN
                    print(f"  Verification error or unknown decision")
                    result["verification_passed"] = False
                    break

                # Check if we exhausted all loops
                if loop_idx == max_verification_loops - 1 and decision != "COMPLETE":
                    print(f"  Verification exhausted {max_verification_loops} loops without completion")
                    result["verification_passed"] = False

            result["verification_iterations"] = verification_iterations
            result["verification_loops_used"] = len(verification_iterations)

        else:
            result["verification_result"] = {"skipped": True}
            result["verification_passed"] = None

        # Update final selected sample info
        result["selected_sample_idx"] = selected_sample_idx
        result["selected_testbench_path"] = selected_testbench_path
        result["selected_golden_path"] = selected_golden_path

        # Step 6: Simulate the generated testbench on mutants and golden DUT
        print(f"\nTask {task_number} - Step 6: Running simulation evaluation")

        simulation_metrics = {
            "eval0_compile_success": False,
            "eval1_module_passes": False,
            "eval2_mutant_detection": {
                "total_mutants": 0,
                "mutants_detected": 0,
                "agreement_rate": 0.0,
                "agreement_80": False,
                "agreement_90": False,
                "agreement_100": False
            },
            "overall_success": False,
            "error": None
        }

        try:
            # CHANGED: benchmark_path sourced from task_info saved earlier instead of
            # hardcoded "test_benchmark_new.json" — we re-read it from output_dir
            benchmark_file = os.path.join(output_dir, "..", "..", "benchmark_path.txt")
            if os.path.exists(benchmark_file):
                with open(benchmark_file, "r") as f:
                    benchmark_file_path = f.read().strip()
            else:
                benchmark_file_path = os.environ.get(
                    "FOLDER_PATH",
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "verilog-eval", "HDLBits", "test_benchmark_new.json")
                )

            if os.path.exists(benchmark_file_path):
                with open(benchmark_file_path, 'r') as f:
                    raw_benchmark = json.load(f)

                # CHANGED: handle both dict and list formats (same as load_benchmark_data)
                if isinstance(raw_benchmark, dict):
                    benchmark_data = [raw_benchmark]
                elif isinstance(raw_benchmark, list):
                    benchmark_data = raw_benchmark
                else:
                    benchmark_data = []

                # Find this task in benchmark
                task_benchmark = None
                for item in benchmark_data:
                    if item.get("task_number") == task_number:
                        task_benchmark = item
                        break

                if task_benchmark:
                    module_code = task_benchmark.get("module_code", "")
                    testbench_code = task_benchmark.get("testbench", "")
                    mutants = task_benchmark.get("mutants", [])
                    # expected_results = task_benchmark.get("result", [])
                    raw_result = task_benchmark.get("result", [])
                    expected_results = [not x for x in raw_result]  # now True = should pass


                    print(f"  Found benchmark data: {len(mutants)} mutants")
                    simulation_metrics["eval2_mutant_detection"]["total_mutants"] = len(mutants)

                    if module_code and testbench_code:
                        # Simulate module_code with testbench
                        print(f"  Step 6.1: Testing module_code compilation and correctness...")

                        # Create simulation directory
                        sim_eval_dir = os.path.join(output_dir, "sim_eval")
                        os.makedirs(sim_eval_dir, exist_ok=True)

                        # Write module and testbench
                        with open(os.path.join(sim_eval_dir, "top_module.v"), 'w') as f:
                            f.write(module_code)
                        with open(os.path.join(sim_eval_dir, "testbench.v"), 'w') as f:
                            f.write(testbench_code)

                        # Copy simulation template
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        sim_template = os.path.join(project_root, "pro_v", f"sim_{circuit_type}")

                        if os.path.exists(sim_template):
                            for fname in ["Makefile", "input.vc", "sim-main.cpp", "rfuzz-harness.h", "harness-generator.py"]:
                                src = os.path.join(sim_template, fname)
                                if os.path.exists(src):
                                    shutil.copy(src, sim_eval_dir)

                        # Write real testbench.json for harness generator
                        if os.path.exists(selected_testbench_path):
                            shutil.copy(selected_testbench_path, os.path.join(sim_eval_dir, "testbench.json"))
                        else:
                            with open(os.path.join(sim_eval_dir, "testbench.json"), 'w') as f:
                                json.dump([], f)

                        # Generate task-specific rfuzz-harness.cpp
                        subprocess.run(
                            ["python3", "harness-generator.py"],
                            cwd=sim_eval_dir,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )

                        # Run simulation
                        try:
                            make_proc = subprocess.run(
                                ["make", "-j1"],
                                cwd=sim_eval_dir,
                                capture_output=True,
                                text=True,
                                timeout=180
                            )
                            sim_output = make_proc.stdout + "\n" + make_proc.stderr

                            # Check eval0: compilation success
                            if make_proc.returncode == 0:
                                simulation_metrics["eval0_compile_success"] = True
                                print(f"  eval0: ✓ PASSED - Compilation successful")

                                # Check eval1: module passes
                                mismatch_match = re.search(r'Mismatches:\s*(\d+)', sim_output)
                                unpass_match = re.search(r'Unpass:\s*(\d+)', sim_output)
                                if mismatch_match:
                                    mismatches = int(mismatch_match.group(1))
                                    simulation_metrics["eval1_module_passes"] = (mismatches == 0)
                                elif unpass_match:
                                    unpass = int(unpass_match.group(1))
                                    simulation_metrics["eval1_module_passes"] = (unpass == 0)

                                if simulation_metrics["eval1_module_passes"]:
                                    print(f"  eval1: ✓ PASSED - Module passes testbench")
                                else:
                                    print(f"  eval1: ✗ FAILED - Module has mismatches")

                                # Step 6.2: Test mutants (eval2)
                                if mutants:
                                    print(f"  Step 6.2: Testing {len(mutants)} mutants...")
                                    mutant_results = []
                                    mutants_detected = 0

                                    for idx, mutant_code in enumerate(mutants):
                                        mutant_dir = os.path.join(output_dir, f"sim_mutant_{idx}")
                                        os.makedirs(mutant_dir, exist_ok=True)

                                        # Write mutant and testbench
                                        with open(os.path.join(mutant_dir, "top_module.v"), 'w') as f:
                                            f.write(mutant_code)
                                        with open(os.path.join(mutant_dir, "testbench.v"), 'w') as f:
                                            f.write(testbench_code)

                                        # Copy simulation files
                                        for fname in ["Makefile", "input.vc", "sim-main.cpp", "rfuzz-harness.h", "harness-generator.py"]:
                                            src = os.path.join(sim_template, fname)
                                            if os.path.exists(src):
                                                shutil.copy(src, mutant_dir)

                                        if os.path.exists(selected_testbench_path):
                                            shutil.copy(selected_testbench_path, os.path.join(mutant_dir, "testbench.json"))
                                        else:
                                            with open(os.path.join(mutant_dir, "testbench.json"), 'w') as f:
                                                json.dump([], f)

                                        # Generate task-specific rfuzz-harness.cpp
                                        subprocess.run(
                                            ["python3", "harness-generator.py"],
                                            cwd=mutant_dir,
                                            capture_output=True,
                                            text=True,
                                            timeout=30
                                        )

                                        # Simulate mutant
                                        try:
                                            mutant_proc = subprocess.run(
                                                ["make", "-j1"],
                                                cwd=mutant_dir,
                                                capture_output=True,
                                                text=True,
                                                timeout=180
                                            )
                                            mutant_output = mutant_proc.stdout + "\n" + mutant_proc.stderr

                                            # Mutant is detected if it fails
                                            mutant_passed = False
                                            if mutant_proc.returncode == 0:
                                                m_match = re.search(r'Mismatches:\s*(\d+)', mutant_output)
                                                u_match = re.search(r'Unpass:\s*(\d+)', mutant_output)
                                                if m_match:
                                                    mutant_passed = (int(m_match.group(1)) == 0)
                                                elif u_match:
                                                    mutant_passed = (int(u_match.group(1)) == 0)

                                            mutant_detected = not mutant_passed
                                            mutant_results.append(mutant_detected)
                                            if mutant_detected:
                                                mutants_detected += 1

                                            # Compare with expected
                                            if idx < len(expected_results):
                                                expected_fails = not expected_results[idx]
                                                status = "✓" if mutant_detected == expected_fails else "✗"
                                                print(f"  Mutant {idx}: detected={mutant_detected}, expected_fails={expected_fails} {status}")

                                        except Exception:
                                            mutant_results.append(True)  # Timeout/error = detected
                                            mutants_detected += 1
                                        finally:
                                            try:
                                                shutil.rmtree(mutant_dir)
                                            except Exception:
                                                pass

                                    # Calculate agreement rate
                                    agreement_count = sum(
                                        1 for actual, expected in zip(mutant_results, expected_results)
                                        if actual == (not expected)
                                    )
                                    agreement_rate = agreement_count / len(mutants) if mutants else 0.0

                                    simulation_metrics["eval2_mutant_detection"]["mutants_detected"] = mutants_detected
                                    simulation_metrics["eval2_mutant_detection"]["agreement_rate"] = agreement_rate
                                    simulation_metrics["eval2_mutant_detection"]["agreement_80"] = (agreement_rate >= 0.80)
                                    simulation_metrics["eval2_mutant_detection"]["agreement_90"] = (agreement_rate >= 0.90)
                                    simulation_metrics["eval2_mutant_detection"]["agreement_100"] = (agreement_rate >= 1.00)

                                    print(f"  eval2: Agreement rate {agreement_rate:.1%} ({agreement_count}/{len(mutants)})")
                                    print(f"    80%: {'✓ PASSED' if simulation_metrics['eval2_mutant_detection']['agreement_80'] else '✗ FAILED'}")
                                    print(f"    90%: {'✓ PASSED' if simulation_metrics['eval2_mutant_detection']['agreement_90'] else '✗ FAILED'}")
                                    print(f"   100%: {'✓ PASSED' if simulation_metrics['eval2_mutant_detection']['agreement_100'] else '✗ FAILED'}")

                            else:
                                print(f"  eval0: ✗ FAILED - Compilation error")
                                simulation_metrics["error"] = "Compilation failed"

                        except subprocess.TimeoutExpired:
                            simulation_metrics["error"] = "Simulation timeout"
                            print(f"  ERROR: Simulation timeout")
                        except Exception as e:
                            simulation_metrics["error"] = f"Simulation error: {str(e)}"
                            print(f"  ERROR: {str(e)}")
                        finally:
                            try:
                                shutil.rmtree(sim_eval_dir)
                            except Exception:
                                pass

                    # Calculate overall success
                    simulation_metrics["overall_success"] = (
                        simulation_metrics["eval0_compile_success"] and
                        simulation_metrics["eval1_module_passes"] and
                        simulation_metrics["eval2_mutant_detection"]["agreement_80"]
                    )
                    print(f"  Overall success: {'✓ PASSED' if simulation_metrics['overall_success'] else '✗ FAILED'}")

                else:
                    simulation_metrics["error"] = f"Task {task_number} not found in benchmark"
                    print(f"  WARNING: Task not found in benchmark")
            else:
                simulation_metrics["error"] = "Benchmark file not found"
                print(f"  WARNING: benchmark file not found at {benchmark_file_path}")

        except Exception as e:
            simulation_metrics["error"] = f"Evaluation error: {str(e)}"
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

        result["simulation_metrics"] = simulation_metrics

        # Mark as successful
        result["success"] = True
        return self._finalize_result(result, start_time, output_dir)

    def _sample_passes_golden(
        self,
        rtl_code: str,
        testbench_json_path: str,
        output_dir: str,
        circuit_type: str,
        sample_idx: int
    ):
        """Check whether one generated JSON testbench passes the benchmark RTL."""
        sample_eval_dir = os.path.join(output_dir, f"sample_select_{sample_idx}")

        try:
            shutil.rmtree(sample_eval_dir, ignore_errors=True)
            os.makedirs(sample_eval_dir, exist_ok=True)

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sim_template = os.path.join(project_root, "pro_v", f"sim_{circuit_type}")

            if not os.path.exists(sim_template):
                return False, f"missing sim template: {sim_template}"

            with open(os.path.join(sample_eval_dir, "top_module.v"), "w") as f:
                f.write(rtl_code)

            try:
                candidate_tb = _load_json_file(testbench_json_path)
            except Exception as exc:
                return False, f"failed to load testbench: {exc}"
            if not isinstance(candidate_tb, list) or not candidate_tb:
                return False, "empty/non-list testbench"
            if circuit_type.lower() == "seq":
                self._complete_seq_inputs_from_rtl(candidate_tb, rtl_code)
                with open(testbench_json_path, "w") as f:
                    json.dump(candidate_tb, f, indent=2)
            if not all(_has_real_expected_output(entry.get("expected_outputs")) for entry in candidate_tb if isinstance(entry, dict)):
                return False, "candidate has empty expected_outputs"

            for fname in ["Makefile", "input.vc", "sim-main.cpp", "rfuzz-harness.h", "rfuzz-harness.cpp", "harness-generator.py"]:
                src = os.path.join(sim_template, fname)
                if os.path.exists(src):
                    shutil.copy(src, sample_eval_dir)
                else:
                    return False, f"missing template file: {fname}"

            shutil.copy(testbench_json_path, os.path.join(sample_eval_dir, "testbench.json"))

            env = os.environ.copy()
            env["PATH"] = f"/scratch/network/ak7587/envs/pro-v/bin:{env.get('PATH', '')}"

            hgen_proc = subprocess.run(
                [sys.executable, "harness-generator.py"],
                cwd=sample_eval_dir,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )

            if hgen_proc.returncode != 0:
                return False, f"harness generation failed: {(hgen_proc.stderr or hgen_proc.stdout)[-500:]}"

            subprocess.run(["make", "clean"], cwd=sample_eval_dir, capture_output=True, text=True, timeout=30, env=env)
            make_timeout = 360 if circuit_type.lower() == "seq" else 180
            make_proc = subprocess.run(
                ["make", "-j1"],
                cwd=sample_eval_dir,
                capture_output=True,
                text=True,
                timeout=make_timeout,
                env=env,
            )

            sim_output = make_proc.stdout + "\n" + make_proc.stderr
            sim_finished = "sim finished" in sim_output
            mismatch_match = re.search(r"Mismatches:\s*(\d+)", sim_output)
            unpass_match = re.search(r"Unpass:\s*(\d+)", sim_output)

            if mismatch_match:
                mismatches = int(mismatch_match.group(1))
                return mismatches == 0, f"Mismatches={mismatches}"

            if unpass_match:
                unpass = int(unpass_match.group(1))
                return unpass == 0, f"Unpass={unpass}"

            if not (make_proc.returncode == 0 or sim_finished):
                return False, f"compile/run failed rc={make_proc.returncode}: {sim_output[-500:]}"

            return make_proc.returncode == 0, f"returncode={make_proc.returncode}"

        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)

    def _extract_output_widths_from_verilog(self, rtl_code: str) -> Dict[str, int]:
        """Best-effort output width extraction from a Verilog module header."""
        text = re.sub(r"//.*", "", rtl_code or "")
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        module_match = re.search(r"module\s+\w+\s*\((.*?)\);", text, re.S)
        search_text = module_match.group(1) if module_match else text

        outputs = {}
        pattern = re.compile(
            r"\boutput\b\s+"
            r"(?:(?:wire|reg|logic|signed|unsigned)\s+)*"
            r"(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?"
            r"([A-Za-z_][A-Za-z0-9_$]*)"
        )
        for msb, lsb, name in pattern.findall(search_text):
            outputs[name] = abs(int(msb) - int(lsb)) + 1 if msb and lsb else 1

        return outputs

    def _extract_input_widths_from_verilog(self, rtl_code: str) -> Dict[str, int]:
        """Best-effort input width extraction from a Verilog module header."""
        text = re.sub(r"//.*", "", rtl_code or "")
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        module_match = re.search(r"module\s+\w+\s*\((.*?)\);", text, re.S)
        search_text = module_match.group(1) if module_match else text

        inputs = {}
        pattern = re.compile(
            r"\binput\b\s+"
            r"(?:(?:wire|reg|logic|signed|unsigned)\s+)*"
            r"(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?"
            r"([A-Za-z_][A-Za-z0-9_$]*)"
        )
        for msb, lsb, name in pattern.findall(search_text):
            if name == "clk":
                continue
            inputs[name] = abs(int(msb) - int(lsb)) + 1 if msb and lsb else 1

        return inputs

    def _deterministic_input_values(
        self,
        name: str,
        width: int,
        cycles: int,
        scenario_idx: int,
    ) -> List[str]:
        """Fill missing inputs with bounded, repeatable stimulus."""
        lname = name.lower()
        width = max(1, int(width or 1))
        cycles = max(1, int(cycles or 1))

        active_low_reset = lname.endswith("n") or lname.endswith("_n")
        if lname in ("reset", "rst", "areset", "arst") or "reset" in lname or lname in ("rst_n", "aresetn", "resetn"):
            asserted = "0" if active_low_reset else "1"
            released = "1" if active_low_reset else "0"
            return [asserted] + [released] * (cycles - 1)

        if width == 1:
            patterns = [
                lambda i: i & 1,
                lambda i: (i + 1) & 1,
                lambda i: 0,
                lambda i: 1,
                lambda i: 1 if i in (1, 3, 4, 7) else 0,
            ]
            fn = patterns[scenario_idx % len(patterns)]
            return [str(fn(i)) for i in range(cycles)]

        mask = (1 << width) - 1
        values = []
        seeds = [0, mask, 1, mask >> 1, 1 << (width - 1)]
        for cycle in range(cycles):
            if cycle < len(seeds):
                value = seeds[(cycle + scenario_idx) % len(seeds)] & mask
            else:
                value = ((scenario_idx + 1) * 1103515245 + (cycle + 3) * 12345) & mask
            values.append(format(value, f"0{width}b"))
        return values

    def _complete_seq_inputs_from_rtl(self, scenarios: List[Dict[str, Any]], rtl_code: str) -> None:
        """Ensure every sequential scenario drives every non-clock RTL input."""
        input_widths = self._extract_input_widths_from_verilog(rtl_code)
        if not input_widths:
            return

        for scenario_idx, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                continue
            cycles = int(scenario.get("clock_cycles", 0) or 0)
            if cycles <= 0:
                cycles = max(
                    [len(v) for v in scenario.values() if isinstance(v, list)] or [1]
                )
                scenario["clock_cycles"] = cycles

            for name, width in input_widths.items():
                if name in scenario and isinstance(scenario[name], list):
                    values = [str(v) for v in scenario[name][:cycles]]
                    if values:
                        values.extend([values[-1]] * (cycles - len(values)))
                    else:
                        values = self._deterministic_input_values(name, width, cycles, scenario_idx)
                    scenario[name] = values
                elif name in scenario:
                    scenario[name] = [str(scenario[name])] * cycles
                else:
                    scenario[name] = self._deterministic_input_values(name, width, cycles, scenario_idx)

    def _zero_output_dict(self, output_widths: Dict[str, int]) -> Dict[str, str]:
        return {name: "0" * max(1, width) for name, width in output_widths.items()}

    def _hex_to_binary(self, hex_value: str, width: int) -> str:
        value = int(hex_value, 16)
        return format(value & ((1 << width) - 1), f"0{width}b")

    def _set_expected_output(
        self,
        testbench_data: Any,
        circuit_type: str,
        output_widths: Dict[str, int],
        index: int,
        name: str,
        value: str,
        cycle: int = None,
        edge: str = None,
    ) -> None:
        if name not in output_widths:
            return

        if circuit_type.lower() == "seq":
            if cycle is None or edge not in ("rising_edge", "falling_edge"):
                return
            scenario = testbench_data[index]
            scenario_outputs = scenario.setdefault("expected_outputs", [])
            while len(scenario_outputs) <= cycle:
                scenario_outputs.append({
                    "rising_edge": self._zero_output_dict(output_widths),
                    "falling_edge": self._zero_output_dict(output_widths),
                })
            scenario_outputs[cycle].setdefault(edge, {})[name] = value
        else:
            entry = testbench_data[index]
            entry.setdefault("expected_outputs", {})[name] = value

    def _parse_oracle_outputs(
        self,
        stdout: str,
        testbench_data: Any,
        circuit_type: str,
        output_widths: Dict[str, int],
    ) -> int:
        """Parse harness stdout and write actual RTL outputs into testbench_data."""
        current_index = None
        current_cycle = None
        wide_target = None
        wide_chunks = {}
        observed = 0

        def flush_wide():
            nonlocal wide_target, wide_chunks, observed
            if not wide_target:
                return
            index, cycle, edge, name = wide_target
            width = output_widths.get(name)
            if not width:
                wide_target = None
                wide_chunks = {}
                return
            value = 0
            for chunk_idx, chunk_value in wide_chunks.items():
                value |= chunk_value << (32 * chunk_idx)
            binary = format(value & ((1 << width) - 1), f"0{width}b")
            self._set_expected_output(
                testbench_data, circuit_type, output_widths,
                index, name, binary, cycle=cycle, edge=edge
            )
            observed += 1
            wide_target = None
            wide_chunks = {}

        for line in stdout.splitlines():
            m = re.search(r"=+ Test Vector (\d+) =+", line)
            if m:
                flush_wide()
                current_index = int(m.group(1))
                current_cycle = None
                continue

            m = re.search(r"=+ Testing Scenario: scenario_(\d+) =+", line)
            if m:
                flush_wide()
                current_index = int(m.group(1))
                current_cycle = None
                continue

            m = re.search(r"--- Cycle (\d+) ---", line)
            if m:
                flush_wide()
                current_cycle = int(m.group(1))
                continue

            m = re.search(
                r"^\s*(Rising edge output|Falling edge output|Output)\s+([A-Za-z_][A-Za-z0-9_$]*):"
                r"\s*expected\(from JSON\)=0x[0-9a-fA-F]+,\s*actual\(from sim\)=0x([0-9a-fA-F]+)",
                line,
            )
            if m and current_index is not None:
                flush_wide()
                kind, name, actual_hex = m.groups()
                edge = None
                if kind.startswith("Rising"):
                    edge = "rising_edge"
                elif kind.startswith("Falling"):
                    edge = "falling_edge"
                width = output_widths.get(name)
                if width:
                    self._set_expected_output(
                        testbench_data, circuit_type, output_widths,
                        current_index, name, self._hex_to_binary(actual_hex, width),
                        cycle=current_cycle, edge=edge,
                    )
                    observed += 1
                continue

            m = re.search(
                r"^\s*(Rising edge output|Falling edge output|Output)\s+([A-Za-z_][A-Za-z0-9_$]*)\s+\(wide\):",
                line,
            )
            if m and current_index is not None:
                flush_wide()
                kind, name = m.groups()
                edge = None
                if kind.startswith("Rising"):
                    edge = "rising_edge"
                elif kind.startswith("Falling"):
                    edge = "falling_edge"
                wide_target = (current_index, current_cycle, edge, name)
                wide_chunks = {}
                continue

            m = re.search(
                r"^\s*\[(\d+)\]\s*expected\(from JSON\)=0x[0-9a-fA-F]+,\s*actual\(from sim\)=0x([0-9a-fA-F]+)",
                line,
            )
            if m and wide_target:
                chunk_idx, actual_hex = m.groups()
                wide_chunks[int(chunk_idx)] = int(actual_hex, 16)
                width = output_widths.get(wide_target[3], 0)
                n_words = (width + 31) // 32 if width else 0
                if n_words and len(wide_chunks) >= n_words:
                    flush_wide()
                continue

        flush_wide()
        return observed

    def _refresh_expected_outputs_from_rtl(
        self,
        rtl_code: str,
        testbench_json_path: str,
        output_dir: str,
        circuit_type: str,
        sample_idx: int,
    ):
        """Run golden RTL once and overwrite testbench expected_outputs with actual outputs."""
        try:
            with open(testbench_json_path, "r") as f:
                testbench_data = json.load(f)
        except Exception as exc:
            return False, f"failed to load testbench: {exc}"

        if not isinstance(testbench_data, list) or not testbench_data:
            return False, "testbench is not a non-empty list"

        output_widths = self._extract_output_widths_from_verilog(rtl_code)
        if not output_widths:
            return False, "could not extract output widths from RTL"

        if circuit_type.lower() == "seq":
            self._complete_seq_inputs_from_rtl(testbench_data, rtl_code)

        placeholder_data = json.loads(json.dumps(testbench_data))
        zeros = self._zero_output_dict(output_widths)
        if circuit_type.lower() == "seq":
            for scenario in placeholder_data:
                cycles = int(scenario.get("clock_cycles", 0) or 0)
                scenario["expected_outputs"] = [
                    {"rising_edge": dict(zeros), "falling_edge": dict(zeros)}
                    for _ in range(cycles)
                ]
        else:
            for entry in placeholder_data:
                entry["expected_outputs"] = dict(zeros)

        oracle_dir = os.path.join(output_dir, f"oracle_sample_{sample_idx}")
        shutil.rmtree(oracle_dir, ignore_errors=True)
        os.makedirs(oracle_dir, exist_ok=True)

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_template = os.path.join(project_root, "pro_v", f"sim_{circuit_type}")
        for fname in ["Makefile", "input.vc", "sim-main.cpp", "rfuzz-harness.h", "harness-generator.py"]:
            src = os.path.join(sim_template, fname)
            if os.path.exists(src):
                shutil.copy(src, oracle_dir)

        with open(os.path.join(oracle_dir, "top_module.v"), "w") as f:
            f.write(rtl_code)
        with open(os.path.join(oracle_dir, "testbench.json"), "w") as f:
            json.dump(placeholder_data, f, indent=2)

        env = os.environ.copy()
        env["PATH"] = f"/scratch/network/ak7587/envs/pro-v/bin:{env.get('PATH', '')}"

        try:
            hgen_proc = subprocess.run(
                [sys.executable, "harness-generator.py"],
                cwd=oracle_dir,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            if hgen_proc.returncode != 0:
                return False, f"harness generation failed: {(hgen_proc.stderr or hgen_proc.stdout)[-300:]}"

            subprocess.run(["make", "clean"], cwd=oracle_dir, capture_output=True, text=True, timeout=15, env=env)
            make_proc = subprocess.run(
                ["make", "-j1"],
                cwd=oracle_dir,
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, "oracle simulation timeout"

        sim_output = (make_proc.stdout or "") + "\n" + (make_proc.stderr or "")
        observed = self._parse_oracle_outputs(sim_output, testbench_data, circuit_type, output_widths)
        if observed == 0:
            return False, f"no oracle outputs parsed; rc={make_proc.returncode}; output={sim_output[-300:]}"

        with open(testbench_json_path, "w") as f:
            json.dump(testbench_data, f, indent=2)

        return True, f"refreshed {observed} output observations"

    def _create_direct_oracle_seed_testbench(
        self,
        stimulus_json_path: str,
        rtl_code: str,
        output_dir: str,
        circuit_type: str,
        sample_idx: int,
    ):
        """Create a testbench skeleton directly from stimulus.json for RTL oracle filling."""
        try:
            with open(stimulus_json_path, "r") as f:
                stimulus_data = json.load(f)
        except Exception as exc:
            return False, f"failed to load stimulus: {exc}", None, None

        if not isinstance(stimulus_data, list) or not stimulus_data:
            return False, "stimulus is not a non-empty list", None, None

        output_widths = self._extract_output_widths_from_verilog(rtl_code)
        if not output_widths:
            return False, "could not extract output widths from RTL", None, None

        stimulus_data = self._augment_direct_stimulus(stimulus_data, circuit_type, rtl_code)

        zeros = self._zero_output_dict(output_widths)
        testbench_data = []

        if circuit_type.lower() == "seq":
            self._complete_seq_inputs_from_rtl(stimulus_data, rtl_code)
            for scenario in stimulus_data:
                if not isinstance(scenario, dict):
                    continue
                clock_cycles = int(scenario.get("clock_cycles", 0) or 0)
                if clock_cycles <= 0:
                    signal_lengths = [
                        len(v) for k, v in scenario.items()
                        if k != "clock_cycles" and isinstance(v, list)
                    ]
                    clock_cycles = max(signal_lengths) if signal_lengths else 1

                entry = {"clock_cycles": clock_cycles}
                for name, values in scenario.items():
                    if name == "clock_cycles":
                        continue
                    if isinstance(values, list):
                        fixed_values = list(values[:clock_cycles])
                        if fixed_values:
                            fixed_values.extend([fixed_values[-1]] * (clock_cycles - len(fixed_values)))
                        else:
                            fixed_values = ["0"] * clock_cycles
                        entry[name] = fixed_values
                    else:
                        entry[name] = [str(values)] * clock_cycles
                entry["expected_outputs"] = [
                    {"rising_edge": dict(zeros), "falling_edge": dict(zeros)}
                    for _ in range(clock_cycles)
                ]
                testbench_data.append(entry)
        else:
            for vector in stimulus_data:
                if not isinstance(vector, dict):
                    continue
                inputs = {k: v for k, v in vector.items() if k != "clock_cycles"}
                testbench_data.append({
                    "inputs": inputs,
                    "expected_outputs": dict(zeros),
                })

        if not testbench_data:
            return False, "no usable stimulus entries", None, None

        testbench_path = os.path.join(output_dir, f"testbench_{sample_idx}.json")
        golden_path = os.path.join(output_dir, f"golden_dut_{sample_idx}.py")

        with open(testbench_path, "w") as f:
            json.dump(testbench_data, f, indent=2)
        with open(golden_path, "w") as f:
            f.write(
                "# Direct stimulus skeleton. Expected outputs are filled from RTL oracle.\n"
                "class GoldenDUT:\n"
                "    pass\n"
            )

        return True, f"created {len(testbench_data)} stimulus entries", golden_path, testbench_path

    def _augment_direct_stimulus(self, stimulus_data: List[Any], circuit_type: str, rtl_code: str = "") -> List[Any]:
        """Add deterministic, bounded scenarios that expose common sequential mutants."""
        if circuit_type.lower() != "seq" or not stimulus_data:
            return stimulus_data

        input_widths = self._extract_input_widths_from_verilog(rtl_code)
        output_widths = self._extract_output_widths_from_verilog(rtl_code)
        keys = set()
        for scenario in stimulus_data:
            if isinstance(scenario, dict):
                keys.update(k for k in scenario.keys() if k != "clock_cycles")

        augmented = list(stimulus_data)

        def find_signal(*candidates):
            lowered = {key.lower(): key for key in keys}
            for candidate in candidates:
                if candidate.lower() in lowered:
                    return lowered[candidate.lower()]
            return None

        reset = find_signal("reset", "areset", "rst", "rst_n", "resetn")
        data = find_signal("data")
        d = find_signal("d")
        in_sig = find_signal("in")
        ground = find_signal("ground")
        dig = find_signal("dig")
        bump_left = find_signal("bump_left")
        bump_right = find_signal("bump_right")

        def bits(value, width):
            mask = (1 << width) - 1
            return format(value & mask, f"0{width}b")

        def reset_values(cycles, asserted_at=(0,)):
            if not reset:
                return None
            lname = reset.lower()
            active_low = lname.endswith("n") or lname.endswith("_n")
            asserted = "0" if active_low else "1"
            released = "1" if active_low else "0"
            values = [released] * cycles
            for idx in asserted_at:
                if 0 <= idx < cycles:
                    values[idx] = asserted
            return values

        # Wide state machines generate enormous logs because every cycle prints
        # many 32-bit chunks. Keep them compact but still include load/boundary cases.
        max_output_width = max(output_widths.values(), default=0)
        if max_output_width > 128:
            compact = []
            width = input_widths.get(data or "", max_output_width)
            patterns = [
                0,
                (1 << width) - 1,
                1,
                1 << (width - 1),
                int(("10" * ((width + 1) // 2))[:width], 2),
                int(("01" * ((width + 1) // 2))[:width], 2),
            ]
            for idx, pattern in enumerate(patterns):
                cycles = 8
                scenario = {"clock_cycles": cycles}
                if reset:
                    scenario[reset] = reset_values(cycles, asserted_at=(0,))
                if data:
                    scenario[data] = [bits(pattern, width)] * cycles
                load = find_signal("load")
                if load:
                    scenario[load] = ["1"] + ["0"] * (cycles - 1)
                for key in keys:
                    if key not in scenario:
                        scenario[key] = self._deterministic_input_values(
                            key, input_widths.get(key, 1), cycles, idx
                        )
                compact.append(scenario)
            return compact

        # Generic DFF coverage: reset value, release, data capture, midstream reset.
        if reset and d and input_widths.get(d, 0) <= 64:
            width = input_widths.get(d, 1)
            patterns = [0, (1 << width) - 1, 0xA5, 0x5A, 1, 1 << (width - 1), (1 << (width - 1)) - 1]
            cycles = len(patterns)
            scenario = {"clock_cycles": cycles, d: [bits(v, width) for v in patterns]}
            scenario[reset] = reset_values(cycles, asserted_at=(0, 4))
            augmented.append(scenario)

            scenario = {"clock_cycles": cycles, d: [bits(patterns[(i + 2) % len(patterns)], width) for i in range(cycles)]}
            scenario[reset] = reset_values(cycles, asserted_at=())
            augmented.append(scenario)

        # Edge-capture coverage: explicit 1->0 transitions, persistence, and reset clear.
        if reset and in_sig and input_widths.get(in_sig, 0) <= 64:
            width = input_widths.get(in_sig, 1)
            values = [
                (1 << width) - 1,
                0,
                (1 << width) - 1,
                int(("10" * ((width + 1) // 2))[:width], 2),
                int(("01" * ((width + 1) // 2))[:width], 2),
                0,
                (1 << (width - 1)),
                0,
                1,
                0,
            ]
            scenario = {"clock_cycles": len(values), in_sig: [bits(v, width) for v in values]}
            scenario[reset] = reset_values(len(values), asserted_at=(0, 6))
            augmented.append(scenario)

        if ground and dig and bump_left and bump_right:
            def base(cycles):
                scenario = {"clock_cycles": cycles}
                for key in keys:
                    if key == ground:
                        scenario[key] = ["1"] * cycles
                    elif key == reset:
                        scenario[key] = ["0"] * cycles
                    else:
                        scenario[key] = ["0"] * cycles
                if reset:
                    scenario[reset][0] = "1"
                return scenario

            # WL -> WR, then FALLR long enough to distinguish >=20 vs >=21.
            cycles = 30
            scenario = base(cycles)
            scenario[bump_left][1] = "1"
            for idx in range(2, 24):
                scenario[ground][idx] = "0"
            for idx in range(24, cycles):
                scenario[ground][idx] = "1"
            augmented.append(scenario)

        return augmented

    def _finalize_result(self, result: Dict[str, Any], start_time: float, output_dir: str) -> Dict[str, Any]:
        """Finalize and save task result

        Args:
            result: Result dictionary
            start_time: Task start time
            output_dir: Output directory

        Returns:
            Finalized result
        """
        result["total_time"] = time.time() - start_time
        result["timestamp"] = time.time()

        # Save result to file
        result_path = os.path.join(output_dir, "task_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"\nTask {result['task_number']} {status} in {result['total_time']:.2f}s")
        print(f"{'='*60}\n")

        return result


class ProVTopAgent:
    """
    Top-level agent that orchestrates the entire Pro-V workflow
    Architecture: __init__ initializes agents once, then distributes tasks to workers
    """

    def __init__(self, args):
        """Initialize the top agent

        Args:
            args: Command line arguments
        """
        self.args = args

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            print("Ray initialized")

        # LLM client configuration
        self.llm_client_config = {
            "model": args.model,
            "vllm_endpoints": args.vllm_endpoints
        }

        print(f"ProVTopAgent initialized for experiment: {args.experiment_name}")

    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all specified tasks using Ray parallelization

        Returns:
            Overall evaluation results
        """
        print(f"\n{'='*70}")
        print(f"Starting Pro-V Evaluation: {self.args.experiment_name}")
        print(f"{'='*70}\n")

        # Determine which tasks to process
        if self.args.task_numbers:
            task_numbers = [int(t.strip()) for t in self.args.task_numbers.split(',')]
        else:
            task_numbers = list(range(1, 157))  # All tasks 1-156

        print(f"Processing {len(task_numbers)} tasks: {task_numbers[:10]}{'...' if len(task_numbers) > 10 else ''}")

        # CHANGED: use args.benchmark_path instead of hardcoded env var
        benchmark_path = self.args.benchmark_path
        benchmark_data = load_benchmark_data(benchmark_path)

        if not benchmark_data:
            print("ERROR: Failed to load benchmark data. Exiting.")
            return {"error": "Failed to load benchmark data"}

        # Prepare tasks to process
        tasks_to_process = []
        for task_num in task_numbers:
            if task_num not in benchmark_data:
                print(f"WARNING: Task {task_num} not found in benchmark data. Skipping.")
                continue
            task = benchmark_data[task_num]
            task_data = {
                "task_number": task_num,
                "task_id": task.get("task_id", f"task_{task_num}"),
                "rtl_code": task.get("module_code", ""),
                "description": task.get("description", ""),
                "header": task.get("header", ""),
                "mutants": task.get("mutants", [])
            }
            tasks_to_process.append(task_data)

        print(f"Successfully prepared {len(tasks_to_process)} tasks for processing")

        # Create output directory
        # Use absolute path to avoid issues with different working directories
        output_base_dir = os.path.abspath(f"outputs/{self.args.experiment_name}")
        os.makedirs(output_base_dir, exist_ok=True)

        # CHANGED: write benchmark_path to a file so TaskWorker.process_task can
        # look it up when running Step 6 simulation evaluation
        with open(os.path.join(output_base_dir, "benchmark_path.txt"), "w") as f:
            f.write(os.path.abspath(benchmark_path))

        # Create worker pool (limited by max_concurrency)
        num_workers = min(len(tasks_to_process), self.args.max_concurrency, os.cpu_count() or 4)
        print(f"\nCreating {num_workers} Ray workers (max_concurrency={self.args.max_concurrency})...")
        workers = [TaskWorker.remote(self.llm_client_config) for _ in range(num_workers)]
        print(f"Workers created successfully\n")

        # Submit tasks to workers (round-robin distribution)
        print(f"Submitting {len(tasks_to_process)} tasks to workers...")
        task_refs = []
        for i, task_data in enumerate(tasks_to_process):
            worker_idx = i % num_workers
            task_ref = workers[worker_idx].process_task.remote(
                task_number=task_data["task_number"],
                rtl_code=task_data["rtl_code"],
                description=task_data["description"],
                output_base_dir=output_base_dir,
                sampling_size=self.args.sampling_size,
                enable_verification=self.args.enable_verification,
                task_id=task_data.get("task_id"),
                header=task_data.get("header"),
                mutants=task_data.get("mutants", [])
            )
            task_refs.append(task_ref)

        # Collect results
        print(f"Processing {len(task_refs)} tasks in parallel...\n")
        all_results = ray.get(task_refs)

        # Calculate overall statistics
        total_tasks = len(all_results)
        successful_tasks = sum(1 for r in all_results if r["success"])

        if self.args.enable_verification:
            verified_tasks = sum(
                1 for r in all_results
                if r.get("verification_result", {}).get("verification_passed", False)
            )
        else:
            verified_tasks = 0

        # Calculate simulation metrics aggregates
        simulation_stats = self._calculate_simulation_aggregates(all_results)

        overall_stats = {
            "experiment_name": self.args.experiment_name,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "verified_tasks": verified_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
            "verification_rate": verified_tasks / total_tasks if total_tasks > 0 else 0.0,
            "total_time": sum(r.get("total_time", 0) for r in all_results),
            "avg_time_per_task": sum(r.get("total_time", 0) for r in all_results) / total_tasks if total_tasks > 0 else 0.0,
            "simulation_metrics": simulation_stats
        }

        # Save results
        with open(os.path.join(output_base_dir, "overall_stats.json"), "w") as f:
            json.dump(overall_stats, f, indent=2)

        with open(os.path.join(output_base_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        # Print summary
        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETED")
        print(f"{'='*70}")
        print(f"Total tasks:       {total_tasks}")
        print(f"Successful tasks:  {successful_tasks} ({overall_stats['success_rate']:.1%})")
        if self.args.enable_verification:
            print(f"Verified tasks:    {verified_tasks} ({overall_stats['verification_rate']:.1%})")
        print(f"Total time:        {overall_stats['total_time']:.2f}s")
        print(f"Avg time/task:     {overall_stats['avg_time_per_task']:.2f}s")

        # Print simulation metrics summary
        if simulation_stats["total_evaluated"] > 0:
            print(f"\n{'='*70}")
            print(f"SIMULATION METRICS SUMMARY")
            print(f"{'='*70}")
            print(f"Total evaluated: {simulation_stats['total_evaluated']}")
            print(f"\nCompile & Simulation Results:")
            print(f"  Compile Success: {simulation_stats['eval0_pass_count']}/{simulation_stats['total_evaluated']} ({simulation_stats['eval0_pass_rate']:.1%})")
            print(f"  Simulation Pass: {simulation_stats['eval1_pass_count']}/{simulation_stats['total_evaluated']} ({simulation_stats['eval1_pass_rate']:.1%})")
            print(f"\nMutant Detection (eval2):")
            print(f"  80% threshold:  {simulation_stats['eval2_80_pass_count']}/{simulation_stats['total_evaluated']} ({simulation_stats['eval2_80_pass_rate']:.1%})")
            print(f"  90% threshold:  {simulation_stats['eval2_90_pass_count']}/{simulation_stats['total_evaluated']} ({simulation_stats['eval2_90_pass_rate']:.1%})")
            print(f"  100% threshold: {simulation_stats['eval2_100_pass_count']}/{simulation_stats['total_evaluated']} ({simulation_stats['eval2_100_pass_rate']:.1%})")
            print(f"\nOverall success: {simulation_stats['overall_success_count']}/{simulation_stats['total_evaluated']} ({simulation_stats['overall_success_rate']:.1%})")
            print(f"{'='*70}")

        print(f"{'='*70}\n")
        return overall_stats

    def _calculate_simulation_aggregates(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate simulation metrics across all tasks

        Args:
            all_results: List of all task results

        Returns:
            Aggregated simulation statistics
        """
        stats = {
            "total_evaluated": 0,
            "eval0_pass_count": 0,
            "eval0_pass_rate": 0.0,
            "eval1_pass_count": 0,
            "eval1_pass_rate": 0.0,
            "eval2_80_pass_count": 0,
            "eval2_80_pass_rate": 0.0,
            "eval2_90_pass_count": 0,
            "eval2_90_pass_rate": 0.0,
            "eval2_100_pass_count": 0,
            "eval2_100_pass_rate": 0.0,
            "overall_success_count": 0,
            "overall_success_rate": 0.0,
            "total_mutants_tested": 0,
            "total_mutants_detected": 0,
            "avg_mutant_agreement_rate": 0.0
        }

        evaluated_count = 0
        total_agreement_rates = []

        for result in all_results:
            sim_metrics = result.get("simulation_metrics")
            if not sim_metrics:
                continue

            evaluated_count += 1

            if sim_metrics.get("eval0_compile_success"):
                stats["eval0_pass_count"] += 1

            if sim_metrics.get("eval1_module_passes"):
                stats["eval1_pass_count"] += 1

            eval2 = sim_metrics.get("eval2_mutant_detection", {})
            if eval2.get("agreement_80"):
                stats["eval2_80_pass_count"] += 1
            if eval2.get("agreement_90"):
                stats["eval2_90_pass_count"] += 1
            if eval2.get("agreement_100"):
                stats["eval2_100_pass_count"] += 1
            if sim_metrics.get("overall_success"):
                stats["overall_success_count"] += 1

            stats["total_mutants_tested"] += eval2.get("total_mutants", 0)
            stats["total_mutants_detected"] += eval2.get("mutants_detected", 0)

            if eval2.get("agreement_rate") is not None:
                total_agreement_rates.append(eval2["agreement_rate"])

        stats["total_evaluated"] = evaluated_count

        if evaluated_count > 0:
            stats["eval0_pass_rate"] = stats["eval0_pass_count"] / evaluated_count
            stats["eval1_pass_rate"] = stats["eval1_pass_count"] / evaluated_count
            stats["eval2_80_pass_rate"] = stats["eval2_80_pass_count"] / evaluated_count
            stats["eval2_90_pass_rate"] = stats["eval2_90_pass_count"] / evaluated_count
            stats["eval2_100_pass_rate"] = stats["eval2_100_pass_count"] / evaluated_count
            stats["overall_success_rate"] = stats["overall_success_count"] / evaluated_count

        if total_agreement_rates:
            stats["avg_mutant_agreement_rate"] = sum(total_agreement_rates) / len(total_agreement_rates)

        return stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Pro-V Top Agent with Ray Support")

    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument("--vllm_endpoints", type=str, required=True, help="Comma-separated vLLM endpoints")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--task_numbers", type=str, help="Comma-separated task numbers to process")
    # CHANGED: added --benchmark_path CLI arg (mirrors simple.py) instead of relying on env var
    parser.add_argument("--benchmark_path", type=str,
                        default="verilog-eval/HDLBits/test_benchmark_new.json",
                        help="Path to benchmark file")
    parser.add_argument("--max_concurrency", type=int, default=8,
                        help="Maximum number of concurrent Ray workers")
    parser.add_argument("--sampling_size", type=int, default=3,
                        help="Number of PyChecker samples to generate per task")
    parser.add_argument("--enable_verification", action="store_true",
                        help="Enable verification loop after PyChecker")

    args = parser.parse_args()

    # Validate max_concurrency
    max_concurrency = min(max(1, args.max_concurrency), 100)
    if args.max_concurrency != max_concurrency:
        print(f"WARNING: max_concurrency adjusted from {args.max_concurrency} to {max_concurrency} (max 100)")
    args.max_concurrency = max_concurrency

    print(f"\n{'='*70}")
    print(f"Pro-V Ray Pipeline: {args.experiment_name}")
    print(f"Max Concurrency: {args.max_concurrency}")
    print(f"Sampling Size:   {args.sampling_size}")
    print(f"Verification:    {args.enable_verification}")
    print(f"{'='*70}\n")

    agent = ProVTopAgent(args)
    result = agent.run_evaluation()

    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
