#!/usr/bin/env python3
"""
Simplified Parallel Simulation and Evaluation System for Mutant Detection
"""

import json
import logging
import shutil
import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from pro_v.mutation_strength import classify_mutant, augment_stimulus_file
    STRENGTH_AVAILABLE = True
except Exception:  # pragma: no cover - package __init__ may be broken in some checkouts
    # Fall back to a direct file-based import that does NOT trigger pro_v/__init__.py
    try:
        import importlib.util as _ilu
        import os as _os
        import sys as _sys
        _ms_path = _os.path.join(_os.path.dirname(__file__), "mutation_strength.py")
        _spec = _ilu.spec_from_file_location("prov_mutation_strength", _ms_path)
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules["prov_mutation_strength"] = _mod  # register before exec (dataclass needs it)
        _spec.loader.exec_module(_mod)
        classify_mutant = _mod.classify_mutant
        augment_stimulus_file = _mod.augment_stimulus_file
        STRENGTH_AVAILABLE = True
    except Exception:  # iverilog/module genuinely unavailable
        STRENGTH_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PID %(process)d] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Store simulation results for a single test"""
    compile_success: bool = False
    passed: bool = False
    error_message: str = ""
    mismatch_count: int = -1
    total_samples: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a task"""
    task_id: str
    task_number: int
    compile_success: bool = False
    module_passes: bool = False
    total_mutants: int = 0
    mutants_detected: int = 0
    mutant_results: List[bool] = field(default_factory=list)
    mutant_compiled: List[bool] = field(default_factory=list)  # per-mutant compile success
    noncompiling_mutants: int = 0
    expected_mutant_results: List[bool] = field(default_factory=list)
    mutant_agreement_80: bool = False
    mutant_agreement_90: bool = False
    mutant_agreement_100: bool = False
    overall_success: bool = False

    # --- testbench-strength analysis (independent differential search) ---
    # Populated only when strength analysis is enabled. A surviving mutant
    # (not killed by the testbench) is classified against the reference RTL:
    #   equivalent   -> no input distinguishes it (excluded from the denominator)
    #   weak_survivor-> a distinguishing input EXISTS (a hole in the testbench)
    #   unknown      -> differential search could not run (e.g. compile collision)
    equivalent_mutants: int = 0
    weak_survivors: int = 0
    unknown_survivors: int = 0
    true_mutation_score: float = -1.0     # killed / (total - equivalent)
    witnesses: List[dict] = field(default_factory=list)

    def calculate_agreement(self, exclude_noncompiling: bool = False):
        """Calculate mutant detection agreement.

        Args:
            exclude_noncompiling: If True, mutants that failed to COMPILE are
                dropped from both numerator and denominator. By default a
                non-compiling mutant counts as "detected" (it fails the run),
                but that detection is the compiler's, not the testbench's -- so
                the stricter option measures pure testbench discrimination.
        """
        if self.total_mutants == 0:
            return

        n = len(self.mutant_results)
        compiled = self.mutant_compiled if self.mutant_compiled else [True] * n
        # positions we score over (aligned across the three per-mutant lists)
        scored = [
            i for i in range(min(n, len(self.expected_mutant_results)))
            if (not exclude_noncompiling) or compiled[i]
        ]
        denom = len(scored)
        if denom == 0:
            logger.info(f"Task {self.task_id}: no scorable mutants after filtering")
            return

        # mutant_results[i]: True = detected (mutant failed the testbench / was killed).
        # expected_mutant_results[i] (benchmark `result`): True = the mutant is a real
        # bug and SHOULD be killed; False = the mutant is equivalent and should survive.
        # (Verified empirically: every code-identical/equivalent mutant is labeled False,
        # and label-True mutants are all functionally distinguishable bugs.)
        # Agreement is therefore detected == should_be_killed, i.e. detected == label.
        # NOTE: the previous version used `(not label)`, which inverted the semantics and
        # scored every correctly-killed mutant as a disagreement -- collapsing eval2.
        agreement_count = sum(
            1 for i in scored
            if self.mutant_results[i] == self.expected_mutant_results[i]
        )
        agreement_rate = agreement_count / denom

        self.mutant_agreement_80 = agreement_rate >= 0.80
        self.mutant_agreement_90 = agreement_rate >= 0.90
        self.mutant_agreement_100 = agreement_rate >= 1.00
        self.overall_success = (
            self.compile_success and
            self.module_passes and
            self.mutant_agreement_80
        )

        suffix = " (excluding non-compiling mutants)" if exclude_noncompiling else ""
        logger.info(
            f"Task {self.task_id}: Agreement rate{suffix}: {agreement_rate:.2%} "
            f"({agreement_count}/{denom})"
        )

    def calculate_true_score(self, exclude_noncompiling: bool = False):
        """True mutation score = killed / (total non-equivalent mutants).

        Equivalent mutants cannot be killed by ANY testbench, so counting them
        as misses understates strength. Excluding them yields the honest number.

        Args:
            exclude_noncompiling: also drop non-compiling mutants (which were
                counted as killed via compile failure) from both numerator and
                denominator, isolating pure testbench discrimination.
        """
        killed = self.mutants_detected
        total = self.total_mutants
        if exclude_noncompiling:
            killed -= self.noncompiling_mutants   # all non-compiling mutants were counted as killed
            total -= self.noncompiling_mutants
        non_equivalent = total - self.equivalent_mutants
        if non_equivalent <= 0:
            self.true_mutation_score = 1.0 if total > 0 else -1.0
            return
        self.true_mutation_score = killed / non_equivalent
        logger.info(
            f"Task {self.task_id}: True mutation score: {self.true_mutation_score:.2%} "
            f"({killed}/{non_equivalent} non-equivalent killed); "
            f"equivalent={self.equivalent_mutants}, weak-survivors={self.weak_survivors}, "
            f"unknown={self.unknown_survivors}"
        )


def run_simulation(module_code: str, sim_dir: Path, timeout: int = 60) -> SimulationResult:
    """
    Run simulation by replacing module and running make

    Args:
        module_code: Verilog module code to test
        sim_dir: Existing simulation directory
        timeout: Simulation timeout in seconds

    Returns:
        SimulationResult
    """
    result = SimulationResult()

    # Replace top_module.v
    module_file = sim_dir / "top_module.v"
    with open(module_file, 'w') as f:
        f.write(module_code)

    # Clean previous build
    subprocess.run("make clean", shell=True, cwd=sim_dir, capture_output=True, timeout=10)

    # Run make to compile and simulate
    try:
        proc = subprocess.run(
            "make -j1",
            shell=True,
            cwd=sim_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        result.stdout = proc.stdout
        result.stderr = proc.stderr

        # `make` compiles AND simulates in one target, so a nonzero return code is
        # ambiguous: it can mean a genuine BUILD error, or that the design compiled
        # and the simulation ran but reported mismatches. Distinguish them -- if the
        # simulator actually executed (it emitted a verdict), compilation succeeded
        # and pass/fail is graded from the output below; only a true build error
        # (no simulation output at all) is a compile failure. Without this a wrong
        # FRM (correct RTL mismatches) is mislabeled as a compile failure -- which
        # understates eval0 (compile != pass) and, worse, under --exclude-noncompiling
        # causes correctly-DETECTED mutants (compiled + mismatched) to be dropped
        # from the eval2 agreement, deflating mutant-detection scores.
        log_content = (proc.stdout or "") + "\n" + (proc.stderr or "")
        sim_ran = any(marker in log_content for marker in (
            "sim finished", "Unpass:", "All tests passed", "Test failed",
            "Mismatches:", "Total mismatched samples", "scenarios passed",
        ))
        result.compile_success = (proc.returncode == 0) or sim_ran

        if not result.compile_success:
            result.error_message = f"Compilation failed: {proc.stderr[:500]}"
            return result
    except subprocess.TimeoutExpired as e:
        result.compile_success = False
        result.passed = False
        result.error_message = f"Simulation timed out after {timeout}s"
        result.stdout = e.stdout.decode() if e.stdout else ""
        result.stderr = e.stderr.decode() if e.stderr else ""
        return result

    # Parse simulation results
    log_content = result.stdout + "\n" + result.stderr

    # Check for mismatch information
    mismatch_pattern = r'Mismatches:\s*(\d+)\s*in\s*(\d+)\s*samples'
    match = re.search(mismatch_pattern, log_content)
    if match:
        result.mismatch_count = int(match.group(1))
        result.total_samples = int(match.group(2))
        result.passed = (result.mismatch_count == 0)
    else:
        alt_pattern = r'Total mismatched samples is\s*(\d+)\s*out of\s*(\d+)\s*samples'
        match2 = re.search(alt_pattern, log_content)
        if match2:
            result.mismatch_count = int(match2.group(1))
            result.total_samples = int(match2.group(2))
            result.passed = (result.mismatch_count == 0)
        elif "All tests passed" in log_content or "✓ All tests passed" in log_content:
            result.passed = True
            result.mismatch_count = 0
        elif "Test failed" in log_content or "✗ Test failed" in log_content or "error(s)" in log_content:
            result.passed = False
            error_match = re.search(r'(\d+)\s+error\(s\)', log_content)
            if error_match:
                result.mismatch_count = int(error_match.group(1))
        else:
            unpass_match = re.search(r'Unpass:\s*(\d+)', log_content)
            if unpass_match:
                result.mismatch_count = int(unpass_match.group(1))
                result.passed = (result.mismatch_count == 0)
            else:
                result.passed = False
                result.error_message = "Could not determine pass/fail status"

    return result


def save_mutant_log(sim_dir: Path, mutant_idx: int, task_id: str, task_number: int, result: SimulationResult):
    """Save mutant simulation log to file"""
    log_dir = sim_dir / "mutant_logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"sim_{mutant_idx}_log.txt"
    with open(log_file, 'w') as f:
        f.write(f"=== Mutant {mutant_idx} Simulation Log ===\n")
        f.write(f"Task: {task_id} (#{task_number})\n")
        f.write(f"Mutant Index: {mutant_idx}\n")
        f.write(f"Compile Success: {result.compile_success}\n")
        f.write(f"Passed: {result.passed}\n")
        f.write(f"Mismatch Count: {result.mismatch_count}\n")
        f.write(f"Total Samples: {result.total_samples}\n")
        f.write(f"\n=== STDOUT ===\n{result.stdout}\n")
        f.write(f"\n=== STDERR ===\n{result.stderr}\n")


def _consolidate_selected_testbench(task_dir: Path, sim_dir: Path,
                                    selected_idx, task_number) -> None:
    """Copy the judge-SELECTED testbench into sim_dir/testbench.json and regenerate
    the harness (rfuzz-harness.cpp) from it.

    The multi-sample (ray) generator writes testbench_<idx>.json per FRM candidate
    and records selected_sample_idx, but never finalizes the chosen one into
    sim_dir/testbench.json. Without this, the scorer compiles a stale/mismatched
    harness and every task fails eval0. Restoring this step makes eval0/1/2 score
    exactly the sample the judge selected -- faithful to Pro-V's single-testbench eval.
    Idempotent; no-op if no candidate testbench is found.
    """
    selected_path = None
    task_result_path = task_dir / "task_result.json"
    if task_result_path.exists():
        try:
            with open(task_result_path, "r") as f:
                task_result = json.load(f)
            raw_selected_path = task_result.get("selected_testbench_path")
            if raw_selected_path:
                selected_path = Path(raw_selected_path)
                if not selected_path.is_absolute():
                    selected_path = task_dir / selected_path
        except Exception as e:
            logger.warning(f"[Task #{task_number}] could not read selected testbench path: {e}")

    candidates = []
    if selected_path is not None:
        candidates.append(selected_path)
    if selected_idx is not None:
        candidates.append(task_dir / f"testbench_{selected_idx}.json")
    candidates += sorted(task_dir.glob("testbench_*.json"))
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        return  # nothing to consolidate; leave whatever the sim dir already has
    try:
        (sim_dir / "testbench.json").write_text(src.read_text())
        for stale_name in ("rfuzz-harness.cpp", "coverage.dat"):
            stale_path = sim_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        shutil.rmtree(sim_dir / "obj_dir", ignore_errors=True)
    except Exception as e:
        logger.warning(f"[Task #{task_number}] testbench consolidation failed: {e}")
        return
    harness_gen = sim_dir / "harness-generator.py"
    if harness_gen.exists():
        try:
            proc = subprocess.run([sys.executable, "harness-generator.py"],
                                  cwd=sim_dir, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                logger.warning(
                    f"[Task #{task_number}] harness regeneration failed: "
                    f"{(proc.stdout or '')[-1000:]} {(proc.stderr or '')[-1000:]}"
                )
        except Exception as e:
            logger.warning(f"[Task #{task_number}] harness regeneration failed: {e}")


def evaluate_single_task(task_data: Dict, experiment_dir: Path, mutant_only: bool,
                         timeout: int = 120, strength_opts: Optional[Dict] = None,
                         exclude_noncompiling: bool = False) -> EvaluationMetrics:
    """
    Evaluate a single task with all its mutants

    Args:
        task_data: Task data from benchmark
        experiment_dir: Experiment outputs directory
        mutant_only: Skip module_code testing if True
        strength_opts: If provided, run independent differential analysis on every
            SURVIVING mutant to separate equivalent mutants from weak-survivors,
            compute a true mutation score, and collect witnesses. Recognized keys:
            enabled(bool), augment_stimulus(bool), and any classify_mutant kwargs
            (max_exhaustive_bits, random_samples, max_witnesses, seq_trials,
            seq_cycles, timeout).

    Returns:
        EvaluationMetrics
    """
    task_id = task_data['task_id']
    task_number = task_data['task_number']

    logger.info(f"[Task #{task_number}] Evaluating {task_id}")

    metrics = EvaluationMetrics(task_id=task_id, task_number=task_number)

    # Find simulation directory
    task_dir = experiment_dir / f"task_{task_number}"
    task_result_path = task_dir / "task_result.json"

    sim_type = "cmb"
    selected_idx = None
    if task_result_path.exists():
        with open(task_result_path, 'r') as f:
            task_result = json.load(f)
            sim_type = (task_result.get('circuit_type') or 'cmb').lower()
            selected_idx = task_result.get('selected_sample_idx')

    sim_dir = task_dir / f"sim_{sim_type}"
    if not sim_dir.exists():
        logger.warning(f"[Task #{task_number}] No sim directory found at: {sim_dir}")
        return metrics

    # Finalize the judge-selected testbench into the sim dir + regen the harness so
    # the scorer evaluates the chosen candidate (faithful to Pro-V's eval0/1/2).
    _consolidate_selected_testbench(task_dir, sim_dir, selected_idx, task_number)

    logger.info(f"[Task #{task_number}] Using sim directory: {sim_dir}")

    module_code = task_data['module_code']
    mutants = task_data.get('mutants', [])
    expected_results = task_data.get('result', [])

    metrics.total_mutants = len(mutants)
    metrics.expected_mutant_results = expected_results

    # Test module_code (skip if mutant_only)
    if not mutant_only:
        logger.info(f"[Task #{task_number}] Testing module_code...")
        result = run_simulation(module_code, sim_dir, timeout=timeout)

        metrics.compile_success = result.compile_success
        metrics.module_passes = result.passed

        if not result.passed:
            logger.warning(f"[Task #{task_number}] Compilation failed: {result.error_message}")
            return metrics

        if result.passed:
            logger.info(f"[Task #{task_number}] Module passed testbench!")
        else:
            logger.warning(f"[Task #{task_number}] Module failed (mismatches: {result.mismatch_count})")
            logger.warning(f"[Task #{task_number}] Skipping mutant evaluation - module did not pass")
            return metrics
    else:
        logger.info(f"[Task #{task_number}] Skipping module_code testing (mutant_only mode)")
        metrics.compile_success = True
        metrics.module_passes = True

    # Test mutants (only if module passed)
    logger.info(f"[Task #{task_number}] Testing {len(mutants)} mutants...")
    surviving_mutants = []  # (index, mutant_code) not killed by the testbench
    for i, mutant_code in enumerate(mutants):
        logger.info(f"[Task #{task_number}] Simulating mutant #{i+1}/{len(mutants)}...")

        mutant_result = run_simulation(mutant_code, sim_dir, timeout=timeout)

        # Save log
        if mutant_result.stdout:
            save_mutant_log(sim_dir, i, task_id, task_number, mutant_result)

        # Mutant detected if it fails
        mutant_detected = not mutant_result.passed
        metrics.mutant_results.append(mutant_detected)
        metrics.mutant_compiled.append(mutant_result.compile_success)
        if not mutant_result.compile_success:
            metrics.noncompiling_mutants += 1

        if mutant_detected:
            metrics.mutants_detected += 1
        else:
            surviving_mutants.append((i, mutant_code))

        # Compare with expected
        if i < len(expected_results):
            expected = expected_results[i]
            actual_fails = mutant_detected
            expected_should_be_killed = bool(expected)
            status = "✓" if actual_fails == expected_should_be_killed else "✗"
            logger.debug(
                f"  Mutant {i}: detected={mutant_detected}, "
                f"expected_should_be_killed={expected_should_be_killed} {status}"
            )

    # ------------------------------------------------------------------
    # Testbench-strength analysis: classify every SURVIVING mutant via an
    # independent differential search (reference RTL vs mutant RTL). This
    # separates truly-equivalent mutants from weak-survivors (holes in the
    # testbench) and yields witnesses that can be injected to close the holes.
    # ------------------------------------------------------------------
    if strength_opts and strength_opts.get("enabled"):
        if not STRENGTH_AVAILABLE:
            logger.warning(f"[Task #{task_number}] Strength analysis requested but "
                           f"mutation_strength (iverilog) unavailable; skipping")
        elif surviving_mutants:
            logger.info(f"[Task #{task_number}] Strength analysis on "
                        f"{len(surviving_mutants)} surviving mutant(s)...")
            classify_kwargs = {k: v for k, v in strength_opts.items()
                               if k in ("max_exhaustive_bits", "random_samples",
                                        "max_witnesses", "seq_trials", "seq_cycles",
                                        "timeout")}
            for i, mutant_code in surviving_mutants:
                verdict = classify_mutant(module_code, mutant_code, sim_type, **classify_kwargs)
                if verdict.unknown:
                    metrics.unknown_survivors += 1
                    logger.debug(f"  Mutant {i}: differential search unknown ({verdict.error[:120]})")
                elif verdict.equivalent:
                    metrics.equivalent_mutants += 1
                    kind = "proven-equivalent" if verdict.certain else "likely-equivalent"
                    logger.debug(f"  Mutant {i}: {kind} (excluded from true score)")
                else:
                    metrics.weak_survivors += 1
                    metrics.witnesses.extend(verdict.witnesses)
                    logger.info(f"  Mutant {i}: WEAK-SURVIVOR - testbench missed a "
                                f"distinguishing input ({len(verdict.witnesses)} witness(es))")

            # Persist witnesses for inspection / feedback
            if metrics.witnesses:
                wpath = sim_dir / "strength_witnesses.json"
                with open(wpath, "w") as f:
                    json.dump(metrics.witnesses, f, indent=2)
                logger.info(f"[Task #{task_number}] Wrote {len(metrics.witnesses)} "
                            f"witness(es) to {wpath}")
                if strength_opts.get("augment_stimulus"):
                    added = augment_stimulus_file(str(sim_dir / "stimulus.json"), metrics.witnesses)
                    logger.info(f"[Task #{task_number}] Injected {added} witness(es) into "
                                f"stimulus.json (re-run PyChecker to refresh testbench.json, "
                                f"then re-evaluate to confirm the holes are closed)")

        metrics.calculate_true_score(exclude_noncompiling=exclude_noncompiling)

    # Calculate metrics
    metrics.calculate_agreement(exclude_noncompiling=exclude_noncompiling)

    logger.info(
        f"[Task #{task_number}] Results: compile={metrics.compile_success}, "
        f"module_passes={metrics.module_passes}, "
        f"mutants_detected={metrics.mutants_detected}/{metrics.total_mutants}, "
        f"agreement_80={metrics.mutant_agreement_80}, "
        f"agreement_90={metrics.mutant_agreement_90}, "
        f"agreement_100={metrics.mutant_agreement_100}"
    )

    return metrics


def worker_function(args):
    """Worker function for parallel execution"""
    task_data, experiment_dir, mutant_only, timeout, strength_opts, exclude_noncompiling = args
    return evaluate_single_task(task_data, Path(experiment_dir), mutant_only,
                                timeout=timeout, strength_opts=strength_opts,
                                exclude_noncompiling=exclude_noncompiling)


def evaluate_all_tasks(
    benchmark_file: str,
    experiment_dir: str,
    mutant_only: bool = False,
    limit: Optional[int] = None,
    start_idx: int = 0,
    num_workers: Optional[int] = None,
    timeout: int = 120,
    strength_opts: Optional[Dict] = None,
    exclude_noncompiling: bool = False,
    task_numbers: Optional[List[int]] = None
) -> List[EvaluationMetrics]:
    """
    Evaluate all tasks in parallel

    Args:
        benchmark_file: Path to test_benchmark_new.json
        experiment_dir: Path to experiment outputs directory
        mutant_only: Skip module_code testing if True
        limit: Maximum number of tasks to evaluate
        start_idx: Starting index in benchmark
        num_workers: Number of parallel workers (default: CPU count - 1)
        task_numbers: Exact 1-based benchmark task numbers to evaluate

    Returns:
        List of EvaluationMetrics
    """
    # Load benchmark
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)

    logger.info(f"Loaded {len(benchmark_data)} tasks from benchmark")
    logger.info(f"Using experiment outputs from: {experiment_dir}")
    if mutant_only:
        logger.info(f"Mutant-only mode: Will skip module_code testing")

    # Select tasks to evaluate. task_numbers are 1-based HDLBits task numbers.
    if task_numbers:
        wanted = set(int(n) for n in task_numbers)
        tasks_to_eval = [
            task for idx, task in enumerate(benchmark_data, start=1)
            if int(task.get("task_number", idx)) in wanted or idx in wanted
        ]
        logger.info(f"Evaluating {len(tasks_to_eval)} explicit task(s): {sorted(wanted)}")
    else:
        tasks_to_eval = benchmark_data[start_idx:]
        if limit:
            tasks_to_eval = tasks_to_eval[:limit]
        logger.info(f"Evaluating {len(tasks_to_eval)} tasks (starting from {start_idx})")

    # Determine number of workers. For exact/subset reports, cap workers to the
    # number of selected tasks so a one-task eval does not spawn the whole CPU.
    if num_workers is None:
        num_workers = max(1, min(len(tasks_to_eval), multiprocessing.cpu_count() - 1))
    else:
        num_workers = max(1, min(num_workers, max(1, len(tasks_to_eval))))

    logger.info(f"Using {num_workers} parallel workers")

    # Prepare worker arguments
    worker_args = [
        (task_data, experiment_dir, mutant_only, timeout, strength_opts, exclude_noncompiling)
        for task_data in tasks_to_eval
    ]

    # Run parallel evaluation
    results = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(worker_function, args): args[0]['task_number']
            for args in worker_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task_number = future_to_task[future]
            try:
                metrics = future.result()
                results.append(metrics)
                completed_count += 1
                logger.info(f"Progress: {completed_count}/{len(tasks_to_eval)} tasks completed")
            except Exception as e:
                logger.error(f"Task #{task_number} failed with exception: {e}")
                completed_count += 1

    # Sort by task_number
    results.sort(key=lambda x: x.task_number)

    return results


def generate_report(results: List[EvaluationMetrics], output_file: Optional[str] = None) -> Dict:
    """
    Generate evaluation report

    Args:
        results: List of EvaluationMetrics
        output_file: Optional file to save report

    Returns:
        Report dictionary
    """
    if not results:
        logger.warning("No results to report")
        return {}

    total_tasks = len(results)

    # Calculate success rates
    compile_success_count = sum(1 for r in results if r.compile_success)
    module_passes_count = sum(1 for r in results if r.module_passes)
    agreement_80_count = sum(1 for r in results if r.mutant_agreement_80)
    agreement_90_count = sum(1 for r in results if r.mutant_agreement_90)
    agreement_100_count = sum(1 for r in results if r.mutant_agreement_100)
    overall_success_count = sum(1 for r in results if r.overall_success)

    # Strength stats (only meaningful when strength analysis ran)
    strength_ran = any(r.true_mutation_score >= 0 for r in results)
    total_equivalent = sum(r.equivalent_mutants for r in results)
    total_weak = sum(r.weak_survivors for r in results)
    total_unknown = sum(r.unknown_survivors for r in results)
    total_killed = sum(r.mutants_detected for r in results)
    total_mutants_all = sum(r.total_mutants for r in results)
    total_noncompiling = sum(r.noncompiling_mutants for r in results)
    non_equiv = total_mutants_all - total_equivalent
    micro_true_score = (total_killed / non_equiv) if non_equiv > 0 else None

    report = {
        "evaluation_date": datetime.now().isoformat(),
        "summary": {
            "total_tasks": total_tasks,
            "eval0_compile_success": {
                "count": compile_success_count,
                "rate": compile_success_count / total_tasks
            },
            "eval1_module_passes": {
                "count": module_passes_count,
                "rate": module_passes_count / total_tasks
            },
            "eval2_mutant_agreement": {
                "80_percent": {
                    "count": agreement_80_count,
                    "rate": agreement_80_count / total_tasks
                },
                "90_percent": {
                    "count": agreement_90_count,
                    "rate": agreement_90_count / total_tasks
                },
                "100_percent": {
                    "count": agreement_100_count,
                    "rate": agreement_100_count / total_tasks
                }
            },
            "overall_success": {
                "count": overall_success_count,
                "rate": overall_success_count / total_tasks
            },
            "testbench_strength": ({
                "analyzed": True,
                "total_mutants": total_mutants_all,
                "killed": total_killed,
                "noncompiling_mutants": total_noncompiling,
                "equivalent_mutants": total_equivalent,
                "weak_survivors": total_weak,
                "unknown_survivors": total_unknown,
                "true_mutation_score_micro": micro_true_score,
                "note": ("true score excludes equivalent mutants; weak_survivors "
                         "are testbench holes with extractable witnesses")
            } if strength_ran else {"analyzed": False})
        },
        "detailed_results": [
            {
                "task_id": r.task_id,
                "task_number": r.task_number,
                "compile_success": r.compile_success,
                "module_passes": r.module_passes,
                "mutants_detected": f"{r.mutants_detected}/{r.total_mutants}",
                "mutant_agreement_80": r.mutant_agreement_80,
                "mutant_agreement_90": r.mutant_agreement_90,
                "mutant_agreement_100": r.mutant_agreement_100,
                "overall_success": r.overall_success,
                "equivalent_mutants": r.equivalent_mutants,
                "weak_survivors": r.weak_survivors,
                "unknown_survivors": r.unknown_survivors,
                "noncompiling_mutants": r.noncompiling_mutants,
                "true_mutation_score": r.true_mutation_score
            }
            for r in results
        ]
    }

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION REPORT")
    logger.info("="*80)
    logger.info(f"Total Tasks: {total_tasks}")
    logger.info(f"")
    logger.info(f"eval0 - Compilation Success: {compile_success_count}/{total_tasks} ({compile_success_count/total_tasks:.1%})")
    logger.info(f"eval1 - Module Passes: {module_passes_count}/{total_tasks} ({module_passes_count/total_tasks:.1%})")
    logger.info(f"eval2 - Mutant Agreement:")
    logger.info(f"  80%+: {agreement_80_count}/{total_tasks} ({agreement_80_count/total_tasks:.1%})")
    logger.info(f"  90%+: {agreement_90_count}/{total_tasks} ({agreement_90_count/total_tasks:.1%})")
    logger.info(f"  100%: {agreement_100_count}/{total_tasks} ({agreement_100_count/total_tasks:.1%})")
    logger.info(f"")
    logger.info(f"Overall Success Rate: {overall_success_count}/{total_tasks} ({overall_success_count/total_tasks:.1%})")
    if strength_ran:
        logger.info(f"")
        logger.info(f"Testbench Strength (independent differential search):")
        logger.info(f"  Mutants killed:      {total_killed}/{total_mutants_all}")
        logger.info(f"  Non-compiling:       {total_noncompiling} (killed by compiler, not testbench)")
        logger.info(f"  Equivalent mutants:  {total_equivalent} (excluded from true score)")
        logger.info(f"  Weak-survivors:      {total_weak} (testbench holes; witnesses extracted)")
        logger.info(f"  Unknown:             {total_unknown} (differential search could not run)")
        if micro_true_score is not None:
            logger.info(f"  TRUE mutation score: {micro_true_score:.1%} "
                        f"(killed / non-equivalent)")
    logger.info("="*80)

    # Save report
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {output_path}")

    return report


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate generated testbenches with mutant detection (parallel)"
    )

    parser.add_argument(
        "--output", "-o",
        default="evaluation_report.json",
        help="Output report file (default: evaluation_report.json)"
    )
    parser.add_argument(
        "--experiment_dir", "-e",
        default="outputs/pro_v_rl",
        help="Path to experiment outputs directory (e.g., outputs/fine_thinking_8B)"
    )
    parser.add_argument(
        "--benchmark_file", "-b",
        default="verilog-eval/HDLBits/merged_benchmark.json",
        help="Path to labeled benchmark JSON with mutants/result labels"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of tasks to evaluate"
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Starting task index (default: 0)"
    )
    parser.add_argument(
        "--task_numbers",
        type=str,
        default="",
        help="Comma-separated 1-based task numbers to evaluate exactly"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--mutant-only", "-m",
        action="store_true",
        help="Only test mutants, skip module_code testing"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=600,
        help="Simulation timeout in seconds (default: 600). Sequential Verilator "
             "builds take ~2 min in isolation and slow further under parallel load; "
             "the old 180s default timed them out and mislabeled passing tasks as "
             "compile failures (false eval0/eval1 negatives)."
    )
    parser.add_argument(
        "--strength",
        action="store_true",
        help="Analyze testbench strength: classify surviving mutants as "
             "equivalent vs weak-survivor via independent differential search, "
             "compute a true mutation score, and extract witnesses (needs iverilog)"
    )
    parser.add_argument(
        "--augment-stimulus",
        action="store_true",
        help="With --strength: inject extracted witnesses back into each task's "
             "stimulus.json (re-run PyChecker afterwards to refresh testbench.json)"
    )
    parser.add_argument(
        "--max-exhaustive-bits",
        type=int,
        default=16,
        help="CMB: exhaustively sweep inputs when total input width <= this "
             "(proves equivalence); otherwise random-sample (default: 16)"
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=50000,
        help="CMB: number of random probe vectors when not exhaustive (default: 50000)"
    )
    parser.add_argument(
        "--exclude-noncompiling",
        action="store_true",
        help="Stricter eval2: drop mutants that fail to COMPILE from agreement and "
             "true-mutation-score (their failure is the compiler's, not the testbench's)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    strength_opts = None
    if args.strength:
        strength_opts = {
            "enabled": True,
            "augment_stimulus": args.augment_stimulus,
            "max_exhaustive_bits": args.max_exhaustive_bits,
            "random_samples": args.random_samples,
        }

    task_numbers = [
        int(x.strip()) for x in (args.task_numbers or "").split(",") if x.strip()
    ]

    # Run evaluation
    results = evaluate_all_tasks(
        benchmark_file=args.benchmark_file,
        experiment_dir=args.experiment_dir,
        mutant_only=args.mutant_only,
        limit=args.limit,
        start_idx=args.start,
        num_workers=args.workers,
        timeout=args.timeout,
        strength_opts=strength_opts,
        exclude_noncompiling=args.exclude_noncompiling,
        task_numbers=task_numbers or None
    )

    # Generate report
    generate_report(results, output_file=args.output)


if __name__ == "__main__":
    main()
