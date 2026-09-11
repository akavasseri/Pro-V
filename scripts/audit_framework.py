#!/usr/bin/env python3
"""Audit Pro-V source invariants and experiment results before long runs."""

import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Audit:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checks = 0

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.error(message)


def python_files():
    return sorted((ROOT / "pro_v").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))


def audit_python(audit):
    for path in python_files():
        relative = path.relative_to(ROOT)
        try:
            source = path.read_text()
            tree = ast.parse(source, filename=str(relative))
            compile(source, str(relative), "exec")
        except Exception as exc:
            audit.error(f"{relative}: Python parse/compile failed: {exc}")
            continue
        audit.checks += 1

        seen_functions = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in seen_functions:
                    audit.error(f"{relative}:{node.lineno}: duplicate top-level function {node.name}")
                seen_functions.add(node.name)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = list(node.args.defaults) + [value for value in node.args.kw_defaults if value]
                if any(isinstance(value, (ast.List, ast.Dict, ast.Set)) for value in defaults):
                    audit.error(f"{relative}:{node.lineno}: mutable default argument in {node.name}")
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_subprocess = (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr in {"run", "call", "check_call", "check_output"}
            )
            if is_subprocess and not any(keyword.arg == "timeout" for keyword in node.keywords):
                audit.warn(f"{relative}:{node.lineno}: subprocess.{func.attr} has no timeout")


def audit_shell(audit):
    paths = sorted((ROOT / "scripts").glob("*.sh")) + sorted((ROOT / "scripts").glob("*.sbatch"))
    for path in paths:
        proc = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, timeout=15
        )
        audit.check(proc.returncode == 0, f"{path.relative_to(ROOT)}: shell syntax failed: {proc.stderr.strip()}")


def audit_pipeline_invariants(audit):
    required_paths = {
        "pipeline": ROOT / "pro_v/prompting_top_agent_ray.py",
        "judge": ROOT / "pro_v/agent/judge.py",
        "merge": ROOT / "scripts/merge_task_results.py",
        "run_script": ROOT / "scripts/run_evaluation_think.sh",
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    for name in missing:
        audit.error(f"missing required reproducibility file: {required_paths[name].relative_to(ROOT)}")
    if missing:
        return
    pipeline = required_paths["pipeline"].read_text()
    judge = required_paths["judge"].read_text()
    merge = required_paths["merge"].read_text()

    required_pipeline_fragments = {
        'PRO_V_UNSAFE_ALLOW_GOLDEN_OUTPUTS", "0"': "unsafe golden-output mode must default off",
        'PRO_V_ALLOW_DIRECT_ORACLE_SEED", "0"': "direct oracle seed must default off",
        'PRO_V_REFRESH_EXPECTED_FROM_RTL", "0"': "golden output refresh must default off",
        'PRO_V_PRE_EVAL_GOLDEN_CHECK", "0"': "pre-eval golden gate must default off",
        'PRO_V_JUDGE_SPEC_EDIT_LOOPS", "1"': "honest spec-only judge refinement must default on",
        'failure_feedback=json.dumps(feedback, indent=2)': "judge refinement must pass explicit critique feedback",
        'rtl_code="",': "judge refinement must stay RTL-blind in honest mode",
        'PRO_V_ALLOW_RTL_GUIDED_REPAIR", "0"': "RTL-guided repair must require explicit unsafe opt-in",
        'if not simulation_metrics["eval1_module_passes"]': "eval2 must be gated by eval1",
        "_calculate_mutant_agreement(": "mutant agreement must use validity-aware scoring",
        'result["pipeline_completed"] = True': "pipeline completion must be recorded separately",
        'result["success"] = bool(simulation_metrics.get("overall_success"))': "task success must reflect evaluation success",
    }
    for fragment, message in required_pipeline_fragments.items():
        audit.check(fragment in pipeline, message)
    audit.check(
        "if module_code and testbench_code" not in pipeline,
        "evaluation must not require an unused external testbench.v",
    )
    run_script = required_paths["run_script"].read_text()
    audit.check(
        '--benchmark_path "${FOLDER_PATH}"' in run_script,
        "run script must pass benchmark_path explicitly",
    )
    audit.check(
        '--benchmark_format "${BENCHMARK_FORMAT}"' in run_script,
        "run script must pass benchmark_format explicitly for RTLLM/Veri-Sure reproducibility",
    )

    token_match = re.search(r'PRO_V_JUDGE_MAX_TOKENS",\s*"(\d+)"', judge)
    audit.check(bool(token_match), "judge token budget default is not discoverable")
    if token_match:
        audit.check(int(token_match.group(1)) >= 384, "judge token budget is too small for its JSON schema")

    score_match = re.search(r"return\s*\(\s*int\(overall\),(.*?)\)", merge, flags=re.S)
    audit.check(bool(score_match), "merge score ordering is not discoverable")


def audit_duplicate_entrypoints(audit):
    pairs = [
        ("gen_tb.py", "pro_v/agent/gen_tb.py"),
        ("pychecker.py", "pro_v/agent/pychecker.py"),
        ("judge.py", "pro_v/agent/judge.py"),
        ("prompting_top_agent_ray.py", "pro_v/prompting_top_agent_ray.py"),
        ("run_evaluation_think.sh", "scripts/run_evaluation_think.sh"),
        ("run_prov_gpu80_eval.sbatch", "scripts/run_prov_gpu80_eval.sbatch"),
    ]
    for legacy_name, canonical_name in pairs:
        legacy = ROOT / legacy_name
        canonical = ROOT / canonical_name
        if legacy.exists() and canonical.exists() and legacy.read_bytes() != canonical.read_bytes():
            audit.warn(f"legacy entry point {legacy_name} differs from canonical {canonical_name}")


def has_expected_output(value):
    if isinstance(value, dict):
        if value and all(isinstance(item, str) and item and set(item) <= {"0", "1"} for item in value.values()):
            return True
        return any(has_expected_output(item) for item in value.values())
    if isinstance(value, list):
        return any(has_expected_output(item) for item in value)
    return False


def iter_result_paths(inputs):
    for raw in inputs:
        path = Path(raw)
        if path.is_file() and path.name == "task_result.json":
            yield path
        elif path.is_dir():
            yield from sorted(path.glob("task_*/task_result.json"))


def audit_results(audit, inputs):
    for path in iter_result_paths(inputs):
        try:
            result = json.loads(path.read_text())
        except Exception as exc:
            audit.error(f"{path}: invalid result JSON: {exc}")
            continue
        task = result.get("task_number", path.parent.name)
        metrics = result.get("simulation_metrics") or {}
        eval1 = metrics.get("eval1_module_passes")
        eval2 = metrics.get("eval2_mutant_detection") or {}
        rate = eval2.get("agreement_rate")
        total = eval2.get("total_mutants") or 0
        details = eval2.get("mutant_eval_details") or []
        valid = sum(item.get("status") == "simulated" for item in details)

        if eval1 is not True and isinstance(rate, (int, float)) and rate > 0:
            audit.error(f"task {task}: eval2 has credit despite eval1 failure")
        if total and details and isinstance(rate, (int, float)) and rate > valid / total + 1e-12:
            audit.error(f"task {task}: eval2 rate {rate:.3f} exceeds maximum from {valid}/{total} valid simulations")
        if eval2.get("invalid_mutant_evals") and not details:
            audit.warn(f"task {task}: invalid mutant count has no per-mutant diagnostics")
        if result.get("success") and metrics and not metrics.get("overall_success"):
            audit.warn(f"task {task}: pipeline success is true while evaluation success is false")

        for sample in result.get("pychecker_results") or []:
            tb_path = Path(sample.get("testbench_json_path", ""))
            if not tb_path.exists():
                audit.error(f"task {task}: selected candidate testbench path is missing: {tb_path}")
                continue
            try:
                testbench = json.loads(tb_path.read_text())
            except Exception as exc:
                audit.error(f"task {task}: invalid candidate testbench JSON: {exc}")
                continue
            if not has_expected_output(testbench):
                audit.error(f"task {task}: candidate testbench has no real expected outputs")


def audit_benchmark(audit, benchmark_path, benchmark_format):
    if not benchmark_path:
        return
    try:
        from pro_v.benchmark_adapters import extract_module_name, load_benchmark_tasks
        from pro_v.mutation_strength import parse_ports
        from pro_v.verification_contract import build_contract
    except Exception:
        benchmark_adapters = _load_module("benchmark_adapters", ROOT / "pro_v/benchmark_adapters.py")
        mutation_strength = _load_module("mutation_strength", ROOT / "pro_v/mutation_strength.py")
        verification_contract = _load_module("verification_contract", ROOT / "pro_v/verification_contract.py")
        extract_module_name = benchmark_adapters.extract_module_name
        load_benchmark_tasks = benchmark_adapters.load_benchmark_tasks
        parse_ports = mutation_strength.parse_ports
        build_contract = verification_contract.build_contract

    try:
        tasks = load_benchmark_tasks(benchmark_path, benchmark_format)
    except Exception as exc:
        audit.error(f"benchmark adapter failed: {exc}")
        return
    audit.check(bool(tasks), "benchmark adapter returned no tasks")

    fake_port_terms = {
        "active", "chip", "defined", "input", "output", "product", "flag",
        "signal", "with", "width", "data", "clock", "reset",
    }
    unsupported_terms = {
        "interface": r"\binterface\b",
        "modport": r"\bmodport\b",
        "package": r"\bpackage\b",
        "typedef_struct": r"\btypedef\s+struct\b",
    }

    seen_numbers = set()
    for idx, task in enumerate(tasks, 1):
        label = task.get("task_id") or task.get("task_number") or idx
        number = task.get("task_number")
        if number in seen_numbers:
            audit.error(f"benchmark task {label}: duplicate task_number {number}")
        seen_numbers.add(number)

        description = str(task.get("description") or "")
        header = str(task.get("header") or "")
        rtl = str(task.get("module_code") or "")
        audit.check(bool(description.strip()), f"benchmark task {label}: missing description/spec")
        audit.check(bool(header.strip()), f"benchmark task {label}: missing module header/interface")
        audit.check(bool(rtl.strip()), f"benchmark task {label}: missing reference RTL/module_code")

        source = "\n".join([header, rtl])
        lowered = source.lower()
        for name, pattern in unsupported_terms.items():
            if re.search(pattern, lowered):
                audit.warn(f"benchmark task {label}: contains SystemVerilog {name}; current harness support is limited")
        if re.search(r"\binout\b", lowered):
            audit.warn(f"benchmark task {label}: contains inout port; current harness treats bidirectional ports conservatively")

        try:
            ports = parse_ports(header or rtl)
        except Exception as exc:
            audit.error(f"benchmark task {label}: shared parser failed: {exc}")
            continue
        if not ports.outputs:
            audit.error(f"benchmark task {label}: parser found no outputs")

        try:
            contract = build_contract(description, header or rtl)
            contract_outputs = set((contract.get("ports") or {}).get("outputs") or {})
        except Exception as exc:
            audit.error(f"benchmark task {label}: contract build failed: {exc}")
            contract_outputs = set()
        parser_outputs = {name for name, _ in ports.outputs}
        if contract_outputs and contract_outputs != parser_outputs:
            audit.error(
                f"benchmark task {label}: contract/parser output mismatch "
                f"{sorted(contract_outputs)} != {sorted(parser_outputs)}"
            )
        suspicious = {name for name in contract_outputs if name.lower() in fake_port_terms and name not in parser_outputs}
        if suspicious:
            audit.error(f"benchmark task {label}: suspicious comment/prose-derived output ports {sorted(suspicious)}")

        module_name = extract_module_name(rtl)
        if module_name and module_name != "top_module":
            audit.error(f"benchmark task {label}: adapter did not normalize module {module_name!r} to top_module")

        mutants = task.get("mutants") or []
        labels = task.get("result") or []
        if mutants and labels and len(mutants) != len(labels):
            audit.error(f"benchmark task {label}: mutants/result label length mismatch {len(mutants)} != {len(labels)}")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", help="Optional experiment folders or task_result.json files")
    parser.add_argument("--benchmark", help="Optional benchmark path to adapter/parser-audit before a long run")
    parser.add_argument(
        "--benchmark_format",
        default="auto",
        choices=[
            "auto", "json", "hdlbits", "hdlbits_json", "rtllm",
            "rtllm_folder", "verilog_eval", "verisure", "folder",
        ],
    )
    parser.add_argument("--strict-warnings", action="store_true")
    args = parser.parse_args()

    audit = Audit()
    audit_python(audit)
    audit_shell(audit)
    audit_pipeline_invariants(audit)
    audit_duplicate_entrypoints(audit)
    audit_benchmark(audit, args.benchmark, args.benchmark_format)
    audit_results(audit, args.results)

    for message in audit.errors:
        print(f"ERROR: {message}")
    for message in audit.warnings:
        print(f"WARN: {message}")
    print(f"audit: {audit.checks} checks, {len(audit.errors)} errors, {len(audit.warnings)} warnings")
    return 1 if audit.errors or (args.strict_warnings and audit.warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
