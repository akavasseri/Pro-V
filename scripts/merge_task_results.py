#!/usr/bin/env python3
"""Merge Pro-V experiment runs by selecting the best result per task.

Usage:
  python scripts/merge_task_results.py \
    --output outputs/merged_best \
    outputs/verilog_eval_full outputs/verilog_eval_recovery
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def task_number_from_dir(path: Path) -> int:
    return int(path.name.split("_", 1)[1])


def result_score(result: Dict[str, Any]) -> Tuple:
    """Higher is better; keep ordering stable and conservative."""
    metrics = result.get("simulation_metrics") or {}
    eval2 = metrics.get("eval2_mutant_detection") or {}
    error = result.get("error")

    judge_pass = error not in (
        "No PyChecker sample passed golden RTL simulation",
        "All PyChecker samples failed",
    )
    eval0 = bool(metrics.get("eval0_compile_success"))
    eval1 = bool(metrics.get("eval1_module_passes"))
    overall = bool(metrics.get("overall_success"))
    agreement = eval2.get("agreement_rate")
    if not isinstance(agreement, (int, float)):
        agreement = -1.0
    mutants = eval2.get("mutants_detected")
    if not isinstance(mutants, int):
        mutants = -1
    total_mutants = eval2.get("total_mutants")
    if not isinstance(total_mutants, int):
        total_mutants = -1

    return (
        int(overall),
        int(eval0),
        int(eval1),
        int(judge_pass),
        float(agreement),
        int(mutants),
        int(total_mutants),
        int(bool(result.get("success"))),
    )


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)

    def agreement(row: Dict[str, Any]) -> float:
        metrics = row.get("simulation_metrics") or {}
        eval2 = metrics.get("eval2_mutant_detection") or {}
        value = eval2.get("agreement_rate")
        return value if isinstance(value, (int, float)) else -1.0

    def metric(row: Dict[str, Any], key: str) -> bool:
        return bool((row.get("simulation_metrics") or {}).get(key))

    eval2_values = [agreement(row) for row in rows]
    return {
        "total_tasks": total,
        "successful_tasks": sum(bool(row.get("success")) for row in rows),
        "eval0_compile_success": sum(metric(row, "eval0_compile_success") for row in rows),
        "eval1_module_passes": sum(metric(row, "eval1_module_passes") for row in rows),
        "overall_success": sum(metric(row, "overall_success") for row in rows),
        "eval2_ge_80": sum(value >= 0.80 for value in eval2_values),
        "eval2_ge_90": sum(value >= 0.90 for value in eval2_values),
        "eval2_eq_100": sum(value >= 1.00 for value in eval2_values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument(
        "--copy-task-dirs",
        action="store_true",
        help="Copy selected task directories into the output folder.",
    )
    args = parser.parse_args()

    candidates: Dict[int, List[Tuple[Path, Dict[str, Any]]]] = {}
    for exp in args.experiments:
        if not exp.exists():
            raise SystemExit(f"missing experiment directory: {exp}")
        for result_path in exp.glob("task_*/task_result.json"):
            task_dir = result_path.parent
            task_num = task_number_from_dir(task_dir)
            candidates.setdefault(task_num, []).append((task_dir, load_json(result_path)))

    if not candidates:
        raise SystemExit("no task_result.json files found")

    args.output.mkdir(parents=True, exist_ok=True)

    selected = {}
    rows = []
    for task_num in sorted(candidates):
        task_dir, result = max(candidates[task_num], key=lambda item: result_score(item[1]))
        selected[task_num] = {
            "task_dir": str(task_dir),
            "score": result_score(result),
            "error": result.get("error"),
        }
        row = dict(result)
        row["merged_from"] = str(task_dir)
        rows.append(row)

        if args.copy_task_dirs:
            dest = args.output / f"task_{task_num}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(task_dir, dest)

    summary = summarize(rows)
    summary["selected_tasks"] = selected

    with (args.output / "merged_task_results.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (args.output / "overall_stats.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
