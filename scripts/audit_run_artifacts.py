#!/usr/bin/env python3
"""Audit Pro-V run artifacts for broken/empty outputs and eval-path invariants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _is_empty(value: Any) -> bool:
    return value is None or value == {} or value == []


def _check_expected_outputs(task: int, path: str, data: Any, issues: list[tuple]) -> None:
    name = os.path.basename(path)
    if not isinstance(data, list) or not data:
        issues.append((task, name, "testbench_not_nonempty_list", type(data).__name__))
        return

    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            issues.append((task, name, "entry_not_dict", idx))
            return
        expected = row.get("expected_outputs")
        if _is_empty(expected):
            issues.append((task, name, "empty_expected_outputs", idx))
            return
        if isinstance(expected, list):
            for cycle_idx, cycle in enumerate(expected):
                if not isinstance(cycle, dict) or not cycle:
                    issues.append((task, name, "empty_seq_cycle_outputs", (idx, cycle_idx)))
                    return
                for phase, outputs in cycle.items():
                    if _is_empty(outputs):
                        issues.append((task, name, "empty_seq_phase_outputs", (idx, cycle_idx, phase)))
                        return
        elif isinstance(expected, dict):
            for out_name, value in expected.items():
                if value is None or value == "":
                    issues.append((task, name, "empty_output_value", (idx, out_name)))
                    return
        else:
            issues.append((task, name, "expected_outputs_bad_type", type(expected).__name__))
            return


def audit_run(run_dir: str) -> tuple[dict, list[tuple]]:
    issues: list[tuple] = []
    summary = {
        "tasks": 0,
        "task_results": 0,
        "pipeline_completed": 0,
        "eval0_fail": 0,
        "eval1_fail": 0,
        "eval2_skipped": 0,
        "coverage_reports_missing": 0,
        "unsafe_golden_policy": 0,
    }

    task_dirs = sorted(
        glob.glob(os.path.join(run_dir, "task_*")),
        key=lambda p: int(re.search(r"task_(\d+)$", p).group(1)),
    )
    for task_dir in task_dirs:
        match = re.search(r"task_(\d+)$", task_dir)
        if not match:
            continue
        task = int(match.group(1))
        summary["tasks"] += 1

        result_path = os.path.join(task_dir, "task_result.json")
        result = None
        if os.path.exists(result_path):
            summary["task_results"] += 1
            try:
                result = _load_json(result_path)
            except Exception as exc:
                issues.append((task, "task_result.json", "bad_json", str(exc)))
        else:
            issues.append((task, "task_result.json", "missing", None))

        if isinstance(result, dict):
            if result.get("pipeline_completed"):
                summary["pipeline_completed"] += 1
                for key in ("selected_golden_path", "selected_testbench_path"):
                    selected = result.get(key)
                    if not selected or not os.path.exists(selected):
                        issues.append((task, key, "missing_selected_path", selected))

            policy = result.get("golden_output_policy") or {}
            if any(
                policy.get(k)
                for k in (
                    "refresh_expected_from_rtl",
                    "direct_oracle_seed",
                    "unsafe_golden_outputs_allowed",
                )
            ):
                summary["unsafe_golden_policy"] += 1
                issues.append((task, "golden_output_policy", "unsafe_enabled", policy))

            metrics = result.get("simulation_metrics") or {}
            eval0 = metrics.get("eval0_compile_success")
            eval1 = metrics.get("eval1_module_passes")
            eval2 = metrics.get("eval2_mutant_detection") or {}
            if eval0 is not True:
                summary["eval0_fail"] += 1
            if eval1 is not True:
                summary["eval1_fail"] += 1
            if eval2.get("skipped_invalid_sample"):
                summary["eval2_skipped"] += 1
            if eval1 is True and eval0 is not True:
                issues.append((task, "simulation_metrics", "eval1_true_but_eval0_false", metrics))
            if eval2 and eval2.get("skipped_invalid_sample") and eval1 is True:
                issues.append((task, "simulation_metrics", "eval2_skipped_despite_eval1_true", eval2))
            if eval2 and not eval2.get("skipped_invalid_sample") and eval1 is not True and eval2.get("total_mutants"):
                issues.append((task, "simulation_metrics", "eval2_not_skipped_despite_eval1_false", eval2))

            coverage = result.get("coverage_closure") or {}
            if coverage.get("enabled") and coverage.get("success"):
                augmented = coverage.get("augmented_testbench_path")
                if augmented != result.get("selected_testbench_path"):
                    issues.append((task, "coverage_closure", "selected_path_mismatch", augmented))

        for tb_path in glob.glob(os.path.join(task_dir, "testbench_*.json")):
            try:
                _check_expected_outputs(task, tb_path, _load_json(tb_path), issues)
            except Exception as exc:
                issues.append((task, os.path.basename(tb_path), "bad_json", str(exc)))

        augmented_path = os.path.join(task_dir, "coverage_augmented_testbench.json")
        if os.path.exists(augmented_path):
            try:
                _check_expected_outputs(task, augmented_path, _load_json(augmented_path), issues)
            except Exception as exc:
                issues.append((task, "coverage_augmented_testbench.json", "bad_json", str(exc)))

        if not os.path.exists(os.path.join(task_dir, "coverage_closure", "coverage_closure_report.json")):
            summary["coverage_reports_missing"] += 1

    return summary, issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to outputs/<experiment> run directory")
    parser.add_argument("--max-issues", type=int, default=100)
    args = parser.parse_args()

    summary, issues = audit_run(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"issues: {len(issues)}")
    for issue in issues[: args.max_issues]:
        print("ISSUE", repr(issue))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
