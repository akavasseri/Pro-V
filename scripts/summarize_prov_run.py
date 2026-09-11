#!/usr/bin/env python3
"""Summarize a Pro-V experiment directory.

Usage:
  python scripts/summarize_prov_run.py outputs/run0
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys


def pct(n: int, d: int) -> str:
    return "n/a" if d == 0 else f"{100.0 * n / d:.1f}%"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_prov_run.py <outputs/experiment_dir>", file=sys.stderr)
        return 2

    base = sys.argv[1]
    paths = sorted(glob.glob(os.path.join(base, "task_*", "task_result.json")))
    rows = []
    for path in paths:
        try:
            with open(path) as f:
                result = json.load(f)
        except Exception as exc:
            print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
            continue
        sim = result.get("simulation_metrics") or {}
        eval2 = sim.get("eval2_mutant_detection") or {}
        rows.append((path, result, sim, eval2))

    print(f"experiment: {base}")
    print(f"tasks_with_result: {len(rows)}")
    if not rows:
        return 0

    overall = sum(result.get("success") is True for _, result, _, _ in rows)
    eval0_known = sum("eval0_compile_success" in sim for _, _, sim, _ in rows)
    eval0_pass = sum(sim.get("eval0_compile_success") is True for _, _, sim, _ in rows)
    eval1_known = sum("eval1_module_passes" in sim for _, _, sim, _ in rows)
    eval1_pass = sum(sim.get("eval1_module_passes") is True for _, _, sim, _ in rows)

    eval2_rates = [
        eval2.get("agreement_rate")
        for _, _, _, eval2 in rows
        if isinstance(eval2.get("agreement_rate"), (int, float))
    ]
    normalized_rates = [rate if rate <= 1 else rate / 100.0 for rate in eval2_rates]

    print(f"overall_success: {overall}/{len(rows)} ({pct(overall, len(rows))})")
    print(f"eval0_compile_success: {eval0_pass}/{eval0_known} ({pct(eval0_pass, eval0_known)})")
    print(f"eval1_module_passes: {eval1_pass}/{eval1_known} ({pct(eval1_pass, eval1_known)})")
    print(f"eval2_known: {len(eval2_rates)}")
    if eval2_rates:
        ge80 = sum(rate >= 0.8 for rate in normalized_rates)
        ge90 = sum(rate >= 0.9 for rate in normalized_rates)
        eq100 = sum(rate == 1.0 for rate in normalized_rates)
        avg = sum(normalized_rates) / len(normalized_rates)
        print(f"eval2_avg: {100.0 * avg:.1f}%")
        print(f"eval2_80: {ge80}/{len(eval2_rates)} ({pct(ge80, len(eval2_rates))})")
        print(f"eval2_90: {ge90}/{len(eval2_rates)} ({pct(ge90, len(eval2_rates))})")
        print(f"eval2_100: {eq100}/{len(eval2_rates)} ({pct(eq100, len(eval2_rates))})")

    causes = collections.Counter()
    examples = collections.defaultdict(list)
    for path, result, sim, eval2 in rows:
        if result.get("success") is True:
            continue
        task = os.path.basename(os.path.dirname(path))
        error = result.get("error") or sim.get("error") or ""
        if not sim:
            cause = "agent_or_pre_eval_failure"
        elif sim.get("eval0_compile_success") is False:
            cause = "eval0_compile_or_runtime_failure"
        elif sim.get("eval1_module_passes") is False:
            cause = "eval1_reference_or_testbench_mismatch"
        elif isinstance(eval2.get("agreement_rate"), (int, float)):
            cause = "eval2_low_agreement"
        else:
            cause = "other_failure"
        causes[cause] += 1
        if len(examples[cause]) < 8:
            examples[cause].append(f"{task}: {error[:120]}")

    print("failure_causes:")
    for cause, count in causes.most_common():
        print(f"  {cause}: {count}")
        for item in examples[cause]:
            print(f"    - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
