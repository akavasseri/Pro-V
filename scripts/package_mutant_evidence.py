#!/usr/bin/env python3
"""Package mutant sources and Eval2 outcomes for a Pro-V run."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil
from collections import OrderedDict


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "task")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("benchmark_json")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    run = pathlib.Path(args.run_dir)
    bench_path = pathlib.Path(args.benchmark_json)
    out = pathlib.Path(args.out_dir)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    bench = json.load(open(bench_path))
    bench_by_task = {int(x["task_number"]): x for x in bench}
    all_results = json.load(open(run / "all_results.json"))
    values = all_results.get("results") if isinstance(all_results, dict) else all_results

    known = []
    for result in values:
        if not isinstance(result, dict):
            continue
        task_number = result.get("task_number")
        eval2 = (result.get("simulation_metrics") or {}).get("eval2_mutant_detection")
        if task_number is not None and eval2 and eval2.get("agreement_rate") is not None:
            known.append(int(task_number))

    summary_rows = []
    readme = [
        "# RTLLM v2 run0 mutant evidence\n\n",
        f"Source run: {run.resolve()}\n\n",
        f"Benchmark labels: {bench_path.resolve()}\n\n",
        "Label meaning: benchmark result=True means non-equivalent mutant, "
        "so it should be detected/killed. result=False means equivalent, "
        "so not detecting it is correct.\n\n",
    ]

    for task_number in sorted(known):
        task_dir = run / f"task_{task_number}"
        task_result = json.load(open(task_dir / "task_result.json"))
        eval2 = task_result["simulation_metrics"]["eval2_mutant_detection"]
        bench_task = bench_by_task[task_number]

        task_id = bench_task.get("task_id") or "task"
        task_out = out / f"task_{task_number:02d}_{_safe_name(task_id)}"
        task_out.mkdir()

        for name in [
            "task_result.json",
            "task_info.json",
            "module_code.v",
            "description.txt",
            "circuit_metadata.json",
        ]:
            src = task_dir / name
            if src.exists():
                shutil.copy2(src, task_out / src.name)

        selected = task_result.get("selected_testbench_path")
        if selected:
            selected_path = pathlib.Path(selected)
            if not selected_path.exists():
                selected_path = task_dir / selected_path.name
            if selected_path.exists():
                shutil.copy2(selected_path, task_out / "selected_testbench.json")

        mutants_dir = task_dir / "mutants"
        if mutants_dir.exists():
            shutil.copytree(mutants_dir, task_out / "mutants")

        benchmark_mutants = task_out / "benchmark_mutants"
        benchmark_mutants.mkdir()
        for idx, code in enumerate(bench_task.get("mutants") or []):
            (benchmark_mutants / f"mutant_{idx}.v").write_text(code)

        labels = bench_task.get("result") or []
        rows = []
        agreement_count = 0
        for fallback_idx, detail in enumerate(eval2.get("mutant_eval_details") or []):
            idx = int(detail.get("mutant_idx", fallback_idx))
            expected_detect = bool(labels[idx]) if idx < len(labels) else None
            actual_detect = detail.get("detected")
            matches_label = (
                actual_detect == expected_detect if expected_detect is not None else None
            )
            if matches_label:
                agreement_count += 1
            row = OrderedDict(
                [
                    ("task_number", task_number),
                    ("task_id", task_id),
                    ("mutant_idx", idx),
                    ("benchmark_result_non_equivalent_should_detect", expected_detect),
                    ("actual_detected_by_generated_tb", actual_detect),
                    ("matches_label", matches_label),
                    ("status", detail.get("status")),
                ]
            )
            rows.append(row)
            summary_rows.append(row.copy())

        if rows:
            with open(task_out / "mutant_outcomes.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        with open(task_out / "mutant_outcomes.json", "w") as f:
            json.dump(rows, f, indent=2)

        task_summary = {
            "task_number": task_number,
            "task_id": task_id,
            "agreement_rate": eval2.get("agreement_rate"),
            "mutants_detected": eval2.get("mutants_detected"),
            "total_mutants": eval2.get("total_mutants"),
            "valid_mutant_evals": eval2.get("valid_mutant_evals"),
            "expected_non_equivalent_count": sum(1 for label in labels if label),
            "expected_equivalent_count": sum(1 for label in labels if not label),
            "recomputed_label_matches": agreement_count,
            "rows": rows,
        }
        with open(task_out / "task_mutant_summary.json", "w") as f:
            json.dump(task_summary, f, indent=2)

        readme.append(
            "Task {task_number} ({task_id}): agreement={agreement}, "
            "detected={detected}/{total}, label_matches={matches}/{row_count}\n".format(
                task_number=task_number,
                task_id=task_id,
                agreement=eval2.get("agreement_rate"),
                detected=eval2.get("mutants_detected"),
                total=eval2.get("total_mutants"),
                matches=agreement_count,
                row_count=len(rows),
            )
        )

    if summary_rows:
        with open(out / "all_mutant_outcomes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    with open(out / "README.md", "w") as f:
        f.write("".join(readme))
    shutil.copy2(run / "all_results.json", out / "all_results.json")
    shutil.copy2(bench_path, out / "benchmark_labeled_clean.json")

    print(out)
    print(f"tasks {len(known)} rows {len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
