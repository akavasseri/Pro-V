#!/usr/bin/env python3
"""Inspect a benchmark after Pro-V adapter normalization."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pro_v.benchmark_adapters import classify_clocking, extract_module_name, load_benchmark_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_path", help="Path to HDLBits JSON file or RTLLM root directory")
    parser.add_argument(
        "--benchmark_format",
        default="auto",
        choices=["auto", "json", "hdlbits", "hdlbits_json", "rtllm", "rtllm_folder", "folder"],
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of task rows to print")
    parser.add_argument("--write_normalized", help="Optional path to write normalized JSON")
    args = parser.parse_args()

    tasks = load_benchmark_tasks(args.benchmark_path, args.benchmark_format)
    counters = Counter()
    rows = []

    for task in tasks:
        rtl = task.get("module_code") or ""
        module_name = extract_module_name(rtl)
        clocking = classify_clocking(rtl)
        mutants = task.get("mutants") or []
        result = task.get("result") or []

        counters["total"] += 1
        counters["with_description"] += int(bool(task.get("description")))
        counters["with_header"] += int(bool(task.get("header")))
        counters["with_rtl"] += int(bool(rtl))
        counters["with_testbench"] += int(bool(task.get("testbench")))
        counters["with_mutants"] += int(bool(mutants))
        counters["with_result_labels"] += int(bool(result))
        counters["sequential"] += int(bool(clocking["sequential"]))
        counters["combinational"] += int(not clocking["sequential"])
        counters["async_reset"] += int(bool(clocking["async_reset"]))
        counters["non_top_module"] += int(bool(module_name and module_name != "top_module"))

        rows.append(
            {
                "task_number": task.get("task_number"),
                "task_id": task.get("task_id"),
                "module": module_name,
                "seq": clocking["sequential"],
                "async_reset": clocking["async_reset"],
                "has_tb": bool(task.get("testbench")),
                "mutants": len(mutants),
                "labels": len(result),
                "source": task.get("source_dir") or task.get("benchmark_source"),
            }
        )

    print(json.dumps(dict(counters), indent=2))
    print()
    for row in rows[: max(0, args.limit)]:
        print(
            f"{row['task_number']:>4} {str(row['task_id'])[:45]:45} "
            f"module={row['module'] or '?':18} seq={int(row['seq'])} "
            f"async={int(row['async_reset'])} tb={int(row['has_tb'])} "
            f"mutants={row['mutants']} labels={row['labels']}"
        )

    if args.write_normalized:
        Path(args.write_normalized).write_text(json.dumps(tasks, indent=2))
        print(f"\nwrote {args.write_normalized}")

    if counters["non_top_module"]:
        print("\nwarning: some modules are not named top_module; Pro-V simulation setup may need module renaming.")
    if counters["with_mutants"] == 0:
        print("note: no mutants found; eval2 will be skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
