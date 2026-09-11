#!/usr/bin/env python3
"""Re-simulate labeled benchmark mutants and audit their labels."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

from pro_v.benchmark_adapters import classify_clocking, load_benchmark_tasks  # noqa: E402
from pro_v.mutation_strength import classify_cmb, classify_seq, parse_ports  # noqa: E402


def _classify(module_code: str, mutant_code: str, ctype: str, args, pass_name: str, seed: int):
    ports = parse_ports(module_code)
    if ctype.lower().startswith("seq"):
        trials = args.seq_trials
        cycles = args.seq_cycles
        if pass_name == "strong":
            trials = args.strong_seq_trials
            cycles = args.strong_seq_cycles
        return classify_seq(
            module_code,
            mutant_code,
            ports,
            seq_trials=trials,
            seq_cycles=cycles,
            timeout=args.timeout,
            rng=random.Random(seed),
        )
    samples = args.random_samples if pass_name != "strong" else args.strong_random_samples
    return classify_cmb(
        module_code,
        mutant_code,
        ports,
        max_exhaustive_bits=args.max_exhaustive_bits,
        random_samples=samples,
        timeout=args.timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_json")
    parser.add_argument("--benchmark_format", default="json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_exhaustive_bits", type=int, default=12)
    parser.add_argument("--random_samples", type=int, default=20000)
    parser.add_argument("--strong_random_samples", type=int, default=60000)
    parser.add_argument("--seq_trials", type=int, default=80)
    parser.add_argument("--seq_cycles", type=int, default=128)
    parser.add_argument("--strong_seq_trials", type=int, default=140)
    parser.add_argument("--strong_seq_cycles", type=int, default=192)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    tasks = load_benchmark_tasks(args.benchmark_json, args.benchmark_format)
    rows = []
    totals = {
        "tasks": len(tasks),
        "tasks_with_mutants": 0,
        "mutants": 0,
        "stored_non_equivalent": 0,
        "stored_equivalent": 0,
        "baseline_matches": 0,
        "repeat_matches": 0,
        "strong_matches": 0,
        "unknown_audits": 0,
        "mismatches": 0,
    }

    for task in tasks:
        module_code = task.get("module_code") or ""
        mutants = task.get("mutants") or []
        labels = task.get("result") or []
        if not mutants:
            continue
        totals["tasks_with_mutants"] += 1
        ctype = "SEQ" if classify_clocking(module_code).startswith("seq") else "CMB"
        for idx, mutant in enumerate(mutants):
            stored = bool(labels[idx]) if idx < len(labels) else None
            if stored is True:
                totals["stored_non_equivalent"] += 1
            elif stored is False:
                totals["stored_equivalent"] += 1

            row = {
                "task_number": task.get("task_number"),
                "task_id": task.get("task_id"),
                "mutant_idx": idx,
                "circuit_type": ctype,
                "stored_label_non_equivalent": stored,
                "passes": {},
            }
            all_match = True
            for pass_name, seed in (("baseline", 0xC0FFEE), ("repeat", 0xBAD5EED), ("strong", 0x5EED1234)):
                verdict = _classify(module_code, mutant, ctype, args, pass_name, seed + idx)
                label = None if verdict.unknown else (not verdict.equivalent)
                matches = (label == stored) if label is not None and stored is not None else None
                row["passes"][pass_name] = {
                    "label_non_equivalent": label,
                    "matches_stored_label": matches,
                    "equivalent": verdict.equivalent,
                    "certain": verdict.certain,
                    "unknown": verdict.unknown,
                    "error": verdict.error[:500] if verdict.error else "",
                    "witness_count": len(verdict.witnesses),
                }
                if matches is True:
                    totals[f"{pass_name}_matches"] += 1
                elif matches is None:
                    totals["unknown_audits"] += 1
                    all_match = False
                else:
                    totals["mismatches"] += 1
                    all_match = False
            row["all_three_passes_match"] = all_match
            rows.append(row)
            totals["mutants"] += 1
            print(
                "task {task} mutant {idx}: stored={stored} all_match={match}".format(
                    task=task.get("task_number"), idx=idx, stored=stored, match=all_match
                ),
                flush=True,
            )

    out = {
        "benchmark_json": args.benchmark_json,
        "settings": vars(args),
        "totals": totals,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    print(json.dumps(totals, indent=2))
    return 0 if totals["mismatches"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
