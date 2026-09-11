#!/usr/bin/env python3
"""
iverilog_eval.py -- eval0/eval1/eval2 via the (working) iverilog path.

The deployed Verilator scoring harness (sim_cmb/rfuzz-harness.cpp) is broken on
this box: it is a static template hardcoded with ports a/b/sel/out that never get
substituted with each task's real ports, so every compile fails. This bypasses it
entirely and scores through iverilog using the pipeline's already-generated,
judge-selected testbench (testbench_{idx}.json), the same 4-state path verified
end-to-end elsewhere.

Per task (that produced a trusted oracle):
  eval0 = correct RTL + selected testbench COMPILES
  eval1 = correct RTL PASSES the FRM's testbench  (FRM oracle is valid)
  eval2 = FRM testbench's mutant detection vs the benchmark's result labels
  true mutation score = killed_of_killable / killable   (labels: True=killable)
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Pro-V"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pro_v.verilog_tb_generator import generate_tb, run_iverilog_tb  # noqa: E402


def _compiled(log: str, passed: bool) -> bool:
    # iverilog TB prints ALL_PASS / MISMATCH / FAIL only after a successful compile+run
    if passed:
        return True
    return any(m in log for m in ("ALL_PASS", "MISMATCH", "FAIL "))


def _run(dut_code: str, tb_code: str):
    passed, log = run_iverilog_tb(dut_code, tb_code)
    return _compiled(log, passed), passed, log


def evaluate(exp_dir: str, bench_path: str, limit=None, verbose=False):
    bench = {t["task_number"]: t for t in json.load(open(bench_path))}
    task_dirs = sorted(glob.glob(os.path.join(exp_dir, "task_*")),
                       key=lambda p: int(p.rsplit("_", 1)[-1]) if p.rsplit("_", 1)[-1].isdigit() else 0)
    rows = []
    n_oracle = 0
    for td in task_dirs:
        tail = td.rsplit("_", 1)[-1]
        if not tail.isdigit():
            continue
        n = int(tail)
        bt = bench.get(n)
        trp = os.path.join(td, "task_result.json")
        if not bt or not os.path.exists(trp):
            continue
        tr = json.load(open(trp))
        idx = tr.get("selected_sample_idx")
        row = {"task": n, "has_oracle": idx is not None}
        if idx is None:
            rows.append(row)
            continue
        tbj = os.path.join(td, "testbench_%s.json" % idx)
        dut_path = os.path.join(td, "module_code.v")
        if not (os.path.exists(tbj) and os.path.exists(dut_path)):
            row["has_oracle"] = False
            rows.append(row)
            continue
        n_oracle += 1
        module_code = bt["module_code"]
        try:
            tb_code = generate_tb(tbj, dut_path)
        except Exception as e:
            row.update(error="tbgen: %s" % e)
            rows.append(row)
            continue
        comp, passed, log = _run(module_code, tb_code)
        row["eval0_compile"] = comp
        row["eval1_module_passes"] = passed
        mutants = bt.get("mutants", []) or []
        labels = bt.get("result", []) or []
        killed = []
        if passed:  # only meaningful to test mutants if the oracle accepts the correct RTL
            for m in mutants:
                c, p, _l = _run(m, tb_code)
                killed.append((not p) and c)
        row["n_mutants"] = len(mutants)
        row["killed"] = sum(killed)
        killable = [i for i, lab in enumerate(labels) if lab]
        row["killable"] = len(killable)
        row["killed_of_killable"] = sum(1 for i in killable if i < len(killed) and killed[i])
        row["agree100"] = bool(killed) and all(
            bool(killed[i]) == bool(labels[i]) for i in range(min(len(killed), len(labels))))
        rows.append(row)
        if verbose:
            print("task %3d: oracle=%s e0=%s e1=%s killed=%d/%d killable=%d" % (
                n, row["has_oracle"], row.get("eval0_compile"), row.get("eval1_module_passes"),
                row["killed"], row["n_mutants"], row["killable"]), flush=True)
        if limit and n_oracle >= limit:
            break
    return summarize(rows)


def summarize(rows):
    total = len(rows)
    oracle = [r for r in rows if r.get("has_oracle")]
    e0 = [r for r in oracle if r.get("eval0_compile")]
    e1 = [r for r in oracle if r.get("eval1_module_passes")]
    agree = [r for r in e1 if r.get("agree100")]
    tot_killable = sum(r.get("killable", 0) for r in e1)
    tot_killed_of_killable = sum(r.get("killed_of_killable", 0) for r in e1)
    tms = (tot_killed_of_killable / tot_killable) if tot_killable else None
    summary = {
        "tasks_evaluated": total,
        "tasks_with_oracle": len(oracle),
        "eval0_compile": "%d/%d" % (len(e0), len(oracle)),
        "eval1_module_passes": "%d/%d" % (len(e1), len(oracle)),
        "eval2_agreement_100": "%d/%d" % (len(agree), len(e1)),
        "true_mutation_score": round(tms, 4) if tms is not None else None,
        "killed_of_killable": "%d/%d" % (tot_killed_of_killable, tot_killable),
        "rows": rows,
    }
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", default="outputs/hdlbits_vlsi")
    ap.add_argument("--benchmark", default="verilog-eval/HDLBits/merged_benchmark.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default="iverilog_eval_report.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    rep = evaluate(a.experiment_dir, a.benchmark, limit=a.limit, verbose=a.verbose)
    with open(a.output, "w") as f:
        json.dump(rep, f, indent=2)
    print("\n=== iverilog eval summary ===")
    for k, v in rep.items():
        if k != "rows":
            print("  %-22s %s" % (k, v))
    print("  report -> %s" % a.output)
