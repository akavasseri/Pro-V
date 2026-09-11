#!/usr/bin/env python3
"""
compare_testbench_quality.py

Head-to-head: is the Functional Coverage Agent's testbench higher quality than
the paper's (LLM/random) testbench?

For each benchmark task we hold everything fixed EXCEPT the stimulus source and
measure how many real (non-equivalent) mutants each stimulus set kills.

    paper arm : the model's own stimulus.json  (Pro-V gen_tb -> random vectors)
    fca  arm : functional_coverage_agent driving the SAME model FRM (golden_dut.py)

Ground-truth oracle = the benchmark's `module_code` (the correct reference RTL),
used ONLY offline to decide, per stimulus set, whether a mutant is distinguished
from the reference. This never enters a deployed checker, so it does not leak.

A mutant M is:
    * EQUIVALENT   : no input distinguishes M from the reference (exhaustive/large
                     probe finds no difference)          -> excluded from the score
    * killed-by-S  : some vector in stimulus set S drives ref and M to different
                     outputs

    true_mutation_score(S) = killed_by_S / (num_mutants - num_equivalent)

We report both arms' true score, raw kills, vector counts, and the set of
mutants that one arm kills and the other misses (the money result), plus two
self-contained structural proxies (input-bit toggle, distinct output classes).

GPU-free: reuses model FRMs + stimulus already produced by a prior eval run.
Requires iverilog (differential engine) + the pro_v package on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# local imports (pro_v package)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pro_v.mutation_strength import (  # noqa: E402
    Ports, parse_ports, _rename_top, _run_iverilog, classify_mutant,
)
from pro_v.functional_coverage_agent import (  # noqa: E402
    build_plan, plan_to_stimulus, Budget,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def detect_type(header: str, module_code: str = "") -> str:
    blob = (header + "\n" + module_code).lower()
    ports = parse_ports(header or module_code)
    if ports.clk_name or "posedge" in blob or " clk" in blob or "clock" in blob:
        return "seq"
    return "cmb"


def _norm_bits(val, width: int) -> str:
    """Normalize a stimulus value to a width-bit binary string (MSB-first)."""
    s = str(val).strip()
    # strip verilog sizing like 4'b0101 / 8'hA / decimal
    if "'" in s:
        s = s.split("'", 1)[1]
        base, digits = s[0].lower(), s[1:]
        if base == "b":
            s = digits
        elif base == "h":
            s = bin(int(digits, 16))[2:]
        elif base == "d":
            s = bin(int(digits))[2:]
        else:
            s = digits
    elif set(s) <= set("01xzXZ"):
        pass  # already binary
    else:
        # decimal integer fallback
        try:
            s = bin(int(s))[2:]
        except ValueError:
            s = "0"
    s = s.replace("x", "0").replace("z", "0").replace("X", "0").replace("Z", "0")
    if len(s) < width:
        s = s.rjust(width, "0")
    elif len(s) > width:
        s = s[-width:]
    return s


def pack_vector(vec: Dict, ports: Ports) -> str:
    """Concatenate per-input binary (MSB-first, port order) into one bitstring."""
    parts = []
    for name, w in ports.inputs:
        parts.append(_norm_bits(vec.get(name, 0), w))
    return "".join(parts)


# ---------------------------------------------------------------------------
# differential kill over a GIVEN stimulus set (not exhaustive)
# ---------------------------------------------------------------------------

def _cmb_kill_tb(ports: Ports, n: int) -> str:
    tw, ow = ports.input_width, ports.output_width
    in_decls = "\n  ".join(f"reg [{w-1}:0] {nm};" for nm, w in ports.inputs)
    ref_outs = "\n  ".join(f"wire [{w-1}:0] ref_{nm};" for nm, w in ports.outputs)
    mut_outs = "\n  ".join(f"wire [{w-1}:0] mut_{nm};" for nm, w in ports.outputs)
    ref_conn = ", ".join([f".{nm}({nm})" for nm, _ in ports.inputs] +
                         [f".{nm}(ref_{nm})" for nm, _ in ports.outputs])
    mut_conn = ", ".join([f".{nm}({nm})" for nm, _ in ports.inputs] +
                         [f".{nm}(mut_{nm})" for nm, _ in ports.outputs])
    lhs = "{" + ", ".join(nm for nm, _ in ports.inputs) + "}"
    refcat = "{" + ", ".join(f"ref_{nm}" for nm, _ in ports.outputs) + "}"
    mutcat = "{" + ", ".join(f"mut_{nm}" for nm, _ in ports.outputs) + "}"
    return f"""`timescale 1ns/1ps
module diff_tb;
  reg [{tw-1}:0] probe;
  {in_decls}
  {ref_outs}
  {mut_outs}
  reg [{tw-1}:0] mem [0:{max(n,1)-1}];
  wire [{ow-1}:0] refcat = {refcat};
  wire [{ow-1}:0] mutcat = {mutcat};
  ref_module refi({ref_conn});
  mut_module muti({mut_conn});
  integer i;
  initial begin
    $readmemb("probe.mem", mem);
    for (i = 0; i < {n}; i = i + 1) begin
      probe = mem[i];
      {lhs} = probe;
      #1;
      if (refcat !== mutcat) begin $display("KILL %0d", i); $finish; end
    end
    $display("SURVIVE");
    $finish;
  end
endmodule
"""


def _seq_kill_tb(ports: Ports, n: int) -> str:
    tw, ow = ports.input_width, ports.output_width
    in_decls = "\n  ".join(f"reg [{w-1}:0] {nm};" for nm, w in ports.inputs)
    ref_outs = "\n  ".join(f"wire [{w-1}:0] ref_{nm};" for nm, w in ports.outputs)
    mut_outs = "\n  ".join(f"wire [{w-1}:0] mut_{nm};" for nm, w in ports.outputs)
    clk = ports.clk_name or "clk"
    ref_conn = ", ".join([f".{clk}(clk)"] +
                         [f".{nm}({nm})" for nm, _ in ports.inputs] +
                         [f".{nm}(ref_{nm})" for nm, _ in ports.outputs])
    mut_conn = ", ".join([f".{clk}(clk)"] +
                         [f".{nm}({nm})" for nm, _ in ports.inputs] +
                         [f".{nm}(mut_{nm})" for nm, _ in ports.outputs])
    lhs = "{" + ", ".join(nm for nm, _ in ports.inputs) + "}"
    refcat = "{" + ", ".join(f"ref_{nm}" for nm, _ in ports.outputs) + "}"
    mutcat = "{" + ", ".join(f"mut_{nm}" for nm, _ in ports.outputs) + "}"
    return f"""`timescale 1ns/1ps
module diff_tb;
  reg clk;
  reg [{tw-1}:0] probe;
  {in_decls}
  {ref_outs}
  {mut_outs}
  reg [{tw-1}:0] mem [0:{max(n,1)-1}];
  wire [{ow-1}:0] refcat = {refcat};
  wire [{ow-1}:0] mutcat = {mutcat};
  ref_module refi({ref_conn});
  mut_module muti({mut_conn});
  integer i;
  initial begin
    $readmemb("probe.mem", mem);
    clk = 0;
    for (i = 0; i < {n}; i = i + 1) begin
      probe = mem[i];
      {lhs} = probe;
      clk = 0; #1; clk = 1; #1;
      if (refcat !== mutcat) begin $display("KILL %0d", i); $finish; end
    end
    $display("SURVIVE");
    $finish;
  end
endmodule
"""


def _stimulus_to_mem(stimulus: List[dict], ports: Ports, ctype: str) -> List[str]:
    """Flatten a stimulus set to packed MSB-first binary lines for probe.mem."""
    lines: List[str] = []
    if ctype == "seq":
        for scn in stimulus:
            if "clock_cycles" in scn:  # scenario form: {clock_cycles, port:[..]}
                cyc = scn.get("clock_cycles") or max(
                    (len(v) for k, v in scn.items() if isinstance(v, list)), default=0)
                for t in range(cyc):
                    step = {nm: (scn.get(nm)[t] if isinstance(scn.get(nm), list)
                                 and t < len(scn.get(nm)) else 0)
                            for nm, _ in ports.inputs}
                    lines.append(pack_vector(step, ports))
            else:  # flat dict per cycle
                lines.append(pack_vector(scn, ports))
    else:
        for vec in stimulus:
            lines.append(pack_vector(vec, ports))
    return lines


def kills(ref_v: str, mut_v: str, ports: Ports, stimulus: List[dict],
          ctype: str, timeout: int = 60) -> Optional[bool]:
    """True if the stimulus set distinguishes mutant from reference. None=error."""
    mem = _stimulus_to_mem(stimulus, ports, ctype)
    if not mem or not ports.inputs or not ports.outputs:
        return None
    n = len(mem)
    tb = _seq_kill_tb(ports, n) if ctype == "seq" else _cmb_kill_tb(ports, n)
    sources = {
        "ref.v": _rename_top(ref_v, "ref_module"),
        "mut.v": _rename_top(mut_v, "mut_module"),
        "diff_tb.v": tb,
        "probe.mem": "\n".join(mem) + "\n",
    }
    with tempfile.TemporaryDirectory() as wd:
        ok, out = _run_iverilog(sources, wd, timeout)
    if not ok:
        return None
    return "KILL" in out


# ---------------------------------------------------------------------------
# structural proxies (self-contained, no verilator)
# ---------------------------------------------------------------------------

def input_toggle_coverage(stimulus: List[dict], ports: Ports, ctype: str) -> float:
    """Fraction of input bits observed at BOTH 0 and 1 across the stimulus set."""
    mem = _stimulus_to_mem(stimulus, ports, ctype)
    tw = ports.input_width
    if not mem or tw == 0:
        return 0.0
    saw0 = [False] * tw
    saw1 = [False] * tw
    for line in mem:
        line = line.rjust(tw, "0")[-tw:]
        for i, ch in enumerate(line):
            if ch == "0":
                saw0[i] = True
            elif ch == "1":
                saw1[i] = True
    toggled = sum(1 for i in range(tw) if saw0[i] and saw1[i])
    return toggled / tw


# ---------------------------------------------------------------------------
# per-task comparison
# ---------------------------------------------------------------------------

@dataclass
class TaskCompare:
    task_id: str
    task_number: int
    ctype: str
    n_mutants: int = 0
    n_equiv: int = 0
    n_unknown: int = 0
    n_effective: int = 0            # non-equivalent, classifiable
    paper_kills: int = 0
    fca_kills: int = 0
    paper_vecs: int = 0
    fca_vecs: int = 0
    paper_toggle: float = 0.0
    fca_toggle: float = 0.0
    fca_only: List[int] = field(default_factory=list)   # mutants FCA kills, paper misses
    paper_only: List[int] = field(default_factory=list)
    note: str = ""

    @property
    def paper_score(self) -> Optional[float]:
        return self.paper_kills / self.n_effective if self.n_effective else None

    @property
    def fca_score(self) -> Optional[float]:
        return self.fca_kills / self.n_effective if self.n_effective else None


def load_first_working_frm(task_dir: str, dut_path: str, header: str,
                           ctype: str) -> Optional[Tuple[str, Dict]]:
    """Try each golden_dut_*.py in task_dir; return (frm_path, plan) for the
    first that yields a usable coverage plan."""
    cands = sorted(glob.glob(os.path.join(task_dir, "golden_dut_*.py"))) or \
        sorted(glob.glob(os.path.join(task_dir, "golden_dut.py")))
    for frm in cands:
        try:
            # completeness policy: exhaustive <=11 input bits, else 2048-sample
            plan = build_plan(frm, dut_path,
                              budget=Budget(max_runtime_seconds=90.0),
                              force_type=ctype)
            stim = plan_to_stimulus(plan)
            if stim:
                return frm, plan
        except Exception as e:  # noqa: BLE001
            continue
    return None


def compare_task(task: Dict, task_dir: str, *, max_exhaustive_bits: int = 18,
                 verbose: bool = True) -> TaskCompare:
    tid = task.get("task_id", "?")
    tnum = task.get("task_number", -1)
    header = task.get("header", "") or task.get("module_code", "")
    module_code = task["module_code"]
    mutants = task.get("mutants", [])
    ctype = task.get("circuit_type") or detect_type(header, module_code)
    ports = parse_ports(module_code)
    tc = TaskCompare(task_id=tid, task_number=tnum, ctype=ctype,
                     n_mutants=len(mutants))

    # 1) paper stimulus
    paper_stim_path = os.path.join(task_dir, "stimulus.json")
    paper_stim: List[dict] = []
    if os.path.exists(paper_stim_path):
        try:
            paper_stim = json.load(open(paper_stim_path))
        except Exception:  # noqa: BLE001
            paper_stim = []
    tc.paper_vecs = len(paper_stim)

    # 2) FCA stimulus from the model FRM
    dut_path = os.path.join(task_dir, "sim_cmb", "top_module.v")
    if not os.path.exists(dut_path):
        # fall back: write module header as the DUT interface source
        dut_path = os.path.join(task_dir, "_dut_iface.v")
        with open(dut_path, "w") as f:
            f.write(module_code)
    frm_plan = load_first_working_frm(task_dir, dut_path, header, ctype)
    fca_stim: List[dict] = []
    if frm_plan:
        _, plan = frm_plan
        try:
            fca_stim = plan_to_stimulus(plan)
        except Exception:  # noqa: BLE001
            fca_stim = []
    else:
        tc.note = "no working FRM"
    tc.fca_vecs = len(fca_stim)

    # 3) per-mutant: equivalence (once) + kill by each arm
    for i, mut in enumerate(mutants):
        verdict = classify_mutant(module_code, mut, ctype,
                                  max_exhaustive_bits=max_exhaustive_bits)
        if verdict.unknown:
            tc.n_unknown += 1
            continue
        if verdict.equivalent:
            tc.n_equiv += 1
            continue
        tc.n_effective += 1
        pk = kills(module_code, mut, ports, paper_stim, ctype) if paper_stim else False
        fk = kills(module_code, mut, ports, fca_stim, ctype) if fca_stim else False
        if pk:
            tc.paper_kills += 1
        if fk:
            tc.fca_kills += 1
        if fk and not pk:
            tc.fca_only.append(i)
        if pk and not fk:
            tc.paper_only.append(i)

    tc.paper_toggle = input_toggle_coverage(paper_stim, ports, ctype)
    tc.fca_toggle = input_toggle_coverage(fca_stim, ports, ctype)

    if verbose:
        ps = f"{tc.paper_score:.2f}" if tc.paper_score is not None else "  - "
        fs = f"{tc.fca_score:.2f}" if tc.fca_score is not None else "  - "
        print(f"[{tnum:>3} {tid:<22}] {ctype}  eff={tc.n_effective:<2} equiv={tc.n_equiv} "
              f"| paper {tc.paper_kills}/{tc.n_effective} score={ps} vec={tc.paper_vecs} tog={tc.paper_toggle:.2f}"
              f" | FCA {tc.fca_kills}/{tc.n_effective} score={fs} vec={tc.fca_vecs} tog={tc.fca_toggle:.2f}"
              + (f" | FCA-only kills: {tc.fca_only}" if tc.fca_only else "")
              + (f" | {tc.note}" if tc.note else ""))
    return tc


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(benchmark_path: str, outputs_dir: str, task_numbers: Optional[List[int]],
        only_cmb: bool, out_json: Optional[str], max_exhaustive_bits: int) -> Dict:
    bench = json.load(open(benchmark_path))
    by_num = {t.get("task_number"): t for t in bench}
    if task_numbers:
        selected = [by_num[n] for n in task_numbers if n in by_num]
    else:
        selected = bench

    results: List[TaskCompare] = []
    t0 = time.time()
    for task in selected:
        tnum = task.get("task_number")
        task_dir = os.path.join(outputs_dir, f"task_{tnum}")
        if not os.path.isdir(task_dir):
            continue
        ctype = task.get("circuit_type") or detect_type(
            task.get("header", ""), task.get("module_code", ""))
        if only_cmb and ctype != "cmb":
            continue
        try:
            results.append(compare_task(task, task_dir,
                                        max_exhaustive_bits=max_exhaustive_bits))
        except Exception as e:  # noqa: BLE001
            print(f"[{tnum}] ERROR {type(e).__name__}: {e}")

    summary = summarize(results, time.time() - t0)
    _print_summary(summary)
    if out_json:
        payload = {"summary": summary,
                   "tasks": [vars(r) for r in results]}
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {out_json}")
    return summary


def summarize(results: List[TaskCompare], elapsed: float) -> Dict:
    scored = [r for r in results if r.n_effective > 0]
    tot_eff = sum(r.n_effective for r in scored)
    tot_paper = sum(r.paper_kills for r in scored)
    tot_fca = sum(r.fca_kills for r in scored)
    tot_equiv = sum(r.n_equiv for r in results)
    fca_wins = sum(1 for r in scored if (r.fca_score or 0) > (r.paper_score or 0))
    paper_wins = sum(1 for r in scored if (r.paper_score or 0) > (r.fca_score or 0))
    ties = len(scored) - fca_wins - paper_wins
    return {
        "n_tasks_scored": len(scored),
        "n_tasks_total": len(results),
        "total_effective_mutants": tot_eff,
        "total_equivalent_excluded": tot_equiv,
        "paper_kills": tot_paper,
        "fca_kills": tot_fca,
        "paper_true_score": (tot_paper / tot_eff) if tot_eff else None,
        "fca_true_score": (tot_fca / tot_eff) if tot_eff else None,
        "fca_wins": fca_wins,
        "paper_wins": paper_wins,
        "ties": ties,
        "fca_only_kills_total": sum(len(r.fca_only) for r in results),
        "paper_only_kills_total": sum(len(r.paper_only) for r in results),
        "mean_paper_vecs": (sum(r.paper_vecs for r in scored) / len(scored)) if scored else 0,
        "mean_fca_vecs": (sum(r.fca_vecs for r in scored) / len(scored)) if scored else 0,
        "mean_paper_toggle": (sum(r.paper_toggle for r in scored) / len(scored)) if scored else 0,
        "mean_fca_toggle": (sum(r.fca_toggle for r in scored) / len(scored)) if scored else 0,
        "elapsed_seconds": round(elapsed, 1),
    }


def _print_summary(s: Dict) -> None:
    print("\n" + "=" * 72)
    print("HEAD-TO-HEAD SUMMARY  (true mutation score = kills / non-equivalent)")
    print("=" * 72)
    ps = s["paper_true_score"]
    fs = s["fca_true_score"]
    print(f"tasks scored           : {s['n_tasks_scored']} / {s['n_tasks_total']}")
    print(f"effective mutants      : {s['total_effective_mutants']}  "
          f"(equivalent excluded: {s['total_equivalent_excluded']})")
    print(f"PAPER true score       : {ps:.3f}  ({s['paper_kills']} killed)" if ps is not None else "PAPER: n/a")
    print(f"FCA   true score       : {fs:.3f}  ({s['fca_kills']} killed)" if fs is not None else "FCA: n/a")
    if ps is not None and fs is not None:
        print(f"delta (FCA - paper)    : {fs - ps:+.3f}")
    print(f"per-task wins           : FCA {s['fca_wins']}  paper {s['paper_wins']}  ties {s['ties']}")
    print(f"FCA-only kills (missed by paper) : {s['fca_only_kills_total']}")
    print(f"paper-only kills (missed by FCA) : {s['paper_only_kills_total']}")
    print(f"mean vectors            : paper {s['mean_paper_vecs']:.1f}  FCA {s['mean_fca_vecs']:.1f}")
    print(f"mean input-toggle cov   : paper {s['mean_paper_toggle']:.2f}  FCA {s['mean_fca_toggle']:.2f}")
    print(f"elapsed                 : {s['elapsed_seconds']}s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True, help="test_benchmark_new.json")
    p.add_argument("--outputs", required=True, help="prior eval outputs dir (task_N/)")
    p.add_argument("--tasks", default=None,
                   help="comma-separated task_numbers (default: all present)")
    p.add_argument("--only-cmb", action="store_true",
                   help="restrict to combinational tasks (clean primary result)")
    p.add_argument("--max-exhaustive-bits", type=int, default=18)
    p.add_argument("--out", default=None, help="write JSON report here")
    args = p.parse_args()
    tns = [int(x) for x in args.tasks.split(",")] if args.tasks else None
    run(args.benchmark, args.outputs, tns, args.only_cmb, args.out,
        args.max_exhaustive_bits)


if __name__ == "__main__":
    main()
