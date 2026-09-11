#!/usr/bin/env python3
"""
verilog_tb_generator.py

Emit a self-checking *Verilog* testbench from testbench.json and run it under a
4-state simulator (Icarus Verilog). This complements the existing C++/Verilator
harness (harness-generator.py), which is fast but 2-state: Verilator cannot model
X/Z propagation, uninitialised-register semantics, or tristate. A 4-state Verilog
TB catches an entire class of bugs/mutants that only manifest as X.

Two entry points:
  * generate_tb()        - build the Verilog TB string from testbench.json
  * run_iverilog_tb()    - compile DUT + TB with iverilog, run, parse pass/fail
  * differential_check() - run BOTH harnesses (Verilator 2-state + iverilog
                           4-state) on the same DUT and flag any disagreement.
                           The disagreement set is exactly where the fast
                           Verilator path is silently hiding an X-bug.

Comparisons use `!==` (4-state), so a DUT output of X against an expected 0/1 is
a mismatch (Verilator would coerce the X and pass).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from pro_v.mutation_strength import parse_ports, Ports, _run_iverilog
except Exception:  # standalone / broken package init
    import importlib.util
    _ms = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutation_strength.py")
    _spec = importlib.util.spec_from_file_location("prov_mutation_strength", _ms)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["prov_mutation_strength"] = _mod
    _spec.loader.exec_module(_mod)
    parse_ports, Ports, _run_iverilog = _mod.parse_ports, _mod.Ports, _mod._run_iverilog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bits(value: str, width: int) -> str:
    """Sanitise a binary string to exactly `width` bits (keep only 0/1)."""
    s = re.sub(r"[^01]", "0", str(value))
    if len(s) < width:
        s = s.zfill(width)
    elif len(s) > width:
        s = s[-width:]
    return s or "0" * max(1, width)


def _lit(value: str, width: int) -> str:
    return f"{width}'b{_bits(value, width)}"


def is_sequential(testbench_data: List[dict]) -> bool:
    return any(isinstance(e, dict) and "clock_cycles" in e for e in testbench_data)


# ---------------------------------------------------------------------------
# Combinational TB
# ---------------------------------------------------------------------------

def build_cmb_tb(ports: Ports, data: List[dict]) -> str:
    in_w = {n: w for n, w in ports.inputs}
    out_w = {n: w for n, w in ports.outputs}
    in_decls = "\n  ".join(f"reg [{w-1}:0] {n};" for n, w in ports.inputs)
    out_decls = "\n  ".join(f"wire [{w-1}:0] {n};" for n, w in ports.outputs)
    conn = ", ".join([f".{n}({n})" for n, _ in ports.inputs] +
                     [f".{n}({n})" for n, _ in ports.outputs])

    body = []
    for idx, tv in enumerate(data):
        ins = tv.get("inputs", {})
        outs = tv.get("expected_outputs", {})
        for n, _ in ports.inputs:
            if n in ins:
                body.append(f"    {n} = {_lit(ins[n], in_w[n])};")
        body.append("    #1;")
        for o, _ in ports.outputs:
            if o in outs:
                exp = _lit(outs[o], out_w[o])
                body.append(
                    f'    if ({o} !== {exp}) begin errors=errors+1; '
                    f'$display("MISMATCH vec {idx} {o}: got %b exp %b", {o}, {exp}); end')
    return _wrap_tb(in_decls, out_decls, conn, "\n".join(body))


# ---------------------------------------------------------------------------
# Sequential TB (best-effort: samples outputs after the rising edge, matching
# the FRM's "rising_edge" expected outputs)
# ---------------------------------------------------------------------------

def build_seq_tb(ports: Ports, data: List[dict]) -> str:
    in_w = {n: w for n, w in ports.inputs}
    out_w = {n: w for n, w in ports.outputs}
    clk = ports.clk_name or "clk"
    in_decls = "\n  ".join(f"reg [{w-1}:0] {n};" for n, w in ports.inputs)
    out_decls = "\n  ".join(f"wire [{w-1}:0] {n};" for n, w in ports.outputs)
    conn = ", ".join([f".{clk}({clk})"] +
                     [f".{n}({n})" for n, _ in ports.inputs] +
                     [f".{n}({n})" for n, _ in ports.outputs])

    body = []
    for sidx, scn in enumerate(data):
        cycles = scn.get("clock_cycles", 0)
        exp_list = scn.get("expected_outputs", [])
        for c in range(cycles):
            for n, _ in ports.inputs:
                seq = scn.get(n)
                if isinstance(seq, list) and c < len(seq):
                    body.append(f"    {n} = {_lit(seq[c], in_w[n])};")
            # one clock: rising edge then settle
            body.append(f"    {clk} = 0; #1; {clk} = 1; #1;")
            if c < len(exp_list):
                rising = exp_list[c].get("rising_edge", {}) if isinstance(exp_list[c], dict) else {}
                for o, _ in ports.outputs:
                    if o in rising:
                        exp = _lit(rising[o], out_w[o])
                        body.append(
                            f'    if ({o} !== {exp}) begin errors=errors+1; '
                            f'$display("MISMATCH scn {sidx} cyc {c} {o}: got %b exp %b", {o}, {exp}); end')
            body.append(f"    {clk} = 0; #1;")
    return _wrap_tb(in_decls, out_decls, conn, "\n".join(body), clk_reg=clk)


def _wrap_tb(in_decls: str, out_decls: str, conn: str, body: str,
             clk_reg: Optional[str] = None) -> str:
    clk_decl = f"reg {clk_reg};" if clk_reg else ""
    return f"""`timescale 1ns/1ps
module tb;
  {clk_decl}
  {in_decls}
  {out_decls}
  integer errors;
  top_module dut({conn});
  initial begin
    errors = 0;
{body}
    if (errors == 0) $display("ALL_PASS");
    else $display("FAIL %0d errors", errors);
    $finish;
  end
endmodule
"""


# ---------------------------------------------------------------------------
# Generation + run
# ---------------------------------------------------------------------------

def generate_tb(testbench_json_path: str, dut_path: str,
                circuit_type: Optional[str] = None) -> str:
    with open(testbench_json_path) as f:
        data = json.load(f)
    with open(dut_path) as f:
        ports = parse_ports(f.read())
    seq = (circuit_type == "seq") if circuit_type else is_sequential(data)
    return build_seq_tb(ports, data) if seq else build_cmb_tb(ports, data)


# ---------------------------------------------------------------------------
# coverage_plan.json  ->  testbench  (SpecKit pipeline, stage 7)
# ---------------------------------------------------------------------------
# Lower a reference-guided coverage_plan.json into an executable testbench. The
# generator invents NO test logic: the functional coverage agent already decided
# what to test and (via golden_dut.py) what the expected outputs are. Expected
# outputs come from the plan (golden-derived), NEVER from the RTL. top_module.v is
# instantiated as a black box; only its interface (parse_ports) is used.
# Every assertion preserves traceability: test name, coverage goal, witness ID,
# and rule ID all appear in the surrounding comment and the failure message.

def _lit_int(val, width: int) -> str:
    w = max(1, int(width))
    return "%d'd%d" % (w, int(val) & ((1 << w) - 1))


def _san(text: str) -> str:
    return re.sub(r'["\n\r]', " ", str(text or "")).strip()[:160]


def _goal_note(covers, goal_desc) -> Tuple[str, str]:
    cg = [c for c in covers if isinstance(c, str) and c.startswith("CG")]
    desc = goal_desc.get(cg[0], "") if cg else ""
    return (" ".join(cg), desc)


def build_tb_from_plan(plan: dict, ports: Ports, include_random: bool = False) -> str:
    """Build a 4-state Verilog testbench from a coverage_plan.json dict."""
    seq = plan.get("design_type") == "sequential"
    in_w = {n: w for n, w in ports.inputs}
    out_w = {n: w for n, w in ports.outputs}
    in_names = [n for n, _ in ports.inputs]
    clk = ports.clk_name or "clk"
    goal_desc = {g.get("id"): g.get("description", "") for g in plan.get("coverage_goals", []) or []}

    in_decls = "\n  ".join(f"reg [{w-1}:0] {n};" for n, w in ports.inputs)
    out_decls = "\n  ".join(f"wire [{w-1}:0] {n};" for n, w in ports.outputs)
    conn_sigs = ([f".{clk}({clk})"] if seq else []) + \
        [f".{n}({n})" for n, _ in ports.inputs] + [f".{n}({n})" for n, _ in ports.outputs]
    conn = ", ".join(conn_sigs)

    body: List[str] = [
        f"    // === directed tests from coverage_plan.json ===",
        f"    // design: {_san(plan.get('design_name'))} | source_of_truth: "
        f"{plan.get('source_of_truth', 'golden_dut.py')} | black-box DUT: "
        f"{plan.get('black_box_dut', 'top_module.v')}",
        f"    // expected outputs are golden-model-derived (never read from the RTL)",
    ]

    def drive_and_check(name, covers, reason, inputs, expected, cyc=None):
        cg_ids, gdesc = _goal_note(covers, goal_desc)
        loc = "" if cyc is None else f" cycle {cyc}"
        body.append("")
        body.append(f"    // Test: {name}{loc}")
        if cg_ids or gdesc:
            body.append(f"    // Goal: {cg_ids} {gdesc}".rstrip())
        if covers:
            body.append(f"    // Covers: {', '.join(str(c) for c in covers)}")
        if reason:
            body.append(f"    // Reason: {_san(reason)}")
        for n in in_names:
            if n in inputs:
                body.append(f"    {n} = {_lit_int(inputs[n], in_w.get(n, 1))};")
        if seq:
            body.append(f"    {clk} = 0; #1; {clk} = 1; #1;")
        else:
            body.append("    #1;")
        invec = " ".join(f"{k}={inputs[k]}" for k in in_names if k in inputs)
        for o, ev in (expected or {}).items():
            if o not in out_w or ev is None:
                continue
            exp = _lit_int(ev, out_w[o])
            tag = ",".join(str(c) for c in covers)
            msg = (f"FAIL {name}{loc} [{tag}] in({invec}): expected {o}={ev} got %0d"
                   f" -- {_san(reason)}")
            body.append(f'    if ({o} !== {exp}) begin errors=errors+1; '
                        f'$display("{msg}", {o}); end')
        if seq:
            body.append(f"    {clk} = 0; #1;")

    if seq:
        for s in plan.get("sequences", []) or []:
            name = s.get("name", "sequence")
            covers = s.get("covers", [])
            reason = s.get("reason", "")
            body.append("")
            body.append(f"    // ---- sequence: {name} ({', '.join(str(c) for c in covers)}) ----")
            for row in s.get("sequence", []) or []:
                drive_and_check(name, covers, reason, row.get("inputs", {}),
                                row.get("expected_outputs", {}), cyc=row.get("cycle"))
            if s.get("expected_final"):
                body.append(f"    // expected final state (model-internal; not observable on "
                            f"black-box RTL): {_san(json.dumps(s['expected_final']))}")
    else:
        for t in plan.get("tests", []) or []:
            drive_and_check(t.get("name", "test"), t.get("covers", []), t.get("reason", ""),
                            t.get("inputs", {}), t.get("expected_outputs", {}))

    # optional random section -- ONLY if explicitly enabled AND the plan carries
    # golden-derived random vectors (kept strictly after the directed tests)
    rand = plan.get("random_tests") or []
    if include_random and rand:
        body.append("")
        body.append("    // === optional random tests (golden-derived, after directed) ===")
        for t in rand:
            drive_and_check(t.get("name", "random"), t.get("covers", ["random"]),
                            t.get("reason", "random fill"), t.get("inputs", {}),
                            t.get("expected_outputs", {}))

    return _wrap_tb(in_decls, out_decls, conn, "\n".join(body), clk_reg=(clk if seq else None))


def generate_tb_from_plan(coverage_plan_path: str, dut_path: str,
                          out_path: Optional[str] = None, include_random: bool = False) -> str:
    """Read coverage_plan.json + the black-box DUT interface, emit a testbench."""
    with open(coverage_plan_path) as f:
        plan = json.load(f)
    with open(dut_path) as f:
        ports = parse_ports(f.read())
    tb = build_tb_from_plan(plan, ports, include_random=include_random)
    if out_path:
        with open(out_path, "w") as f:
            f.write(tb)
    return tb


def run_iverilog_tb(dut_code: str, tb_code: str, timeout: int = 120) -> Tuple[bool, str]:
    """Compile DUT + TB with iverilog, run, return (passed, log). 4-state."""
    sources = {"top_module.v": dut_code, "tb.v": tb_code}
    with tempfile.TemporaryDirectory() as wd:
        ok, out = _run_iverilog(sources, wd, timeout)
    if not ok:
        return False, out
    if "ALL_PASS" in out:
        return True, out
    return False, out  # FAIL or unparseable => treat as fail


# ---------------------------------------------------------------------------
# Differential 2-state vs 4-state check
# ---------------------------------------------------------------------------

def differential_check(dut_code: str, sim_dir, timeout: int = 120) -> Dict:
    """Run the DUT under BOTH harnesses on the same testbench.json and compare.

    Returns {passed_2state, passed_4state, agree, note}. A disagreement where
    2-state PASSES but 4-state FAILS is Verilator hiding an X/Z bug.
    """
    from pathlib import Path
    from pro_v.simulate_and_evaluate_mutants import run_simulation
    sim_dir = Path(sim_dir)

    tbj = sim_dir / "testbench.json"
    dutv = sim_dir / "top_module.v"
    if not tbj.exists() or not dutv.exists():
        return {"error": "sim_dir missing testbench.json or top_module.v"}

    with open(dutv) as f:
        ports = parse_ports(f.read())
    with open(tbj) as f:
        data = json.load(f)
    seq = is_sequential(data)
    tb_code = build_seq_tb(ports, data) if seq else build_cmb_tb(ports, data)

    # 4-state (iverilog) first -- runs in its own temp dir, does not touch sim_dir
    passed_4state, log4 = run_iverilog_tb(dut_code, tb_code, timeout=timeout)

    # 2-state (Verilator) -- writes dut into sim_dir and runs make
    res = run_simulation(dut_code, sim_dir, timeout=timeout)
    passed_2state = res.passed

    agree = (passed_2state == passed_4state)
    note = ""
    if not agree:
        if passed_2state and not passed_4state:
            note = "Verilator (2-state) PASSES but iverilog (4-state) FAILS -> likely X/Z bug hidden by Verilator"
        else:
            note = "iverilog (4-state) PASSES but Verilator (2-state) FAILS -> check harness/expected mismatch"
    return {
        "passed_2state": passed_2state,
        "passed_4state": passed_4state,
        "agree": agree,
        "note": note,
        "iverilog_log_tail": log4[-400:],
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    logging.basicConfig(level=logging.INFO)

    dut_path = os.path.join(tempfile.gettempdir(), "prov_selftest_dut.v")
    tbj_path = os.path.join(tempfile.gettempdir(), "prov_selftest_tb.json")
    # AND gate + a testbench.json covering all 4 vectors
    with open(dut_path, "w") as f:
        f.write("module top_module(input a, input b, output y); assign y=a&b; endmodule")
    data = [
        {"inputs": {"a": "0", "b": "0"}, "expected_outputs": {"y": "0"}},
        {"inputs": {"a": "0", "b": "1"}, "expected_outputs": {"y": "0"}},
        {"inputs": {"a": "1", "b": "0"}, "expected_outputs": {"y": "0"}},
        {"inputs": {"a": "1", "b": "1"}, "expected_outputs": {"y": "1"}},
    ]
    with open(tbj_path, "w") as f:
        json.dump(data, f)

    tb = generate_tb(tbj_path, dut_path)

    good = "module top_module(input a, input b, output y); assign y=a&b; endmodule"
    orm  = "module top_module(input a, input b, output y); assign y=a|b; endmodule"
    # X-bug: outputs X when a==0 (Verilator would coerce; iverilog flags it)
    xbug = "module top_module(input a, input b, output y); assign y = a ? (a&b) : 1'bx; endmodule"

    p_good, _ = run_iverilog_tb(good, tb)
    p_or, log_or = run_iverilog_tb(orm, tb)
    p_x, log_x = run_iverilog_tb(xbug, tb)

    print("correct AND  -> pass:", p_good, "(expect True)")
    print("OR mutant    -> pass:", p_or, "(expect False)  |", log_or.strip().splitlines()[-1] if log_or.strip() else "")
    print("X-bug DUT    -> pass:", p_x, "(expect False, 4-state catches X)  |",
          [l for l in log_x.splitlines() if "MISMATCH" in l][:1])

    ok = p_good and (not p_or) and (not p_x)
    print("SELFTEST:", "PASS" if ok else "FAIL",
          "(4-state Verilog TB checks correct designs, kills OR mutant, and catches X)")
    return 0 if ok else 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Self-checking 4-state Verilog TB generator")
    parser.add_argument("--testbench", "-b", required=True, help="testbench.json path")
    parser.add_argument("--dut", "-d", required=True, help="top_module.v path")
    parser.add_argument("--out", "-o", default="tb.v")
    parser.add_argument("--type", choices=["cmb", "seq"], default=None)
    parser.add_argument("--run", action="store_true", help="also compile+run with iverilog")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    tb = generate_tb(args.testbench, args.dut, circuit_type=args.type)
    with open(args.out, "w") as f:
        f.write(tb)
    logger.info(f"Wrote Verilog testbench to {args.out}")
    if args.run:
        with open(args.dut) as f:
            dut = f.read()
        passed, log = run_iverilog_tb(dut, tb)
        print("iverilog result:", "PASS" if passed else "FAIL")
        print(log[-600:])


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    main()
