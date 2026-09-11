#!/usr/bin/env python3
"""
Unit tests for the Verilator coverage-closure loop.

The coverage.dat parser + percentage/uncovered logic are tested without any
tools. The end-to-end closure is tested only when verilator is on PATH (skipped
otherwise so the suite stays green in tool-less environments).

Run:
    python pro_v/tests/test_coverage_closure_loop.py
"""

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pro_v.coverage_closure_loop as L
from pro_v.coverage_closure_loop import (
    parse_coverage_dat, CoverageReport, close_coverage, _gen_harness, _have_tools,
)
from pro_v.mutation_strength import parse_ports


# --------------------------------------------------------------------------
# coverage.dat parsing + per-category math (no tools required)
# --------------------------------------------------------------------------

def _dat_line(page, f, l, name, count):
    payload = f"\x01f\x02{f}\x01l\x02{l}\x01n\x02{name}\x01page\x02{page}"
    return f"C '{payload}' {count}\n"


def test_parse_and_percentages():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "coverage.dat")
        with open(path, "w") as fh:
            fh.write("# SystemC::Coverage-3\n")
            fh.write(_dat_line("v_line/top_module", "top.v", 5, "blk", 3))   # covered
            fh.write(_dat_line("v_line/top_module", "top.v", 6, "blk", 0))   # uncovered
            fh.write(_dat_line("v_toggle/top_module", "top.v", 5, "a[0]", 1))
            fh.write(_dat_line("v_toggle/top_module", "top.v", 5, "a[1]", 0))
            fh.write(_dat_line("v_branch/top_module", "top.v", 7, "if", 2))
        rep = parse_coverage_dat(path)
        pct = rep.percentages()
        # categories mapped to friendly names
        assert set(pct) == {"line/block", "toggle", "branch"}
        assert pct["line/block"]["total"] == 2 and pct["line/block"]["covered"] == 1
        assert abs(pct["line/block"]["pct"] - 0.5) < 1e-9
        assert abs(pct["toggle"]["pct"] - 0.5) < 1e-9
        assert pct["branch"]["pct"] == 1.0
        # uncovered points are reported with file/line
        unc = rep.uncovered("line/block")
        assert len(unc) == 1 and unc[0].line == 6
    print("PASS test_parse_and_percentages")


def test_parse_missing_file_is_empty():
    rep = parse_coverage_dat("/nonexistent/coverage.dat")
    assert isinstance(rep, CoverageReport) and rep.points == []
    assert rep.percentages() == {}
    print("PASS test_parse_missing_file_is_empty")


def test_harness_generation_shapes():
    ports = parse_ports("module top_module(input [1:0] sel, input [3:0] a, "
                        "output [3:0] y); endmodule")
    cpp = _gen_harness(ports, is_seq=False,
                       flat=[{"sel": 1, "a": 5}, {"sel": 2, "a": 6}])
    assert "Vtop_module" in cpp
    assert "V_sel[] = { 0x1U, 0x2U }" in cpp
    assert "V_a[] = { 0x5U, 0x6U }" in cpp
    assert "coverage.dat" in cpp
    # sequential harness toggles the clock
    sports = parse_ports("module top_module(input clk, input en, "
                         "output [1:0] q); endmodule")
    scpp = _gen_harness(sports, is_seq=True, flat=[{"en": 1}])
    assert "top->clk = 1" in scpp and "top->clk = 0" in scpp
    print("PASS test_harness_generation_shapes")


# --------------------------------------------------------------------------
# end-to-end closure (requires verilator)
# --------------------------------------------------------------------------

MUX_V = (
    "module top_module(input [1:0] sel, input [3:0] a, input [3:0] b,\n"
    "                  input [3:0] c, input [3:0] d, output reg [3:0] y);\n"
    "  always @(*) begin\n    case (sel)\n      2'd0: y=a;\n      2'd1: y=b;\n"
    "      2'd2: y=c;\n      default: y=d;\n    endcase\n  end\nendmodule\n"
)
MUX_G = (
    "class GoldenDUT:\n"
    "    def load(self, inputs):\n"
    "        sel=int(inputs['sel'],2)\n"
    "        v=[int(inputs['a'],2),int(inputs['b'],2),int(inputs['c'],2),int(inputs['d'],2)]\n"
    "        return {'y': format(v[sel],'04b')}\n"
)


def test_closure_end_to_end():
    if not _have_tools():
        print("SKIP test_closure_end_to_end (verilator not on PATH)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        dp = os.path.join(tmp, "top_module.v"); open(dp, "w").write(MUX_V)
        gp = os.path.join(tmp, "golden_dut.py"); open(gp, "w").write(MUX_G)
        # weak seed: sel fixed 0 -> must iterate to close the case branches
        weak = {"design_type": "combinational",
                "tests": [{"inputs": {"sel": 0, "a": i, "b": 0, "c": 0, "d": 0}}
                          for i in range(4)],
                "sequences": []}
        res = close_coverage(gp, dp, plan=weak, target=0.90, max_iters=5)
        assert res.history[0]["coverage"]["line/block"] < 0.90, "seed should undercover"
        assert res.reached_target, f"loop failed to close: {res.per_category}"
        assert res.iterations >= 1, "closing a weak seed should take >=1 iteration"
    print("PASS test_closure_end_to_end")


def test_closure_with_llm_callback():
    if not _have_tools():
        print("SKIP test_closure_with_llm_callback (verilator not on PATH)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        dp = os.path.join(tmp, "top_module.v"); open(dp, "w").write(MUX_V)
        gp = os.path.join(tmp, "golden_dut.py"); open(gp, "w").write(MUX_G)
        calls = {"n": 0}

        def fake_llm(request):
            calls["n"] += 1
            assert "uncovered_points" in request and "deficient_categories" in request
            return [{"sel": s, "a": 1, "b": 2, "c": 3, "d": 4} for s in range(4)]

        weak = {"design_type": "combinational",
                "tests": [{"inputs": {"sel": 0, "a": 0, "b": 0, "c": 0, "d": 0}}],
                "sequences": []}
        # scope to line/block: the stub sweeps sel (closing the case branches)
        # but its fixed operands don't toggle every data bit, so target that
        # category to verify the callback is invoked and closes what it targets.
        res = close_coverage(gp, dp, plan=weak, target=0.90, max_iters=5,
                             categories=["line/block"], llm_generate=fake_llm)
        assert calls["n"] >= 1, "the LLM callback was never invoked"
        assert res.reached_target, f"line/block not closed: {res.per_category}"
    print("PASS test_closure_with_llm_callback")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
