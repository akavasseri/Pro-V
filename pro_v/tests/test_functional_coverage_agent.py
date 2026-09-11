#!/usr/bin/env python3
"""
Unit tests for the Reference-Guided Functional Coverage Agent.

Covers the behaviors the agent promises:
  * exhaustive enumeration on small input spaces
  * bounded sampling on large input spaces (no blow-up, budget respected)
  * output-to-input backtracking (a chosen input actually causes each output class)
  * removal of redundant tests (greedy minimization drops zero-gain tests)
  * sequential BFS path discovery (shortest sequence into a target state)

Run:
    python -m pytest pro_v/tests/test_functional_coverage_agent.py -q
    # or, dependency-free:
    python pro_v/tests/test_functional_coverage_agent.py
"""

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pro_v.functional_coverage_agent import (
    Budget, GoldenModelAdapter, build_plan, generate_candidates,
    greedy_cover, plan_to_stimulus, _Item, _Clock,
)
from pro_v.mutation_strength import parse_ports


# --------------------------------------------------------------------------
# helpers: write a golden_dut.py + top_module.v pair into a temp dir
# --------------------------------------------------------------------------

def _pair(tmp, name, golden_src, verilog):
    gp = os.path.join(tmp, f"{name}_golden.py")
    dp = os.path.join(tmp, f"{name}.v")
    open(gp, "w").write(golden_src)
    open(dp, "w").write(verilog)
    return gp, dp


AND_SRC = (
    "class GoldenDUT:\n"
    "    def load(self, inputs):\n"
    "        return {'y': str(int(inputs['a'],2) & int(inputs['b'],2))}\n"
)
AND_V = "module top_module(input a, input b, output y); assign y=a&b; endmodule"

ADD_SRC = (
    "class GoldenDUT:\n"
    "    def load(self, inputs):\n"
    "        a=int(inputs['a'],2); b=int(inputs['b'],2)\n"
    "        s=(a+b) & 0xFF\n"
    "        return {'sum': format(s,'08b')}\n"
)
ADD_V = "module top_module(input [7:0] a, input [7:0] b, output [7:0] sum); endmodule"

CMP_SRC = (
    "class GoldenDUT:\n"
    "    def load(self, inputs):\n"
    "        a=int(inputs['a'],2); b=int(inputs['b'],2)\n"
    "        return {'le': '1' if a<=b else '0'}\n"
)
CMP_V = "module top_module(input [3:0] a, input [3:0] b, output le); endmodule"

COUNTER_SRC = (
    "class GoldenDUT:\n"
    "    def __init__(self):\n"
    "        self.count = 0\n"
    "    def load(self, clk, inputs):\n"
    "        if clk==1 and int(inputs.get('en','0'),2):\n"
    "            self.count=(self.count+1)%4\n"
    "        return {'q': format(self.count,'02b')}\n"
)
COUNTER_V = "module top_module(input clk, input en, output [1:0] q); endmodule"


# --------------------------------------------------------------------------
# 1. exhaustive enumeration on small input spaces
# --------------------------------------------------------------------------

def test_exhaustive_small_space():
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "and", AND_SRC, AND_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=64))
        assert plan["summary"]["exhaustive"] is True
        cands, exhaustive = generate_candidates(
            GoldenModelAdapter(_import(gp), parse_ports(AND_V)), Budget())
        # 2 one-bit inputs -> exactly 4 candidate vectors, all combinations
        assert exhaustive is True
        combos = {(c["a"], c["b"]) for c in cands}
        assert combos == {(0, 0), (0, 1), (1, 0), (1, 1)}
    print("PASS test_exhaustive_small_space")


# --------------------------------------------------------------------------
# 2. bounded sampling on large input spaces
# --------------------------------------------------------------------------

def test_bounded_sampling_large_space():
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "add", ADD_SRC, ADD_V)
        adapter = GoldenModelAdapter(_import(gp), parse_ports(ADD_V))
        # 16-bit total space (65536) must NOT be enumerated exhaustively; the
        # completeness policy instead emits a large capped SAMPLE (not a minimized
        # handful) so nothing distinguishing is left untested by luck.
        budget = Budget(sample_cap=2048)
        cands, exhaustive = generate_candidates(adapter, budget)
        assert exhaustive is False
        assert len(cands) == budget.sample_cap          # filled to the hard cap
        # sampling must still include the key boundary values for each input
        avals = {c["a"] for c in cands}
        assert {0, 255}.issubset(avals)          # 0 and all-ones present
        plan = build_plan(gp, dp, budget=Budget(sample_cap=2048))
        assert plan["summary"]["num_tests"] >= 512       # not minimized to a few
        assert plan["summary"]["exhaustive"] is False
    print("PASS test_bounded_sampling_large_space")


# --------------------------------------------------------------------------
# 3. output-to-input backtracking: chosen input truly causes each output class
# --------------------------------------------------------------------------

def test_output_to_input_backtracking():
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "add", ADD_SRC, ADD_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=60, max_candidates=800))
        adapter = GoldenModelAdapter(_import(gp), parse_ports(ADD_V))
        # every output-class goal that is claimed covered must have a test whose
        # golden-model output really lands in that class
        covered = set(plan["summary"]["covered_goals"])
        assert any(g.startswith("out[sum]=zero") for g in covered)
        # re-run the golden model on each test's inputs and confirm the expected
        # output matches (backtracking is grounded in the reference, not faked)
        for t in plan["tests"]:
            out = adapter.run(t["inputs"])
            assert out["sum"] == t["expected_outputs"]["sum"], \
                f"expected output not reproduced by golden model for {t['name']}"
        # a zero-output test must exist and be a real preimage of sum==0
        zeros = [t for t in plan["tests"] if t["expected_outputs"]["sum"] == 0]
        assert zeros, "no input found that drives sum to the zero output class"
        assert (zeros[0]["inputs"]["a"] + zeros[0]["inputs"]["b"]) % 256 == 0
    print("PASS test_output_to_input_backtracking")


# --------------------------------------------------------------------------
# 4. removal of redundant tests (greedy keeps only new-coverage tests)
# --------------------------------------------------------------------------

def test_redundant_test_removal():
    # three items: A covers {g1,g2}, B covers {g2} (redundant vs A), C covers {g3}
    items = [
        _Item(goals={"g1", "g2"}, cost=1, payload={"v": 0}, reason="A", kind="single_cycle"),
        _Item(goals={"g2"}, cost=1, payload={"v": 1}, reason="B", kind="single_cycle"),
        _Item(goals={"g3"}, cost=1, payload={"v": 2}, reason="C", kind="single_cycle"),
    ]
    universe = {"g1", "g2", "g3"}
    chosen = greedy_cover(items, universe, max_units=10, unit_of=lambda p: [p])
    reasons = {c.reason for c in chosen}
    assert reasons == {"A", "C"}, f"redundant B should be dropped, got {reasons}"

    # and end-to-end: exhaustive AND has 4 possible vectors but the minimized
    # plan should not contain duplicate input vectors, and should be small.
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "and", AND_SRC, AND_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=64))
        keys = [tuple(sorted(t["inputs"].items())) for t in plan["tests"]]
        assert len(keys) == len(set(keys)), "plan contains duplicate test vectors"
    print("PASS test_redundant_test_removal")


# --------------------------------------------------------------------------
# 5. sequential BFS path discovery
# --------------------------------------------------------------------------

def test_sequential_bfs_path_discovery():
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "counter", COUNTER_SRC, COUNTER_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=40, max_sequence_depth=8))
        assert plan["design_type"] == "sequential"
        # a mod-4 counter has exactly 4 reachable states
        assert plan["summary"]["states_reached"] == 4
        # some sequence must drive q to 3 (binary 11) -- the farthest state,
        # requiring en=1 held for 3 cycles: this is the BFS path.
        reached3 = any(s["expected_outputs"].get("q") == 3
                       for seq in plan["sequences"] for s in seq["steps"])
        assert reached3, "BFS did not find a sequence reaching counter state 3"
        # shortest path to state 3 is 3 cycles of en=1; the reaching sequence
        # must not be longer than the depth budget.
        for seq in plan["sequences"]:
            assert len(seq["steps"]) <= 8
        # sequences are clocked (multi-cycle), not single vectors
        assert all(seq["kind"] == "multi_cycle" for seq in plan["sequences"])
    print("PASS test_sequential_bfs_path_discovery")


# --------------------------------------------------------------------------
# 6. distinguishing selection + equality case (< vs <=) + budget respected
# --------------------------------------------------------------------------

def test_distinguishing_and_equality():
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "and", AND_SRC, AND_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=16))
        vecs = {(t["inputs"]["a"], t["inputs"]["b"]) for t in plan["tests"]}
        # the whole point: 01 and 10 (the AND-vs-OR separators) are selected
        assert (0, 1) in vecs and (1, 0) in vecs

    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "cmp", CMP_SRC, CMP_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=60))
        # <= vs < needs an equality case; the agent must include a==b
        assert any(t["inputs"]["a"] == t["inputs"]["b"] for t in plan["tests"]), \
            "no equality case to separate <= from <"
        # and the equality goal is in the covered universe
        assert any(g.startswith("eq[") for g in plan["summary"]["covered_goals"])
    print("PASS test_distinguishing_and_equality")


# --------------------------------------------------------------------------
# 7. budget: max_tests is a hard cap on emitted tests
# --------------------------------------------------------------------------

def test_budget_max_tests_cap():
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "add", ADD_SRC, ADD_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=5, max_candidates=800))
        assert plan["summary"]["num_tests"] <= 5
    print("PASS test_budget_max_tests_cap")


# --------------------------------------------------------------------------
# 8. plan -> current stimulus.json down-conversion (integration point)
# --------------------------------------------------------------------------

def test_stimulus_downconversion():
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "counter", COUNTER_SRC, COUNTER_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=20, max_sequence_depth=6))
        stim = plan_to_stimulus(plan)
        # sequential -> list of scenarios with clock_cycles + per-signal bit lists
        assert all("clock_cycles" in s for s in stim)
        assert all(isinstance(s["en"], list) for s in stim)
        assert all(all(c in "01" for c in bit) for s in stim for bit in s["en"])
    print("PASS test_stimulus_downconversion")


# --------------------------------------------------------------------------
# 9. header is authoritative; a wrong/mismatched FRM is flagged, not trusted
# --------------------------------------------------------------------------

# FRM reads a port ('c') that the header does not declare, ignores declared 'b',
# and never produces declared output 'y' (produces 'z' instead).
WRONG_SRC = (
    "class GoldenDUT:\n"
    "    def load(self, inputs):\n"
    "        return {'z': str(int(inputs['a'],2) & int(inputs.get('c','0'),2))}\n"
)
WRONG_V = "module top_module(input a, input b, output y); assign y=a&b; endmodule"


def test_header_authoritative_and_frm_consistency():
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "wrong", WRONG_SRC, WRONG_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=16))
        # interface comes from the HEADER, not the FRM: ports are a,b / y
        assert set(plan["interface"]["inputs"]) == {"a", "b"}
        assert set(plan["interface"]["outputs"]) == {"y"}
        con = plan["frm_consistency"]
        assert con["status"] == "warnings"
        assert con["interface_source"].startswith("top_module.v header")
        assert "c" in con["reads_unknown_inputs"]        # FRM reads a phantom port
        assert "b" in con["ignored_inputs"]              # FRM ignores a real input
        assert "y" in con["missing_outputs"]             # FRM never produces y
        # and because y is never produced, expected_outputs[y] is null (not faked 0)
        assert all(t["expected_outputs"]["y"] is None for t in plan["tests"])

    # a CORRECT FRM has a clean bill of health
    with tempfile.TemporaryDirectory() as tmp:
        gp, dp = _pair(tmp, "and", AND_SRC, AND_V)
        plan = build_plan(gp, dp, budget=Budget(max_tests=16))
        assert plan["frm_consistency"]["status"] == "ok"
        assert plan["frm_consistency"]["warnings"] == []
    print("PASS test_header_authoritative_and_frm_consistency")


# --------------------------------------------------------------------------
# util: import a GoldenDUT class from a file path
# --------------------------------------------------------------------------

def _import(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("t_golden_" + os.path.basename(path), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GoldenDUT


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
