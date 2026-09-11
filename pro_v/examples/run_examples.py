#!/usr/bin/env python3
"""
Demonstrations of the Reference-Guided Functional Coverage Agent.

Run:
    python -m pro_v.examples.run_examples
    # or
    python pro_v/examples/run_examples.py

Example 1 (AND vs OR): shows that the naive {00, 11} suite gives "50% input
coverage" and sees both output values, yet cannot tell AND from OR -- because
AND and OR agree on 00 and 11. The agent, driving golden_dut.py, selects the
mixed cases 01 and 10, which are exactly where the two operations disagree.

Example 2 (request/grant FSM): shows the agent doing BFS over the sequential
reference model to find the shortest clocked input sequence that reaches the
DONE state, plus transition-covering sequences.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pro_v.functional_coverage_agent import build_plan, Budget, write_plan


def _load_golden(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ex_golden", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GoldenDUT


def and_vs_or():
    print("=" * 72)
    print("EXAMPLE 1  --  AND vs OR: why {00, 11} is insufficient")
    print("=" * 72)

    and_g = _load_golden(os.path.join(_HERE, "and_gate", "golden_dut.py"))
    or_g = _load_golden(os.path.join(_HERE, "or_gate", "golden_dut.py"))

    def ev(g, a, b):
        return int(g().load({"a": str(a), "b": str(b)})["y"], 2)

    print("\nNaive suite {00, 11} -- outputs from each model:")
    print("  input | AND | OR")
    for a, b in [(0, 0), (1, 1)]:
        print(f"    {a}{b}  |  {ev(and_g, a, b)}  | {ev(or_g, a, b)}")
    print("  -> AND and OR are IDENTICAL on 00 and 11. 50% input coverage,")
    print("     both output classes seen, yet AND and OR are indistinguishable.")

    print("\nDistinguishing cases {01, 10}:")
    print("  input | AND | OR")
    for a, b in [(0, 1), (1, 0)]:
        print(f"    {a}{b}  |  {ev(and_g, a, b)}  | {ev(or_g, a, b)}")
    print("  -> These are exactly where AND and OR DISAGREE.")

    plan = build_plan(
        os.path.join(_HERE, "and_gate", "golden_dut.py"),
        os.path.join(_HERE, "and_gate", "top_module.v"),
        budget=Budget(max_tests=16),
    )
    selected = sorted((t["inputs"]["a"], t["inputs"]["b"]) for t in plan["tests"])
    print(f"\nAgent-selected input vectors (driving golden_dut.py): {selected}")
    print("Distinguishing tests the agent produced:")
    for t in plan["tests"]:
        if "distinguish" in t["name"]:
            print(f"  - {t['name']}: a={t['inputs']['a']} b={t['inputs']['b']} "
                  f"-> y={t['expected_outputs']['y']}")
            print(f"      reason: {t['reason']}")
    out = os.path.join(_HERE, "and_gate", "coverage_plan.json")
    write_plan(plan, out)
    print(f"\nWrote {out}")
    print(f"Summary: {json.dumps(plan['summary'], indent=2)}")
    return plan


def traffic_fsm():
    print("\n" + "=" * 72)
    print("EXAMPLE 2  --  request/grant FSM: BFS to a target state")
    print("=" * 72)
    plan = build_plan(
        os.path.join(_HERE, "traffic_fsm", "golden_dut.py"),
        os.path.join(_HERE, "traffic_fsm", "top_module.v"),
        budget=Budget(max_tests=40, max_sequence_depth=10, max_bfs_states=64),
        interface_meta={"target_states": ["DONE"]},
    )
    print(f"\nStates reached by BFS over the reference model: "
          f"{plan['summary']['states_reached']}")
    print(f"State labels -> reference state snapshot: "
          f"{json.dumps(plan['state_labels'])}")
    # find and print a sequence that observes done=1
    for seq in plan["sequences"]:
        if any(s["expected_outputs"].get("done") == 1 for s in seq["steps"]):
            print(f"\nSequence reaching DONE: {seq['name']} "
                  f"(target {seq['target_state']})")
            for s in seq["steps"]:
                print(f"  cycle {s['cycle']}: inputs={s['inputs']} "
                      f"-> expected={s['expected_outputs']}")
            print(f"  covers: {seq['covers']}")
            print(f"  reason: {seq['reason']}")
            break
    out = os.path.join(_HERE, "traffic_fsm", "coverage_plan.json")
    write_plan(plan, out)
    print(f"\nWrote {out}")
    print(f"Summary: {json.dumps(plan['summary'], indent=2)}")
    return plan


if __name__ == "__main__":
    and_vs_or()
    traffic_fsm()
