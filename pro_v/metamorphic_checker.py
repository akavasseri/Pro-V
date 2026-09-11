#!/usr/bin/env python3
"""
metamorphic_checker.py  --  relation-based golden-model checks (Pro-V stage 13).

Checks golden_dut.py against RELATION properties derived from the behavior contract
(and invariants.json). These catch bugs even when exact witness examples are sparse
-- e.g. an adder that is not commutative, an AND that does not annihilate on 0, a mux
that leaks the unselected input, a counter that ignores enable.

Meant to run inside golden_model_verifier.py after the exact witness checks (an
opt-in hook is provided there). Standalone here for testing. Uses the verifier's
API-agnostic ModelAdapter to drive the model; never touches top_module.v.

Output: metamorphic_report.json with per-property pass/fail and, on failure, the
inputs/trace, actual outputs, the expected relation, associated rule IDs, and a
repair hint.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    from pro_v.golden_model_verifier import ModelAdapter, _load_golden_class, _load, _to_int, _w
    from pro_v.witness_generator import _detect_operation, _is_control, _extract_fsm, _bfs_paths
except Exception:  # pragma: no cover
    from golden_model_verifier import ModelAdapter, _load_golden_class, _load, _to_int, _w
    from witness_generator import _detect_operation, _is_control, _extract_fsm, _bfs_paths


def _vals(width: int) -> List[int]:
    w = max(1, width)
    cand = {0, 1, 2, 3, (1 << w) - 1, (1 << w) - 2, 1 << (w - 1)}
    return sorted(v for v in cand if 0 <= v < (1 << w))


class MetamorphicChecker:

    def run(self, golden_dut_path: str, contract: Any, invariants: Any = None,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        contract = _load(contract)
        inv = _load(invariants) or {}
        in_names = list((contract.get("inputs") or {}).keys())
        in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        out_names = list((contract.get("outputs") or {}).keys())
        out_w = {n: _w(s) for n, s in (contract.get("outputs") or {}).items()}
        rules = contract.get("rules", []) or []
        rule_ids = [r.get("id") for r in rules if r.get("id")]
        seq = contract.get("design_type") == "sequential"

        props: List[dict] = []
        try:
            cls = _load_golden_class(golden_dut_path)
            adapter = ModelAdapter(cls, contract)
        except Exception as e:
            return self._finish([{"id": "P0", "name": "load", "status": "fail",
                                  "detail": "cannot load golden model: %s" % e}], output_dir)

        controls = [n for n in in_names if _is_control(n)]
        data = [n for n in in_names if n not in controls]
        sel = next((n for n in in_names if _is_control(n) and n.lower().startswith("sel")), None)
        op = _detect_operation(rules)
        o0 = out_names[0] if out_names else None

        if not seq and o0:
            two = data[:2] if len(data) >= 2 else in_names[:2]
            if op in ("add", "sub", "and", "or", "xor", "mul") and len(two) >= 2:
                self._comb_op(adapter, contract, in_names, in_w, out_w, o0, two, op, rule_ids, props)
            if op in ("compare", "compare_eq") and len(two) >= 2:
                self._compare(adapter, contract, in_names, in_w, o0, two, rule_ids, props)
            if sel and len([d for d in data]) >= 2:
                self._mux(adapter, contract, in_names, in_w, o0, sel, [d for d in data], rule_ids, props)

        if seq:
            if o0 and self._is_counter(contract, rules):
                self._counter(adapter, contract, in_names, in_w, out_w, o0, rule_ids, props)
            if contract.get("states"):
                self._fsm(adapter, contract, in_names, in_w, rules, rule_ids, props)

        return self._finish(props, output_dir)

    # -- combinational op relations ---------------------------------------

    def _run(self, adapter, contract, in_names, in_w, assign: Dict[str, int]) -> Dict[str, Optional[int]]:
        full = {n: 0 for n in in_names}
        full.update({k: (v & ((1 << in_w.get(k, 1)) - 1)) for k, v in assign.items() if k in in_w})
        return adapter.eval(adapter.cls(), full)

    def _comb_op(self, adapter, contract, in_names, in_w, out_w, o, two, op, rule_ids, props):
        a, b = two
        va, vb = _vals(in_w[a]), _vals(in_w[b])
        omask = (1 << out_w.get(o, 1)) - 1
        widths_match = out_w.get(o, 1) >= max(in_w[a], in_w[b])

        # commutative (add/and/or/xor/mul)
        if op in ("add", "and", "or", "xor", "mul"):
            self._check(props, "commutative_%s" % op, op, "f(a,b) == f(b,a)", rule_ids,
                        self._pairs_equal(adapter, contract, in_names, in_w, a, b, va, vb),
                        "Make the operator commutative: swapping %s and %s must not change the output." % (a, b))
        # identity / annihilator / self-inverse
        checks = {
            "and": [("annihilator_zero", "f(a,0) == 0", lambda x: 0),
                    ("identity_max", "f(a,max) == a", lambda x: x & omask if widths_match else None)],
            "or": [("identity_zero", "f(a,0) == a", lambda x: x & omask if widths_match else None),
                   ("saturate_max", "f(a,max) == max", lambda x: omask)],
            "xor": [("identity_zero", "f(a,0) == a", lambda x: x & omask if widths_match else None),
                    ("self_inverse", "f(a,a) == 0", "self0")],
            "add": [("identity_zero", "f(a,0) == a", lambda x: x & omask if widths_match else None)],
            "sub": [("identity_zero", "f(a,0) == a", lambda x: x & omask if widths_match else None),
                    ("self_zero", "f(a,a) == 0", "self0")],
        }
        for pid, rel, expect in checks.get(op, []):
            fail = None
            for x in va:
                if expect == "self0":
                    out = self._run(adapter, contract, in_names, in_w, {a: x, b: x})
                    got = out.get(o)
                    if got not in (None, 0):
                        fail = {"inputs": {a: x, b: x}, "actual": {o: got}, "expected": {o: 0}}
                        break
                else:
                    exp = expect(x)
                    if exp is None:
                        fail = "skip"
                        break
                    bval = 0 if "zero" in pid else (1 << in_w[b]) - 1   # _zero -> b=0, _max -> b=max
                    out = self._run(adapter, contract, in_names, in_w, {a: x, b: bval})
                    got = out.get(o)
                    if got is not None and got != exp:
                        fail = {"inputs": {a: x, b: bval}, "actual": {o: got}, "expected": {o: exp}}
                        break
            if fail == "skip":
                self._skip(props, "%s_%s" % (pid, op), op, rel, rule_ids, "output width < operand width")
            else:
                self._check(props, "%s_%s" % (pid, op), op, rel, rule_ids, fail,
                            "Fix %s: %s must hold." % (op, rel))

        # overflow semantics (only if contract states wrap/saturate)
        sem = self._overflow_sem(contract)
        if op == "add" and sem:
            mx = (1 << in_w[a]) - 1
            out = self._run(adapter, contract, in_names, in_w, {a: mx, b: 1})
            got = out.get(o)
            exp = 0 if sem == "wrap" else omask
            fail = None if (got is None or got == exp) else {"inputs": {a: mx, b: 1}, "actual": {o: got}, "expected": {o: exp}}
            self._check(props, "overflow_%s" % sem, op, "overflow %s: f(max,1) == %d" % (sem, exp), rule_ids, fail,
                        "Implement %s overflow as the contract states." % sem)

    def _pairs_equal(self, adapter, contract, in_names, in_w, a, b, va, vb):
        n = 0
        for x in va:
            for y in vb:
                if n >= 20:
                    return None
                n += 1
                o1 = self._run(adapter, contract, in_names, in_w, {a: x, b: y})
                o2 = self._run(adapter, contract, in_names, in_w, {a: y, b: x})
                if o1 != o2:
                    return {"inputs": {a: x, b: y}, "actual": {"f(a,b)": o1, "f(b,a)": o2}, "expected": "equal"}
        return None

    def _compare(self, adapter, contract, in_names, in_w, o, two, rule_ids, props):
        a, b = two
        vals = _vals(in_w[a])
        # antisymmetry: for a != b, f(a,b) and f(b,a) not both true
        fail = None
        for x in vals:
            for y in vals:
                if x == y:
                    continue
                f1 = self._run(adapter, contract, in_names, in_w, {a: x, b: y}).get(o)
                f2 = self._run(adapter, contract, in_names, in_w, {a: y, b: x}).get(o)
                if f1 and f2:
                    fail = {"inputs": {a: x, b: y}, "actual": {"f(a,b)": f1, "f(b,a)": f2},
                            "expected": "not both true (antisymmetry)"}
                    break
            if fail:
                break
        self._check(props, "compare_antisymmetry", "compare", "a!=b => not (f(a,b) and f(b,a))", rule_ids, fail,
                    "A strict/ordered comparator must be antisymmetric.")
        # reflexivity consistency: f(a,a) constant across a
        eqvals = {self._run(adapter, contract, in_names, in_w, {a: x, b: x}).get(o) for x in vals}
        fail2 = None if len(eqvals) <= 1 else {"inputs": "a==b cases", "actual": sorted(str(v) for v in eqvals),
                                               "expected": "constant"}
        self._check(props, "compare_equality_consistent", "compare", "f(a,a) constant (defines <,<=,==,!=)",
                    rule_ids, fail2, "The equality case must resolve consistently to <, <=, ==, or !=.")

    def _mux(self, adapter, contract, in_names, in_w, o, sel, data, rule_ids, props):
        omask = (1 << _w((contract.get("outputs") or {}).get(o, {}))) - 1

        # select-sensitivity: with DISTINCT data, the output must change as sel varies
        # (catches a mux that ignores sel and always routes one input -- which would
        # still satisfy defined-source + independence).
        outs_by_sel = []
        for sv in range(min(1 << in_w[sel], 4)):
            assign = {sel: sv}
            for i, d in enumerate(data):
                assign[d] = (i + 1) & ((1 << in_w[d]) - 1)
            outs_by_sel.append(self._run(adapter, contract, in_names, in_w, assign).get(o))
        sens_fail = None if len(set(outs_by_sel)) > 1 else {
            "inputs": "distinct data, sel swept", "actual": {o: outs_by_sel},
            "expected": "output must change with sel"}
        self._check(props, "mux_select_sensitivity", "mux", "output depends on sel", rule_ids, sens_fail,
                    "The mux ignores select: output must change when sel routes a different source.")

        source_fail = None
        indep_fail = None
        for sv in range(min(1 << in_w[sel], 4)):
            assign = {sel: sv}
            for i, d in enumerate(data):
                assign[d] = (i + 1) & ((1 << in_w[d]) - 1)
            out = self._run(adapter, contract, in_names, in_w, assign).get(o)
            selected = [d for d in data if (assign[d] & omask) == (out if out is not None else -1)]
            if not selected:
                source_fail = source_fail or {"inputs": dict(assign), "actual": {o: out},
                                              "expected": "output equals one selected data input"}
                continue
            for d in data:
                if d in selected:
                    continue
                var = dict(assign); var[d] = ((1 << in_w[d]) - 1) ^ assign[d]
                out2 = self._run(adapter, contract, in_names, in_w, var).get(o)
                if out2 != out:
                    indep_fail = indep_fail or {"inputs": dict(var), "actual": {o: out2},
                                                "expected": "unchanged (unselected %s)" % d,
                                                "sel": sv}
                    break
        self._check(props, "mux_defined_source", "mux", "every sel maps output to a defined data input", rule_ids,
                    source_fail, "Ensure each select value routes a specific data input to the output.")
        self._check(props, "mux_independence", "mux", "unselected input does not affect output", rule_ids,
                    indep_fail, "The unselected data input must not influence the output.")

    # -- sequential relations ---------------------------------------------

    def _is_counter(self, contract, rules) -> bool:
        blob = (contract.get("design_name", "") + " " +
                " ".join(str(r.get("description", "")) for r in rules)).lower()
        import re as _re
        return bool(_re.search(r"count|increment", blob))

    def _seq_reset_name(self, contract, in_names):
        r = (contract.get("reset") or {})
        return r.get("name") or next((n for n in in_names if n.lower() in ("reset", "rst")), None), \
            bool(r.get("active_high", True))

    def _counter(self, adapter, contract, in_names, in_w, out_w, o, rule_ids, props):
        rname, ah = self._seq_reset_name(contract, in_names)
        en = next((n for n in in_names if n.lower() in ("enable", "en") or n.lower().endswith("_en")), None)
        W = out_w.get(o, 1)
        rasrt = 1 if ah else 0

        def ins(**kw):
            d = {n: 0 for n in in_names}
            for k, v in kw.items():
                if k in d:
                    d[k] = v
            return d

        # reset returns initial count
        if rname:
            inst = adapter.new()
            if en:
                adapter.step(inst, ins(**{en: 1}))
                adapter.step(inst, ins(**{en: 1}))
            out = adapter.step(inst, ins(**{rname: rasrt}))
            got = out.get(o)
            fail = None if got in (None, 0) else {"trace": "assert reset", "actual": {o: got}, "expected": {o: 0}}
            self._check(props, "counter_reset", "counter", "reset -> initial count 0", rule_ids, fail,
                        "Reset must return the counter to its initial value.")
        # enable increments
        if en:
            inst = adapter.new()
            c0 = adapter.step(inst, ins(**{en: 1})).get(o)
            c1 = adapter.step(inst, ins(**{en: 1})).get(o)
            fail = None
            if c0 is not None and c1 is not None and c1 != (c0 + 1) % (1 << W):
                fail = {"trace": "two enabled cycles", "actual": {o: [c0, c1]}, "expected": "c1 == (c0+1) mod 2**%d" % W}
            self._check(props, "counter_increment", "counter", "enable -> +1", rule_ids, fail,
                        "With enable asserted the counter must increment by one.")
            # disable holds
            inst = adapter.new()
            adapter.step(inst, ins(**{en: 1}))
            h0 = adapter.step(inst, ins(**{en: 0})).get(o)
            h1 = adapter.step(inst, ins(**{en: 0})).get(o)
            fail = None if (h0 is None or h1 is None or h0 == h1) else {
                "trace": "two disabled cycles", "actual": {o: [h0, h1]}, "expected": "held constant"}
            self._check(props, "counter_hold", "counter", "disable -> hold", rule_ids, fail,
                        "With enable low the counter must hold its value.")
            # wrap / saturate at max
            sem = self._overflow_sem(contract) or "wrap"
            inst = adapter.new()
            last = None
            for _ in range(1 << W):   # from 0, 2**W enabled steps land on the wrap (…max->0) / saturation point
                last = adapter.step(inst, ins(**{en: 1})).get(o)
            exp = 0 if sem == "wrap" else (1 << W) - 1
            fail = None if (last is None or last == exp) else {"trace": "overflow", "actual": {o: last},
                                                               "expected": {o: exp}, "semantics": sem}
            self._check(props, "counter_%s" % sem, "counter", "max+1 %ss" % sem, rule_ids, fail,
                        "At max the counter must %s per the contract." % sem)

    def _fsm(self, adapter, contract, in_names, in_w, rules, rule_ids, props):
        states = contract.get("states", []) or []
        init = contract.get("initial_state") or (states[0] if states else None)
        rname, ah = self._seq_reset_name(contract, in_names)
        if not adapter.has_get_state:
            self._skip(props, "fsm_reset", "fsm", "reset -> initial state", rule_ids, "no get_state()")
            return

        def ins(**kw):
            d = {n: 0 for n in in_names}
            for k, v in kw.items():
                if k in d:
                    d[k] = v
            return d

        # reset reaches initial state
        if rname and init is not None:
            inst = adapter.new()
            adapter.step(inst, ins(**{rname: 1 if ah else 0}))
            st = adapter.state(inst)
            got = (st or {}).get("state", next(iter((st or {}).values()), None)) if st else None
            fail = None if (got is None or str(got) == str(init)) else {"trace": "assert reset",
                                                                        "actual": {"state": got}, "expected": {"state": init}}
            self._check(props, "fsm_reset", "fsm", "reset -> initial state %s" % init, rule_ids, fail,
                        "Reset must place the FSM in its initial state.")
        # each transition exercisable
        fsm = _extract_fsm(rules, states)
        if fsm["edges"] and init is not None:
            paths = _bfs_paths(fsm["edges"], init, states)
            for (src, guard, dst) in fsm["edges"]:
                path = paths.get(src)
                if path is None:
                    continue
                inst = adapter.new()
                if rname:
                    adapter.step(inst, ins(**{rname: 1 if ah else 0}))
                    adapter.step(inst, ins(**{rname: 0 if ah else 1}))
                for (_s, g, _d) in path:
                    adapter.step(inst, ins(**{k: int(v, 2) if isinstance(v, str) else v for k, v in g.items()}))
                adapter.step(inst, ins(**{k: int(v, 2) if isinstance(v, str) else v for k, v in guard.items()}))
                st = adapter.state(inst)
                got = (st or {}).get("state", next(iter((st or {}).values()), None)) if st else None
                fail = None if (got is None or str(got) == str(dst)) else {
                    "trace": "%s --%s--> %s" % (src, guard, dst), "actual": {"state": got}, "expected": {"state": dst}}
                self._check(props, "fsm_transition_%s_%s" % (src, dst), "fsm",
                            "%s --%s--> %s exercisable" % (src, guard, dst), rule_ids, fail,
                            "Ensure the transition %s -> %s fires under its guard." % (src, dst))

    # -- bookkeeping -------------------------------------------------------

    def _overflow_sem(self, contract):
        blob = " ".join([str(x) for x in (contract.get("assumptions") or [])] +
                        [str(x) for x in (contract.get("undefined_behavior") or [])] +
                        [str(r.get("description", "")) for r in (contract.get("rules") or [])]).lower()
        if "saturat" in blob:
            return "saturate"
        if "wrap" in blob or "modulo" in blob:
            return "wrap"
        return None

    def _check(self, props, pid, op, relation, rule_ids, fail, repair_hint):
        if fail == "skip":
            return self._skip(props, pid, op, relation, rule_ids, "not applicable")
        entry = {"id": "P_%s" % pid, "name": pid, "op": op, "relation": relation,
                 "covers_rules": rule_ids, "status": "pass" if not fail else "fail"}
        if fail:
            entry["repair_hint"] = repair_hint
            if isinstance(fail, dict):
                entry.update({k: fail[k] for k in ("inputs", "trace", "actual", "expected", "sel", "semantics")
                              if k in fail})
        props.append(entry)

    def _skip(self, props, pid, op, relation, rule_ids, why):
        props.append({"id": "P_%s" % pid, "name": pid, "op": op, "relation": relation,
                      "covers_rules": rule_ids, "status": "skip", "note": why})

    def _finish(self, props, output_dir) -> Dict[str, Any]:
        passed = sum(1 for p in props if p["status"] == "pass")
        failed = sum(1 for p in props if p["status"] == "fail")
        report = {
            "status": "fail" if failed else "pass",
            "num_properties": len(props),
            "passed": passed, "failed": failed,
            "skipped": sum(1 for p in props if p["status"] == "skip"),
            "properties": props,
            "failures": [p for p in props if p["status"] == "fail"],
        }
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "metamorphic_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            report["json_path"] = os.path.join(output_dir, "metamorphic_report.json")
        return report


def check_metamorphic(golden_dut_path, contract, invariants=None, output_dir=None) -> Dict[str, Any]:
    return MetamorphicChecker().run(golden_dut_path, contract, invariants, output_dir)


# ---------------------------------------------------------------------------
# self-test (writes tiny golden models; no LLM, no RTL)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    ok = True
    M = MetamorphicChecker()

    def write(d, src):
        p = os.path.join(d, "g.py")
        with open(p, "w") as f:
            f.write(src)
        return p

    and_c = {"design_name": "and_gate", "design_type": "combinational",
             "inputs": {"a": {"width": 4}, "b": {"width": 4}}, "outputs": {"y": {"width": 4}},
             "rules": [{"id": "R1", "effect": "y = a & b"}]}
    with tempfile.TemporaryDirectory() as d:
        gp = write(d, "class GoldenDUT:\n    def eval(self,i): return {'y': (i['a'] & i['b']) & 15}\n")
        r = M.run(gp, and_c, output_dir=d)
        ok = ok and r["status"] == "pass" and r["failed"] == 0
        ok = ok and any(p["name"] == "commutative_and" and p["status"] == "pass" for p in r["properties"])
        ok = ok and any(p["name"] == "annihilator_zero_and" for p in r["properties"])
        ok = ok and os.path.exists(r["json_path"])
        # OR model on an AND contract -> annihilator f(a,0)==0 fails (OR gives a)
        gp = write(d, "class GoldenDUT:\n    def eval(self,i): return {'y': (i['a'] | i['b']) & 15}\n")
        r2 = M.run(gp, and_c)
        ok = ok and r2["status"] == "fail"
        ok = ok and any(p["name"] == "annihilator_zero_and" and p["status"] == "fail" for p in r2["properties"])
        f = next(p for p in r2["failures"] if p["name"] == "annihilator_zero_and")
        ok = ok and "repair_hint" in f and "expected" in f

    # XOR self-inverse; non-commutative subtractor stub fails commutative-less (sub has no commutative check)
    xor_c = {"design_name": "xor", "design_type": "combinational",
             "inputs": {"a": {"width": 3}, "b": {"width": 3}}, "outputs": {"y": {"width": 3}},
             "rules": [{"id": "R1", "effect": "y = a ^ b"}]}
    with tempfile.TemporaryDirectory() as d:
        gp = write(d, "class GoldenDUT:\n    def eval(self,i): return {'y': (i['a'] ^ i['b']) & 7}\n")
        rx = M.run(gp, xor_c)
        ok = ok and rx["status"] == "pass"
        ok = ok and any(p["name"] == "self_inverse_xor" and p["status"] == "pass" for p in rx["properties"])

    # mux: correct routes + independence; broken mux ignores sel -> independence/source fail
    mux_c = {"design_name": "mux2", "design_type": "combinational",
             "inputs": {"a": {"width": 2}, "b": {"width": 2}, "sel": {"width": 1}},
             "outputs": {"y": {"width": 2}},
             "rules": [{"id": "R1", "condition": "sel==0", "effect": "y = a"},
                       {"id": "R2", "condition": "sel==1", "effect": "y = b"}]}
    with tempfile.TemporaryDirectory() as d:
        gp = write(d, "class GoldenDUT:\n    def eval(self,i): return {'y': (i['b'] if i['sel'] else i['a']) & 3}\n")
        rm = M.run(gp, mux_c)
        ok = ok and rm["status"] == "pass"
        ok = ok and any(p["name"] == "mux_independence" and p["status"] == "pass" for p in rm["properties"])
        gp = write(d, "class GoldenDUT:\n    def eval(self,i): return {'y': i['a'] & 3}\n")  # ignores sel & b
        rmb = M.run(gp, mux_c)
        ok = ok and rmb["status"] == "fail"

    # counter: correct increments/holds/reset; broken ignores enable
    cnt_c = {"design_name": "counter", "design_type": "sequential",
             "inputs": {"reset": {"width": 1}, "en": {"width": 1}}, "outputs": {"q": {"width": 2}},
             "reset": {"name": "reset", "active_high": True}, "states": [],
             "rules": [{"id": "R1", "description": "increment when en"}]}
    good_cnt = ("class GoldenDUT:\n    def __init__(self): self.reset()\n    def reset(self): self.q=0\n"
                "    def step(self,i):\n        if i.get('reset'): self.q=0\n        elif i.get('en'): self.q=(self.q+1)&3\n"
                "        return {'q': self.q}\n    def get_state(self): return {'q': self.q}\n")
    with tempfile.TemporaryDirectory() as d:
        gp = write(d, good_cnt)
        rc = M.run(gp, cnt_c)
        ok = ok and rc["status"] == "pass"
        ok = ok and any(p["name"] == "counter_increment" and p["status"] == "pass" for p in rc["properties"])
        gp = write(d, good_cnt.replace("(self.q+1)&3", "self.q"))  # ignores en
        rcb = M.run(gp, cnt_c)
        ok = ok and rcb["status"] == "fail"
        ok = ok and any(p["name"] == "counter_increment" and p["status"] == "fail" for p in rcb["properties"])

    print("metamorphic_checker selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
