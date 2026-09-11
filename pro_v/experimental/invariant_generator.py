#!/usr/bin/env python3
"""
invariant_generator.py  --  deterministic invariant synthesis (Pro-V stage 1c).

Reads `behavior_contract.json` and produces `invariants.json`. Where witnesses are
concrete point examples, invariants are broader properties that must hold across
many inputs / cycles. They are used by the Golden Model Verifier (stage 4) to
validate `golden_dut.py` before the functional coverage agent is allowed to trust
it. Never uses `top_module.v` internals -- only the contract.

Each invariant carries a machine-readable `check_strategy` so the verifier can
execute it, plus a `severity` (critical / warning / optional) and an `assumption`
flag. Invariants inferred by convention (not stated in the spec) are marked
`assumption=true`, kept non-critical, and collected under `assumption_based_invariants`.
Nothing is emitted that contradicts the spec.

Invariant types: width, reset, hold, priority, reachability, output_domain,
arithmetic, metamorphic, safety, liveness.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from pro_v.witness_generator import _detect_operation, _is_control, _extract_fsm, _width_of
except Exception:  # pragma: no cover
    from witness_generator import _detect_operation, _is_control, _extract_fsm, _width_of


# ---------------------------------------------------------------------------
# spec-property detectors (only assert what the spec supports)
# ---------------------------------------------------------------------------

def _rules_blob(contract: dict) -> str:
    parts = []
    for r in contract.get("rules", []) or []:
        parts += [str(r.get("description", "")), str(r.get("condition", "")), str(r.get("effect", ""))]
    parts += [str(x) for x in contract.get("undefined_behavior", []) or []]
    parts += [str(x) for x in contract.get("assumptions", []) or []]
    return " ".join(parts).lower()


def _arith_semantics(blob: str) -> Optional[str]:
    if "saturat" in blob:
        return "saturate"
    if "wrap" in blob or "modulo" in blob or "overflow ignored" in blob or "truncat" in blob:
        return "wrap"
    return None


def _enable_hold_specified(blob: str) -> bool:
    return bool(re.search(r"(hold|retain|unchanged|stall|no update|does not (?:change|update))", blob))


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

class InvariantGenerator:

    def generate(self, contract: dict) -> Dict[str, Any]:
        self._n = 0
        inv: List[dict] = []
        inputs = contract.get("inputs", {}) if isinstance(contract.get("inputs"), dict) else {}
        outputs = contract.get("outputs", {}) if isinstance(contract.get("outputs"), dict) else {}
        rules = contract.get("rules", []) if isinstance(contract.get("rules"), list) else []
        states = contract.get("states", []) or []
        dtype = contract.get("design_type", "combinational")
        seq = (dtype == "sequential")
        domain = "sequential" if seq else "combinational"
        in_names = list(inputs.keys())
        out_names = list(outputs.keys())
        rule_ids = [r.get("id") for r in rules if r.get("id")]
        blob = _rules_blob(contract)

        reset = contract.get("reset", {}) if isinstance(contract.get("reset"), dict) else {}
        reset_name = reset.get("name") or next(
            (n for n in in_names if n.lower() in ("reset", "rst") or "rst" in n.lower().split("_")), None)
        enable = next((n for n in in_names if n.lower() in ("enable", "en") or n.lower().endswith("_en")), None)
        init_state = contract.get("initial_state") or (states[0] if states else None)

        # 1) WIDTH invariants (spec-derived: declared widths) -- critical
        for o in out_names:
            w = _width_of(outputs[o])
            inv.append(self._inv(
                "WIDTH_%s" % o.upper(), "width", domain,
                "Output %s must always fit in its declared %d-bit width." % (o, w),
                rule_ids, severity="critical", assumption=False,
                check="0 <= %s < 2**%d and len(bits(%s)) == %d" % (o, w, o, w),
                strategy={"kind": "width", "output": o, "width": w}))

        # 2) OUTPUT DOMAIN invariants -- only if the spec states a domain
        for o in out_names:
            spec = outputs.get(o, {})
            dom = spec.get("domain") if isinstance(spec, dict) else None
            if dom:
                inv.append(self._inv(
                    "DOMAIN_%s" % o.upper(), "output_domain", domain,
                    "Output %s must stay within its specified domain: %s." % (o, dom),
                    rule_ids, severity="critical", assumption=False,
                    check="%s in domain(%s)" % (o, dom),
                    strategy={"kind": "output_domain", "output": o, "domain": dom}))

        # 3) RESET invariants (sequential) -- critical if reset specified
        if seq and reset_name:
            polarity_amb = reset.get("active_high") in (None, "ambiguous")
            inv.append(self._inv(
                "RESET_INIT", "reset", "sequential",
                "When reset is asserted, the model returns to the initial state%s."
                % ((" %s" % init_state) if init_state else ""),
                rule_ids, severity="critical", assumption=polarity_amb,
                check="after reset, state == initial_state",
                strategy={"kind": "reset", "reset": reset_name,
                          "active_high": bool(reset.get("active_high", True)),
                          "initial_state": init_state,
                          "synchronous": reset.get("synchronous", "ambiguous")},
                note=("reset polarity ambiguous -> assumption" if polarity_amb else None)))

        # 4) HOLD / ENABLE invariants
        if seq and enable:
            specified = _enable_hold_specified(blob)
            inv.append(self._inv(
                "ENABLE_HOLD", "hold", "sequential",
                "When enable is 0 (and reset inactive), state/output must not update.",
                rule_ids, severity="critical" if specified else "warning",
                assumption=not specified,
                check="if %s == 0 and reset == 0: next_state == current_state" % enable,
                strategy={"kind": "hold", "enable": enable, "reset": reset_name}))

        # 5) CONTROL PRIORITY invariants
        priorities = contract.get("priority_rules", []) or []
        if priorities:
            for i, p in enumerate(priorities):
                inv.append(self._inv(
                    "PRIORITY_%d" % (i + 1), "priority", domain,
                    "Control priority: %s." % p, rule_ids,
                    severity="critical", assumption=False,
                    check=str(p), strategy={"kind": "priority", "rule": str(p)}))
        elif seq and reset_name and enable:
            # convention: reset dominates enable -- assumption unless stated
            inv.append(self._inv(
                "PRIORITY_RESET_OVER_ENABLE", "priority", "sequential",
                "When reset and enable are both asserted, reset takes priority.",
                rule_ids, severity="warning", assumption=True,
                check="if reset asserted: reset behavior regardless of %s" % enable,
                strategy={"kind": "priority", "high": reset_name, "low": enable, "expect": "reset_wins"}))

        # 6) STATE REACHABILITY invariants (sequential)
        if seq and states:
            inv.append(self._inv(
                "STATE_REACHABILITY", "reachability", "sequential",
                "Every declared state must be reachable from the initial state.",
                rule_ids, severity="warning", assumption=False,
                check="for s in states: exists input path init -> s",
                strategy={"kind": "reachability", "states": list(states), "initial_state": init_state}))

        # 7) ARITHMETIC invariants
        op = _detect_operation(rules)
        if op in ("add", "sub", "mul"):
            sem = _arith_semantics(blob)
            if sem:
                inv.append(self._inv(
                    "ARITH_%s" % sem.upper(), "arithmetic", domain,
                    "Arithmetic result uses %s semantics on overflow." % sem,
                    rule_ids, severity="critical", assumption=False,
                    check="result follows %s on overflow" % sem,
                    strategy={"kind": "arithmetic", "semantics": sem, "op": op}))
            else:
                inv.append(self._inv(
                    "ARITH_WRAP_ASSUMED", "arithmetic", domain,
                    "Overflow behavior unspecified; assuming modular wrap to output width.",
                    rule_ids, severity="optional", assumption=True,
                    check="result wraps modulo 2**width",
                    strategy={"kind": "arithmetic", "semantics": "wrap", "op": op, "assumed": True}))

        # 8) COUNTER STEP (sequential arithmetic) invariant
        if seq and (re.search(r"count", contract.get("design_name", ""), re.I) or "increment" in blob):
            cnt = out_names[0] if out_names else None
            if cnt:
                inv.append(self._inv(
                    "COUNTER_STEP", "arithmetic", "sequential",
                    "Counter increments by one per enabled cycle unless reset/hold active.",
                    rule_ids, severity="critical" if "increment" in blob else "warning",
                    assumption="increment" not in blob,
                    check="if reset==0 and (%s): next_%s == (%s + 1) mod 2**width"
                    % ((enable + "==1") if enable else "enabled", cnt, cnt),
                    strategy={"kind": "counter_step", "count": cnt, "enable": enable, "reset": reset_name,
                              "width": _width_of(outputs.get(cnt, {}))}))

        # 9) METAMORPHIC invariants
        if op in ("add", "and", "or", "xor", "mul") and len(in_names) >= 2:
            stated = "commutat" in blob
            inv.append(self._inv(
                "%s_COMMUTATIVE" % op.upper(), "metamorphic", domain,
                "%s is commutative: f(a,b) == f(b,a)." % op, rule_ids,
                severity="warning", assumption=not stated,
                check="f(%s,%s) == f(%s,%s)" % (in_names[0], in_names[1], in_names[1], in_names[0]),
                strategy={"kind": "metamorphic", "relation": "commutative",
                          "operands": [in_names[0], in_names[1]]}))
        sel = next((n for n in in_names if _is_control(n) and n.lower().startswith("sel")), None)
        if sel:
            data = [n for n in in_names if n != sel and not _is_control(n)]
            if len(data) >= 2:
                inv.append(self._inv(
                    "MUX_UNSELECTED_INDEPENDENCE", "metamorphic", domain,
                    "Changing the unselected input must not change the output.",
                    rule_ids, severity="critical", assumption=False,   # direct consequence of the selection rules
                    check="if %s selects one input, the other input does not affect output" % sel,
                    strategy={"kind": "metamorphic", "relation": "mux_independence",
                              "select": sel, "data": data}))

        # 10) SAFETY invariants (output tied to a state)  &  bounded LIVENESS
        for (out_sig, st) in self._output_state_bindings(rules, out_names, states):
            inv.append(self._inv(
                "SAFETY_%s_IN_%s" % (out_sig.upper(), st.upper()), "safety", "sequential",
                "%s may only be asserted in state %s." % (out_sig, st), rule_ids,
                severity="critical", assumption=False,
                check="%s == 1 implies state == %s" % (out_sig, st),
                strategy={"kind": "safety", "antecedent": {"output": out_sig, "value": "1"},
                          "consequent": {"state": st}}))
        if seq and states:
            fsm = _extract_fsm(rules, states)
            terminal = next((s for s in states if re.search(r"done|finish|accept|complete", s, re.I)), None)
            if terminal and fsm["edges"]:
                inv.append(self._inv(
                    "LIVENESS_%s" % terminal.upper(), "liveness", "sequential",
                    "State %s is reachable from reset within a bounded number of cycles." % terminal,
                    rule_ids, severity="optional", assumption=False,
                    check="exists input sequence reaching %s within %d cycles" % (terminal, max(2, len(states) + 1)),
                    strategy={"kind": "liveness_bounded", "target_state": terminal,
                              "max_cycles": max(2, len(states) + 1), "initial_state": init_state}))

        return self._package(inv)

    # -- helpers -----------------------------------------------------------

    def _output_state_bindings(self, rules, out_names, states):
        """Detect (output, state) pairs where the output is asserted only in a state."""
        pairs = []
        state_set = set(states or [])
        for r in rules:
            eff = (r.get("effect") or "")
            cond = (r.get("condition") or "") + " " + (r.get("description") or "")
            m_out = re.search(r"([A-Za-z_]\w*)\s*=\s*1\b", eff)
            m_st = re.search(r"state\s*==\s*([A-Za-z_]\w*)", cond) or \
                re.search(r"\bin\s+(?:state\s+)?([A-Za-z_]\w*)", cond, re.I)
            if m_out and m_out.group(1) in out_names and m_st and m_st.group(1) in state_set:
                pair = (m_out.group(1), m_st.group(1))
                if pair not in pairs:
                    pairs.append(pair)
        return pairs

    def _inv(self, suffix, itype, domain, description, covers, severity, assumption,
             check, strategy, note=None) -> dict:
        self._n += 1
        d = {
            "id": "INV_%s" % suffix,
            "type": itype,
            "description": description,
            "covers_rules": list(covers) if covers else [],
            "domain": domain,
            "severity": severity,
            "assumption": bool(assumption),
            "check": check,
            "check_strategy": strategy,
        }
        if note:
            d["note"] = note
        return d

    def _package(self, inv: List[dict]) -> Dict[str, Any]:
        critical = [i for i in inv if i["severity"] == "critical"]
        assumption_based = [i for i in inv if i["assumption"]]
        by_type: Dict[str, int] = {}
        for i in inv:
            by_type[i["type"]] = by_type.get(i["type"], 0) + 1
        return {
            "invariants": inv,
            "critical_invariants": critical,
            "assumption_based_invariants": assumption_based,
            "summary": {
                "total": len(inv),
                "critical": len(critical),
                "warning": sum(1 for i in inv if i["severity"] == "warning"),
                "optional": sum(1 for i in inv if i["severity"] == "optional"),
                "assumption_based": len(assumption_based),
                "by_type": by_type,
            },
        }

    def run(self, contract_path: str, out_path: Optional[str] = None) -> Dict[str, Any]:
        with open(contract_path) as f:
            contract = json.load(f)
        result = self.generate(contract)
        out_path = out_path or os.path.join(os.path.dirname(contract_path) or ".", "invariants.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        result["path"] = out_path
        return result


def generate_invariants(contract: dict) -> Dict[str, Any]:
    return InvariantGenerator().generate(contract)


# ---------------------------------------------------------------------------
# self-test (no LLM, no RTL)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    ok = True

    # mux: width(y) + mux_independence(critical, not assumption)
    mux = {"design_name": "mux2", "design_type": "combinational",
           "inputs": {"a": {"width": 1}, "b": {"width": 1}, "sel": {"width": 1}},
           "outputs": {"y": {"width": 1}},
           "rules": [{"id": "R1", "condition": "sel == 0", "effect": "y = a"},
                     {"id": "R2", "condition": "sel == 1", "effect": "y = b"}]}
    r = generate_invariants(mux)
    ids = {i["id"] for i in r["invariants"]}
    ok = ok and "INV_WIDTH_Y" in ids and "INV_MUX_UNSELECTED_INDEPENDENCE" in ids
    mi = next(i for i in r["invariants"] if i["id"] == "INV_MUX_UNSELECTED_INDEPENDENCE")
    ok = ok and mi["severity"] == "critical" and mi["assumption"] is False
    ok = ok and mi["check_strategy"]["kind"] == "metamorphic"
    ok = ok and any(i["type"] == "width" for i in r["critical_invariants"])

    # adder with no stated overflow -> commutative(assumption) + wrap-assumed(optional,assumption)
    add = {"design_name": "adder", "design_type": "combinational",
           "inputs": {"a": {"width": 4}, "b": {"width": 4}}, "outputs": {"s": {"width": 4}},
           "rules": [{"id": "R1", "effect": "s = a + b", "description": "sum of a and b"}]}
    ra = generate_invariants(add)
    ids = {i["id"] for i in ra["invariants"]}
    ok = ok and "INV_ADD_COMMUTATIVE" in ids and "INV_ARITH_WRAP_ASSUMED" in ids
    wrap = next(i for i in ra["invariants"] if i["id"] == "INV_ARITH_WRAP_ASSUMED")
    ok = ok and wrap["assumption"] is True and wrap in ra["assumption_based_invariants"]

    # adder with stated saturation -> critical saturate, not assumption
    sat = json.loads(json.dumps(add))
    sat["rules"][0]["description"] = "sum of a and b, saturating on overflow"
    rs = generate_invariants(sat)
    sa = next(i for i in rs["invariants"] if i["type"] == "arithmetic")
    ok = ok and sa["check_strategy"]["semantics"] == "saturate" and sa["severity"] == "critical"

    # FSM: reset(critical) + reachability + safety(done in DONE) + liveness(DONE)
    fsm = {"design_name": "ctrl", "design_type": "sequential",
           "inputs": {"reset": {"width": 1}, "enable": {"width": 1}, "start": {"width": 1}},
           "outputs": {"done": {"width": 1}},
           "reset": {"name": "reset", "active_high": True},
           "states": ["IDLE", "RUN", "DONE"], "initial_state": "IDLE",
           "rules": [
               {"id": "R1", "condition": "reset", "effect": "next_state = IDLE"},
               {"id": "R2", "condition": "state == IDLE and start == 1", "effect": "next_state = RUN"},
               {"id": "R3", "condition": "state == RUN", "effect": "next_state = DONE"},
               {"id": "R4", "condition": "state == DONE", "effect": "done = 1", "description": "done high in DONE"}]}
    rf = generate_invariants(fsm)
    ids = {i["id"] for i in rf["invariants"]}
    ok = ok and "INV_RESET_INIT" in ids and "INV_STATE_REACHABILITY" in ids
    ok = ok and "INV_ENABLE_HOLD" in ids and "INV_LIVENESS_DONE" in ids
    ok = ok and any(i["type"] == "safety" and i["check_strategy"]["consequent"]["state"] == "DONE"
                    for i in rf["invariants"])
    # enable hold not spec-stated here -> assumption/warning
    eh = next(i for i in rf["invariants"] if i["id"] == "INV_ENABLE_HOLD")
    ok = ok and eh["assumption"] is True and eh["severity"] == "warning"

    print("invariant_generator selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        res = InvariantGenerator().run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print("wrote", res["path"], "-", res["summary"])
    else:
        raise SystemExit(_selftest())
