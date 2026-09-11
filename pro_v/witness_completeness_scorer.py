#!/usr/bin/env python3
"""
witness_completeness_scorer.py  --  witness strength scoring (Pro-V stage 9).

Measures whether the generated witnesses are strong enough to prevent VACUOUS
correctness -- a model that passes only because it was exercised on non-distinguishing
inputs (AND and OR both give 0 on 00 and 1 on 11). Reads behavior_contract.json +
witnesses.json (+ coverage_plan.json if available) and reports per-category coverage
with a summary score and an optional pipeline gate.

Categories: rule, operation, control_mode, boundary_value, output_class,
distinguishing_case, negative_case, state, transition, reset_enable_hold, metamorphic.
Never uses top_module.v. "Expected" totals come from the contract; "covered" comes
from the witnesses.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from pro_v.witness_generator import (_detect_operation, boundary_values, _is_control,
                                         _extract_fsm, _output_exprs, eval_effect)
except Exception:  # pragma: no cover
    from witness_generator import (_detect_operation, boundary_values, _is_control,
                                   _extract_fsm, _output_exprs, eval_effect)

# distinguishing weight is doubled: these are the categories that actually stop
# vacuous correctness.
_WEIGHTS = {"distinguishing_case": 2.0, "operation": 2.0}


def _to_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s and set(s) <= set("01"):
            return int(s, 2)
        if re.fullmatch(r"\d+", s):
            return int(s)
    return None


def _w(spec: Any) -> int:
    try:
        return max(1, int(spec.get("width", 1))) if isinstance(spec, dict) else 1
    except Exception:
        return 1


def _point_inputs(wl: List[dict]) -> List[Tuple[dict, dict]]:
    out = []
    for wit in wl:
        if "sequence" in wit or wit.get("relation"):
            continue
        ins = {k: _to_int(v) for k, v in (wit.get("inputs") or {}).items()}
        ins = {k: v for k, v in ins.items() if v is not None}
        outs = {k: _to_int(v) for k, v in (wit.get("expected_outputs") or {}).items()}
        outs = {k: v for k, v in outs.items() if v is not None}
        out.append((ins, outs))
    return out


class WitnessCompletenessScorer:

    def score(self, contract: Any, witnesses: Any, coverage_plan: Any = None,
              mode: str = "permissive", threshold: float = 0.75,
              output_dir: Optional[str] = None) -> Dict[str, Any]:
        contract = _load(contract)
        wl = _wl(_load(witnesses))
        plan = _load(coverage_plan) if coverage_plan is not None else None
        seq = contract.get("design_type") == "sequential"
        in_names = list((contract.get("inputs") or {}).keys())
        in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        out_names = list((contract.get("outputs") or {}).keys())
        out_w = {n: _w(s) for n, s in (contract.get("outputs") or {}).items()}
        rules = contract.get("rules", []) or []
        pts = _point_inputs(wl)
        warnings: List[str] = []
        cats: Dict[str, dict] = {}

        cats["rule_coverage"] = self._cat_rule(rules, wl)
        cats["operation_coverage"] = self._cat_operation(rules, pts, warnings)
        cats["control_mode_coverage"] = self._cat_control(in_names, in_w, pts, warnings)
        cats["boundary_value_coverage"] = self._cat_boundary(in_names, in_w, contract, pts)
        cats["output_class_coverage"] = self._cat_output(contract, in_names, in_w, out_names, out_w, rules, wl, plan)
        cats["distinguishing_case"] = self._cat_distinguishing(rules, in_names, in_w, pts, warnings)
        cats["negative_case"] = self._cat_negative(pts, out_names)
        if seq:
            cats["state_coverage"] = self._cat_state(contract, wl, warnings)
            cats["transition_coverage"] = self._cat_transition(contract, rules, wl)
            cats["reset_enable_hold_coverage"] = self._cat_reset_hold(contract, in_names, in_w, wl, pts, warnings)
        cats["metamorphic_coverage"] = self._cat_metamorphic(rules, in_names, contract, wl, seq)

        # weighted score over applicable categories (total > 0)
        num = den = 0.0
        for name, c in cats.items():
            if c["total"] <= 0:
                c["severity"] = c.get("severity", "n/a")
                continue
            wgt = _WEIGHTS.get(name.replace("_coverage", "").replace("_case", "_case"), 1.0)
            wgt = _WEIGHTS.get(name, wgt)
            ratio = c["covered"] / c["total"]
            num += wgt * ratio
            den += wgt
        summary = round(num / den, 4) if den else 1.0

        if summary >= threshold:
            gate = "pass"
        else:
            gate = "fail" if mode == "strict" else "warn"

        report = {
            "design_name": contract.get("design_name"),
            "design_type": contract.get("design_type"),
            "summary_score": summary,
            "gate": gate, "mode": mode, "threshold": threshold,
            "categories": cats,
            "warnings": warnings,
        }
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "witness_completeness_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            with open(os.path.join(output_dir, "witness_completeness_report.md"), "w") as f:
                f.write(self._render_md(report))
            report["json_path"] = os.path.join(output_dir, "witness_completeness_report.json")
            report["md_path"] = os.path.join(output_dir, "witness_completeness_report.md")
        return report

    # -- categories --------------------------------------------------------

    def _mk(self, covered, total, missing, severity, note=None) -> dict:
        d = {"covered": covered, "total": total, "missing": missing[:8], "severity": severity}
        if len(missing) > 8:
            d["missing_truncated"] = len(missing) - 8
        if note:
            d["note"] = note
        return d

    def _cat_rule(self, rules, wl) -> dict:
        rule_ids = [r.get("id") for r in rules if r.get("id")]
        covered_ids = set()
        for wit in wl:
            covered_ids |= set(wit.get("covers_rules", []) or [])
        missing = ["Rule %s has no witness" % r for r in rule_ids if r not in covered_ids]
        return self._mk(len(rule_ids) - len([m for m in missing]), len(rule_ids), missing,
                        "warning" if missing else "ok")

    def _cat_operation(self, rules, pts, warnings) -> dict:
        op = _detect_operation(rules)
        if not op:
            return self._mk(0, 0, [], "n/a", "no single dominant operation detected")
        # operation is "exercised" iff witnesses produce more than one output class
        outclasses = {tuple(sorted(o.items())) for _, o in pts if o}
        if len(outclasses) >= 2:
            return self._mk(1, 1, [], "ok")
        warnings.append("Operation '%s' is not meaningfully exercised: witnesses produce a single output "
                        "class, so the operation could be anything with that constant output." % op)
        return self._mk(0, 1, ["witnesses do not exercise distinct outputs of the %s operation" % op], "critical")

    def _cat_control(self, in_names, in_w, pts, warnings) -> dict:
        controls = [n for n in in_names if _is_control(n)]
        total = 0
        want: Dict[str, set] = {}
        for c in controls:
            vals = set(range(1 << in_w[c])) if in_w[c] <= 4 else {0, (1 << in_w[c]) - 1}
            want[c] = vals
            total += len(vals)
        seen: Dict[str, set] = {c: set() for c in controls}
        for ins, _ in pts:
            for c in controls:
                if c in ins and ins[c] in want[c]:
                    seen[c].add(ins[c])
        covered = sum(len(seen[c]) for c in controls)
        missing = []
        for c in controls:
            for v in sorted(want[c] - seen[c]):
                missing.append("control %s=%d not witnessed" % (c, v))
        if missing and any(n.lower().startswith("sel") for n in controls):
            warnings.append("Not all mux modes are witnessed.")
        return self._mk(covered, total, missing, "warning" if missing else "ok")

    def _cat_boundary(self, in_names, in_w, contract, pts) -> dict:
        signed = {n: bool((contract.get("inputs") or {}).get(n, {}).get("signed")) for n in in_names}
        want: List[Tuple[str, str, int]] = []
        for n in in_names:
            for label, val in boundary_values(in_w[n], signed[n]):
                want.append((n, label, val))
        seen = set()
        for ins, _ in pts:
            for (n, label, val) in want:
                if ins.get(n) == val:
                    seen.add((n, label))
        missing = ["boundary %s on %s" % (label, n) for (n, label, _v) in want if (n, label) not in seen]
        return self._mk(len(want) - len(missing), len(want), missing, "warning" if missing else "ok")

    def _cat_output(self, contract, in_names, in_w, out_names, out_w, rules, wl, plan) -> dict:
        # totals from coverage_plan output_class goals if available; else from rule
        # evaluation over a small input space (deploy-safe, contract-only).
        expected = None
        if plan and isinstance(plan.get("coverage_goals"), list):
            oc = [g for g in plan["coverage_goals"] if g.get("type") == "output_class"]
            if oc:
                expected = set(g["id"] for g in oc)
                covered_ids = {c for t in (plan.get("tests", []) + plan.get("sequences", []))
                               for c in t.get("covers", []) if isinstance(c, str)}
                covered = len([g for g in expected if g in covered_ids])
                missing = ["output class %s not in plan tests" % g for g in expected if g not in covered_ids]
                return self._mk(covered, len(expected), missing, "warning" if missing else "ok",
                                note="from coverage_plan output-class goals")
        exprs = _output_exprs(rules, out_names)
        total_bits = sum(in_w.values())
        if not exprs or not out_names or total_bits > 12:
            return self._mk(0, 0, [], "n/a", "output-class totals need the golden model or a small space")
        classes = set()
        for combo in range(1 << total_bits):
            env = self._decode(combo, in_names, in_w)
            out = {}
            ok = True
            for o in out_names:
                if o in exprs:
                    b = eval_effect(o + " = " + exprs[o], env, out_w[o])
                    if b is None:
                        ok = False
                        break
                    out[o] = int(b, 2)
            if ok and out:
                classes.add(tuple(sorted(out.items())))
        wclasses = {tuple(sorted(o.items())) for _, o in _point_inputs(wl) if o}
        missing = ["output class %s not witnessed" % dict(c) for c in classes if c not in wclasses]
        return self._mk(len(classes) - len(missing), len(classes), missing, "warning" if missing else "ok",
                        note="derived from contract rules over the input space")

    def _cat_distinguishing(self, rules, in_names, in_w, pts, warnings) -> dict:
        op = _detect_operation(rules)
        two = in_names[:2]
        want: List[Tuple[str, Any]] = []          # (description, predicate over point inputs)
        def has(pred): return any(pred(ins) for ins, _ in pts)

        if op in ("and", "or", "nand", "nor") and len(two) >= 2:
            a, b = two
            want.append(("mixed input 01 (distinguishes %s from its dual)" % op,
                         lambda i: i.get(a) == 0 and i.get(b, 0) not in (0, None)))
            want.append(("mixed input 10 (distinguishes %s from its dual)" % op,
                         lambda i: i.get(a, 0) not in (0, None) and i.get(b) == 0))
        elif op == "xor" and len(two) >= 2:
            a, b = two
            want.append(("both-high 11 (distinguishes XOR from OR)",
                         lambda i: i.get(a, 0) and i.get(b, 0)))
            want.append(("mixed 01/10 (distinguishes XOR from AND)",
                         lambda i: (i.get(a) == 0) != (i.get(b) == 0)))
        elif op in ("compare", "compare_eq") and len(two) >= 2:
            a, b = two
            want.append(("equality a==b (distinguishes < from <=)", lambda i: i.get(a) == i.get(b)))
        elif op in ("add", "sub") and len(two) >= 2:
            a, b = two
            want.append(("asymmetric operands (distinguishes + from -)",
                         lambda i: i.get(a) != i.get(b) and (i.get(a) or i.get(b))))
        elif op == "not" and two:
            want.append(("input 0 (distinguishes NOT from identity)", lambda i: i.get(two[0]) == 0))
            want.append(("input 1 (distinguishes NOT from identity)", lambda i: i.get(two[0]) == 1))

        missing = []
        covered = 0
        for desc, pred in want:
            if has(pred):
                covered += 1
            else:
                missing.append("No %s" % desc)

        # specific vacuous-correctness warnings (prompt examples)
        outclasses = {tuple(sorted(o.items())) for _, o in pts if o}
        if op in ("and", "or") and covered < len(want) and len(outclasses) >= 2:
            warnings.append("These witnesses produce both output classes but do not distinguish AND from OR. "
                            "Add 01 and/or 10.")
        if op in ("compare", "compare_eq") and missing:
            warnings.append("Cannot distinguish < from <= without a == b witness.")
        sev = "critical" if missing else ("ok" if want else "n/a")
        return self._mk(covered, len(want), missing, sev)

    def _cat_negative(self, pts, out_names) -> dict:
        # a "negative" case = a witness where every output is the inactive value 0
        has_neg = any(o and all(v == 0 for v in o.values()) for _, o in pts)
        total = 1 if pts else 0
        missing = [] if has_neg else ["no witness produces an all-zero (inactive) output"]
        return self._mk(1 if has_neg else 0, total, missing, "warning" if (total and not has_neg) else "ok")

    def _cat_state(self, contract, wl, warnings) -> dict:
        states = contract.get("states", []) or []
        reached = set()
        for wit in wl:
            st = (wit.get("expected_final") or {}).get("state")
            if st is not None:
                reached.add(str(st))
        missing = []
        for s in states:
            if str(s) not in reached:
                missing.append("State %s has no reachability witness." % s)
                warnings.append("State %s has no reachability witness." % s)
        return self._mk(len(states) - len(missing), len(states), missing, "warning" if missing else "ok")

    def _cat_transition(self, contract, rules, wl) -> dict:
        fsm = _extract_fsm(rules, contract.get("states", []) or [])
        edges = fsm["edges"]
        # a transition is covered if a state_transition witness names it or its dst is reached
        covered_pairs = set()
        for wit in wl:
            if wit.get("type") == "state_transition":
                m = re.search(r"([A-Za-z_]\w*)\s*--.*-->\s*([A-Za-z_]\w*)", wit.get("description", ""))
                if m:
                    covered_pairs.add((m.group(1), m.group(2)))
        missing = []
        covered = 0
        for (src, _g, dst) in edges:
            if (src, dst) in covered_pairs:
                covered += 1
            else:
                missing.append("No witness for %s -> %s transition" % (src, dst))
        return self._mk(covered, len(edges), missing, "warning" if missing else "ok")

    def _cat_reset_hold(self, contract, in_names, in_w, wl, pts, warnings) -> dict:
        reset = contract.get("reset") or {}
        rname = reset.get("name") or next((n for n in in_names if n.lower() in ("reset", "rst")), None)
        enable = next((n for n in in_names if n.lower() in ("enable", "en") or n.lower().endswith("_en")), None)
        items, covered, missing = 0, 0, []
        if rname:
            items += 1
            if any(wit.get("type") == "reset_hold_enable" for wit in wl) or \
               any(rel_reset(wit, rname) for wit in wl):
                covered += 1
            else:
                missing.append("no reset witness")
        if enable:
            items += 1
            # enable-hold covered by an enable_hold metamorphic witness OR a sequence
            # that drives enable=0 while data changes
            if any(wit.get("relation") == "enable_hold" for wit in wl) or self._enable_tested(wl, enable, in_names):
                covered += 1
            else:
                missing.append("enable-hold not witnessed")
                warnings.append("Cannot detect enable-ignored bug.")
        return self._mk(covered, items, missing, "warning" if missing else "ok")

    def _enable_tested(self, wl, enable, in_names) -> bool:
        for wit in wl:
            steps = wit.get("sequence") or []
            saw_en0, data_changed = False, False
            prev = None
            for s in steps:
                ins = {k: _to_int(v) for k, v in (s.get("inputs") or {}).items()}
                if ins.get(enable) == 0:
                    saw_en0 = True
                    if prev is not None and any(ins.get(n) != prev.get(n) for n in in_names if n != enable):
                        data_changed = True
                    prev = ins
            if saw_en0 and data_changed:
                return True
        return False

    def _cat_metamorphic(self, rules, in_names, contract, wl, seq) -> dict:
        want = []
        op = _detect_operation(rules)
        if op in ("add", "and", "or", "xor", "mul") and len(in_names) >= 2:
            want.append("commutative")
        if any(_is_control(n) and n.lower().startswith("sel") for n in in_names):
            want.append("mux_independence")
        if seq:
            if any(n.lower() in ("enable", "en") for n in in_names):
                want.append("enable_hold")
            if (contract.get("reset") or {}).get("name"):
                want.append("reset_idempotence")
        present = {wit.get("relation") for wit in wl if wit.get("relation")}
        missing = ["no %s metamorphic witness" % r for r in want if r not in present]
        return self._mk(len(want) - len(missing), len(want), missing, "warning" if missing else "ok")

    # -- helpers -----------------------------------------------------------

    def _decode(self, combo, in_names, in_w):
        env, shift = {}, 0
        for n in reversed(in_names):
            w = in_w[n]
            env[n] = (combo >> shift) & ((1 << w) - 1)
            shift += w
        return env

    def _render_md(self, report) -> str:
        L = ["# Witness Completeness — score %.2f (gate: %s)" % (report["summary_score"], report["gate"]), ""]
        L.append("- design: %s (%s) | mode: %s | threshold: %.2f" % (
            report["design_name"], report["design_type"], report["mode"], report["threshold"]))
        L.append("")
        L.append("| category | covered | total | severity |")
        L.append("|---|---|---|---|")
        for name, c in report["categories"].items():
            L.append("| %s | %s | %s | %s |" % (name, c["covered"], c["total"], c["severity"]))
        miss = [(n, c) for n, c in report["categories"].items() if c["missing"]]
        if miss:
            L.append("")
            L.append("## Missing")
            for n, c in miss:
                L.append("- **%s**:" % n)
                for m in c["missing"]:
                    L.append("  - %s" % m)
        if report["warnings"]:
            L.append("")
            L.append("## Vacuous-correctness warnings")
            for w in report["warnings"]:
                L.append("- %s" % w)
        return "\n".join(L) + "\n"


def rel_reset(wit, rname) -> bool:
    return wit.get("relation") in ("reset_idempotence",) or (
        "reset" in (wit.get("type") or "") and rname in json.dumps(wit.get("sequence", [])))


def _load(x):
    if isinstance(x, str) and os.path.exists(x):
        with open(x) as f:
            return json.load(f)
    return x if x is not None else {}


def _wl(w):
    if isinstance(w, dict):
        w = w.get("witnesses", [])
    return [x for x in (w or []) if isinstance(x, dict)]


def score_completeness(contract, witnesses, coverage_plan=None, **kw) -> Dict[str, Any]:
    return WitnessCompletenessScorer().score(contract, witnesses, coverage_plan, **kw)


# ---------------------------------------------------------------------------
# self-test (no LLM, no RTL)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    ok = True
    S = WitnessCompletenessScorer()

    and_c = {"design_name": "and_gate", "design_type": "combinational",
             "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "effect": "y = a & b"}]}

    # vacuous: only 00 and 11 -> distinguishing missing, AND/OR warning fires
    vac = {"witnesses": [
        {"id": "W1", "type": "positive", "covers_rules": ["R1"], "inputs": {"a": "0", "b": "0"},
         "expected_outputs": {"y": "0"}},
        {"id": "W2", "type": "positive", "covers_rules": ["R1"], "inputs": {"a": "1", "b": "1"},
         "expected_outputs": {"y": "1"}}]}
    r = S.score(and_c, vac, mode="strict", threshold=0.9)
    ok = ok and r["categories"]["distinguishing_case"]["covered"] == 0
    ok = ok and r["categories"]["distinguishing_case"]["severity"] == "critical"
    ok = ok and any("distinguish AND from OR" in w for w in r["warnings"])
    ok = ok and r["gate"] == "fail" and r["summary_score"] < 0.9

    # strong: add 01 and 10 -> distinguishing full, score high
    strong = {"witnesses": vac["witnesses"] + [
        {"id": "W3", "type": "distinguishing", "covers_rules": ["R1"], "inputs": {"a": "0", "b": "1"},
         "expected_outputs": {"y": "0"}},
        {"id": "W4", "type": "distinguishing", "covers_rules": ["R1"], "inputs": {"a": "1", "b": "0"},
         "expected_outputs": {"y": "0"}}]}
    r2 = S.score(and_c, strong, threshold=0.75)
    ok = ok and r2["categories"]["distinguishing_case"]["covered"] == 2
    ok = ok and not any("distinguish AND from OR" in w for w in r2["warnings"])
    ok = ok and r2["summary_score"] > r["summary_score"]

    # comparator without equality -> warning
    cmp_c = {"design_name": "lt", "design_type": "combinational",
             "inputs": {"a": {"width": 2}, "b": {"width": 2}}, "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "effect": "y = (a < b)"}]}
    cmp_w = {"witnesses": [{"id": "W1", "covers_rules": ["R1"], "inputs": {"a": "01", "b": "10"},
                            "expected_outputs": {"y": "1"}},
                           {"id": "W2", "covers_rules": ["R1"], "inputs": {"a": "10", "b": "01"},
                            "expected_outputs": {"y": "0"}}]}
    rc = S.score(cmp_c, cmp_w)
    ok = ok and any("== b" in w for w in rc["warnings"])
    ok = ok and rc["categories"]["distinguishing_case"]["covered"] == 0

    # mux missing a select value -> "Not all mux modes are witnessed."
    mux_c = {"design_name": "mux2", "design_type": "combinational",
             "inputs": {"a": {"width": 1}, "b": {"width": 1}, "sel": {"width": 1}},
             "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "condition": "sel == 0", "effect": "y = a"},
                       {"id": "R2", "condition": "sel == 1", "effect": "y = b"}]}
    mux_w = {"witnesses": [{"id": "W1", "type": "control_mode", "covers_rules": ["R1"],
                            "inputs": {"a": "1", "b": "0", "sel": "0"}, "expected_outputs": {"y": "1"}}]}
    rm = S.score(mux_c, mux_w)
    ok = ok and any("mux modes" in w for w in rm["warnings"])
    ok = ok and any("sel=1" in m for m in rm["categories"]["control_mode_coverage"]["missing"])

    # FSM: state with no reachability witness
    fsm_c = {"design_name": "ctrl", "design_type": "sequential",
             "inputs": {"reset": {"width": 1}, "start": {"width": 1}}, "outputs": {"done": {"width": 1}},
             "reset": {"name": "reset", "active_high": True}, "states": ["IDLE", "RUN", "DONE"],
             "initial_state": "IDLE",
             "rules": [{"id": "R1", "condition": "reset", "effect": "next_state = IDLE"},
                       {"id": "R2", "condition": "state == IDLE and start == 1", "effect": "next_state = RUN"}]}
    fsm_w = {"witnesses": [{"id": "WS", "type": "sequential_reachability", "covers_rules": ["R2"],
                            "sequence": [{"cycle": 0, "inputs": {"reset": "1"}},
                                         {"cycle": 1, "inputs": {"reset": "0", "start": "1"}}],
                            "expected_final": {"state": "RUN"}}]}
    rf = S.score(fsm_c, fsm_w)
    ok = ok and any("DONE has no reachability" in w for w in rf["warnings"])
    ok = ok and rf["categories"]["state_coverage"]["covered"] == 1  # only RUN reached (+IDLE not final of any)

    # enable-ignored: sequential with enable, no enable=0-while-data-changes -> warning
    en_c = {"design_name": "reg", "design_type": "sequential",
            "inputs": {"reset": {"width": 1}, "en": {"width": 1}, "d": {"width": 4}},
            "outputs": {"q": {"width": 4}}, "reset": {"name": "reset", "active_high": True},
            "states": [], "rules": [{"id": "R1", "description": "load d when en"}]}
    en_w = {"witnesses": [{"id": "W1", "type": "sequential_reachability", "covers_rules": ["R1"],
                           "sequence": [{"cycle": 0, "inputs": {"reset": "1", "en": "0", "d": "0000"}},
                                        {"cycle": 1, "inputs": {"reset": "0", "en": "1", "d": "1010"}}]}]}
    re_ = S.score(en_c, en_w)
    ok = ok and any("enable-ignored" in w for w in re_["warnings"])

    print("witness_completeness_scorer selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
