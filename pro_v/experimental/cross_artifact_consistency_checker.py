#!/usr/bin/env python3
"""
cross_artifact_consistency_checker.py  --  cross-artifact consistency (Pro-V stage 8).

Each SpecKit artifact can look valid alone yet conflict with another: the contract
says AND but a witness expects OR's answer; the contract says reset active-high but a
witness asserts reset=0; an invariant's width contradicts a witness value; the
coverage plan drops a required witness; the golden model cites a rule that does not
exist. This checker cross-validates them and reports minimal conflicting artifacts
with human-readable explanations and suggested fixes.

Run it after: (1) contract/witnesses/invariants are created, (2) golden_dut.py is
generated, (3) coverage_plan.json is generated.

Checks: rule coverage, witness-vs-contract, invariant-vs-witness, ambiguity gate,
golden traceability (+ extra-rule), coverage-plan completeness, black-box rule.
It never treats top_module.v internals as truth.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from pro_v.witness_generator import _output_exprs, eval_effect
except Exception:  # pragma: no cover
    from witness_generator import _output_exprs, eval_effect


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _load(x):
    if isinstance(x, str) and os.path.exists(x):
        with open(x) as f:
            return json.load(f)
    return x if x is not None else {}


def _wl(w):
    if isinstance(w, dict):
        w = w.get("witnesses", [])
    return [x for x in (w or []) if isinstance(x, dict)]


def _il(i):
    if isinstance(i, dict):
        i = i.get("invariants", [])
    return [x for x in (i or []) if isinstance(x, dict)]


# ---------------------------------------------------------------------------
# checker
# ---------------------------------------------------------------------------

# artifacts that mention RTL internals as a *justification* are a black-box violation
_RTL_TOKENS = re.compile(
    r"\btop_module\b|\bnetlist\b|gate[- ]level|\bthe rtl\b|rtl internal|internal (?:wire|signal|net)"
    r"|because .*\bassign\b|\balways\s*@|posedge\s+clk\b", re.I)


class CrossArtifactConsistencyChecker:

    def run(self, spec: str = "", contract: Any = None, witnesses: Any = None,
            invariants: Any = None, golden_dut_path: Optional[str] = None,
            coverage_plan: Any = None, ambiguities_md: Optional[str] = None,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        contract = _load(contract)
        wl = _wl(_load(witnesses))
        inv = _il(_load(invariants))
        plan = _load(coverage_plan) if coverage_plan is not None else None
        if ambiguities_md is None and output_dir:
            ambiguities_md = _read(os.path.join(output_dir, "ambiguities.md"))

        conflicts: List[dict] = []
        warnings: List[dict] = []
        checks: Dict[str, str] = {}

        blocking = self._ck_ambiguity(contract, ambiguities_md or "", conflicts, checks)
        self._ck_rule_coverage(contract, wl, inv, warnings, checks)
        self._ck_witness_contract(contract, wl, conflicts, checks)
        self._ck_invariant_witness(contract, inv, wl, conflicts, checks)
        self._ck_reset_polarity(contract, wl, conflicts, warnings, checks)
        if golden_dut_path:
            self._ck_golden_traceability(golden_dut_path, contract, conflicts, warnings, checks)
        if plan is not None:
            self._ck_coverage_completeness(plan, wl, conflicts, checks)
        self._ck_black_box(spec, contract, wl, inv, conflicts, checks)

        hard = [c for c in conflicts if c.get("severity") in ("critical", "error")]
        status = "blocked" if blocking else ("inconsistent" if hard else "consistent")
        report = {
            "status": status, "blocking": blocking,
            "num_conflicts": len(conflicts), "num_warnings": len(warnings),
            "conflicts": conflicts, "warnings": warnings, "checks": checks,
        }
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "cross_artifact_consistency_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            with open(os.path.join(output_dir, "cross_artifact_consistency_report.md"), "w") as f:
                f.write(self._render_md(report))
            report["json_path"] = os.path.join(output_dir, "cross_artifact_consistency_report.json")
            report["md_path"] = os.path.join(output_dir, "cross_artifact_consistency_report.md")
        return report

    # -- 4) ambiguity gate -------------------------------------------------

    def _ck_ambiguity(self, contract, amb_md, conflicts, checks) -> bool:
        blocked = contract.get("can_generate_golden_model") is False
        if not blocked and re.search(r"^>\s*\*\*BLOCKING", amb_md, re.M):
            blocked = True
        if blocked:
            conflicts.append({
                "id": "X_AMBIGUITY", "severity": "critical", "check": "ambiguity_gate",
                "artifacts": ["behavior_contract.json", "ambiguities.md"],
                "explanation": "A critical ambiguity is unresolved (can_generate_golden_model=false).",
                "suggested_fix": "Resolve the critical ambiguities before generating golden_dut.py."})
        checks["ambiguity_gate"] = "block" if blocked else "pass"
        return blocked

    # -- 1) rule coverage --------------------------------------------------

    def _ck_rule_coverage(self, contract, wl, inv, warnings, checks) -> None:
        covered = set()
        for src in (wl, inv):
            for item in src:
                for rid in item.get("covers_rules", []) or []:
                    covered.add(rid)
        missing = []
        for r in contract.get("rules", []) or []:
            rid = r.get("id")
            if not rid or r.get("untestable"):
                continue
            if rid not in covered:
                missing.append(rid)
        for rid in missing:
            warnings.append({
                "id": "X_UNCOVERED_%s" % rid, "severity": "warning", "check": "rule_coverage",
                "artifacts": ["behavior_contract.%s" % rid],
                "explanation": "Rule %s is not covered by any witness or invariant." % rid,
                "suggested_fix": "Add a witness/invariant that covers %s, or mark it untestable." % rid})
        checks["rule_coverage"] = "warn" if missing else "pass"

    # -- 2) witness vs contract --------------------------------------------

    def _ck_witness_contract(self, contract, wl, conflicts, checks) -> None:
        out_names = list((contract.get("outputs") or {}).keys())
        out_w = {n: _w(s) for n, s in (contract.get("outputs") or {}).items()}
        exprs = _output_exprs(contract.get("rules", []) or [], out_names)
        rule_of_out = self._rule_for_output(contract, out_names)
        bad = 0
        for wit in wl:
            if "sequence" in wit or wit.get("relation") or not wit.get("expected_outputs"):
                continue
            env = {k: _to_int(v) for k, v in (wit.get("inputs") or {}).items()}
            env = {k: v for k, v in env.items() if v is not None}
            for o, ev in wit["expected_outputs"].items():
                if o not in exprs:
                    continue
                computed = eval_effect(o + " = " + exprs[o], env, out_w.get(o, 1))
                exp_i = _to_int(ev)
                if computed is not None and exp_i is not None and _to_int(computed) != exp_i:
                    bad += 1
                    rid = rule_of_out.get(o, "R?")
                    conflicts.append({
                        "id": "X_WIT_%s_%s" % (wit.get("id"), o), "severity": "critical",
                        "check": "witness_vs_contract",
                        "artifacts": ["behavior_contract.%s (%s = %s)" % (rid, o, exprs[o]),
                                      "witnesses.%s (%s -> %s=%s)" % (wit.get("id"), wit.get("inputs"), o, ev)],
                        "explanation": "Contract %s computes %s=%s for inputs %s, but witness %s expects %s=%s. "
                                       "These cannot both be true."
                                       % (rid, o, _to_int(computed), wit.get("inputs"), wit.get("id"), o, exp_i),
                        "suggested_fix": "Change %s's expected %s to %s, or revise %s if the intended operation "
                                         "differs (e.g. OR vs AND)." % (wit.get("id"), o, _to_int(computed), rid)})
        checks["witness_vs_contract"] = "fail" if bad else "pass"

    # -- 3) invariant vs witness -------------------------------------------

    def _ck_invariant_witness(self, contract, inv, wl, conflicts, checks) -> None:
        bad = 0
        # width invariant vs witness values
        for i in inv:
            strat = i.get("check_strategy", {})
            if strat.get("kind") == "width":
                o, W = strat.get("output"), strat.get("width", 1)
                for wit in wl:
                    for src in ([wit.get("expected_outputs")] +
                                [s.get("expected_outputs") for s in wit.get("sequence", []) or []]):
                        if not isinstance(src, dict) or o not in src:
                            continue
                        v = _to_int(src[o])
                        if v is not None and v >= (1 << W):
                            bad += 1
                            conflicts.append({
                                "id": "X_INV_%s_%s" % (i.get("id"), wit.get("id")), "severity": "critical",
                                "check": "invariant_vs_witness",
                                "artifacts": ["invariants.%s (%s < 2**%d)" % (i.get("id"), o, W),
                                              "witnesses.%s (%s=%s)" % (wit.get("id"), o, src[o])],
                                "explanation": "Invariant %s bounds %s to %d bits, but witness %s expects %s=%s "
                                               "(out of range)." % (i.get("id"), o, W, wit.get("id"), o, v),
                                "suggested_fix": "Fix the witness value or the declared width of %s." % o})
        # commutativity invariant vs witness pair
        for i in inv:
            strat = i.get("check_strategy", {})
            if strat.get("kind") == "metamorphic" and strat.get("relation") == "commutative":
                ops = strat.get("operands") or []
                if len(ops) == 2:
                    bad += self._commutativity_conflicts(i, ops, wl, contract, conflicts)
        checks["invariant_vs_witness"] = "fail" if bad else "pass"

    def _commutativity_conflicts(self, inv, ops, wl, contract, conflicts) -> int:
        a, b = ops
        table: Dict[Tuple[int, int], Dict[str, int]] = {}
        for wit in wl:
            ins = {k: _to_int(v) for k, v in (wit.get("inputs") or {}).items()}
            outs = {k: _to_int(v) for k, v in (wit.get("expected_outputs") or {}).items()}
            if a in ins and b in ins and outs:
                table[(ins[a], ins[b])] = outs
        found = 0
        for (x, y), o1 in table.items():
            o2 = table.get((y, x))
            if o2 is not None and x != y and o1 != o2:
                found += 1
                conflicts.append({
                    "id": "X_COMM_%s_%d_%d" % (inv.get("id"), x, y), "severity": "critical",
                    "check": "invariant_vs_witness",
                    "artifacts": ["invariants.%s (commutative)" % inv.get("id"),
                                  "witnesses (%s=%d,%s=%d -> %s ; swapped -> %s)" % (a, x, b, y, o1, o2)],
                    "explanation": "Invariant %s asserts %s/%s commute, but witnesses give different outputs "
                                   "for (%d,%d) vs (%d,%d)." % (inv.get("id"), a, b, x, y, y, x),
                    "suggested_fix": "Correct the witness outputs or drop the commutativity invariant."})
                break
        return found

    # -- reset polarity (contract vs witnesses) ----------------------------

    def _ck_reset_polarity(self, contract, wl, conflicts, warnings, checks) -> None:
        reset = contract.get("reset") or {}
        rname = reset.get("name")
        ah = reset.get("active_high")
        if not rname or ah in (None, "ambiguous"):
            checks["reset_polarity"] = "skip"
            return
        asserted = "1" if ah else "0"
        issue = False
        for wit in wl:
            if wit.get("type") != "reset_hold_enable":
                continue
            steps = wit.get("sequence") or []
            if not steps:
                continue
            first = str((steps[0].get("inputs") or {}).get(rname, ""))
            # the reset witness's first cycle should ASSERT reset
            if first and first not in ("", asserted) and set(first) <= set("01"):
                issue = True
                conflicts.append({
                    "id": "X_RESET_%s" % wit.get("id"), "severity": "error", "check": "reset_polarity",
                    "artifacts": ["behavior_contract.reset (active_%s)" % ("high" if ah else "low"),
                                  "witnesses.%s (asserts %s=%s)" % (wit.get("id"), rname, first)],
                    "explanation": "Contract says reset is active-%s (asserted=%s), but reset witness %s drives "
                                   "%s=%s on its assert cycle." % ("high" if ah else "low", asserted,
                                                                   wit.get("id"), rname, first),
                    "suggested_fix": "Align the witness with the contract polarity, or clarify reset polarity."})
        checks["reset_polarity"] = "fail" if issue else "pass"

    # -- 5) golden traceability --------------------------------------------

    def _ck_golden_traceability(self, golden_path, contract, conflicts, warnings, checks) -> None:
        src = _read(golden_path)
        if not src:
            checks["golden_traceability"] = "warn"
            warnings.append({"id": "X_GOLDEN_UNREADABLE", "severity": "warning",
                             "check": "golden_traceability", "artifacts": [golden_path],
                             "explanation": "Could not read golden_dut.py.", "suggested_fix": "Ensure the path is valid."})
            return
        rule_ids = {r.get("id") for r in (contract.get("rules") or []) if r.get("id")}
        cited = set(re.findall(r"#[^\n]*?\b(R\d+)\b", src))
        m = re.search(r"IMPLEMENTED_RULES\s*=\s*\[([^\]]*)\]", src)
        if m:
            cited |= set(re.findall(r"R\d+", m.group(1)))
        status = "pass"
        if not cited:
            status = "warn"
            warnings.append({"id": "X_GOLDEN_NOTRACE", "severity": "warning", "check": "golden_traceability",
                             "artifacts": ["golden_dut.py"],
                             "explanation": "golden_dut.py cites no rule IDs (no '# Implements Rx' or IMPLEMENTED_RULES).",
                             "suggested_fix": "Add rule-ID comments so behavior is traceable to the contract."})
        extra = sorted(cited - rule_ids)
        if extra:
            status = "fail"
            conflicts.append({
                "id": "X_GOLDEN_EXTRA", "severity": "critical", "check": "golden_traceability",
                "artifacts": ["golden_dut.py (cites %s)" % ", ".join(extra), "behavior_contract.json"],
                "explanation": "golden_dut.py cites rule(s) %s not present in the behavior contract." % ", ".join(extra),
                "suggested_fix": "Remove behavior not in the contract, or add the missing rule to the contract."})
        checks["golden_traceability"] = status

    # -- 6) coverage-plan completeness -------------------------------------

    def _ck_coverage_completeness(self, plan, wl, conflicts, checks) -> None:
        refs = set()
        plan_inputs = []
        for t in (plan.get("tests", []) or []) + (plan.get("sequences", []) or []):
            for c in t.get("covers", []) or []:
                if isinstance(c, str) and c.startswith("W"):
                    refs.add(c)
            if isinstance(t.get("inputs"), dict):
                plan_inputs.append({k: _to_int(v) for k, v in t["inputs"].items()})
        missing = []
        for wit in wl:
            if not wit.get("required"):
                continue
            wid = wit.get("id")
            # golden-only / not RTL-observable witnesses need not appear as directed vectors
            golden_only = bool(wit.get("relation")) or wit.get("type") == "metamorphic" or \
                (not wit.get("expected_outputs") and not any(
                    s.get("expected_outputs") for s in wit.get("sequence", []) or []))
            if wid in refs or golden_only:
                continue
            wi = {k: _to_int(v) for k, v in (wit.get("inputs") or {}).items()}
            if wi and any(all(pi.get(k) == v for k, v in wi.items()) for pi in plan_inputs):
                continue
            missing.append(wid)
        for wid in missing:
            conflicts.append({
                "id": "X_PLAN_MISSING_%s" % wid, "severity": "error", "check": "coverage_completeness",
                "artifacts": ["witnesses.%s (required)" % wid, "coverage_plan.json"],
                "explanation": "Required witness %s is not present in the coverage plan and is not marked "
                               "golden-only/not-RTL-observable." % wid,
                "suggested_fix": "Add %s to coverage_plan.json, or mark it golden-only (metamorphic/"
                                 "not observable on the RTL)." % wid})
        checks["coverage_completeness"] = "fail" if missing else "pass"

    # -- 7) black-box rule -------------------------------------------------

    def _ck_black_box(self, spec, contract, wl, inv, conflicts, checks) -> None:
        hits = []
        def scan(text, where):
            if isinstance(text, str) and _RTL_TOKENS.search(text):
                hits.append((where, text[:120]))
        for r in contract.get("rules", []) or []:
            scan(r.get("description"), "behavior_contract.%s.description" % r.get("id"))
        for tr in (contract.get("spec_traceability") or {}).values() if isinstance(
                contract.get("spec_traceability"), dict) else []:
            scan((tr or {}).get("source_text"), "spec_traceability")
        for wit in wl:
            scan(wit.get("reason"), "witnesses.%s.reason" % wit.get("id"))
        for i in inv:
            scan(i.get("description"), "invariants.%s.description" % i.get("id"))
            scan(i.get("note"), "invariants.%s.note" % i.get("id"))
        for where, snippet in hits:
            conflicts.append({
                "id": "X_BLACKBOX_%d" % len(conflicts), "severity": "error", "check": "black_box_rule",
                "artifacts": [where],
                "explanation": "An artifact justifies behavior using RTL internals: \"%s\". "
                               "top_module.v must be treated as a black box." % snippet,
                "suggested_fix": "Re-ground this reason in the spec/contract, not in the RTL."})
        checks["black_box_rule"] = "fail" if hits else "pass"

    # -- helpers -----------------------------------------------------------

    def _rule_for_output(self, contract, out_names) -> Dict[str, str]:
        m = {}
        for r in contract.get("rules", []) or []:
            eff = (r.get("effect") or "")
            if "=" in eff:
                lhs = re.sub(r"\[.*", "", eff.split("=", 1)[0]).strip()
                if lhs in out_names and lhs not in m:
                    m[lhs] = r.get("id", "R?")
        return m

    def _render_md(self, report) -> str:
        L = ["# Cross-Artifact Consistency — %s" % report["status"].upper(), ""]
        L.append("- conflicts: %d | warnings: %d" % (report["num_conflicts"], report["num_warnings"]))
        L.append("- checks: " + ", ".join("%s=%s" % (k, v) for k, v in report["checks"].items()))
        L.append("")
        if report["conflicts"]:
            L.append("## Conflicts")
            for c in report["conflicts"]:
                L.append("### [%s] %s (%s)" % (c.get("severity", "").upper(), c.get("id"), c.get("check")))
                L.append("- Artifacts: %s" % "; ".join(c.get("artifacts", [])))
                L.append("- %s" % c.get("explanation"))
                L.append("- **Suggested fix:** %s" % c.get("suggested_fix"))
                L.append("")
        if report["warnings"]:
            L.append("## Warnings")
            for w in report["warnings"]:
                L.append("- **%s** (%s): %s — _fix:_ %s" % (
                    w.get("id"), w.get("check"), w.get("explanation"), w.get("suggested_fix")))
        return "\n".join(L) + "\n"


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def check_consistency(**kw) -> Dict[str, Any]:
    return CrossArtifactConsistencyChecker().run(**kw)


# ---------------------------------------------------------------------------
# self-test (no LLM, no RTL)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    ok = True
    C = CrossArtifactConsistencyChecker()

    contract = {"design_name": "and_gate", "design_type": "combinational",
                "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"y": {"width": 1}},
                "rules": [{"id": "R1", "effect": "y = a & b", "description": "y = a AND b"}],
                "can_generate_golden_model": True}

    # consistent set
    good_w = {"witnesses": [
        {"id": "W1", "type": "distinguishing", "required": True, "covers_rules": ["R1"],
         "inputs": {"a": "0", "b": "1"}, "expected_outputs": {"y": "0"}, "reason": "AND vs OR"},
        {"id": "W2", "type": "distinguishing", "required": True, "covers_rules": ["R1"],
         "inputs": {"a": "1", "b": "1"}, "expected_outputs": {"y": "1"}, "reason": "1&1"}]}
    good_inv = {"invariants": [{"id": "INV_WIDTH_Y", "covers_rules": ["R1"],
                                "check_strategy": {"kind": "width", "output": "y", "width": 1}}]}
    plan = {"tests": [{"name": "t1", "inputs": {"a": 0, "b": 1}, "expected_outputs": {"y": 0}, "covers": ["W1", "R1"]},
                      {"name": "t2", "inputs": {"a": 1, "b": 1}, "expected_outputs": {"y": 1}, "covers": ["W2", "R1"]}],
            "sequences": []}
    r = C.run(contract=contract, witnesses=good_w, invariants=good_inv, coverage_plan=plan)
    ok = ok and r["status"] == "consistent" and r["num_conflicts"] == 0

    # witness vs contract: W1 expects y=1 (OR answer) -> conflict
    bad_w = json.loads(json.dumps(good_w))
    bad_w["witnesses"][0]["expected_outputs"]["y"] = "1"
    r2 = C.run(contract=contract, witnesses=bad_w, invariants=good_inv)
    ok = ok and r2["status"] == "inconsistent"
    wc = next((c for c in r2["conflicts"] if c["check"] == "witness_vs_contract"), None)
    ok = ok and wc and "cannot both be true" in wc["explanation"] and "OR vs AND" in wc["suggested_fix"]

    # invariant vs witness: width 1 but witness expects y=2
    wide_w = {"witnesses": [{"id": "W3", "required": True, "covers_rules": ["R1"],
                             "inputs": {"a": "1", "b": "1"}, "expected_outputs": {"y": "10"}}]}
    r3 = C.run(contract=contract, witnesses=wide_w, invariants=good_inv)
    ok = ok and any(c["check"] == "invariant_vs_witness" for c in r3["conflicts"])

    # rule coverage gap -> warning (R2 uncovered)
    c2 = json.loads(json.dumps(contract))
    c2["rules"].append({"id": "R2", "effect": "y = a | b"})
    r4 = C.run(contract=c2, witnesses=good_w, invariants=good_inv)
    ok = ok and any(w["check"] == "rule_coverage" and "R2" in w["id"] for w in r4["warnings"])

    # golden cites extra rule + missing traceability
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "golden_dut.py")
        with open(gp, "w") as f:
            f.write("class GoldenDUT:\n    def eval(self, i):\n        # Implements R9\n        return {'y': 0}\n")
        r5 = C.run(contract=contract, witnesses=good_w, invariants=good_inv, golden_dut_path=gp)
        ok = ok and any(c["check"] == "golden_traceability" and "R9" in c["explanation"] for c in r5["conflicts"])

    # coverage plan omits required witness W2
    plan_missing = {"tests": [{"name": "t1", "inputs": {"a": 0, "b": 1}, "expected_outputs": {"y": 0}, "covers": ["W1"]}],
                    "sequences": []}
    r6 = C.run(contract=contract, witnesses=good_w, invariants=good_inv, coverage_plan=plan_missing)
    ok = ok and any(c["check"] == "coverage_completeness" and "W2" in c["id"] for c in r6["conflicts"])

    # black-box violation: witness reason cites top_module
    bb_w = json.loads(json.dumps(good_w))
    bb_w["witnesses"][0]["reason"] = "because top_module assign says so"
    r7 = C.run(contract=contract, witnesses=bb_w, invariants=good_inv)
    ok = ok and any(c["check"] == "black_box_rule" for c in r7["conflicts"])

    # reset polarity: contract active-high, reset witness asserts reset=0
    seq_c = {"design_name": "cnt", "design_type": "sequential",
             "inputs": {"reset": {"width": 1}, "en": {"width": 1}}, "outputs": {"q": {"width": 2}},
             "reset": {"name": "reset", "active_high": True}, "states": ["A"], "initial_state": "A",
             "rules": [{"id": "R1", "description": "reset to A"}], "can_generate_golden_model": True}
    rp_w = {"witnesses": [{"id": "WR", "type": "reset_hold_enable", "required": True, "covers_rules": ["R1"],
                           "sequence": [{"cycle": 0, "inputs": {"reset": "0", "en": "0"}},
                                        {"cycle": 1, "inputs": {"reset": "1", "en": "0"}}]}]}
    r8 = C.run(contract=seq_c, witnesses=rp_w, invariants={"invariants": []})
    ok = ok and any(c["check"] == "reset_polarity" for c in r8["conflicts"])

    # ambiguity gate blocks
    amb_c = json.loads(json.dumps(contract)); amb_c["can_generate_golden_model"] = False
    r9 = C.run(contract=amb_c, witnesses=good_w, invariants=good_inv)
    ok = ok and r9["status"] == "blocked" and r9["blocking"]

    # report files written
    with tempfile.TemporaryDirectory() as d:
        rr = C.run(contract=contract, witnesses=bad_w, invariants=good_inv, output_dir=d)
        ok = ok and os.path.exists(rr["json_path"]) and os.path.exists(rr["md_path"])

    print("cross_artifact_consistency selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
