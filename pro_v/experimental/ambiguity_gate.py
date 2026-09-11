#!/usr/bin/env python3
"""
ambiguity_gate.py  --  block unsafe generation (Pro-V stage 3-gate).

Runs after speckit_helper_agent.py and before pychecker_guidance_agent.py. It
blocks golden_dut.py generation when the spec is too ambiguous to implement safely.

It trusts upstream signals (behavior_contract.can_generate_golden_model and the
critical entries in ambiguities.md) AND independently runs structural detectors for
the critical-ambiguity classes, so a gap that slipped past SpecKit still stops the
pipeline. Non-critical ambiguities are surfaced as questions but do not block.

Output: pass/fail (passed=True means generation may proceed) and, when blocked,
clarification_request.md in the "Pipeline blocked / Ask user" format.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from pro_v.witness_generator import _detect_operation, _is_control
except Exception:  # pragma: no cover
    from witness_generator import _detect_operation, _is_control


# ---------------------------------------------------------------------------
# parsing ambiguities.md
# ---------------------------------------------------------------------------

def _parse_ambiguities_md(md: str) -> Tuple[List[dict], bool]:
    """Return ([{id,question,critical}], blocking_banner_present)."""
    if not md:
        return [], False
    banner = bool(re.search(r"^>\s*\*\*BLOCKING", md, re.M))
    items: List[dict] = []
    section_critical = None
    for line in md.splitlines():
        h = re.match(r"^##\s+(.*)", line)
        if h:
            title = h.group(1).lower()
            if "critical" in title and "non-critical" not in title and "non critical" not in title:
                section_critical = True
            elif "non-critical" in title or "non critical" in title:
                section_critical = False
            elif "repair loop" in title:
                section_critical = True
            else:
                section_critical = None
            continue
        m = re.match(r"^\s*[-*]\s+\*\*\[?([A-Za-z0-9_]+)\]?\s*(.*?)\*\*", line)
        if m:
            crit = bool(section_critical) or banner and section_critical is None
            items.append({"id": m.group(1), "question": m.group(2).strip(), "critical": crit})
    return items, banner


# ---------------------------------------------------------------------------
# structural detectors  (each returns (severity, reason, question))
# ---------------------------------------------------------------------------

def _resolved(contract: dict, *keywords: str) -> bool:
    """True if any assumption / undefined_behavior / rule text resolves the keyword."""
    blob = " ".join([str(x) for x in (contract.get("assumptions") or [])] +
                     [str(x) for x in (contract.get("undefined_behavior") or [])] +
                     [str(r.get("description", "")) + " " + str(r.get("effect", ""))
                      for r in (contract.get("rules") or [])]).lower()
    return any(k in blob for k in keywords)


def _structural_criticals(contract: dict) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    dtype = contract.get("design_type")
    inputs = contract.get("inputs", {}) or {}
    outputs = contract.get("outputs", {}) or {}
    in_names = list(inputs)
    rules = contract.get("rules", []) or []
    op = _detect_operation(rules)

    if dtype == "ambiguous":
        out.append(("critical", "The spec does not make clear whether the design is combinational or sequential.",
                    "Is the output combinational, or available after one clock cycle?"))

    if dtype == "sequential":
        has_clock = bool(contract.get("clock")) or any(n.lower() in ("clk", "clock") for n in in_names)
        if not has_clock:
            out.append(("critical", "Sequential design but no clock is specified.",
                        "Which signal is the clock, and on which edge does state update?"))
        reset = contract.get("reset") or {}
        rname = reset.get("name") or next((n for n in in_names if n.lower() in ("reset", "rst")), None)
        if rname:
            if reset.get("active_high") in (None, "ambiguous"):
                out.append(("critical", 'Spec references reset but does not specify active-high vs active-low.',
                            "Is reset active high?"))
            if reset.get("synchronous") in (None, "ambiguous") and not _resolved(contract, "synchron", "asynchron"):
                out.append(("question", "Reset timing (synchronous vs asynchronous) is unspecified.",
                            "Is reset synchronous or asynchronous?"))
        if contract.get("latency") is None and _resolved(contract, "registered", "next cycle", "one cycle") is False \
                and any("next" in str(r.get("description", "")).lower() for r in rules):
            out.append(("question", "Spec says \"next output\" but does not state the register latency.",
                        "Is the output combinational or registered by one clock cycle?"))

    # unknown output width
    for o, spec in outputs.items():
        w = spec.get("width") if isinstance(spec, dict) else None
        if not w or int(w) <= 0:
            out.append(("critical", "Output width for '%s' is unknown." % o,
                        "What is the bit width of output '%s'?" % o))

    # arithmetic overflow semantics
    if op in ("add", "sub", "mul") and not _resolved(contract, "wrap", "saturat", "modulo", "overflow", "truncat"):
        out.append(("critical", 'Spec performs arithmetic but does not state whether overflow wraps or saturates.',
                    "Should overflow wrap to the output width or saturate?"))

    # signedness where comparison depends on it
    if op in ("compare", "compare_eq"):
        signed_specified = any(isinstance(s, dict) and "signed" in s for s in inputs.values()) or \
            _resolved(contract, "signed", "unsigned", "two's complement", "twos complement")
        if not signed_specified:
            out.append(("critical", "Comparison result depends on signedness, which is unspecified.",
                        "Are the compared inputs signed or unsigned?"))

    # control priority when multiple controls can be asserted together
    controls = [n for n in in_names if _is_control(n)]
    if dtype == "sequential" and len(controls) >= 2 and not (contract.get("priority_rules")):
        out.append(("question", "Multiple controls (%s) can be asserted together but no priority is stated."
                    % ", ".join(controls),
                    "Which control takes priority when %s are asserted simultaneously?" % " and ".join(controls[:2])))

    # memory timing
    name_blob = (contract.get("design_name", "") + " " + " ".join(str(r.get("description", "")) for r in rules)).lower()
    if re.search(r"\b(memory|ram|register file|regfile)\b", name_blob) and not _resolved(contract, "synchronous read", "async read", "read latency", "combinational read"):
        out.append(("question", "Memory design but read timing (synchronous vs asynchronous) is unspecified.",
                    "Is memory read synchronous (registered) or asynchronous (combinational)?"))

    # valid/ready handshake semantics
    if any(n.lower() in ("valid", "ready") for n in in_names) and not _resolved(contract, "handshake", "consumed", "accepted when"):
        out.append(("question", "valid/ready present but handshake semantics (when data is consumed) are unspecified.",
                    "When is data consumed — on valid&&ready in the same cycle?"))

    # conflicting rules
    conflict = _conflicting_rules(rules, list(outputs))
    if conflict:
        a, b, o = conflict
        out.append(("critical", "Rules %s and %s give conflicting effects for output '%s' under the same condition."
                    % (a, b, o),
                    "Which rule is correct for '%s' (the spec has conflicting statements)?" % o))
    return out


def _conflicting_rules(rules: List[dict], out_names: List[str]) -> Optional[Tuple[str, str, str]]:
    by: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for r in rules:
        eff = (r.get("effect") or "")
        if "=" not in eff:
            continue
        lhs = re.sub(r"\[.*", "", eff.split("=", 1)[0]).strip()
        rhs = eff.split("=", 1)[1].strip()
        if lhs in out_names:
            cond = re.sub(r"\s+", " ", (r.get("condition") or "always").strip().lower())
            by.setdefault((lhs, cond), []).append((r.get("id", "R?"), rhs))
    for (o, _cond), lst in by.items():
        rhss = {rhs for _id, rhs in lst}
        if len(lst) >= 2 and len(rhss) >= 2:
            return lst[0][0], lst[1][0], o
    return None


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

class AmbiguityGate:

    def run(self, contract: Any, ambiguities_md: Optional[str] = None,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        contract = _load(contract)
        if ambiguities_md is None and output_dir:
            ambiguities_md = _read(os.path.join(output_dir, "ambiguities.md"))
        md_items, banner = _parse_ambiguities_md(ambiguities_md or "")
        md_critical = [a for a in md_items if a["critical"]]

        contract_flag_block = contract.get("can_generate_golden_model") is False
        structural = _structural_criticals(contract)
        struct_critical = [s for s in structural if s[0] == "critical"]

        blocked = bool(contract_flag_block or banner or md_critical or struct_critical)

        # assemble reasons + questions (deduped, order-preserving)
        reasons: List[str] = []
        questions: List[str] = []
        def add(reason, question):
            if reason and reason not in reasons:
                reasons.append(reason)
            if question and question not in questions:
                questions.append(question)

        for a in md_critical:
            add(a["question"], a["question"])
        for sev, reason, q in structural:
            if sev == "critical":
                add(reason, q)
        # non-blocking questions still surface for the user to answer in one pass
        for sev, reason, q in structural:
            if sev != "critical":
                if q not in questions:
                    questions.append(q)
        for a in md_items:
            if not a["critical"] and a["question"] and a["question"] not in questions:
                questions.append(a["question"])

        report = {
            "passed": not blocked,
            "blocked": blocked,
            "design_name": contract.get("design_name"),
            "reasons": reasons,
            "questions": questions,
            "critical_ambiguities": [{"id": a["id"], "question": a["question"]} for a in md_critical]
                                    + [{"id": "S%d" % i, "question": q} for i, (_s, _r, q) in enumerate(struct_critical)],
            "sources": {
                "contract_flag": contract_flag_block,
                "ambiguities_md_critical": len(md_critical),
                "ambiguities_md_banner": banner,
                "structural_critical": len(struct_critical),
            },
        }
        if output_dir and blocked:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "clarification_request.md")
            with open(path, "w") as f:
                f.write(self._render_request(contract.get("design_name", "design"), reasons, questions))
            report["clarification_request_path"] = path
        return report

    def _render_request(self, name: str, reasons: List[str], questions: List[str]) -> str:
        L = ["# Pipeline blocked — clarification required (%s)" % name, ""]
        L.append("Cannot safely generate `golden_dut.py` because:")
        for i, r in enumerate(reasons, 1):
            L.append("%d. %s" % (i, r))
        if not reasons:
            L.append("1. The specification has unresolved critical ambiguities.")
        L.append("")
        L.append("## Ask user")
        for q in questions:
            L.append("- %s" % q)
        if not questions:
            L.append("- Please clarify the flagged ambiguities before generation.")
        L.append("")
        L.append("_golden_dut.py generation is blocked until these are resolved._")
        return "\n".join(L) + "\n"


def _load(x):
    if isinstance(x, str) and os.path.exists(x):
        with open(x) as f:
            return json.load(f)
    return x if x is not None else {}


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def check_ambiguity(contract, ambiguities_md=None, output_dir=None) -> Dict[str, Any]:
    return AmbiguityGate().run(contract, ambiguities_md, output_dir)


# ---------------------------------------------------------------------------
# self-test (no LLM, no RTL)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    ok = True
    G = AmbiguityGate()

    # clean, fully specified AND -> pass
    and_c = {"design_name": "and_gate", "design_type": "combinational",
             "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "effect": "y = a & b"}], "can_generate_golden_model": True}
    r = G.run(and_c)
    ok = ok and r["passed"] and not r["blocked"]

    # contract flag blocks
    blk = dict(and_c); blk["can_generate_golden_model"] = False
    ok = ok and G.run(blk)["blocked"]

    # ambiguities.md critical entry blocks + question extracted
    md = ("# Ambiguities — x\n> **BLOCKING:** critical.\n\n## Critical (blocking)\n"
          "- **[A1] Is reset synchronous or asynchronous?**\n  - Why: timing\n")
    ra = G.run(and_c, ambiguities_md=md)
    ok = ok and ra["blocked"] and any("synchronous" in q for q in ra["questions"])

    # adder without overflow semantics -> blocked with overflow question
    add_c = {"design_name": "adder", "design_type": "combinational",
             "inputs": {"a": {"width": 4}, "b": {"width": 4}}, "outputs": {"s": {"width": 4}},
             "rules": [{"id": "R1", "effect": "s = a + b", "description": "add a and b"}],
             "can_generate_golden_model": True, "assumptions": []}
    r_add = G.run(add_c)
    ok = ok and r_add["blocked"] and any("wrap" in q.lower() and "saturate" in q.lower() for q in r_add["questions"])

    # adder with a wrap assumption -> overflow resolved (may still pass if nothing else)
    add_ok = json.loads(json.dumps(add_c)); add_ok["assumptions"] = ["overflow wraps modulo the output width"]
    r_add2 = G.run(add_ok)
    ok = ok and r_add2["passed"]

    # sequential with ambiguous reset polarity -> blocked, "Is reset active high?"
    seq_c = {"design_name": "ctrl", "design_type": "sequential", "clock": "clk",
             "inputs": {"reset": {"width": 1}, "start": {"width": 1}}, "outputs": {"done": {"width": 1}},
             "reset": {"name": "reset", "active_high": "ambiguous"}, "states": ["A", "B"], "initial_state": "A",
             "rules": [{"id": "R1", "description": "reset to A"}], "can_generate_golden_model": True}
    r_seq = G.run(seq_c)
    ok = ok and r_seq["blocked"] and any("active high" in q.lower() for q in r_seq["questions"])

    # unknown output width -> blocked
    now = {"design_name": "z", "design_type": "combinational",
           "inputs": {"a": {"width": 2}}, "outputs": {"y": {}}, "rules": [{"id": "R1", "effect": "y = a"}],
           "can_generate_golden_model": True}
    ok = ok and G.run(now)["blocked"]

    # compare without signedness -> blocked with signedness question
    cmp_c = {"design_name": "lt", "design_type": "combinational",
             "inputs": {"a": {"width": 4}, "b": {"width": 4}}, "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "effect": "y = (a < b)"}], "can_generate_golden_model": True}
    r_cmp = G.run(cmp_c)
    ok = ok and r_cmp["blocked"] and any("signed" in q.lower() for q in r_cmp["questions"])
    # compare WITH signedness declared -> passes
    cmp_ok = json.loads(json.dumps(cmp_c))
    cmp_ok["inputs"]["a"]["signed"] = False; cmp_ok["inputs"]["b"]["signed"] = False
    ok = ok and G.run(cmp_ok)["passed"]

    # conflicting rules -> blocked
    conf = {"design_name": "c", "design_type": "combinational",
            "inputs": {"a": {"width": 1}}, "outputs": {"y": {"width": 1}},
            "rules": [{"id": "R1", "condition": "always", "effect": "y = a"},
                      {"id": "R2", "condition": "always", "effect": "y = ~a"}],
            "can_generate_golden_model": True}
    r_conf = G.run(conf)
    ok = ok and r_conf["blocked"] and any("conflicting" in rr.lower() for rr in r_conf["reasons"])

    # clarification_request.md written on block, with the expected format
    with tempfile.TemporaryDirectory() as d:
        rr = G.run(add_c, output_dir=d)
        ok = ok and os.path.exists(rr["clarification_request_path"])
        txt = _read(rr["clarification_request_path"])
        ok = ok and "Pipeline blocked" in txt and "## Ask user" in txt and "Cannot safely generate" in txt

    print("ambiguity_gate selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
