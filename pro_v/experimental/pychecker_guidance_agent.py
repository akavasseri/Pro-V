#!/usr/bin/env python3
"""
pychecker_guidance_agent.py  --  contract -> implementation prompt (Pro-V stage 2).

Sits between SpecKit (stage 1) and the PyChecker golden-model generator (stage 3).
It does NOT generate golden_dut.py. It assembles a strict, rule-grounded prompt
(`pychecker_prompt.md`) that forces the PyChecker agent to implement ONLY the
behavior in behavior_contract.json, satisfy every witness and invariant, expose a
standard adapter API, and never inspect top_module.v internals.

Deterministic assembly (no LLM). Inputs are the stage-1 artifacts; an optional
prior Golden Model Verifier report turns the prompt into a targeted repair prompt.

API contract this prompt demands of the generated model:
  * combinational:  GoldenDUT().eval(inputs: dict[str,int]) -> dict[str,int]
  * sequential:     GoldenDUT().reset(); .step(inputs) -> outputs; optional .get_state()
Values in this prompt are shown as INTEGERS (the eval/step API), converted from the
binary-string witness values used elsewhere in the pipeline.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# value helpers
# ---------------------------------------------------------------------------

def _to_int(v: Any) -> Any:
    if isinstance(v, str) and v and set(v) <= set("01"):
        return int(v, 2)
    if isinstance(v, str) and set(v) <= set("01xzXZ") and v:
        return v  # keep x/z visible
    return v


def _int_map(m: Any) -> Dict[str, Any]:
    if not isinstance(m, dict):
        return {}
    return {k: _to_int(v) for k, v in m.items()}


def _fmt_map(m: Dict[str, Any]) -> str:
    return "{" + ", ".join("%s: %s" % (k, v) for k, v in m.items()) + "}"


# ---------------------------------------------------------------------------
# agent
# ---------------------------------------------------------------------------

class PyCheckerGuidanceAgent:
    """Builds pychecker_prompt.md from the SpecKit artifacts (+ optional repair report)."""

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client  # unused; kept for interface symmetry

    # -- public ------------------------------------------------------------

    def run(self, artifacts_dir: Optional[str] = None, output_dir: Optional[str] = None,
            contract: Optional[dict] = None, witnesses: Optional[Any] = None,
            invariants: Optional[Any] = None, ambiguities_md: Optional[str] = None,
            prior_report: Optional[dict] = None) -> Dict[str, Any]:
        if artifacts_dir:
            contract = contract if contract is not None else _load_json(artifacts_dir, "behavior_contract.json", {})
            witnesses = witnesses if witnesses is not None else _load_json(artifacts_dir, "witnesses.json", {})
            invariants = invariants if invariants is not None else _load_json(artifacts_dir, "invariants.json", {})
            if ambiguities_md is None:
                ambiguities_md = _load_text(artifacts_dir, "ambiguities.md", "")
        contract = contract or {}
        wl = _witness_list(witnesses)
        inv = _invariant_list(invariants)

        blocked = contract.get("can_generate_golden_model") is False
        prompt = (self._blocked_prompt(contract, ambiguities_md or "")
                  if blocked else
                  self._build_prompt(contract, wl, inv, ambiguities_md or "", prior_report))

        out_dir = output_dir or artifacts_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "pychecker_prompt.md")
        with open(path, "w") as f:
            f.write(prompt)
        return {
            "path": path, "prompt": prompt, "blocked": blocked,
            "design_type": contract.get("design_type"),
            "is_repair": bool(prior_report),
            "num_witnesses": len(wl), "num_invariants": len(inv),
        }

    def build_prompt(self, contract: dict, witnesses: Any = None, invariants: Any = None,
                     ambiguities_md: str = "", prior_report: Optional[dict] = None) -> str:
        if contract.get("can_generate_golden_model") is False:
            return self._blocked_prompt(contract, ambiguities_md)
        return self._build_prompt(contract, _witness_list(witnesses),
                                  _invariant_list(invariants), ambiguities_md, prior_report)

    # -- prompt sections ---------------------------------------------------

    def _build_prompt(self, contract: dict, wl: List[dict], inv: List[dict],
                      ambiguities_md: str, prior_report: Optional[dict]) -> str:
        seq = contract.get("design_type") == "sequential"
        name = contract.get("design_name", "design")
        S: List[str] = []

        S.append("# PyChecker Implementation Prompt — `%s`" % name)
        S.append("")
        S.append("You are the PyChecker golden-model generator. Produce `golden_dut.py` that "
                 "implements **only** the behavior in the behavior contract below. This is a "
                 "SpecKit-guided, witness-checked task: your model will be executed against every "
                 "witness and invariant before it is trusted.")
        S.append("")
        S.append("## Non-negotiable rules")
        S.append("1. Implement `golden_dut.py` **only** from the behavior contract. Do not add behavior.")
        S.append("2. Every branch/expression in your code must map back to a **rule ID** (cite it in a comment, e.g. `# R2`).")
        S.append("3. **`top_module.v` is a black box.** Do not read, infer from, or reference RTL internals. "
                 "Use only the contract's port names and widths.")
        S.append("4. If behavior is ambiguous, **do not guess silently** — see the ASSUMPTIONS rule.")
        S.append("5. Any unavoidable assumption goes in an explicit `# ASSUMPTIONS` comment block at the top, "
                 "each line tied to what forced it. Do not bury assumptions in code.")
        S.append("6. Expose the **standard adapter API** exactly as specified below.")
        S.append("7. Your model **must pass all witnesses and invariants** listed here.")
        S.append("8. The model must be **deterministic** — no randomness, no time, no I/O, no global state leakage.")
        S.append("9. Handle **bit widths exactly** — mask every output to its declared width.")
        S.append("10. Cleanly separate combinational vs sequential behavior (this design is **%s**)."
                 % ("sequential" if seq else "combinational"))
        S.append("")

        S.append(self._api_section(seq))
        S.append(self._contract_section(contract, seq))
        S.append(self._width_signed_section(contract, seq))
        if seq:
            S.append(self._sequential_rules_section(contract))
        S.append(self._witness_section(wl, seq))
        S.append(self._invariant_section(inv))
        S.append(self._forbidden_section())
        if contract.get("assumptions"):
            S.append("## Pre-recorded assumptions (from spec analysis)")
            for a in contract["assumptions"]:
                S.append("- %s" % a)
            S.append("Carry these forward in your `# ASSUMPTIONS` block; do not silently re-decide them.")
            S.append("")
        if prior_report:
            S.append(self._repair_section(prior_report))
        S.append(self._closing_section(seq))
        return "\n".join(S).rstrip() + "\n"

    def _api_section(self, seq: bool) -> str:
        if seq:
            return (
                "## Required API (sequential)\n\n"
                "```python\n"
                "class GoldenDUT:\n"
                "    def __init__(self):\n"
                "        self.reset()\n\n"
                "    def reset(self):\n"
                "        '''Reset internal model state to the initial state.'''\n\n"
                "    def step(self, inputs: dict) -> dict:\n"
                "        '''Advance exactly one clock cycle; return this cycle's outputs.\n"
                "        inputs: mapping input name -> integer value.'''\n\n"
                "    def get_state(self) -> dict:   # optional but recommended\n"
                "        '''Return internal model state for verification/debugging.'''\n"
                "```\n"
                "`inputs`/return values are integer maps keyed by the exact contract signal names.\n")
        return (
            "## Required API (combinational)\n\n"
            "```python\n"
            "class GoldenDUT:\n"
            "    def __init__(self):\n"
            "        pass\n\n"
            "    def eval(self, inputs: dict) -> dict:\n"
            "        '''inputs: mapping input name -> integer value.\n"
            "        returns: mapping output name -> integer value.'''\n"
            "```\n"
            "`eval` must be a pure function of its inputs (no stored state).\n")

    def _contract_section(self, contract: dict, seq: bool) -> str:
        L = ["## Behavior contract", "", "**Design:** `%s`  •  **type:** %s"
             % (contract.get("design_name", "design"), contract.get("design_type", "combinational"))]
        ins = contract.get("inputs", {}) or {}
        outs = contract.get("outputs", {}) or {}
        L.append("")
        L.append("**Inputs:** " + (", ".join("`%s`[%s%s]" % (
            n, _w(s), ",signed" if _sgn(s) else "") for n, s in ins.items()) or "(none)"))
        L.append("")
        L.append("**Outputs:** " + (", ".join("`%s`[%s]" % (n, _w(s)) for n, s in outs.items()) or "(none)"))
        if seq:
            if contract.get("clock"):
                L.append("")
                L.append("**Clock:** `%s`" % contract["clock"])
            r = contract.get("reset") or {}
            if r:
                L.append("")
                L.append("**Reset:** `%s` (active %s, %s)" % (
                    r.get("name", "reset"),
                    "high" if r.get("active_high", True) else "low",
                    r.get("synchronous", "sync/async unspecified")))
            if contract.get("states"):
                L.append("")
                L.append("**States:** %s  •  **initial:** %s" % (
                    ", ".join(contract["states"]), contract.get("initial_state", "?")))
            if contract.get("latency") is not None:
                L.append("")
                L.append("**Latency:** %s" % contract["latency"])
        L.append("")
        L.append("**Rules — implement each, cite its ID:**")
        for rule in contract.get("rules", []) or []:
            rid = rule.get("id", "R?")
            desc = rule.get("description") or ""
            cond = rule.get("condition")
            eff = rule.get("effect")
            line = "- **%s:** %s" % (rid, desc)
            if cond or eff:
                line += "  ⟶  `%s%s`" % (("when %s: " % cond) if cond else "", eff or "")
            L.append(line)
        if contract.get("priority_rules"):
            L.append("")
            L.append("**Priority (when controls collide):** " + "; ".join(contract["priority_rules"]))
        if contract.get("operation_modes"):
            L.append("")
            L.append("**Operation modes:** " + ", ".join(map(str, contract["operation_modes"])))
        if contract.get("undefined_behavior"):
            L.append("")
            L.append("**Explicitly undefined (do NOT invent behavior):** "
                     + "; ".join(map(str, contract["undefined_behavior"])))
        L.append("")
        return "\n".join(L)

    def _width_signed_section(self, contract: dict, seq: bool) -> str:
        outs = contract.get("outputs", {}) or {}
        ins = contract.get("inputs", {}) or {}
        L = ["## Width, signedness & masking", ""]
        L.append("- Mask every output to its declared width: `out &= (1 << W) - 1`.")
        for n, s in outs.items():
            L.append("  - `%s`: %d-bit → `%s &= 0x%X`" % (n, _w(s), n, (1 << _w(s)) - 1))
        signed_ins = [n for n, s in ins.items() if _sgn(s)]
        if signed_ins:
            L.append("- Signed inputs %s: interpret as two's complement of their width before use."
                     % ", ".join("`%s`" % n for n in signed_ins))
        signed_outs = [n for n, s in outs.items() if _sgn(s)]
        if signed_outs:
            L.append("- Signed outputs %s: produce the correct two's-complement bit pattern within width."
                     % ", ".join("`%s`" % n for n in signed_outs))
        L.append("- Do not overflow silently past the declared width; follow the contract's overflow rule "
                 "(wrap/saturate) if one is stated, otherwise record a wrap assumption.")
        L.append("")
        return "\n".join(L)

    def _sequential_rules_section(self, contract: dict) -> str:
        r = contract.get("reset") or {}
        L = ["## Reset, latency & state-update rules", ""]
        L.append("- `reset()` sets the model to the initial state `%s`." % contract.get("initial_state", "?"))
        if r:
            L.append("- Reset `%s` is active-%s; when asserted, `step` must drive the reset behavior%s."
                     % (r.get("name", "reset"), "high" if r.get("active_high", True) else "low",
                        " regardless of other controls" if not contract.get("priority_rules") else ""))
        L.append("- `step(inputs)` computes next-state from current state + inputs, updates internal state, "
                 "and returns this cycle's outputs. Registered outputs must reflect the specified latency.")
        if contract.get("latency") is not None:
            L.append("- Honor the stated latency of `%s` cycle(s) between stimulus and observable output."
                     % contract["latency"])
        L.append("- Do not read future inputs; each `step` sees only the current cycle.")
        L.append("")
        return "\n".join(L)

    def _witness_section(self, wl: List[dict], seq: bool) -> str:
        L = ["## Witnesses your model MUST satisfy", ""]
        if not wl:
            L.append("_(No witnesses provided — still satisfy the contract and invariants.)_")
            L.append("")
            return "\n".join(L)
        required = [w for w in wl if w.get("required")]
        optional = [w for w in wl if not w.get("required")]
        L.append("These are not optional examples; a model that fails a required witness is rejected.")
        L.append("")
        L.append("### Required")
        for w in required or []:
            L += self._render_witness(w)
        if not required:
            L.append("_(none marked required)_")
        if optional:
            L.append("")
            L.append("### Recommended")
            for w in optional[:40]:
                L += self._render_witness(w)
        L.append("")
        return "\n".join(L)

    def _render_witness(self, w: dict) -> List[str]:
        wid = w.get("id", "W?")
        wtype = w.get("type", "")
        reason = w.get("reason", "")
        rules = ",".join(w.get("covers_rules", []) or [])
        tag = " [%s]" % rules if rules else ""
        if "sequence" in w:
            head = "- **%s** (%s)%s: sequential" % (wid, wtype, tag)
            lines = [head]
            for step in w["sequence"]:
                ins = _fmt_map(_int_map(step.get("inputs", {})))
                exp = step.get("expected_outputs")
                cyc = step.get("cycle", "?")
                extra = " → outputs %s" % _fmt_map(_int_map(exp)) if exp else ""
                lines.append("    - Cycle %s: inputs %s%s" % (cyc, ins, extra))
            if w.get("expected_final"):
                lines.append("    - Expected final: %s" % _fmt_map(w["expected_final"]))
            if reason:
                lines.append("    - Reason: %s" % reason)
            return lines
        if w.get("relation"):
            return ["- **%s** (metamorphic)%s: relation `%s` %s — %s" % (
                wid, tag, w["relation"], json.dumps(w.get("params", {})), reason)]
        ins = _fmt_map(_int_map(w.get("inputs", {})))
        if w.get("expected_outputs"):
            outs = _fmt_map(_int_map(w["expected_outputs"]))
            body = "inputs %s → outputs %s" % (ins, outs)
        else:
            body = "inputs %s → outputs must be derivable deterministically from the rules" % ins
        line = "- **%s** (%s)%s: %s" % (wid, wtype, tag, body)
        if reason:
            line += "  _(%s)_" % reason
        return [line]

    def _invariant_section(self, inv: List[dict]) -> str:
        L = ["## Invariants your model MUST uphold", ""]
        if not inv:
            L.append("_(No invariants provided.)_")
            L.append("")
            return "\n".join(L)
        crit = [i for i in inv if i.get("severity") == "critical"]
        other = [i for i in inv if i.get("severity") != "critical"]
        L.append("### Critical (rejection if violated)")
        for i in crit or []:
            L.append("- **%s** (%s): %s — `%s`" % (
                i.get("id"), i.get("type"), i.get("description"), i.get("check")))
        if not crit:
            L.append("_(none)_")
        if other:
            L.append("")
            L.append("### Warning / optional (assumption-based invariants included)")
            for i in other:
                flag = " _(assumption)_" if i.get("assumption") else ""
                L.append("- **%s** (%s, %s)%s: %s — `%s`" % (
                    i.get("id"), i.get("type"), i.get("severity"), flag,
                    i.get("description"), i.get("check")))
        L.append("")
        return "\n".join(L)

    def _forbidden_section(self) -> str:
        return ("## Forbidden behavior\n\n"
                "- Do **not** add features, ports, or modes not in the contract.\n"
                "- Do **not** infer anything from `top_module.v`.\n"
                "- Do **not** rename or drop output signals; use the exact contract names.\n"
                "- Do **not** ignore bit widths or emit out-of-width values.\n"
                "- Do **not** implement random, time-dependent, or non-deterministic behavior.\n"
                "- Do **not** silently decide overflow/saturation, reset polarity, or signedness if the "
                "contract did not specify them — record an explicit assumption instead.\n")

    def _repair_section(self, report: dict) -> str:
        L = ["## REPAIR MODE — a previous model failed verification", ""]
        msg = report.get("message") or report.get("summary")
        if msg:
            L.append("> %s" % msg)
            L.append("")
        failing = report.get("failing_witnesses") or report.get("failures") or []
        passing = report.get("passing_witnesses") or []
        fail_inv = report.get("failing_invariants") or []
        if failing:
            L.append("**Failing witnesses — fix these:**")
            for w in failing:
                if isinstance(w, dict):
                    wid = w.get("id", "W?")
                    got = w.get("got") or w.get("actual")
                    exp = w.get("expected") or w.get("expected_outputs")
                    detail = ""
                    if exp is not None or got is not None:
                        detail = " (expected %s, model produced %s)" % (
                            _fmt_map(_int_map(exp)) if isinstance(exp, dict) else exp,
                            _fmt_map(_int_map(got)) if isinstance(got, dict) else got)
                    L.append("- %s%s" % (wid, detail))
                else:
                    L.append("- %s" % w)
            L.append("")
        if fail_inv:
            L.append("**Failing invariants — fix these:**")
            for i in fail_inv:
                L.append("- %s" % (i.get("id") if isinstance(i, dict) else i))
            L.append("")
        L.append("**Repair instructions:**")
        L.append("- Change **only** the logic responsible for the failing witnesses/invariants.")
        L.append("- **Preserve** all currently-passing behavior%s unless it contradicts the contract."
                 % (" (%d witnesses already pass)" % len(passing) if passing else ""))
        L.append("- Re-cite the rule ID for every line you touch; keep the same API and output names.")
        L.append("")
        return "\n".join(L)

    def _closing_section(self, seq: bool) -> str:
        call = "GoldenDUT().step({...})" if seq else "GoldenDUT().eval({...})"
        return ("## Output\n\n"
                "Return a single ```python``` block defining `class GoldenDUT` with the required API "
                "and a top `# ASSUMPTIONS` block (empty if none). It must import nothing beyond the "
                "standard library, be deterministic, and satisfy every required witness and critical "
                "invariant above. A quick sanity call `%s` must run without error." % call)

    def _blocked_prompt(self, contract: dict, ambiguities_md: str) -> str:
        L = ["# BLOCKED — clarification required before generating `%s`"
             % contract.get("design_name", "design"), ""]
        L.append("`can_generate_golden_model = false`: the spec has unresolved **critical** ambiguities. "
                 "Do not generate `golden_dut.py` yet — generating from an under-specified contract risks a "
                 "wrong reference model. Resolve the following first.")
        L.append("")
        crit = [a for a in (contract.get("_critical_ambiguities") or []) if isinstance(a, dict)]
        if crit:
            L.append("## Critical questions")
            for a in crit:
                L.append("- **%s** %s" % (a.get("id", "A?"), a.get("question", "")))
        if ambiguities_md.strip():
            L.append("")
            L.append("## Full ambiguity report")
            L.append(ambiguities_md.strip())
        L.append("")
        return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------------------
# loaders / normalizers
# ---------------------------------------------------------------------------

def _w(spec: Any) -> int:
    try:
        return max(1, int(spec.get("width", 1))) if isinstance(spec, dict) else 1
    except Exception:
        return 1


def _sgn(spec: Any) -> bool:
    return bool(isinstance(spec, dict) and spec.get("signed"))


def _witness_list(witnesses: Any) -> List[dict]:
    if isinstance(witnesses, dict):
        witnesses = witnesses.get("witnesses", [])
    return [w for w in (witnesses or []) if isinstance(w, dict)]


def _invariant_list(invariants: Any) -> List[dict]:
    if isinstance(invariants, dict):
        invariants = invariants.get("invariants", [])
    return [i for i in (invariants or []) if isinstance(i, dict)]


def _load_json(d: str, name: str, default):
    p = os.path.join(d, name)
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def _load_text(d: str, name: str, default: str) -> str:
    p = os.path.join(d, name)
    try:
        with open(p) as f:
            return f.read()
    except Exception:
        return default


def build_guidance_prompt(contract: dict, witnesses: Any = None, invariants: Any = None,
                          ambiguities_md: str = "", prior_report: Optional[dict] = None) -> str:
    return PyCheckerGuidanceAgent().build_prompt(contract, witnesses, invariants,
                                                 ambiguities_md, prior_report)


# ---------------------------------------------------------------------------
# self-test (no LLM, no RTL)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    ok = True
    agent = PyCheckerGuidanceAgent()

    # combinational AND with distinguishing witnesses
    and_c = {"design_name": "and_gate", "design_type": "combinational",
             "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "description": "y = a AND b", "effect": "y = a & b"}],
             "assumptions": [], "can_generate_golden_model": True}
    wl = {"witnesses": [
        {"id": "W1", "type": "distinguishing", "covers_rules": ["R1"], "required": True,
         "inputs": {"a": "0", "b": "1"}, "expected_outputs": {"y": "0"},
         "reason": "distinguishes AND from OR"},
        {"id": "W2", "type": "distinguishing", "covers_rules": ["R1"], "required": True,
         "inputs": {"a": "1", "b": "0"}, "expected_outputs": {"y": "0"},
         "reason": "distinguishes AND from OR"}]}
    iv = {"invariants": [{"id": "INV_WIDTH_Y", "type": "width", "severity": "critical",
                          "description": "y fits 1 bit", "check": "0<=y<2", "assumption": False}]}
    p = agent.build_prompt(and_c, wl, iv)
    ok = ok and "def eval(self, inputs: dict)" in p and "def step(" not in p
    ok = ok and "inputs {a: 0, b: 1} → outputs {y: 0}" in p   # bitstring->int rendered
    ok = ok and "top_module.v` is a black box" in p and "map back to a **rule ID**" in p
    ok = ok and "INV_WIDTH_Y" in p and "Forbidden behavior" in p

    # sequential API selection + cycle rendering
    seq_c = {"design_name": "ctrl", "design_type": "sequential",
             "inputs": {"reset": {"width": 1}, "start": {"width": 1}, "valid": {"width": 1}},
             "outputs": {"done": {"width": 1}},
             "reset": {"name": "reset", "active_high": True}, "states": ["IDLE", "LOAD", "COMPUTE"],
             "initial_state": "IDLE", "can_generate_golden_model": True,
             "rules": [{"id": "R1", "description": "reset->IDLE"}]}
    sw = {"witnesses": [{"id": "WS", "type": "sequential_reachability", "required": True,
                         "covers_rules": ["R1"], "expected_final": {"state": "COMPUTE"},
                         "sequence": [{"cycle": 0, "inputs": {"reset": "1"}},
                                      {"cycle": 1, "inputs": {"reset": "0", "start": "1"}},
                                      {"cycle": 2, "inputs": {"valid": "1"}}],
                         "reason": "reach COMPUTE"}]}
    ps = agent.build_prompt(seq_c, sw, {"invariants": []})
    ok = ok and "def step(self, inputs: dict)" in ps and "def reset(self)" in ps
    ok = ok and "Cycle 0: inputs {reset: 1}" in ps and "Expected final: {state: COMPUTE}" in ps
    ok = ok and "Reset, latency & state-update rules" in ps

    # blocked path
    blk = json.loads(json.dumps(and_c)); blk["can_generate_golden_model"] = False
    blk["_critical_ambiguities"] = [{"id": "A1", "question": "sync or async reset?"}]
    pb = agent.build_prompt(blk, wl, iv, ambiguities_md="# Ambiguities\n- reset timing")
    ok = ok and pb.startswith("# BLOCKED") and "sync or async reset?" in pb

    # repair path
    report = {"message": "1 witness failed", "failing_witnesses": [
        {"id": "W1", "expected": {"y": "0"}, "got": {"y": "1"}}],
        "passing_witnesses": [{"id": "W2"}]}
    pr = agent.build_prompt(and_c, wl, iv, prior_report=report)
    ok = ok and "REPAIR MODE" in pr and "expected {y: 0}, model produced {y: 1}" in pr
    ok = ok and "Preserve" in pr and "1 witnesses already pass" in pr

    # file write
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for fn, obj in (("behavior_contract.json", and_c), ("witnesses.json", wl),
                        ("invariants.json", iv)):
            with open(os.path.join(d, fn), "w") as f:
                json.dump(obj, f)
        with open(os.path.join(d, "ambiguities.md"), "w") as f:
            f.write("# Ambiguities\nnone")
        res = agent.run(artifacts_dir=d)
        ok = ok and os.path.exists(res["path"]) and res["blocked"] is False and res["num_witnesses"] == 2

    print("pychecker_guidance selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        r = PyCheckerGuidanceAgent().run(artifacts_dir=sys.argv[1])
        print("wrote", r["path"], "| blocked=%s repair=%s" % (r["blocked"], r["is_repair"]))
    else:
        raise SystemExit(_selftest())
