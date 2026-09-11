#!/usr/bin/env python3
"""
speckit_helper_agent.py  --  SpecKit-style spec decomposition (Pro-V stage 1).

Runs BEFORE any golden-model / pychecker generation. It converts the original
natural-language hardware spec into structured, checkable artifacts so that
downstream reference-model generation is guided by an explicit contract instead
of jumping from vague prose to code.

Philosophy (SpecKit):
  1. Specify behavior clearly.
  2. Clarify ambiguity BEFORE implementation.
  3. Produce a contract that later agents must follow.
  4. Never jump directly from vague NL to code.

It emits five artifacts (never `golden_dut.py`):
  1. behavior_contract.json   -- the machine-readable behavior contract
  2. witnesses.json           -- distinguishing witnesses (witness-aware abstraction)
  3. invariants.json          -- properties that must always hold
  4. ambiguities.md           -- critical missing information, human-readable
  5. spec_traceability.json   -- every rule traced back to spec text + confidence

Design rules:
  * No hallucinated behavior. Every rule traces to spec text OR is an explicit
    recorded assumption.
  * If a *critical* ambiguity remains, set can_generate_golden_model=false in the
    contract and stop -- downstream must ask for clarification, not force codegen.
  * `top_module.v` is a black box. Interface metadata (header) is used only for
    port names / widths / clock-reset wiring, never to infer intent.

Witness-aware abstraction (the research idea):
  A model can look correct while being wrong if it is only tested on vacuous
  inputs. AND and OR both give 0 on 00 and 1 on 11 -- those rows distinguish
  nothing. The meaningful witnesses are 01 and 10. This agent must generate
  witnesses that FORCE meaningful behavior and SEPARATE near-miss implementations.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # structural, deploy-safe port parse (names/widths only)
    from pro_v.mutation_strength import parse_ports
except Exception:  # pragma: no cover
    try:
        from mutation_strength import parse_ports
    except Exception:
        parse_ports = None  # type: ignore

DESIGN_TYPES = {"combinational", "sequential", "ambiguous"}

ARTIFACT_FILES = {
    "behavior_contract": "behavior_contract.json",
    "witnesses": "witnesses.json",
    "invariants": "invariants.json",
    "ambiguities": "ambiguities.md",
    "spec_traceability": "spec_traceability.json",
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a SpecKit-style hardware specification analyst. You DECOMPOSE a natural-language hardware spec into structured, checkable artifacts. You DO NOT write RTL and you DO NOT write the Python reference model (golden_dut.py). Your only job is to specify, clarify, and contract.

Hard rules:
- NEVER hallucinate behavior. Every behavioral rule MUST either quote/trace to the spec text, or be listed as an explicit assumption. Do not invent protocols, states, counters, reset polarity, latency, or signedness that the spec does not state.
- If a CRITICAL detail that would change golden_dut.py is missing (combinational vs sequential, reset sync/async, reset polarity, initial state, arithmetic wrap/saturate/carry, signedness, output width, registered vs immediate output, latency, memory read sync/async, when valid/ready consumes data, control priority), record it as an ambiguity and mark it critical.
- If ANY critical ambiguity exists, set "can_generate_golden_model": false.
- Generate DISTINGUISHING WITNESSES. A witness set must force meaningful behavior and separate near-miss implementations. Vacuous rows are forbidden as the ONLY evidence: AND and OR both give 0 on 00 and 1 on 11, so those rows distinguish nothing; the distinguishing witnesses are 01 and 10. For every rule, include at least one witness that would FAIL if the rule were subtly wrong (wrong operator, off-by-one priority, swapped branch, wrong polarity).
- All signal values are BINARY STRINGS made of '0'/'1' (e.g. "0", "1", "1011"). Match the module header's exact signal names.

Output: a SINGLE JSON object (no prose, no code fences needed) with EXACTLY these top-level keys:
{
  "behavior_contract": {
     "design_name": str,
     "design_type": "combinational" | "sequential" | "ambiguous",
     "inputs":  { "<name>": {"width": int, "signed": bool (optional), "domain": str (optional)} },
     "outputs": { "<name>": {"width": int, "signed": bool (optional)} },
     "clock":  str (optional),
     "reset":  {"name": str, "active_high": bool, "synchronous": bool|"ambiguous"} (optional),
     "latency": str|int (optional),
     "states": [str] (sequential only),
     "initial_state": str (optional),
     "rules": [ {"id":"R1","description":str,"condition":str,"effect":str,"confidence":float} ],
     "priority_rules": [str] (optional; e.g. "reset overrides enable"),
     "operation_modes": [str] (optional),
     "undefined_behavior": [str],
     "assumptions": [str],
     "can_generate_golden_model": bool
  },
  "witnesses": [
     {"id":"W1","purpose":str,"distinguishes":[str] (optional),
      "inputs":{"<name>":"<bits>"}, "expected_outputs":{"<name>":"<bits>"},
      "traces_rule":"R1","confidence":float}
     // For SEQUENTIAL designs a witness may instead carry an ordered "sequence":
     // {"id":"W2","purpose":str,"sequence":[{"inputs":{...},"expected_outputs":{...}}, ...],"traces_rule":"R2","confidence":float}
  ],
  "invariants": [
     {"id":"I1","description":str,"kind":"width"|"range"|"onehot"|"reset"|"stability"|"relation"|"custom","expr":str,"confidence":float}
  ],
  "ambiguities": [
     {"id":"A1","question":str,"why_it_matters":str,"critical":bool,"default_assumption":str (optional)}
  ],
  "spec_traceability": {
     "R1": {"source_text": str, "confidence": float}
  }
}

Only include fields you can justify. Prefer marking something ambiguous over guessing."""


USER_TEMPLATE = """<spec>
{description}
</spec>

<module_header>
{header}
</module_header>

<interface_ports_parsed>
{ports}
</interface_ports_parsed>
{examples}{prior}
Decompose this spec into the five-artifact JSON object described in the system prompt.
Remember: trace every rule to the spec, record assumptions explicitly, generate distinguishing (non-vacuous) witnesses, and set can_generate_golden_model=false if any critical ambiguity remains."""


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"</?think>", "", text)


def extract_json_object(response: str) -> Optional[dict]:
    """Pull the first well-formed top-level JSON object out of an LLM response.
    Tolerates <think> tags, ```json fences, and leading/trailing prose."""
    if not response:
        return None
    text = _strip_think(response)

    # 1) fenced ```json ... ``` or ``` ... ```
    for pat in (r"```json\s*(.*?)\s*```", r"```\s*(\{.*?\})\s*```"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            obj = _try_load(m.group(1))
            if obj is not None:
                return obj

    # 2) first balanced {...} scan
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        obj = _try_load(text[start:i + 1])
                        if obj is not None:
                            return obj
                        break
        start = text.find("{", start + 1)
    return None


def _try_load(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Schema validation  (each returns (ok, [errors]))
# ---------------------------------------------------------------------------

def validate_behavior_contract(c: Any) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not isinstance(c, dict):
        return False, ["behavior_contract is not an object"]
    if not isinstance(c.get("design_name"), str) or not c.get("design_name"):
        errs.append("design_name missing/invalid")
    dt = c.get("design_type")
    if dt not in DESIGN_TYPES:
        errs.append(f"design_type must be one of {sorted(DESIGN_TYPES)}, got {dt!r}")
    for key in ("inputs", "outputs"):
        if not isinstance(c.get(key), dict):
            errs.append(f"{key} must be an object of name->spec")
    if not isinstance(c.get("rules"), list):
        errs.append("rules must be a list")
    else:
        for i, r in enumerate(c["rules"]):
            if not isinstance(r, dict) or not r.get("id"):
                errs.append(f"rule[{i}] missing id")
            elif not (r.get("description") or r.get("effect") or r.get("condition")):
                errs.append(f"rule {r.get('id')} has no description/condition/effect")
    for key in ("undefined_behavior", "assumptions"):
        if not isinstance(c.get(key, []), list):
            errs.append(f"{key} must be a list")
    if not isinstance(c.get("can_generate_golden_model"), bool):
        errs.append("can_generate_golden_model must be a bool")
    if dt == "sequential" and not isinstance(c.get("states", []), list):
        errs.append("sequential design should carry a states list")
    return (not errs), errs


def validate_witnesses(w: Any) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not isinstance(w, list):
        return False, ["witnesses must be a list"]
    for i, item in enumerate(w):
        if not isinstance(item, dict) or not item.get("id"):
            errs.append(f"witness[{i}] missing id")
            continue
        has_single = isinstance(item.get("inputs"), dict) and isinstance(item.get("expected_outputs"), dict)
        seq = item.get("sequence")
        has_seq = isinstance(seq, list) and seq and all(
            isinstance(s, dict) and isinstance(s.get("inputs"), dict) for s in seq
        )
        if not (has_single or has_seq):
            errs.append(f"witness {item['id']} needs inputs+expected_outputs or a non-empty sequence")
        for field, val in _iter_witness_bitmaps(item):
            for name, bits in val.items():
                if not _is_bitstring(bits):
                    errs.append(f"witness {item['id']} {field}.{name}={bits!r} is not a binary string")
    return (not errs), errs


def validate_invariants(inv: Any) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not isinstance(inv, list):
        return False, ["invariants must be a list"]
    for i, item in enumerate(inv):
        if not isinstance(item, dict) or not item.get("id"):
            errs.append(f"invariant[{i}] missing id")
        elif not item.get("description") and not item.get("expr"):
            errs.append(f"invariant {item.get('id')} needs description or expr")
    return (not errs), errs


def validate_traceability(t: Any) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if not isinstance(t, dict):
        return False, ["spec_traceability must be an object"]
    for rid, entry in t.items():
        if not isinstance(entry, dict) or "source_text" not in entry:
            errs.append(f"traceability[{rid}] must carry source_text")
    return (not errs), errs


def _is_bitstring(v: Any) -> bool:
    return isinstance(v, str) and len(v) > 0 and set(v) <= set("01xzXZ")


def _iter_witness_bitmaps(item: dict):
    if isinstance(item.get("inputs"), dict):
        yield "inputs", item["inputs"]
    if isinstance(item.get("expected_outputs"), dict):
        yield "expected_outputs", item["expected_outputs"]
    for s in item.get("sequence") or []:
        if isinstance(s, dict):
            if isinstance(s.get("inputs"), dict):
                yield "sequence.inputs", s["inputs"]
            if isinstance(s.get("expected_outputs"), dict):
                yield "sequence.expected_outputs", s["expected_outputs"]


# ---------------------------------------------------------------------------
# Interface merge (structural, deploy-safe)
# ---------------------------------------------------------------------------

def _parse_interface(header: str) -> Dict[str, Dict[str, int]]:
    """Return {'inputs': {name: width}, 'outputs': {name: width}, 'clk': name|None,
    'reset': name|None} from the module header, using the shared structural parser."""
    out = {"inputs": {}, "outputs": {}, "clk": None, "reset": None}
    if not header or parse_ports is None:
        return out
    try:
        ports = parse_ports(header)
    except Exception:
        return out
    out["inputs"] = {n: w for n, w in getattr(ports, "inputs", [])}
    out["outputs"] = {n: w for n, w in getattr(ports, "outputs", [])}
    out["clk"] = getattr(ports, "clk_name", None)
    out["reset"] = getattr(ports, "reset_name", None)
    return out


def _reconcile_interface(contract: dict, iface: Dict[str, Any]) -> None:
    """Fill missing widths from the parsed header and record any name/width
    disagreement as an explicit assumption (never silently overwrite intent)."""
    if not iface or not (iface["inputs"] or iface["outputs"]):
        return
    assumptions = contract.setdefault("assumptions", [])
    for direction in ("inputs", "outputs"):
        cmap = contract.setdefault(direction, {})
        if not isinstance(cmap, dict):
            continue
        for name, width in iface[direction].items():
            spec = cmap.get(name)
            if spec is None:
                cmap[name] = {"width": width}
                assumptions.append(f"{direction[:-1]} '{name}' (width {width}) taken from module header interface")
            elif isinstance(spec, dict):
                if spec.get("width") in (None, 0):
                    spec["width"] = width
                elif spec.get("width") != width:
                    assumptions.append(
                        f"width mismatch on {name}: contract={spec.get('width')} vs header={width}; "
                        f"kept contract value")


# ---------------------------------------------------------------------------
# ambiguities.md rendering
# ---------------------------------------------------------------------------

def render_ambiguities_md(design_name: str, ambiguities: List[dict], can_generate: bool) -> str:
    lines = [f"# Ambiguities — {design_name}", ""]
    crit = [a for a in ambiguities if isinstance(a, dict) and a.get("critical")]
    noncrit = [a for a in ambiguities if isinstance(a, dict) and not a.get("critical")]
    if not can_generate:
        lines += ["> **BLOCKING:** critical ambiguities remain. `can_generate_golden_model = false`.",
                  "> Downstream must request clarification before generating golden_dut.py.", ""]
    else:
        lines += ["> No blocking ambiguities. Non-critical items are resolved via recorded assumptions.", ""]

    def _emit(title: str, items: List[dict]) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        for a in items:
            aid = a.get("id", "?")
            lines.append(f"- **[{aid}] {a.get('question', '(unspecified)')}**")
            if a.get("why_it_matters"):
                lines.append(f"  - Why it matters: {a['why_it_matters']}")
            if a.get("default_assumption"):
                lines.append(f"  - Default assumption if unresolved: {a['default_assumption']}")
        lines.append("")

    _emit("Critical (blocking)", crit)
    _emit("Non-critical", noncrit)
    if not ambiguities:
        lines.append("No ambiguities detected.")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SpecKitHelperAgent:
    """Stage-1 SpecKit decomposition agent. LLM-driven, schema-validated."""

    def __init__(self, llm_client: Any = None, max_retries: int = 2):
        self.llm_client = llm_client
        self.max_retries = max_retries
        logger.info("SpecKitHelperAgent initialized (max_retries=%d, llm=%s)",
                    max_retries, "yes" if llm_client else "no")

    # -- public ------------------------------------------------------------

    def run(self, description: str, header: str = "", output_dir: str = ".",
            examples: Optional[str] = None, prior_artifacts: Optional[dict] = None,
            design_name_hint: Optional[str] = None) -> Dict[str, Any]:
        """Decompose `description` into the five artifacts and write them to
        output_dir. Returns a dict with success, can_generate_golden_model, the
        parsed artifacts, file paths, and any validation warnings."""
        iface = _parse_interface(header)
        raw = None
        last_err = ""
        for attempt in range(self.max_retries + 1):
            response = self._call_llm(description, header, iface, examples, prior_artifacts, repair=last_err)
            raw = extract_json_object(response) if response else None
            if raw is None:
                last_err = "response was not valid JSON"
                logger.warning("SpecKit attempt %d: %s", attempt + 1, last_err)
                continue
            result = self._assemble(raw, iface, output_dir, design_name_hint)
            if result["success"]:
                return result
            last_err = "; ".join(result.get("warnings", [])) or "schema validation failed"
            logger.warning("SpecKit attempt %d invalid: %s", attempt + 1, last_err)
        # all attempts failed -> emit a blocking ambiguity so the pipeline stops safely
        return self._fallback(iface, output_dir, design_name_hint, last_err)

    # -- assembly ----------------------------------------------------------

    def _assemble(self, raw: dict, iface: Dict[str, Any], output_dir: str,
                  name_hint: Optional[str]) -> Dict[str, Any]:
        contract = raw.get("behavior_contract") if isinstance(raw.get("behavior_contract"), dict) else {}
        witnesses = raw.get("witnesses") if isinstance(raw.get("witnesses"), list) else []
        invariants = raw.get("invariants") if isinstance(raw.get("invariants"), list) else []
        ambiguities = raw.get("ambiguities") if isinstance(raw.get("ambiguities"), list) else []
        traceability = raw.get("spec_traceability") if isinstance(raw.get("spec_traceability"), dict) else {}

        if name_hint and not contract.get("design_name"):
            contract["design_name"] = name_hint
        contract.setdefault("undefined_behavior", [])
        contract.setdefault("assumptions", [])
        _reconcile_interface(contract, iface)

        # can_generate: honor the model, but force False if a critical ambiguity slipped through
        has_critical = any(isinstance(a, dict) and a.get("critical") for a in ambiguities)
        if "can_generate_golden_model" not in contract or not isinstance(
                contract["can_generate_golden_model"], bool):
            contract["can_generate_golden_model"] = not has_critical
        elif contract["can_generate_golden_model"] and has_critical:
            contract["can_generate_golden_model"] = False
            contract["assumptions"].append(
                "can_generate_golden_model forced to false: unresolved critical ambiguity present")

        warnings: List[str] = []
        for label, ok_errs in (
            ("behavior_contract", validate_behavior_contract(contract)),
            ("witnesses", validate_witnesses(witnesses)),
            ("invariants", validate_invariants(invariants)),
            ("spec_traceability", validate_traceability(traceability)),
        ):
            ok, errs = ok_errs
            if not ok:
                warnings += [f"{label}: {e}" for e in errs]

        contract_ok, _ = validate_behavior_contract(contract)
        artifacts = {
            "behavior_contract": contract,
            "witnesses": witnesses,
            "invariants": invariants,
            "ambiguities": ambiguities,
            "spec_traceability": traceability,
        }
        paths = self._write_artifacts(artifacts, output_dir, contract)
        return {
            "success": contract_ok,
            "can_generate_golden_model": bool(contract.get("can_generate_golden_model")),
            "design_name": contract.get("design_name"),
            "design_type": contract.get("design_type"),
            "num_rules": len(contract.get("rules", [])),
            "num_witnesses": len(witnesses),
            "num_invariants": len(invariants),
            "critical_ambiguities": [a for a in ambiguities if isinstance(a, dict) and a.get("critical")],
            "artifacts": artifacts,
            "paths": paths,
            "warnings": warnings,
        }

    def _write_artifacts(self, artifacts: dict, output_dir: str, contract: dict) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}
        for key in ("behavior_contract", "witnesses", "invariants", "spec_traceability"):
            p = os.path.join(output_dir, ARTIFACT_FILES[key])
            with open(p, "w") as f:
                json.dump(artifacts[key], f, indent=2)
            paths[key] = p
        md_path = os.path.join(output_dir, ARTIFACT_FILES["ambiguities"])
        with open(md_path, "w") as f:
            f.write(render_ambiguities_md(
                contract.get("design_name", "design"),
                artifacts["ambiguities"],
                bool(contract.get("can_generate_golden_model"))))
        paths["ambiguities"] = md_path
        return paths

    def _fallback(self, iface: Dict[str, Any], output_dir: str,
                  name_hint: Optional[str], reason: str) -> Dict[str, Any]:
        """Could not obtain a valid decomposition -> block generation, but still
        write artifacts so the pipeline has something to reason about."""
        contract = {
            "design_name": name_hint or "unknown",
            "design_type": "ambiguous",
            "inputs": {n: {"width": w} for n, w in iface.get("inputs", {}).items()},
            "outputs": {n: {"width": w} for n, w in iface.get("outputs", {}).items()},
            "rules": [],
            "undefined_behavior": [],
            "assumptions": [f"SpecKit could not produce a valid contract: {reason}"],
            "can_generate_golden_model": False,
        }
        ambiguities = [{
            "id": "A0",
            "question": "The specification could not be decomposed into a reliable contract.",
            "why_it_matters": "Generating golden_dut.py from an un-decomposable spec risks a wrong reference model.",
            "critical": True,
        }]
        artifacts = {
            "behavior_contract": contract, "witnesses": [], "invariants": [],
            "ambiguities": ambiguities, "spec_traceability": {},
        }
        paths = self._write_artifacts(artifacts, output_dir, contract)
        return {
            "success": False, "can_generate_golden_model": False,
            "design_name": contract["design_name"], "design_type": "ambiguous",
            "num_rules": 0, "num_witnesses": 0, "num_invariants": 0,
            "critical_ambiguities": ambiguities, "artifacts": artifacts,
            "paths": paths, "warnings": [reason],
        }

    # -- llm ---------------------------------------------------------------

    def _build_user_prompt(self, description: str, header: str, iface: Dict[str, Any],
                           examples: Optional[str], prior: Optional[dict], repair: str) -> str:
        ports_txt = json.dumps({"inputs": iface.get("inputs", {}),
                                "outputs": iface.get("outputs", {}),
                                "clk": iface.get("clk"), "reset": iface.get("reset")}, indent=2)
        ex = f"\n<user_examples>\n{examples}\n</user_examples>\n" if examples else ""
        pr = ""
        if prior:
            pr = f"\n<previous_artifacts>\n{json.dumps(prior, indent=2)[:4000]}\n</previous_artifacts>\n"
        if repair:
            pr += (f"\n<fix_required>\nYour previous output was rejected: {repair}. "
                   f"Return corrected JSON only.\n</fix_required>\n")
        return USER_TEMPLATE.format(description=description or "", header=header or "(none)",
                                    ports=ports_txt, examples=ex, prior=pr)

    def _call_llm(self, description: str, header: str, iface: Dict[str, Any],
                  examples: Optional[str], prior: Optional[dict], repair: str = "") -> str:
        if self.llm_client is None:
            logger.warning("SpecKitHelperAgent: no llm_client configured")
            return ""
        user = self._build_user_prompt(description, header, iface, examples, prior, repair)
        try:
            return self.llm_client.chat(system=SYSTEM_PROMPT, user=user)
        except Exception as e:  # pragma: no cover
            logger.error("SpecKit LLM call failed: %s", e)
            return ""


# convenience functional entry point
def run_speckit(description: str, header: str = "", output_dir: str = ".",
                llm_client: Any = None, **kw) -> Dict[str, Any]:
    return SpecKitHelperAgent(llm_client=llm_client).run(
        description=description, header=header, output_dir=output_dir, **kw)


# ---------------------------------------------------------------------------
# Offline self-test (no LLM): exercises parse -> validate -> write path.
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    ok = True

    # 1) JSON extraction survives <think> + fences
    raw = extract_json_object('<think>reasoning</think>\n```json\n{"behavior_contract": {"design_name":"x"}}\n```')
    ok = ok and raw is not None and raw["behavior_contract"]["design_name"] == "x"

    # 2) full assemble on a canonical AND-gate contract with distinguishing witnesses
    canned = {
        "behavior_contract": {
            "design_name": "and_gate", "design_type": "combinational",
            "inputs": {"a": {"width": 1}, "b": {"width": 1}},
            "outputs": {"y": {"width": 1}},
            "rules": [{"id": "R1", "description": "y = a & b", "condition": "always",
                       "effect": "y = a & b", "confidence": 0.97}],
            "undefined_behavior": [], "assumptions": [], "can_generate_golden_model": True,
        },
        "witnesses": [
            {"id": "W1", "purpose": "distinguish AND from OR (vacuous rows excluded)",
             "distinguishes": ["and", "or"], "inputs": {"a": "0", "b": "1"},
             "expected_outputs": {"y": "0"}, "traces_rule": "R1", "confidence": 0.95},
            {"id": "W2", "purpose": "force the 1&1 case", "inputs": {"a": "1", "b": "1"},
             "expected_outputs": {"y": "1"}, "traces_rule": "R1", "confidence": 0.95},
        ],
        "invariants": [{"id": "I1", "description": "y is 1 bit", "kind": "width",
                        "expr": "len(y)==1", "confidence": 0.99}],
        "ambiguities": [],
        "spec_traceability": {"R1": {"source_text": "Output y should be a AND b", "confidence": 0.95}},
    }
    agent = SpecKitHelperAgent(llm_client=None)
    with tempfile.TemporaryDirectory() as d:
        res = agent._assemble(canned, _parse_interface("module top_module(input a, input b, output y);"), d, None)
        ok = ok and res["success"] and res["can_generate_golden_model"]
        ok = ok and all(os.path.exists(p) for p in res["paths"].values())
        # header width should have flowed into the contract
        ok = ok and res["artifacts"]["behavior_contract"]["inputs"]["a"]["width"] == 1

    # 3) critical ambiguity forces can_generate=false even if model said true
    amb = json.loads(json.dumps(canned))
    amb["ambiguities"] = [{"id": "A1", "question": "sync or async reset?", "critical": True}]
    amb["behavior_contract"]["can_generate_golden_model"] = True
    with tempfile.TemporaryDirectory() as d:
        res = agent._assemble(amb, {"inputs": {}, "outputs": {}, "clk": None, "reset": None}, d, None)
        ok = ok and res["can_generate_golden_model"] is False

    # 4) validators reject a non-binary witness value
    bad_ok, _ = validate_witnesses([{"id": "W", "inputs": {"a": "2"}, "expected_outputs": {"y": "0"}}])
    ok = ok and bad_ok is False

    print("speckit_helper selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
