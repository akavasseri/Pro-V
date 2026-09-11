#!/usr/bin/env python3
"""
golden_model_repair_loop.py  --  automatic golden-model repair (Pro-V stage 4b).

Runs after golden_model_verifier.py fails and BEFORE the functional coverage agent
is allowed to run. It closes the loop:

    verify -> (fail) -> targeted repair prompt -> regenerate golden_dut.py -> verify ...

until one of: pass, max iterations, ambiguity, or a stalled failure that needs human
clarification. The coverage agent must not run unless this returns approved=True.

Regeneration is injected as `regenerate_fn(prompt, attempt) -> golden_dut source`
(default: an llm_client.chat call + code extraction), keeping the loop decoupled from
the PyChecker internals and testable offline.

Never inspects top_module.v -- the repair prompt explicitly forbids it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Callable, Dict, List, Optional

try:
    from pro_v.golden_model_verifier import GoldenModelVerifier, _load, _witness_list, _invariant_list
    from pro_v.pychecker_guidance_agent import PyCheckerGuidanceAgent
except Exception:  # pragma: no cover
    from golden_model_verifier import GoldenModelVerifier, _load, _witness_list, _invariant_list
    from pychecker_guidance_agent import PyCheckerGuidanceAgent


REPAIR_SYSTEM = (
    "You are the PyChecker golden-model generator operating in REPAIR mode. Repair "
    "golden_dut.py exactly as the instructions require: fix only the failing behavior, "
    "preserve passing behavior, keep the same API and output names, and never inspect or "
    "adapt to top_module.v. Output ONLY one ```python code block defining class GoldenDUT."
)


def _extract_code(response: str) -> str:
    if not response:
        return ""
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    for pat in (r"```python\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pat, text, re.DOTALL)
        if m and "class GoldenDUT" in m.group(1):
            return m.group(1).strip()
    idx = text.find("class GoldenDUT")
    return text[idx:].strip() if idx != -1 else ""


class GoldenModelRepairLoop:

    def __init__(self, verifier: Optional[GoldenModelVerifier] = None,
                 guidance: Optional[PyCheckerGuidanceAgent] = None,
                 regenerate_fn: Optional[Callable[[str, int], str]] = None,
                 llm_client: Any = None, max_iters: int = 3,
                 verifier_kwargs: Optional[dict] = None):
        self.verifier = verifier or GoldenModelVerifier(**(verifier_kwargs or {}))
        self.guidance = guidance or PyCheckerGuidanceAgent()
        self.regenerate_fn = regenerate_fn
        self.llm_client = llm_client
        self.max_iters = max_iters

    # -- public ------------------------------------------------------------

    def run(self, golden_path: str, contract: Any, witnesses: Any = None,
            invariants: Any = None, spec: str = "", header: str = "",
            output_dir: Optional[str] = None, ambiguities_md: Optional[str] = None,
            initial_report: Optional[dict] = None) -> Dict[str, Any]:
        contract = _load(contract)
        wl_obj = _load(witnesses) or {}
        inv_obj = _load(invariants) or {}
        wl = _witness_list(wl_obj)
        out_dir = output_dir or (os.path.dirname(golden_path) or ".")
        if ambiguities_md is None:
            ambiguities_md = _read(os.path.join(out_dir, "ambiguities.md"))

        report = initial_report or self._verify(golden_path, contract, wl_obj, inv_obj, out_dir)
        history: List[dict] = []
        prev_ids: Optional[tuple] = None
        stop_reason = None
        it = 0

        while True:
            ids = self._failing_ids(report)
            history.append(self._record(it, report))
            if report["status"] == "pass":
                stop_reason = "passed"
                break
            if report.get("checks", {}).get("ambiguity") == "fail":
                stop_reason = "ambiguity"
                break
            if it >= self.max_iters:
                stop_reason = "max_iterations"
                break
            if prev_ids is not None and ids == prev_ids and ids:
                stop_reason = "stalled_needs_clarification"
                break

            prior = report.get("repair_report") or self.verifier._repair_report(report, contract, wl)
            prompt = self.guidance.build_prompt(contract, wl_obj, inv_obj, ambiguities_md or "",
                                                prior_report=prior)
            try:
                new_src = self._regenerate(prompt, it)
            except Exception as e:
                stop_reason = "regeneration_failed"
                history[-1]["regeneration_error"] = str(e)
                break
            if not new_src or "class GoldenDUT" not in new_src:
                stop_reason = "regeneration_failed"
                break

            self._save_iteration(golden_path, new_src, out_dir, it)
            history[-1]["repair_prompt_excerpt"] = self._concise_prompt(report, contract)
            prev_ids = ids
            it += 1
            report = self._verify(golden_path, contract, wl_obj, inv_obj, out_dir)

        approved = report["status"] == "pass"
        summary = {
            "approved": approved,
            "iterations": it,
            "final_status": report["status"],
            "stop_reason": stop_reason,
            "history": history,
            "witnesses_passed": report.get("witnesses_passed"),
            "witnesses_failed": report.get("witnesses_failed"),
            "final_report_paths": {"json": report.get("json_path"), "md": report.get("md_path")},
        }
        self._write_outputs(out_dir, summary, report, contract, approved, stop_reason)
        summary["final_report"] = report
        return summary

    # -- steps -------------------------------------------------------------

    def _verify(self, golden_path, contract, wl_obj, inv_obj, out_dir) -> dict:
        return self.verifier.run(golden_path, contract, wl_obj, inv_obj, output_dir=out_dir)

    def _regenerate(self, prompt: str, attempt: int) -> str:
        if self.regenerate_fn is not None:
            return self.regenerate_fn(prompt, attempt) or ""
        if self.llm_client is None:
            raise RuntimeError("no regenerate_fn and no llm_client provided")
        resp = self.llm_client.chat(system=REPAIR_SYSTEM, user=prompt)
        return _extract_code(resp)

    def _save_iteration(self, golden_path: str, new_src: str, out_dir: str, it: int) -> None:
        # keep an audit copy of the pre-repair model and each repaired revision
        if os.path.exists(golden_path):
            shutil.copyfile(golden_path, os.path.join(out_dir, "golden_dut_pre_repair_%d.py" % it))
        with open(golden_path, "w") as f:
            f.write(new_src if new_src.endswith("\n") else new_src + "\n")
        with open(os.path.join(out_dir, "golden_dut_repair_%d.py" % it), "w") as f:
            f.write(new_src)

    # -- analysis ----------------------------------------------------------

    def _failing_ids(self, report: dict) -> tuple:
        ids = []
        for f in report.get("failures", []):
            if f.get("witness_id"):
                ids.append("W:%s" % f["witness_id"])
            elif f.get("invariant_id"):
                ids.append("I:%s" % f["invariant_id"])
            elif f.get("check"):
                ids.append("C:%s" % f["check"])
        return tuple(sorted(ids))

    def _record(self, it: int, report: dict) -> dict:
        return {
            "iteration": it,
            "status": report["status"],
            "witnesses_failed": report.get("witnesses_failed"),
            "invariants_failed": report.get("invariants_failed"),
            "failing_ids": list(self._failing_ids(report)),
        }

    def _concise_prompt(self, report: dict, contract: dict) -> str:
        """A short human-readable repair note in the required example format."""
        rules = {r.get("id"): (r.get("effect") or r.get("description") or "") for r in contract.get("rules", [])}
        for f in report.get("failures", []):
            if not f.get("witness_id"):
                continue
            rids = f.get("covers_rules") or []
            rtext = "; ".join("%s says %s" % (rid, rules.get(rid, "")) for rid in rids) or "the relevant rule"
            return ("The current golden model fails %s. %s. For input %s, expected %s but actual %s. "
                    "Repair only %s. Preserve all passing witnesses. Do not inspect top_module.v."
                    % (f["witness_id"], rtext, f.get("inputs"), f.get("expected"), f.get("actual"),
                       ", ".join(rids) or "the failing behavior"))
        return "Repair the failing invariants/structural checks; preserve passing behavior; do not inspect top_module.v."

    # -- outputs -----------------------------------------------------------

    def _write_outputs(self, out_dir, summary, report, contract, approved, stop_reason) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "golden_repair_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if approved:
            with open(os.path.join(out_dir, "GOLDEN_APPROVED"), "w") as f:
                f.write("approved after %d repair iteration(s)\n" % summary["iterations"])
            return
        # unapproved: explain, and if the spec looks underspecified, ask for clarification
        self._write_explanation(out_dir, summary, report, contract, stop_reason)
        if stop_reason in ("stalled_needs_clarification", "ambiguity"):
            self._append_ambiguities(out_dir, report, contract, stop_reason)

    def _write_explanation(self, out_dir, summary, report, contract, stop_reason) -> None:
        L = ["# Golden model NOT approved", "",
             "**Stop reason:** %s" % stop_reason,
             "**Iterations:** %d of max %d" % (summary["iterations"], self.max_iters), ""]
        reasons = {
            "max_iterations": "The repair loop hit the iteration cap without passing verification.",
            "stalled_needs_clarification": "The same failures persisted across repairs, so the model is "
                                           "not converging — the spec is likely underspecified for this behavior.",
            "regeneration_failed": "The generator did not return a usable golden_dut.py.",
            "ambiguity": "The contract has can_generate_golden_model=false (critical ambiguity).",
        }
        L.append(reasons.get(stop_reason, "See the verification report."))
        L.append("")
        L.append("## Remaining failures")
        for f in report.get("failures", []):
            L.append("- " + json.dumps(f))
        L.append("")
        L.append("The functional coverage agent must NOT run: `golden_dut.py` is unapproved.")
        with open(os.path.join(out_dir, "golden_repair_explanation.md"), "w") as f:
            f.write("\n".join(L) + "\n")

    def _append_ambiguities(self, out_dir, report, contract, stop_reason) -> None:
        path = os.path.join(out_dir, "ambiguities.md")
        existing = _read(path)
        add = ["", "## Added by repair loop (%s)" % stop_reason,
               "The golden model could not be made to satisfy the following after automated repair. "
               "This usually means the spec does not pin the intended behavior — please clarify:"]
        rules = {r.get("id"): (r.get("effect") or r.get("description") or "") for r in contract.get("rules", [])}
        for f in report.get("failures", []):
            if f.get("witness_id"):
                rids = f.get("covers_rules") or []
                add.append("- **%s** (rules %s): for input %s the spec-derived expectation was %s but the "
                           "model produced %s. What is the intended behavior here?"
                           % (f["witness_id"], rids, f.get("inputs"), f.get("expected"), f.get("actual")))
            elif f.get("invariant_id"):
                add.append("- **invariant %s**: %s" % (f["invariant_id"], f.get("reason")))
        with open(path, "w") as fh:
            fh.write((existing.rstrip() + "\n" if existing else "") + "\n".join(add) + "\n")


def repair_golden_model(golden_path, contract, witnesses=None, invariants=None,
                        llm_client=None, regenerate_fn=None, output_dir=None,
                        max_iters=3, **kw) -> Dict[str, Any]:
    return GoldenModelRepairLoop(llm_client=llm_client, regenerate_fn=regenerate_fn,
                                 max_iters=max_iters, **kw).run(
        golden_path, contract, witnesses, invariants, output_dir=output_dir)


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# self-test (fake regenerator; no LLM, no RTL)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    ok = True
    contract = {"design_name": "and_gate", "design_type": "combinational",
                "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"y": {"width": 1}},
                "rules": [{"id": "R1", "effect": "y = a & b", "description": "y = a AND b"}],
                "can_generate_golden_model": True}
    witnesses = {"witnesses": [
        {"id": "W1", "type": "distinguishing", "required": True, "covers_rules": ["R1"],
         "inputs": {"a": "0", "b": "1"}, "expected_outputs": {"y": "0"}, "reason": "AND vs OR"},
        {"id": "W2", "type": "distinguishing", "required": True, "covers_rules": ["R1"],
         "inputs": {"a": "1", "b": "1"}, "expected_outputs": {"y": "1"}, "reason": "1&1"}]}
    invariants = {"invariants": [{"id": "INV_WIDTH_Y", "severity": "critical",
                                  "check_strategy": {"kind": "width", "output": "y", "width": 1}}]}
    OR_BUG = ("class GoldenDUT:\n    def __init__(self): pass\n    def eval(self, inputs):\n"
              "        # Implements R1\n        return {'y': (inputs['a'] | inputs['b']) & 1}\n")
    AND_FIX = ("```python\nclass GoldenDUT:\n    def __init__(self): pass\n    def eval(self, inputs):\n"
               "        # Implements R1\n        return {'y': (inputs['a'] & inputs['b']) & 1}\n```")

    # 1) converges: first repair returns the correct AND model
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "golden_dut.py")
        with open(gp, "w") as f: f.write(OR_BUG)
        calls = {"n": 0}
        def regen(prompt, attempt):
            calls["n"] += 1
            assert "top_module.v` is a black box" in prompt and "REPAIR MODE" in prompt
            assert "W1" in prompt
            return _extract_code(AND_FIX)
        loop = GoldenModelRepairLoop(regenerate_fn=regen, max_iters=3)
        res = loop.run(gp, contract, witnesses, invariants, output_dir=d)
        ok = ok and res["approved"] and res["stop_reason"] == "passed" and res["iterations"] == 1
        ok = ok and calls["n"] == 1 and os.path.exists(os.path.join(d, "GOLDEN_APPROVED"))
        # the repaired file on disk is the AND model
        ok = ok and "a'] & inputs['b" in _read(gp)

    # 2) stalls: regenerator never fixes -> unapproved + ambiguities updated + explanation
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "golden_dut.py")
        with open(gp, "w") as f: f.write(OR_BUG)
        with open(os.path.join(d, "ambiguities.md"), "w") as f: f.write("# Ambiguities\n")
        def regen_bad(prompt, attempt):
            return OR_BUG  # never fixes
        loop = GoldenModelRepairLoop(regenerate_fn=regen_bad, max_iters=3)
        res = loop.run(gp, contract, witnesses, invariants, output_dir=d)
        ok = ok and not res["approved"]
        ok = ok and res["stop_reason"] in ("stalled_needs_clarification", "max_iterations")
        ok = ok and os.path.exists(os.path.join(d, "golden_repair_explanation.md"))
        amb = _read(os.path.join(d, "ambiguities.md"))
        ok = ok and "repair loop" in amb and "W1" in amb
        ok = ok and not os.path.exists(os.path.join(d, "GOLDEN_APPROVED"))

    # 3) regeneration failure -> graceful unapproved
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "golden_dut.py")
        with open(gp, "w") as f: f.write(OR_BUG)
        loop = GoldenModelRepairLoop(regenerate_fn=lambda p, a: "", max_iters=3)
        res = loop.run(gp, contract, witnesses, invariants, output_dir=d)
        ok = ok and not res["approved"] and res["stop_reason"] == "regeneration_failed"

    # 4) ambiguity gate short-circuits with no repair attempts
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "golden_dut.py")
        with open(gp, "w") as f: f.write(OR_BUG)
        amb_contract = dict(contract); amb_contract["can_generate_golden_model"] = False
        calls = {"n": 0}
        loop = GoldenModelRepairLoop(regenerate_fn=lambda p, a: (calls.__setitem__("n", calls["n"] + 1) or AND_FIX),
                                     max_iters=3)
        res = loop.run(gp, amb_contract, witnesses, invariants, output_dir=d)
        ok = ok and not res["approved"] and res["stop_reason"] == "ambiguity" and calls["n"] == 0

    print("golden_model_repair_loop selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
