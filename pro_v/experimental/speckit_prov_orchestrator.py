#!/usr/bin/env python3
"""
speckit_prov_orchestrator.py  --  the full SpecKit witness-guided pipeline.

Wires every stage together with the pipeline gates and writes an artifacts/ directory:

  1  input spec
  2  speckit_helper_agent          -> behavior_contract/witnesses/invariants/ambiguities/traceability
  3  witness_generator             -> witnesses.json (deterministic, merged with SpecKit seeds)
  4  invariant_generator           -> invariants.json
  5  cross_artifact_consistency    (pre)
  6  witness_completeness_scorer   (pre)
  7  ambiguity_gate                -- GATE: stop for clarification if critical ambiguity
  8  pychecker_guidance_agent      -> pychecker_prompt.md
  9  pychecker generation          -> golden_dut.py            (injected generate_fn / llm)
 10  golden_model_verifier         (+ optional metamorphic)
 11  golden_model_repair_loop      (only if verify fails)
 12  approve golden model          -- GATE: no coverage unless approved
 13  functional_coverage_agent     -> coverage_plan.json  (golden_dut.py is the source of truth)
 14  cross_artifact_consistency    (post, with plan)
 15  witness_completeness_scorer   (post, with plan)
 16  testbench generator           -> testbench.sv        (from coverage_plan.json)
 17  RTL simulation                -- black-box top_module.v vs golden-derived expected

Gates enforced: no pychecker if the ambiguity gate fails; no coverage agent if the
golden model is unapproved; top_module.v is never used for behavior (only interface +
simulation); random tests never substitute for required witnesses.

The two generative steps are injectable so the whole pipeline runs offline in tests.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:                    # allow `python pro_v/speckit_prov_orchestrator.py`
    sys.path.insert(0, _REPO_ROOT)

PYCHECKER_SYS = ("You are the PyChecker golden-model generator. Implement golden_dut.py strictly from "
                 "the prompt/contract. Output ONLY one ```python code block defining class GoldenDUT "
                 "with the required API. Never inspect top_module.v.")


def _extract_code(response: str) -> str:
    if not response:
        return ""
    t = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    t = re.sub(r"</?think>", "", t)
    for pat in (r"```python\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pat, t, re.DOTALL)
        if m and "class GoldenDUT" in m.group(1):
            return m.group(1).strip()
    idx = t.find("class GoldenDUT")
    return t[idx:].strip() if idx != -1 else ""


class SpecKitProVOrchestrator:

    def __init__(self, llm_client: Any = None,
                 pychecker_generate_fn: Optional[Callable[[str], str]] = None,
                 budget: Any = None, completeness_threshold: float = 0.75,
                 completeness_mode: str = "permissive", max_repair_iters: int = 3,
                 run_metamorphic: bool = True, verifier_kwargs: Optional[dict] = None):
        self.llm_client = llm_client
        self.pychecker_generate_fn = pychecker_generate_fn
        self.budget = budget
        self.completeness_threshold = completeness_threshold
        self.completeness_mode = completeness_mode
        self.max_repair_iters = max_repair_iters
        self.run_metamorphic = run_metamorphic
        self.verifier_kwargs = dict(verifier_kwargs or {})
        self.verifier_kwargs.setdefault("run_metamorphic", run_metamorphic)

    # -- public ------------------------------------------------------------

    def run(self, description: str, header: str = "", dut_path: Optional[str] = None,
            dut_code: Optional[str] = None, output_dir: str = "artifacts",
            examples: Optional[str] = None) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        rep: Dict[str, Any] = {"artifacts_dir": output_dir, "steps": [], "approved": False,
                               "status": "running"}

        # resolve DUT (interface + simulation only)
        if dut_code and not dut_path:
            dut_path = os.path.join(output_dir, "top_module.v")
            with open(dut_path, "w") as f:
                f.write(dut_code)
        elif dut_path and not dut_code and os.path.exists(dut_path):
            dut_code = open(dut_path).read()

        # 2) SpecKit decomposition
        from pro_v.speckit_helper_agent import SpecKitHelperAgent
        sk = SpecKitHelperAgent(llm_client=self.llm_client).run(
            description, header, output_dir=output_dir, examples=examples)
        contract = sk["artifacts"]["behavior_contract"]
        self._step(rep, 2, "speckit_helper", "ok" if sk["success"] else "warn",
                   {"design": sk.get("design_name"), "rules": sk.get("num_rules"),
                    "can_generate": sk.get("can_generate_golden_model")})

        # 3) witnesses (deterministic + merge SpecKit seeds)
        from pro_v.witness_generator import generate_witnesses
        wobj = generate_witnesses(contract, existing=sk["artifacts"].get("witnesses"))
        _dump(output_dir, "witnesses.json", wobj)
        self._step(rep, 3, "witness_generator", "ok", {"num_witnesses": len(wobj["witnesses"]),
                                                        "completeness": wobj.get("completeness")})

        # 4) invariants
        from pro_v.invariant_generator import generate_invariants
        iobj = generate_invariants(contract)
        _dump(output_dir, "invariants.json", iobj)
        self._step(rep, 4, "invariant_generator", "ok", {"num_invariants": iobj["summary"]["total"]})

        # 5) consistency (pre)
        from pro_v.cross_artifact_consistency_checker import CrossArtifactConsistencyChecker
        cons = CrossArtifactConsistencyChecker().run(
            spec=description, contract=contract, witnesses=wobj, invariants=iobj, output_dir=output_dir)
        self._step(rep, 5, "cross_artifact_consistency(pre)", cons["status"],
                   {"conflicts": cons["num_conflicts"], "warnings": cons["num_warnings"]})

        # 6) completeness (pre)
        from pro_v.witness_completeness_scorer import WitnessCompletenessScorer
        comp = WitnessCompletenessScorer().score(contract, wobj, mode=self.completeness_mode,
                                                  threshold=self.completeness_threshold, output_dir=output_dir)
        self._step(rep, 6, "witness_completeness(pre)", comp["gate"],
                   {"score": comp["summary_score"], "warnings": comp["warnings"][:4]})

        # 7) ambiguity GATE
        from pro_v.ambiguity_gate import AmbiguityGate
        gate = AmbiguityGate().run(contract, ambiguities_md=_read(os.path.join(output_dir, "ambiguities.md")),
                                   output_dir=output_dir)
        self._step(rep, 7, "ambiguity_gate", "pass" if gate["passed"] else "block",
                   {"questions": gate["questions"][:6]})
        if not gate["passed"]:
            rep["status"] = "blocked_ambiguity"
            rep["clarification"] = gate.get("clarification_request_path")
            rep["questions"] = gate["questions"]
            return self._finalize(rep, contract, wobj, iobj, comp, None, None, None, output_dir)

        # 8) guidance prompt
        from pro_v.pychecker_guidance_agent import PyCheckerGuidanceAgent
        guide = PyCheckerGuidanceAgent().run(artifacts_dir=output_dir, output_dir=output_dir)
        self._step(rep, 8, "pychecker_guidance", "ok", {"prompt": guide["path"]})

        # 9) generate golden_dut.py
        golden_path = os.path.join(output_dir, "golden_dut.py")
        src = self._pychecker(guide["prompt"])
        if not src or "class GoldenDUT" not in src:
            rep["status"] = "generation_failed"
            self._step(rep, 9, "pychecker_generate", "fail", {"reason": "no GoldenDUT produced"})
            return self._finalize(rep, contract, wobj, iobj, comp, None, None, None, output_dir)
        with open(golden_path, "w") as f:
            f.write(src if src.endswith("\n") else src + "\n")
        self._step(rep, 9, "pychecker_generate", "ok", {"path": golden_path})

        # 10) verify
        from pro_v.golden_model_verifier import GoldenModelVerifier
        ver = GoldenModelVerifier(**self.verifier_kwargs).run(golden_path, contract, wobj, iobj, output_dir=output_dir)
        self._step(rep, 10, "golden_model_verifier", ver["status"],
                   {"witnesses": "%d/%d" % (ver["witnesses_passed"], ver["num_witnesses"]),
                    "checks": ver["checks"]})

        # 11) repair if needed
        repaired = None
        if ver["status"] != "pass":
            from pro_v.golden_model_repair_loop import GoldenModelRepairLoop
            loop = GoldenModelRepairLoop(
                regenerate_fn=lambda p, a: self._pychecker(p),
                max_iters=self.max_repair_iters, verifier_kwargs=self.verifier_kwargs)
            repaired = loop.run(golden_path, contract, wobj, iobj, spec=description, output_dir=output_dir,
                                initial_report=ver)
            ver = repaired["final_report"]
            self._step(rep, 11, "golden_model_repair_loop",
                       "approved" if repaired["approved"] else "unapproved",
                       {"iterations": repaired["iterations"], "stop_reason": repaired["stop_reason"]})

        # 12) approval GATE
        approved = ver["status"] == "pass"
        rep["verification"] = {"status": ver["status"], "witnesses_passed": ver["witnesses_passed"],
                               "num_witnesses": ver["num_witnesses"], "metamorphic": ver.get("metamorphic")}
        if not approved:
            rep["status"] = "golden_unapproved"
            self._step(rep, 12, "approval_gate", "block", {"reason": "golden model failed verification"})
            return self._finalize(rep, contract, wobj, iobj, comp, ver, None, None, output_dir)
        self._step(rep, 12, "approval_gate", "pass", {})

        # 13) coverage plan (golden_dut.py is the source of truth)
        from pro_v.functional_coverage_agent import build_coverage_plan
        plan = build_coverage_plan(golden_path, contract, witnesses=wobj, invariants=iobj,
                                   dut_path=(dut_path or "top_module.v"), budget=self.budget, output_dir=output_dir)
        self._step(rep, 13, "functional_coverage_agent", "ok",
                   {"tests": plan["summary"]["num_tests"], "sequences": plan["summary"]["num_sequences"],
                    "goals": plan["summary"].get("num_goals"),
                    "uncovered": len(plan["summary"]["uncovered_goals"])})

        # 14) consistency (post, with plan)
        cons2 = CrossArtifactConsistencyChecker().run(spec=description, contract=contract, witnesses=wobj,
                                                      invariants=iobj, golden_dut_path=golden_path,
                                                      coverage_plan=plan)
        self._step(rep, 14, "cross_artifact_consistency(post)", cons2["status"],
                   {"conflicts": cons2["num_conflicts"]})

        # 15) completeness (post, with plan)
        comp2 = WitnessCompletenessScorer().score(contract, wobj, coverage_plan=plan,
                                                  mode=self.completeness_mode,
                                                  threshold=self.completeness_threshold, output_dir=output_dir)
        self._step(rep, 15, "witness_completeness(post)", comp2["gate"], {"score": comp2["summary_score"]})

        # 16) testbench from coverage_plan.json
        sim = None
        tb_path = os.path.join(output_dir, "testbench.sv")
        try:
            from pro_v.verilog_tb_generator import build_tb_from_plan, run_iverilog_tb
            from pro_v.mutation_strength import parse_ports
            ports_src = dut_code or header
            if ports_src:
                ports = parse_ports(ports_src)
                tb = build_tb_from_plan(plan, ports)   # include_random defaults False (no random substitute)
                with open(tb_path, "w") as f:
                    f.write(tb)
                self._step(rep, 16, "testbench_generator", "ok", {"path": tb_path})
                # 17) RTL simulation (black box)
                if dut_code:
                    import shutil
                    if shutil.which("iverilog"):
                        passed, log = run_iverilog_tb(dut_code, tb)
                        sim = {"ran": True, "passed": passed, "log_tail": log.strip().splitlines()[-4:]}
                        self._step(rep, 17, "rtl_simulation", "pass" if passed else "fail",
                                   {"note": "black-box top_module.v vs golden-derived expected"})
                    else:
                        sim = {"ran": False, "reason": "iverilog not installed"}
                        self._step(rep, 17, "rtl_simulation", "skip", {"reason": "iverilog not installed"})
                else:
                    sim = {"ran": False, "reason": "no DUT code provided"}
                    self._step(rep, 17, "rtl_simulation", "skip", {"reason": "no DUT code"})
            else:
                self._step(rep, 16, "testbench_generator", "skip", {"reason": "no interface (header/dut) provided"})
        except Exception as e:
            self._step(rep, 16, "testbench_generator", "error", {"error": str(e)})

        rep["approved"] = True
        rep["status"] = "approved"
        rep["coverage"] = plan["summary"]
        rep["simulation"] = sim
        return self._finalize(rep, contract, wobj, iobj, comp2, ver, plan, sim, output_dir)

    # -- helpers -----------------------------------------------------------

    def _pychecker(self, prompt: str) -> str:
        if self.pychecker_generate_fn is not None:
            return self.pychecker_generate_fn(prompt) or ""
        if self.llm_client is None:
            raise RuntimeError("no pychecker_generate_fn and no llm_client provided")
        return _extract_code(self.llm_client.chat(system=PYCHECKER_SYS, user=prompt))

    def _step(self, rep, n, name, status, detail):
        rep["steps"].append({"n": n, "name": name, "status": status, "detail": detail})

    def _finalize(self, rep, contract, wobj, iobj, comp, ver, plan, sim, output_dir) -> Dict[str, Any]:
        md = self._final_md(rep, contract, wobj, iobj, comp, ver, plan, sim)
        path = os.path.join(output_dir, "final_pipeline_report.md")
        with open(path, "w") as f:
            f.write(md)
        rep["report_path"] = path
        return rep

    def _final_md(self, rep, contract, wobj, iobj, comp, ver, plan, sim) -> str:
        L = ["# Pro-V SpecKit Pipeline Report — %s" % rep["status"].upper(), ""]
        L.append("**Design:** %s (%s)  •  **Approved:** %s"
                 % (contract.get("design_name"), contract.get("design_type"), rep.get("approved")))
        L.append("")
        L.append("## Pipeline steps")
        for s in rep["steps"]:
            L.append("- %2d. %-32s %s" % (s["n"], s["name"], s["status"]))
        L.append("")
        L.append("## Rules extracted")
        for r in contract.get("rules", []) or []:
            L.append("- **%s**: %s%s" % (r.get("id"), r.get("description") or "",
                                         "  ⟶  `%s`" % r["effect"] if r.get("effect") else ""))
        if not contract.get("rules"):
            L.append("- (none)")
        L.append("")
        L.append("## Ambiguities")
        if rep.get("questions"):
            for q in rep["questions"]:
                L.append("- %s" % q)
        else:
            crit = [a for a in (contract.get("assumptions") or [])]
            L.append("- No blocking ambiguities." + (" Recorded assumptions: %d." % len(crit) if crit else ""))
        L.append("")
        L.append("## Witnesses generated")
        by_type: Dict[str, int] = {}
        for w in wobj.get("witnesses", []):
            by_type[w.get("type", "?")] = by_type.get(w.get("type", "?"), 0) + 1
        L.append("- total: %d  (%s)" % (len(wobj.get("witnesses", [])),
                                        ", ".join("%s:%d" % (k, v) for k, v in sorted(by_type.items()))))
        if comp:
            L.append("- completeness score: %.2f (gate: %s)" % (comp["summary_score"], comp["gate"]))
            for w in comp.get("warnings", [])[:6]:
                L.append("  - ⚠ %s" % w)
        L.append("")
        L.append("## Golden model")
        if ver:
            L.append("- verification: **%s** (%d/%d witnesses)"
                     % (ver["status"], ver["witnesses_passed"], ver["num_witnesses"]))
            if ver.get("metamorphic"):
                L.append("- metamorphic: %s (%d passed / %d failed)"
                         % (ver["metamorphic"]["status"], ver["metamorphic"]["passed"], ver["metamorphic"]["failed"]))
            for f in ver.get("failures", [])[:6]:
                L.append("  - ✗ %s" % json.dumps(f)[:200])
        else:
            L.append("- not generated (pipeline stopped before generation)")
        L.append("")
        L.append("## Coverage")
        if plan:
            L.append("- goals created: %d" % plan["summary"].get("num_goals", 0))
            L.append("- tests: %d  •  sequences: %d" % (plan["summary"]["num_tests"], plan["summary"]["num_sequences"]))
            for g in plan.get("coverage_goals", [])[:12]:
                L.append("  - %s [%s]: %s" % (g["id"], g["type"], g["description"]))
            unc = plan["summary"]["uncovered_goals"]
            L.append("- **uncovered goals:** %s" % (", ".join(unc) if unc else "none"))
        else:
            L.append("- coverage not generated (golden model not approved)")
        L.append("")
        L.append("## RTL simulation (black box)")
        if sim and sim.get("ran"):
            L.append("- result: **%s** (top_module.v vs golden-derived expected)"
                     % ("PASS" if sim["passed"] else "FAIL"))
            for line in sim.get("log_tail", []):
                L.append("  - %s" % line)
        elif sim:
            L.append("- skipped: %s" % sim.get("reason"))
        else:
            L.append("- not run")
        return "\n".join(L) + "\n"


def _dump(d, name, obj):
    with open(os.path.join(d, name), "w") as f:
        json.dump(obj, f, indent=2)


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def run_pipeline(description, header="", dut_code=None, output_dir="artifacts",
                 llm_client=None, pychecker_generate_fn=None, **kw) -> Dict[str, Any]:
    return SpecKitProVOrchestrator(llm_client=llm_client, pychecker_generate_fn=pychecker_generate_fn, **kw).run(
        description, header=header, dut_code=dut_code, output_dir=output_dir)


# ---------------------------------------------------------------------------
# self-test (fake SpecKit LLM + injected pychecker; runs iverilog if present)
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, speckit_json: str):
        self._j = speckit_json

    def chat(self, system="", user="", **kw):
        return self._j if "SpecKit" in system else ""


def _selftest() -> int:
    import tempfile, shutil
    ok = True

    speckit = json.dumps({
        "behavior_contract": {
            "design_name": "and_gate", "design_type": "combinational",
            "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"y": {"width": 1}},
            "rules": [{"id": "R1", "description": "y = a AND b", "condition": "always",
                       "effect": "y = a & b", "confidence": 0.97}],
            "undefined_behavior": [], "assumptions": [], "can_generate_golden_model": True},
        "witnesses": [{"id": "W1", "type": "distinguishing", "required": True, "covers_rules": ["R1"],
                       "inputs": {"a": "0", "b": "1"}, "expected_outputs": {"y": "0"}, "reason": "AND vs OR"}],
        "invariants": [], "ambiguities": [],
        "spec_traceability": {"R1": {"source_text": "y should be a AND b", "confidence": 0.95}}})
    AND_SRC = ("class GoldenDUT:\n    def __init__(self): pass\n    def eval(self, inputs):\n"
               "        # Implements R1\n        return {'y': (inputs['a'] & inputs['b']) & 1}\n"
               "IMPLEMENTED_RULES=['R1']\n")
    header = "module top_module(input a, input b, output y);"
    good_dut = "module top_module(input a, input b, output y); assign y = a & b; endmodule\n"
    or_dut = "module top_module(input a, input b, output y); assign y = a | b; endmodule\n"

    # full happy path with a correct DUT
    with tempfile.TemporaryDirectory() as d:
        orch = SpecKitProVOrchestrator(llm_client=_FakeLLM(speckit),
                                       pychecker_generate_fn=lambda p: AND_SRC)
        r = orch.run("y is a AND b", header=header, dut_code=good_dut, output_dir=d)
        ok = ok and r["approved"] and r["status"] == "approved"
        for fn in ("behavior_contract.json", "witnesses.json", "invariants.json", "ambiguities.md",
                   "spec_traceability.json", "pychecker_prompt.md", "golden_dut.py",
                   "golden_verification_report.json", "witness_completeness_report.json",
                   "coverage_plan.json", "final_pipeline_report.md"):
            ok = ok and os.path.exists(os.path.join(d, fn))
        ok = ok and os.path.exists(os.path.join(d, "testbench.sv"))
        ok = ok and r["verification"]["status"] == "pass"
        ok = ok and r["coverage"]["num_tests"] >= 2
        if shutil.which("iverilog"):
            ok = ok and r["simulation"]["ran"] and r["simulation"]["passed"]

    # buggy DUT: golden still approved, but RTL simulation FAILS (bug caught)
    with tempfile.TemporaryDirectory() as d:
        orch = SpecKitProVOrchestrator(llm_client=_FakeLLM(speckit),
                                       pychecker_generate_fn=lambda p: AND_SRC)
        r = orch.run("y is a AND b", header=header, dut_code=or_dut, output_dir=d)
        ok = ok and r["approved"]
        if shutil.which("iverilog"):
            ok = ok and r["simulation"]["ran"] and r["simulation"]["passed"] is False

    # ambiguity gate blocks: contract can_generate=false -> no golden_dut.py, pychecker not run
    amb = json.loads(speckit); amb["behavior_contract"]["can_generate_golden_model"] = False
    amb["ambiguities"] = [{"id": "A1", "question": "sync or async reset?", "critical": True}]
    with tempfile.TemporaryDirectory() as d:
        called = {"n": 0}
        orch = SpecKitProVOrchestrator(llm_client=_FakeLLM(json.dumps(amb)),
                                       pychecker_generate_fn=lambda p: (called.__setitem__("n", called["n"] + 1) or AND_SRC))
        r = orch.run("adder maybe", header=header, output_dir=d)
        ok = ok and not r["approved"] and r["status"] == "blocked_ambiguity"
        ok = ok and called["n"] == 0 and not os.path.exists(os.path.join(d, "golden_dut.py"))
        ok = ok and os.path.exists(os.path.join(d, "final_pipeline_report.md"))

    # repair path: first generation is OR (fails W1), repair returns AND -> approved
    with tempfile.TemporaryDirectory() as d:
        state = {"n": 0}
        def gen(prompt):
            state["n"] += 1
            or_src = AND_SRC.replace("inputs['a'] & inputs['b']", "inputs['a'] | inputs['b']")
            return or_src if state["n"] == 1 else AND_SRC
        orch = SpecKitProVOrchestrator(llm_client=_FakeLLM(speckit), pychecker_generate_fn=gen)
        r = orch.run("y is a AND b", header=header, dut_code=good_dut, output_dir=d)
        ok = ok and r["approved"] and any(s["name"] == "golden_model_repair_loop" for s in r["steps"])

    print("speckit_prov_orchestrator selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
