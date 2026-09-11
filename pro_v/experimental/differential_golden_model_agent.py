#!/usr/bin/env python3
"""
differential_golden_model_agent.py  --  two-model cross-check (Pro-V stage, optional).

Generates TWO independent Python reference models from the same behavior contract
(using two different prompt styles) and compares them. If two independently derived
models disagree, neither is trusted until the disagreement is resolved -- this catches
hallucinated reference models before any RTL verification.

Runs after pychecker_guidance_agent.py and before final golden-model approval. It is
optional (2x generation cost).

Inputs : behavior_contract.json, witnesses.json, invariants.json
Process: generate A and B -> run both on witnesses + boundary + random + sequential
traces -> compare. Agree AND both pass witnesses/invariants -> approve one as
golden_dut.py. Disagree -> golden_disagreement_report.json with the SMALLEST
disagreeing input/sequence, arbitrated against the contract (which model is wrong),
mapped to rule IDs, with a repair/clarification action.

Generation is injected as generate_fn(prompt, style, attempt) -> source (default:
llm_client.chat with a style-specific system prompt), so this is testable offline.
Never inspects top_module.v.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from pro_v.golden_model_verifier import (ModelAdapter, _load_golden_class, _load, _to_int, _w,
                                             _witness_list, _invariant_list, GoldenModelVerifier)
    from pro_v.witness_generator import generate_witnesses, boundary_values, eval_effect, _output_exprs
    from pro_v.pychecker_guidance_agent import PyCheckerGuidanceAgent
except Exception:  # pragma: no cover
    from golden_model_verifier import (ModelAdapter, _load_golden_class, _load, _to_int, _w,
                                       _witness_list, _invariant_list, GoldenModelVerifier)
    from witness_generator import generate_witnesses, boundary_values, eval_effect, _output_exprs
    from pychecker_guidance_agent import PyCheckerGuidanceAgent


STYLE_A = ("## Implementation style A\nImplement the behavior with DIRECT algebraic / boolean "
           "expressions (e.g. y = a & b). Prefer a concise closed-form per output.")
STYLE_B = ("## Implementation style B\nImplement the behavior by EXPLICIT case / truth-table "
           "enumeration over the inputs (branch on each relevant condition). Do not simplify to "
           "an algebraic form; enumerate the cases the contract describes.")

SYS = ("You are the PyChecker golden-model generator. Implement golden_dut.py strictly from the "
       "contract. Output ONLY one ```python code block defining class GoldenDUT with the required API.")


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


class DifferentialGoldenModelAgent:

    def __init__(self, guidance: Optional[PyCheckerGuidanceAgent] = None,
                 generate_fn: Optional[Callable[[str, str, int], str]] = None,
                 llm_client: Any = None, max_random: int = 64, max_trace_depth: int = 12,
                 num_random_traces: int = 6, seed: int = 0, verifier_kwargs: Optional[dict] = None):
        self.guidance = guidance or PyCheckerGuidanceAgent()
        self.generate_fn = generate_fn
        self.llm_client = llm_client
        self.max_random = max_random
        self.max_trace_depth = max_trace_depth
        self.num_random_traces = num_random_traces
        self.seed = seed
        self.verifier_kwargs = verifier_kwargs or {}

    # -- public ------------------------------------------------------------

    def run(self, contract: Any, witnesses: Any = None, invariants: Any = None,
            output_dir: Optional[str] = None) -> Dict[str, Any]:
        import random
        contract = _load(contract)
        wl_obj = _load(witnesses) or {}
        inv_obj = _load(invariants) or {}
        wl = _witness_list(wl_obj)
        out_dir = output_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        rng = random.Random(self.seed)
        seq = contract.get("design_type") == "sequential"

        # 1-2) generate A and B with two prompt styles
        base = self.guidance.build_prompt(contract, wl_obj, inv_obj)
        pathA = os.path.join(out_dir, "golden_dut_A.py")
        pathB = os.path.join(out_dir, "golden_dut_B.py")
        srcA = self._generate(base + "\n\n" + STYLE_A, "A", 0)
        srcB = self._generate(base + "\n\n" + STYLE_B, "B", 0)
        for p, s in ((pathA, srcA), (pathB, srcB)):
            with open(p, "w") as f:
                f.write((s or "") + ("\n" if s and not s.endswith("\n") else ""))

        adapters = {}
        load_errs = {}
        for tag, path in (("A", pathA), ("B", pathB)):
            try:
                adapters[tag] = ModelAdapter(_load_golden_class(path), contract)
            except Exception as e:
                load_errs[tag] = str(e)
        if load_errs:
            return self._write(out_dir, {
                "approved": False, "status": "generation_failed",
                "load_errors": load_errs, "models": {"A": pathA, "B": pathB},
                "suggested_action": "regenerate the model(s) that failed to load"})

        # 3-4) run both on candidates and compare
        if seq:
            disagreements = self._compare_sequential(adapters, contract, wl, rng)
            n_compared = len(self._seq_traces(contract, wl, rng))
        else:
            disagreements = self._compare_combinational(adapters, contract, wl, rng)
            n_compared = len(self._comb_candidates(contract, wl, rng))

        if disagreements:
            report = self._disagreement_report(contract, disagreements, pathA, pathB, n_compared)
            return self._write(out_dir, report)

        # 5) agree -> require both pass witnesses/invariants, then approve one
        ver = GoldenModelVerifier(**self.verifier_kwargs)
        rA = ver.run(pathA, contract, wl_obj, inv_obj)
        rB = ver.run(pathB, contract, wl_obj, inv_obj)
        if rA["status"] == "pass":
            chosen, other = "A", "B"
        elif rB["status"] == "pass":
            chosen, other = "B", "A"
        else:
            # agree on the SAME (wrong) behavior -> both fail witnesses/invariants
            return self._write(out_dir, {
                "approved": False, "status": "agree_but_fail_verification",
                "explanation": "Both models agree on every candidate but both fail witness/invariant "
                               "verification -- they likely share the same wrong behavior.",
                "verification": {"A": rA["status"], "B": rB["status"]},
                "failing_A": rA.get("failures", [])[:6], "failing_B": rB.get("failures", [])[:6],
                "models": {"A": pathA, "B": pathB}, "num_candidates_compared": n_compared,
                "suggested_action": "repair using the failing witnesses (both models are wrong the same way)"})

        golden = os.path.join(out_dir, "golden_dut.py")
        with open(golden, "w") as f:
            f.write(open(pathA if chosen == "A" else pathB).read())
        return self._write(out_dir, {
            "approved": True, "status": "agree", "chosen": chosen,
            "golden_dut_path": golden, "models": {"A": pathA, "B": pathB},
            "verification": {"A": rA["status"], "B": rB["status"]},
            "num_candidates_compared": n_compared,
            "note": "Two independently-derived models agree on all %d candidates and %s passes verification."
                    % (n_compared, chosen)}, report_name=None)

    # -- comparison --------------------------------------------------------

    def _comb_candidates(self, contract, wl, rng) -> List[Dict[str, int]]:
        in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        names = list(in_w)
        seen, out = set(), []

        def add(vec):
            key = tuple(vec.get(n, 0) for n in names)
            if key not in seen:
                seen.add(key); out.append({n: vec.get(n, 0) for n in names})

        for wit in wl:
            if isinstance(wit.get("inputs"), dict) and "sequence" not in wit and not wit.get("relation"):
                add({k: _to_int(v) for k, v in wit["inputs"].items() if _to_int(v) is not None})
        for n in names:
            for _lbl, val in boundary_values(in_w[n], False):
                b = {k: 0 for k in names}; b[n] = val; add(b)
        add({n: 0 for n in names})
        add({n: (1 << in_w[n]) - 1 for n in names})
        total_bits = sum(in_w.values())
        if total_bits <= 8:
            for combo in range(1 << total_bits):
                v, shift = {}, 0
                for n in reversed(names):
                    w = in_w[n]; v[n] = (combo >> shift) & ((1 << w) - 1); shift += w
                add(v)
        else:
            for _ in range(self.max_random):
                add({n: rng.randrange(1 << in_w[n]) for n in names})
        return out

    def _compare_combinational(self, adapters, contract, wl, rng) -> List[dict]:
        out_names = list((contract.get("outputs") or {}).keys())
        diffs = []
        for inp in self._comb_candidates(contract, wl, rng):
            try:
                oa = adapters["A"].eval(adapters["A"].cls(), inp)
                ob = adapters["B"].eval(adapters["B"].cls(), inp)
            except Exception:
                continue
            oa = {k: oa.get(k) for k in out_names}
            ob = {k: ob.get(k) for k in out_names}
            if oa != ob:
                diffs.append({"kind": "single_cycle", "inputs": inp, "output_A": oa, "output_B": ob,
                              "magnitude": sum(inp.values())})
        return diffs

    def _seq_traces(self, contract, wl, rng) -> List[List[Dict[str, int]]]:
        in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        names = list(in_w)
        traces: List[List[Dict[str, int]]] = []
        for wit in wl:
            if "sequence" in wit:
                tr = []
                for step in wit["sequence"][:self.max_trace_depth]:
                    ints = {k: _to_int(v) for k, v in (step.get("inputs") or {}).items()}
                    tr.append({n: (ints.get(n) if ints.get(n) is not None else 0) for n in names})
                if tr:
                    traces.append(tr)
        for _ in range(self.num_random_traces):
            depth = rng.randint(2, self.max_trace_depth)
            traces.append([{n: rng.randrange(1 << in_w[n]) for n in names} for _ in range(depth)])
        return traces

    def _compare_sequential(self, adapters, contract, wl, rng) -> List[dict]:
        out_names = list((contract.get("outputs") or {}).keys())
        diffs = []
        for trace in self._seq_traces(contract, wl, rng):
            ia, ib = adapters["A"].new(), adapters["B"].new()
            for c, row in enumerate(trace):
                try:
                    oa = adapters["A"].step(ia, row)
                    ob = adapters["B"].step(ib, row)
                except Exception:
                    break
                oa = {k: oa.get(k) for k in out_names}
                ob = {k: ob.get(k) for k in out_names}
                if oa != ob:
                    diffs.append({"kind": "sequence", "sequence": trace[:c + 1], "cycle": c,
                                  "output_A": oa, "output_B": ob, "magnitude": c + 1})
                    break
        return diffs

    # -- disagreement report ----------------------------------------------

    def _disagreement_report(self, contract, diffs, pathA, pathB, n_compared) -> dict:
        diffs.sort(key=lambda d: (0 if d["kind"] == "single_cycle" else 1, d["magnitude"]))
        smallest = diffs[0]
        arb = self._arbitrate(contract, smallest)
        smallest = {**smallest, **arb}
        return {
            "approved": False, "status": "disagreement",
            "num_disagreements": len(diffs),
            "smallest_disagreement": smallest,
            "disagreements": [self._brief(d) for d in diffs[:12]],
            "models": {"A": pathA, "B": pathB},
            "num_candidates_compared": n_compared,
            "suggested_action": arb.get("suggested_action", "repair the wrong model or clarify the spec"),
        }

    def _arbitrate(self, contract, diff) -> dict:
        """Use the contract as arbiter: whichever model matches the rule-computed
        output is right; the other is wrong. Map to rule IDs."""
        out_names = list((contract.get("outputs") or {}).keys())
        out_w = {n: _w(s) for n, s in (contract.get("outputs") or {}).items()}
        exprs = _output_exprs(contract.get("rules", []) or [], out_names)
        rule_of = self._rule_for_output(contract, out_names)
        if diff["kind"] == "single_cycle":
            env = dict(diff["inputs"])
        else:  # arbitrate the last cycle of the trace (where they first differ)
            env = dict(diff["sequence"][-1])
        expected, ruleset, verdict = {}, set(), {"A": 0, "B": 0}
        oa, ob = diff["output_A"], diff["output_B"]
        for o in out_names:
            if o not in exprs or oa.get(o) == ob.get(o):
                continue
            comp = eval_effect(o + " = " + exprs[o], env, out_w.get(o, 1))
            if comp is None:
                continue
            ev = int(comp, 2)
            expected[o] = ev
            ruleset.add(rule_of.get(o, "R?"))
            if oa.get(o) == ev:
                verdict["A"] += 1
            if ob.get(o) == ev:
                verdict["B"] += 1
        if not expected:
            return {"contract_expected": None, "likely_wrong": None, "rule_ids": [],
                    "explanation": "Models disagree but the contract rule is not evaluable here; "
                                   "human clarification is required.",
                    "suggested_action": "clarify the intended behavior for this input"}
        wrong = "B" if verdict["A"] > verdict["B"] else ("A" if verdict["B"] > verdict["A"] else None)
        if wrong is None:
            expl = ("Models disagree and neither fully matches the contract on this input; "
                    "both may be wrong -- clarify.")
            action = "clarify the intended behavior; both models are suspect"
        else:
            right = "A" if wrong == "B" else "B"
            expl = ("Models disagree on %s. The contract (%s) computes %s, which matches model %s; "
                    "model %s is wrong." % (self._where(diff), ", ".join(sorted(ruleset)) or "rules",
                                            expected, right, wrong))
            action = "repair model %s (contract-derived output is %s)" % (wrong, expected)
        return {"contract_expected": expected, "likely_wrong": wrong, "rule_ids": sorted(ruleset),
                "explanation": expl, "suggested_action": action}

    def _where(self, diff) -> str:
        if diff["kind"] == "single_cycle":
            return "input %s" % diff["inputs"]
        return "cycle %d of the sequence (inputs %s)" % (diff.get("cycle", 0), diff["sequence"][-1])

    def _brief(self, d) -> dict:
        b = {"kind": d["kind"], "output_A": d["output_A"], "output_B": d["output_B"]}
        if d["kind"] == "single_cycle":
            b["inputs"] = d["inputs"]
        else:
            b["cycle"] = d.get("cycle"); b["sequence"] = d["sequence"]
        return b

    def _rule_for_output(self, contract, out_names) -> Dict[str, str]:
        m = {}
        for r in contract.get("rules", []) or []:
            eff = (r.get("effect") or "")
            if "=" in eff:
                lhs = re.sub(r"\[.*", "", eff.split("=", 1)[0]).strip()
                if lhs in out_names and lhs not in m:
                    m[lhs] = r.get("id", "R?")
        return m

    # -- generation / io ---------------------------------------------------

    def _generate(self, prompt: str, style: str, attempt: int) -> str:
        if self.generate_fn is not None:
            return self.generate_fn(prompt, style, attempt) or ""
        if self.llm_client is None:
            raise RuntimeError("no generate_fn and no llm_client provided")
        return _extract_code(self.llm_client.chat(system=SYS, user=prompt))

    def _write(self, out_dir, report, report_name="golden_disagreement_report.json") -> dict:
        if report_name and not report.get("approved"):
            path = os.path.join(out_dir, report_name)
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            report["report_path"] = path
        return report


def run_differential(contract, witnesses=None, invariants=None, output_dir=None,
                     llm_client=None, generate_fn=None, **kw) -> Dict[str, Any]:
    return DifferentialGoldenModelAgent(llm_client=llm_client, generate_fn=generate_fn, **kw).run(
        contract, witnesses, invariants, output_dir)


# ---------------------------------------------------------------------------
# self-test (fake generators; no LLM, no RTL)
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
    AND = "```python\nclass GoldenDUT:\n    def eval(self, i):\n        # Implements R1\n        return {'y': (i['a'] & i['b']) & 1}\n```"
    OR = "```python\nclass GoldenDUT:\n    def eval(self, i):\n        # Implements R1\n        return {'y': (i['a'] | i['b']) & 1}\n```"

    # 1) agree: A and B both AND -> approved, chosen A
    with tempfile.TemporaryDirectory() as d:
        agent = DifferentialGoldenModelAgent(generate_fn=lambda p, style, a: _extract_code(AND))
        r = agent.run(contract, witnesses, {"invariants": []}, output_dir=d)
        ok = ok and r["approved"] and r["status"] == "agree" and r["chosen"] == "A"
        ok = ok and os.path.exists(os.path.join(d, "golden_dut.py"))
        ok = ok and "a'] & i['b" in open(os.path.join(d, "golden_dut.py")).read()

    # 2) disagree: A=AND, B=OR -> report, smallest = a=0,b=1, contract says B wrong (R1)
    with tempfile.TemporaryDirectory() as d:
        agent = DifferentialGoldenModelAgent(
            generate_fn=lambda p, style, a: _extract_code(AND if style == "A" else OR))
        r = agent.run(contract, witnesses, {"invariants": []}, output_dir=d)
        ok = ok and not r["approved"] and r["status"] == "disagreement"
        sm = r["smallest_disagreement"]
        ok = ok and sm["kind"] == "single_cycle" and sm["inputs"] == {"a": 0, "b": 1}
        ok = ok and sm["output_A"] == {"y": 0} and sm["output_B"] == {"y": 1}
        ok = ok and sm["likely_wrong"] == "B" and sm["rule_ids"] == ["R1"]
        ok = ok and sm["contract_expected"] == {"y": 0}
        ok = ok and "model B is wrong" in sm["explanation"]
        ok = ok and os.path.exists(r["report_path"])

    # 3) agree but both wrong (both OR on an AND contract) -> agree_but_fail_verification
    with tempfile.TemporaryDirectory() as d:
        agent = DifferentialGoldenModelAgent(generate_fn=lambda p, style, a: _extract_code(OR))
        r = agent.run(contract, witnesses, {"invariants": []}, output_dir=d)
        ok = ok and not r["approved"] and r["status"] == "agree_but_fail_verification"

    # 4) generation failure -> reported
    with tempfile.TemporaryDirectory() as d:
        agent = DifferentialGoldenModelAgent(generate_fn=lambda p, style, a: "" if style == "B" else _extract_code(AND))
        r = agent.run(contract, witnesses, {"invariants": []}, output_dir=d)
        ok = ok and not r["approved"] and r["status"] == "generation_failed" and "B" in r["load_errors"]

    # 5) sequential disagreement: A counts, B ignores enable
    seq_c = {"design_name": "cnt", "design_type": "sequential",
             "inputs": {"reset": {"width": 1}, "en": {"width": 1}}, "outputs": {"q": {"width": 2}},
             "reset": {"name": "reset", "active_high": True}, "states": [],
             "rules": [{"id": "R1", "description": "increment when en"}], "can_generate_golden_model": True}
    seq_w = {"witnesses": [{"id": "WS", "type": "sequential_reachability", "required": True, "covers_rules": ["R1"],
                            "sequence": [{"cycle": 0, "inputs": {"reset": "1", "en": "0"}},
                                         {"cycle": 1, "inputs": {"reset": "0", "en": "1"}},
                                         {"cycle": 2, "inputs": {"reset": "0", "en": "1"}}]}]}
    A_CNT = ("```python\nclass GoldenDUT:\n    def __init__(self): self.reset()\n    def reset(self): self.q=0\n"
             "    def step(self,i):\n        if i.get('reset'): self.q=0\n        elif i.get('en'): self.q=(self.q+1)&3\n"
             "        return {'q': self.q}\n    def get_state(self): return {'q': self.q}\n```")
    B_CNT = A_CNT.replace("(self.q+1)&3", "self.q")  # ignores enable
    with tempfile.TemporaryDirectory() as d:
        agent = DifferentialGoldenModelAgent(
            generate_fn=lambda p, style, a: _extract_code(A_CNT if style == "A" else B_CNT))
        r = agent.run(seq_c, seq_w, {"invariants": []}, output_dir=d)
        ok = ok and not r["approved"] and r["status"] == "disagreement"
        ok = ok and r["smallest_disagreement"]["kind"] == "sequence"

    print("differential_golden_model_agent selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
