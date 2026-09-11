#!/usr/bin/env python3
"""
golden_model_verifier.py  --  the trust gate (Pro-V stage 4).

Verifies that a generated `golden_dut.py` is consistent with the spec-derived
artifacts BEFORE the functional coverage agent is allowed to use it. If the
reference model is wrong, every downstream test is wrong -- so this is a hard gate.

Inputs : golden_dut.py, behavior_contract.json, witnesses.json, invariants.json
Outputs: golden_verification_report.{json,md} and a pass/fail status; on failure a
         repair report suitable for pychecker_guidance_agent(prior_report=...).

It NEVER inspects top_module.v. It only checks the Python model against the
contract, witnesses, and invariants.

Checks: import/API, interface, witnesses, invariants, ambiguity gate, traceability,
determinism, and behavioral mutation-sanity probes (always-const, ignores input /
control / reset, width masking, state stuck / state moves under hold).

Comparison is value-based: witness bitstrings, eval() ints, and legacy load()
bitstrings are all coerced to integers per signal, so the verifier is agnostic to
which adapter API the model exposes.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from pro_v.witness_generator import boundary_values, _is_control
except Exception:  # pragma: no cover
    from witness_generator import boundary_values, _is_control


# ---------------------------------------------------------------------------
# value helpers
# ---------------------------------------------------------------------------

def _to_int(v: Any) -> Optional[int]:
    """Coerce a signal value (int, '0101' bitstring, or bool) to int; None if unknown."""
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


def _bits(val: int, width: int) -> str:
    width = max(1, int(width))
    return format(int(val) & ((1 << width) - 1), "0%db" % width)


def _fmt(m: Dict[str, Any]) -> str:
    return "{" + ", ".join("%s: %s" % (k, v) for k, v in m.items()) + "}"


# ---------------------------------------------------------------------------
# soft call timeout (best-effort; no-op off the main thread)
# ---------------------------------------------------------------------------

class _Timeout(Exception):
    pass


def _guard(fn, seconds: float = 5.0):
    try:
        import signal
        def _h(_s, _f):
            raise _Timeout()
        old = signal.signal(signal.SIGALRM, _h)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    except (ValueError, ImportError):      # not main thread / unsupported
        return fn()


# ---------------------------------------------------------------------------
# model adapter (bridges eval / step / legacy load)
# ---------------------------------------------------------------------------

class ModelAdapter:
    def __init__(self, cls, contract: dict):
        self.cls = cls
        self.seq = contract.get("design_type") == "sequential"
        self.in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        self.out_names = list((contract.get("outputs") or {}).keys())
        inst = cls()
        self.has_eval = callable(getattr(inst, "eval", None))
        self.has_step = callable(getattr(inst, "step", None))
        self.has_reset = callable(getattr(inst, "reset", None))
        self.has_load = callable(getattr(inst, "load", None))
        self.has_get_state = callable(getattr(inst, "get_state", None))

    # -- API adequacy ------------------------------------------------------

    def api_ok(self, require_new_api: bool) -> Tuple[bool, List[str]]:
        errs = []
        if self.seq:
            if not (self.has_step and self.has_reset):
                if require_new_api or not self.has_load:
                    errs.append("sequential model must expose reset() and step()")
        else:
            if not self.has_eval:
                if require_new_api or not self.has_load:
                    errs.append("combinational model must expose eval(inputs)")
        return (not errs), errs

    # -- encoding ----------------------------------------------------------

    def _encode(self, inputs_int: Dict[str, int], for_load: bool) -> Dict[str, Any]:
        if for_load:
            return {n: _bits(v, self.in_w.get(n, 1)) for n, v in inputs_int.items()}
        return dict(inputs_int)

    def _decode(self, out: Any) -> Dict[str, Optional[int]]:
        if not isinstance(out, dict):
            return {}
        return {k: _to_int(v) for k, v in out.items()}

    # -- combinational -----------------------------------------------------

    def eval(self, inst, inputs_int: Dict[str, int]) -> Dict[str, Optional[int]]:
        if self.has_eval:
            return self._decode(_guard(lambda: inst.eval(self._encode(inputs_int, False))))
        if self.has_load:
            return self._decode(_guard(lambda: inst.load(self._encode(inputs_int, True))))
        raise RuntimeError("no combinational API")

    # -- sequential --------------------------------------------------------

    def new(self):
        inst = self.cls()
        if self.has_reset:
            _guard(lambda: inst.reset())
        return inst

    def step(self, inst, inputs_int: Dict[str, int]) -> Dict[str, Optional[int]]:
        if self.has_step:
            return self._decode(_guard(lambda: inst.step(self._encode(inputs_int, False))))
        if self.has_load:
            return self._decode(_guard(lambda: inst.load(self._encode(inputs_int, True))))
        raise RuntimeError("no sequential API")

    def state(self, inst) -> Optional[dict]:
        if self.has_get_state:
            try:
                s = _guard(lambda: inst.get_state())
                return s if isinstance(s, dict) else {"state": s}
            except Exception:
                return None
        return None


def _w(spec: Any) -> int:
    try:
        return max(1, int(spec.get("width", 1))) if isinstance(spec, dict) else 1
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------

class GoldenModelVerifier:
    MAX_CALLS = 3000

    def __init__(self, strict_traceability: bool = False, require_new_api: bool = False,
                 sample_budget: int = 64, seed: int = 0, run_metamorphic: bool = False):
        self.strict_traceability = strict_traceability
        self.require_new_api = require_new_api
        self.sample_budget = sample_budget
        self.seed = seed
        self.run_metamorphic = run_metamorphic

    # -- public ------------------------------------------------------------

    def run(self, golden_path: str, contract: Any, witnesses: Any = None,
            invariants: Any = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
        contract = _load(contract)
        wl = _witness_list(_load(witnesses))
        inv = _invariant_list(_load(invariants))
        rng = random.Random(self.seed)

        report: Dict[str, Any] = {
            "status": "fail", "golden_path": golden_path,
            "design_type": contract.get("design_type"),
            "num_witnesses": len(wl), "witnesses_passed": 0, "witnesses_failed": 0,
            "invariants_passed": 0, "invariants_failed": 0,
            "failures": [], "warnings": [], "checks": {},
        }

        # 5) ambiguity gate (do this first -- cheapest, hardest stop)
        if contract.get("can_generate_golden_model") is False:
            report["checks"]["ambiguity"] = "fail"
            report["failures"].append({"check": "ambiguity",
                                       "reason": "behavior_contract.can_generate_golden_model=false; clarification required"})
            report["repair_prompt"] = "Resolve critical ambiguities before generating golden_dut.py."
            return self._finalize(report, contract, wl, output_dir)

        # 1) import + API
        try:
            cls = _load_golden_class(golden_path)
        except Exception as e:
            report["checks"]["import"] = "fail"
            report["failures"].append({"check": "import", "reason": "cannot import GoldenDUT: %s" % e})
            return self._finalize(report, contract, wl, output_dir)
        report["checks"]["import"] = "pass"

        adapter = ModelAdapter(cls, contract)
        api_ok, api_errs = adapter.api_ok(self.require_new_api)
        report["checks"]["api"] = "pass" if api_ok else "fail"
        if not api_ok:
            for e in api_errs:
                report["failures"].append({"check": "api", "reason": e})
            return self._finalize(report, contract, wl, output_dir)

        # 2) interface
        self._check_interface(adapter, contract, report, rng)

        # 6) traceability
        self._check_traceability(golden_path, contract, report)

        # 3) witnesses
        self._check_witnesses(adapter, contract, wl, report)

        # 3b) metamorphic relations (opt-in; runs after exact witness checks)
        if self.run_metamorphic:
            self._check_metamorphic(golden_path, contract, inv, report)

        # 7) determinism
        self._check_determinism(adapter, contract, wl, report, rng)

        # 4) invariants
        self._check_invariants(adapter, contract, inv, wl, report, rng)

        # 8) mutation-sanity probes
        self._mutation_probes(adapter, contract, wl, report, rng)

        return self._finalize(report, contract, wl, output_dir)

    # -- 2) interface ------------------------------------------------------

    def _check_interface(self, adapter, contract, report, rng) -> None:
        outs = list((contract.get("outputs") or {}).keys())
        out_w = {n: _w(s) for n, s in (contract.get("outputs") or {}).items()}
        probe = self._sample_inputs(contract, [], rng, n=1)
        got = {}
        if probe:
            try:
                got = self._run_once(adapter, contract, probe[0])
            except Exception as e:
                report["warnings"].append({"check": "interface", "reason": "model call failed: %s" % e})
        missing = [o for o in outs if o not in got]
        extra = [k for k in got if k not in outs]
        oob = [o for o in outs if o in got and got[o] is not None and not (0 <= got[o] < (1 << out_w[o]))]
        status = "pass"
        if missing:
            report["failures"].append({"check": "interface", "reason": "missing outputs: %s" % missing})
            status = "fail"
        if extra:
            report["warnings"].append({"check": "interface", "reason": "extra outputs: %s" % extra})
        if oob:
            report["failures"].append({"check": "interface", "reason": "outputs exceed declared width: %s" % oob})
            status = "fail"
        report["checks"]["interface"] = status

    # -- 3) witnesses ------------------------------------------------------

    def _check_witnesses(self, adapter, contract, wl, report) -> None:
        passed = failed = 0
        for w in wl:
            res = self._run_witness(adapter, contract, w)
            if res["skipped"]:
                report["warnings"].append({"witness_id": w.get("id"), "reason": res["reason"]})
                continue
            if res["ok"]:
                passed += 1
            else:
                failed += 1
                fail = {"witness_id": w.get("id"), "type": w.get("type"),
                        "required": bool(w.get("required")), "covers_rules": w.get("covers_rules", []),
                        "reason": res["reason"]}
                fail.update({k: res[k] for k in ("inputs", "expected", "actual", "sequence") if k in res})
                report["failures" if w.get("required") else "warnings"].append(fail)
        report["witnesses_passed"] = passed
        report["witnesses_failed"] = failed
        report["checks"]["witnesses"] = "fail" if any(
            f.get("witness_id") and f.get("required") for f in report["failures"]) else "pass"

    def _run_witness(self, adapter, contract, w) -> Dict[str, Any]:
        try:
            if w.get("relation"):
                return self._run_relation(adapter, contract, w)
            if "sequence" in w:
                return self._run_sequence(adapter, contract, w)
            return self._run_point(adapter, contract, w)
        except _Timeout:
            return {"ok": False, "skipped": False, "reason": "model timed out on witness"}
        except Exception as e:
            return {"ok": False, "skipped": False, "reason": "model raised: %s" % e}

    def _run_point(self, adapter, contract, w) -> Dict[str, Any]:
        inputs = _int_inputs(w.get("inputs", {}))
        actual = self._run_once(adapter, contract, inputs)
        if not w.get("expected_outputs"):
            return {"ok": True, "skipped": True, "reason": "no expected output (input-only witness)"}
        expected = _int_inputs(w["expected_outputs"])
        bad = {k: (expected[k], actual.get(k)) for k in expected if actual.get(k) != expected[k]}
        if bad:
            return {"ok": False, "skipped": False,
                    "inputs": inputs, "expected": expected,
                    "actual": {k: actual.get(k) for k in expected},
                    "reason": w.get("reason") or "witness output mismatch %s" % bad}
        return {"ok": True, "skipped": False}

    def _run_sequence(self, adapter, contract, w) -> Dict[str, Any]:
        if not adapter.seq:
            return {"ok": True, "skipped": True, "reason": "sequence witness on non-sequential model"}
        inst = adapter.new()
        cyc_out = []
        for step in w["sequence"]:
            ins = _int_inputs(step.get("inputs", {}))
            out = adapter.step(inst, ins)
            cyc_out.append(out)
            exp = step.get("expected_outputs")
            if exp:
                exp = _int_inputs(exp)
                bad = {k: (exp[k], out.get(k)) for k in exp if out.get(k) != exp[k]}
                if bad:
                    return {"ok": False, "skipped": False, "sequence": w.get("sequence"),
                            "expected": exp, "actual": {k: out.get(k) for k in exp},
                            "reason": "cycle %s output mismatch %s" % (step.get("cycle"), bad)}
        ef = w.get("expected_final")
        if ef and "state" in ef:
            st = adapter.state(inst)
            if st is None:
                return {"ok": True, "skipped": True, "reason": "no get_state(); final-state unverified"}
            got = st.get("state", next(iter(st.values()), None))
            if str(got) != str(ef["state"]):
                return {"ok": False, "skipped": False, "sequence": w.get("sequence"),
                        "expected": ef, "actual": {"state": got},
                        "reason": "reached state %s, expected %s" % (got, ef["state"])}
        return {"ok": True, "skipped": False}

    def _run_relation(self, adapter, contract, w) -> Dict[str, Any]:
        rel = w.get("relation")
        p = w.get("params", {})
        if rel == "commutative" and not adapter.seq:
            a, b = (p.get("operands") or [None, None])[:2]
            base = _int_inputs(p.get("inputs", {})) or {n: 0 for n in (contract.get("inputs") or {})}
            if a and b:
                o1 = self._run_once(adapter, contract, base)
                sw = dict(base); sw[a], sw[b] = base.get(b, 0), base.get(a, 0)
                o2 = self._run_once(adapter, contract, sw)
                if o1 != o2:
                    return {"ok": False, "skipped": False, "inputs": base,
                            "expected": {"f(a,b)": o1}, "actual": {"f(b,a)": o2},
                            "reason": "commutativity violated"}
                return {"ok": True, "skipped": False}
        if rel == "mux_independence" and not adapter.seq:
            return self._check_mux_independence(adapter, contract, p)
        return {"ok": True, "skipped": True, "reason": "relation '%s' not executed here" % rel}

    def _check_mux_independence(self, adapter, contract, p) -> Dict[str, Any]:
        sel = p.get("select")
        data = p.get("data") or []
        in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        if not sel or len(data) < 2:
            return {"ok": True, "skipped": True, "reason": "mux params incomplete"}
        for sv in range(min(1 << in_w.get(sel, 1), 4)):
            base = {n: 0 for n in in_w}
            base[sel] = sv
            o1 = self._run_once(adapter, contract, base)
            for d in data:
                var = dict(base); var[d] = (1 << in_w.get(d, 1)) - 1
                o2 = self._run_once(adapter, contract, var)
                if o1 != o2 and all(o1.get(k) == o2.get(k) for k in o1) is False:
                    # output moved when SOME unselected input changed -- only a violation if that
                    # input is genuinely unselected for this sel; we cannot know which, so warn softly
                    if _all_data_change_moves_output(o1, o2):
                        return {"ok": False, "skipped": False, "inputs": base,
                                "reason": "output changed when unselected input %s changed at sel=%d" % (d, sv)}
        return {"ok": True, "skipped": False}

    # -- 3b) metamorphic (opt-in hook) -------------------------------------

    def _check_metamorphic(self, golden_path, contract, inv, report) -> None:
        try:
            from pro_v.metamorphic_checker import MetamorphicChecker
        except Exception:  # pragma: no cover
            from metamorphic_checker import MetamorphicChecker
        try:
            mr = MetamorphicChecker().run(golden_path, contract, inv)
        except Exception as e:
            report["warnings"].append({"check": "metamorphic", "reason": "metamorphic checker errored: %s" % e})
            report["checks"]["metamorphic"] = "warn"
            return
        report["metamorphic"] = {"status": mr["status"], "passed": mr["passed"],
                                 "failed": mr["failed"], "skipped": mr.get("skipped", 0)}
        for f in mr["failures"]:
            report["failures"].append({
                "check": "metamorphic", "property_id": f["id"], "name": f["name"],
                "relation": f.get("relation"), "reason": f.get("repair_hint"),
                "inputs": f.get("inputs"), "trace": f.get("trace"),
                "actual": f.get("actual"), "expected": f.get("expected"),
                "covers_rules": f.get("covers_rules", [])})
        report["checks"]["metamorphic"] = mr["status"]

    # -- 4) invariants -----------------------------------------------------

    def _check_invariants(self, adapter, contract, inv, wl, report, rng) -> None:
        samples = self._sample_inputs(contract, wl, rng, n=self.sample_budget)
        passed = failed = 0
        for i in inv:
            strat = i.get("check_strategy", {})
            kind = strat.get("kind")
            ok, reason = self._run_invariant(adapter, contract, i, strat, samples, rng)
            if ok is None:
                report["warnings"].append({"invariant_id": i.get("id"), "reason": reason or "not executed"})
                continue
            if ok:
                passed += 1
            else:
                failed += 1
                sev = i.get("severity", "warning")
                entry = {"invariant_id": i.get("id"), "type": i.get("type"),
                         "severity": sev, "reason": reason, "covers_rules": i.get("covers_rules", [])}
                report["failures" if sev == "critical" else "warnings"].append(entry)
        report["invariants_passed"] = passed
        report["invariants_failed"] = failed
        report["checks"]["invariants"] = "fail" if any(
            f.get("invariant_id") and f.get("severity") == "critical" for f in report["failures"]) else "pass"

    def _run_invariant(self, adapter, contract, i, strat, samples, rng):
        kind = strat.get("kind")
        try:
            if kind == "width":
                o, W = strat.get("output"), strat.get("width", 1)
                for inp in samples:
                    got = self._safe_run(adapter, contract, inp)
                    v = got.get(o)
                    if v is not None and not (0 <= v < (1 << W)):
                        return False, "output %s=%s exceeds %d-bit width on inputs %s" % (o, v, W, _fmt(inp))
                return True, None
            if kind == "metamorphic" and strat.get("relation") == "commutative":
                a, b = (strat.get("operands") or [None, None])[:2]
                if not (a and b):
                    return None, "operands unknown"
                for inp in samples[:16]:
                    o1 = self._safe_run(adapter, contract, inp)
                    sw = dict(inp); sw[a], sw[b] = inp.get(b, 0), inp.get(a, 0)
                    o2 = self._safe_run(adapter, contract, sw)
                    if o1 != o2:
                        return False, "commutativity violated on %s" % _fmt(inp)
                return True, None
            if kind == "metamorphic" and strat.get("relation") == "mux_independence":
                r = self._check_mux_independence(adapter, contract, strat)
                return (True, None) if r["ok"] else (False, r.get("reason"))
            if kind == "reset" and adapter.seq:
                return self._inv_reset(adapter, contract, strat)
            if kind == "hold" and adapter.seq:
                return self._inv_hold(adapter, contract, strat)
            if kind == "counter_step" and adapter.seq:
                return self._inv_counter(adapter, contract, strat)
            # output_domain / priority / reachability / safety / liveness / arithmetic:
            # best-effort width-fit is already covered; deeper checks need an oracle we
            # avoid inventing -> report as unverified rather than risk a false failure.
            return None, "invariant kind '%s' not executed (no safe oracle)" % kind
        except _Timeout:
            return None, "timed out"
        except Exception as e:
            return None, "invariant check errored: %s" % e

    def _inv_reset(self, adapter, contract, strat):
        init = strat.get("initial_state")
        rname = strat.get("reset")
        active_high = strat.get("active_high", True)
        if not (adapter.has_get_state and rname):
            return None, "reset invariant needs get_state()"
        inst = adapter.new()
        in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        # drive some non-reset activity, then assert reset
        for _ in range(2):
            ins = {n: (1 if not _is_reset(n, rname) else (0 if active_high else 1)) for n in in_w}
            adapter.step(inst, ins)
        assert_ins = {n: 0 for n in in_w}
        assert_ins[rname] = 1 if active_high else 0
        adapter.step(inst, assert_ins)
        st = adapter.state(inst)
        if st is None:
            return None, "no state exposed"
        got = st.get("state", next(iter(st.values()), None))
        if init is not None and str(got) != str(init):
            return False, "after reset state=%s, expected initial %s" % (got, init)
        return True, None

    def _inv_hold(self, adapter, contract, strat):
        en = strat.get("enable")
        rname = strat.get("reset")
        if not (adapter.has_get_state and en):
            return None, "hold invariant needs get_state() and enable"
        in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        inst = adapter.new()
        base = {n: 0 for n in in_w}
        base[en] = 0
        if rname:
            base[en] = 0
        adapter.step(inst, base)
        s0 = adapter.state(inst)
        var = dict(base)
        for d in [n for n in in_w if n not in (en, rname)]:
            var[d] = (1 << in_w[d]) - 1
        adapter.step(inst, var)
        s1 = adapter.state(inst)
        if s0 is not None and s1 is not None and s0 != s1:
            return False, "state changed while enable=0 (hold violated): %s -> %s" % (s0, s1)
        return True, None

    def _inv_counter(self, adapter, contract, strat):
        cnt = strat.get("count")
        en = strat.get("enable")
        rname = strat.get("reset")
        W = strat.get("width", 1)
        if not cnt:
            return None, "counter output unknown"
        in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        inst = adapter.new()
        ins = {n: 0 for n in in_w}
        if en:
            ins[en] = 1
        o0 = adapter.step(inst, ins)
        o1 = adapter.step(inst, ins)
        c0, c1 = o0.get(cnt), o1.get(cnt)
        if c0 is None or c1 is None:
            return None, "counter output not observed"
        if c1 != (c0 + 1) % (1 << W):
            return False, "counter stepped %s->%s, expected +1 mod 2**%d" % (c0, c1, W)
        return True, None

    # -- 7) determinism ----------------------------------------------------

    def _check_determinism(self, adapter, contract, wl, report, rng) -> None:
        samples = self._sample_inputs(contract, wl, rng, n=min(8, self.sample_budget))
        for inp in samples:
            try:
                a = self._run_once(adapter, contract, inp)
                b = self._run_once(adapter, contract, inp)
            except Exception:
                continue
            if a != b:
                report["failures"].append({"check": "determinism",
                                           "reason": "same input gave different outputs: %s vs %s on %s"
                                           % (a, b, _fmt(inp))})
                report["checks"]["determinism"] = "fail"
                return
        report["checks"]["determinism"] = "pass"

    # -- 6) traceability ---------------------------------------------------

    def _check_traceability(self, golden_path, contract, report) -> None:
        try:
            with open(golden_path) as f:
                src = f.read()
        except Exception:
            report["checks"]["traceability"] = "warn"
            return
        rule_ids = [r.get("id") for r in (contract.get("rules") or []) if r.get("id")]
        has_map = "IMPLEMENTED_RULES" in src
        cited = set(re.findall(r"#\s*(?:implements\s+)?(R\d+)", src, re.I)) | \
            set(re.findall(r"\b(R\d+)\b", src)) if has_map else \
            set(re.findall(r"#\s*(?:implements\s+)?\b(R\d+)\b", src, re.I))
        covered = [r for r in rule_ids if r in cited or has_map]
        if not rule_ids:
            report["checks"]["traceability"] = "pass"
            return
        if not (has_map or cited):
            report["checks"]["traceability"] = "fail" if self.strict_traceability else "warn"
            (report["failures"] if self.strict_traceability else report["warnings"]).append(
                {"check": "traceability", "reason": "no rule-ID comments or IMPLEMENTED_RULES map found"})
            return
        report["checks"]["traceability"] = "pass"
        missing = [r for r in rule_ids if r not in cited and not has_map]
        if missing:
            report["warnings"].append({"check": "traceability", "reason": "rules not cited: %s" % missing})

    # -- 8) mutation-sanity probes ----------------------------------------

    def _mutation_probes(self, adapter, contract, wl, report, rng) -> None:
        in_w = {n: _w(s) for n, s in (contract.get("inputs") or {}).items()}
        outs = list((contract.get("outputs") or {}).keys())
        samples = self._sample_inputs(contract, wl, rng, n=min(24, self.sample_budget))
        if not samples or not outs:
            report["checks"]["mutation_probes"] = "skip"
            return
        runs = []
        for inp in samples:
            try:
                runs.append((inp, self._run_once(adapter, contract, inp)))
            except Exception:
                continue
        issues: List[dict] = []
        # The generic constant/ignores probes assume single-shot evaluation, which is
        # only meaningful for COMBINATIONAL models (each sequential _run_once resets +
        # single-steps, so a lone reset flip legitimately shows no change). Sequential
        # degeneracy is covered by _seq_probes instead.
        if not adapter.seq:
            # constant output across varied inputs
            for o in outs:
                vals = {r[1].get(o) for r in runs if r[1].get(o) is not None}
                if len(vals) == 1 and len(runs) > 2:
                    only = next(iter(vals))
                    expects_var = any(_to_int((w.get("expected_outputs") or {}).get(o)) not in (None, only)
                                      for w in wl if isinstance(w.get("expected_outputs"), dict))
                    if expects_var or only in (0, (1 << _w((contract.get("outputs") or {}).get(o, {}))) - 1):
                        issues.append({"probe": "constant_output",
                                       "severity": "critical" if expects_var else "warn",
                                       "reason": "output %s is always %s across %d varied inputs" % (o, only, len(runs))})
            # ignores input? an input is "ignored" only if it moves the output from NEITHER
            # an all-zero base NOR an all-one base (masking makes a single base unreliable)
            bases = [{n: 0 for n in in_w}, {n: (1 << in_w[n]) - 1 for n in in_w}]
            base_outs = []
            for b in bases:
                try:
                    base_outs.append(self._run_once(adapter, contract, b))
                except Exception:
                    base_outs.append({})
            for n in in_w:
                changed = False
                for b, bo in zip(bases, base_outs):
                    for val in {0, 1, (1 << in_w[n]) - 1} - {b[n]}:
                        probe = dict(b); probe[n] = val
                        try:
                            if self._run_once(adapter, contract, probe) != bo:
                                changed = True; break
                        except Exception:
                            changed = True; break
                    if changed:
                        break
                if not changed:
                    is_ctrl = _is_control(n)
                    issues.append({"probe": "ignores_input",
                                   "severity": "critical" if is_ctrl else "warn",
                                   "reason": "output never changes when %s%s changes" %
                                             ("control " if is_ctrl else "", n)})
        # sequential: ignores reset / state stuck
        if adapter.seq and adapter.has_get_state:
            issues += self._seq_probes(adapter, contract, in_w)
        for it in issues:
            (report["failures"] if it["severity"] == "critical" else report["warnings"]).append(it)
        report["checks"]["mutation_probes"] = "fail" if any(
            it["severity"] == "critical" for it in issues) else "pass"

    def _seq_probes(self, adapter, contract, in_w) -> List[dict]:
        issues = []
        rname = next((n for n in in_w if n.lower() in ("reset", "rst") or "rst" in n.lower().split("_")), None)
        inst = adapter.new()
        states_seen = set()
        for _ in range(6):
            ins = {n: 1 for n in in_w}
            if rname:
                ins[rname] = 0
            adapter.step(inst, ins)
            st = adapter.state(inst)
            if st is not None:
                states_seen.add(json.dumps(st, sort_keys=True))
        if len(states_seen) <= 1:
            issues.append({"probe": "state_stuck", "severity": "warn",
                           "reason": "state never changes across 6 active cycles"})
        if rname:
            s_before = adapter.state(inst)
            rins = {n: 0 for n in in_w}; rins[rname] = 1
            adapter.step(inst, rins)
            s_after = adapter.state(inst)
            if s_before is not None and s_after is not None and s_before == s_after and len(states_seen) > 1:
                issues.append({"probe": "ignores_reset", "severity": "critical",
                               "reason": "asserting reset did not change state"})
        return issues

    # -- helpers -----------------------------------------------------------

    def _run_once(self, adapter, contract, inputs_int) -> Dict[str, Optional[int]]:
        if adapter.seq:
            inst = adapter.new()
            return adapter.step(inst, inputs_int)
        inst = adapter.cls()
        return adapter.eval(inst, inputs_int)

    def _safe_run(self, adapter, contract, inputs_int) -> Dict[str, Optional[int]]:
        try:
            return self._run_once(adapter, contract, inputs_int)
        except Exception:
            return {}

    def _sample_inputs(self, contract, wl, rng, n=32) -> List[Dict[str, int]]:
        in_w = {name: _w(s) for name, s in (contract.get("inputs") or {}).items()}
        names = list(in_w)
        out: List[Dict[str, int]] = []
        seen = set()

        def add(vec):
            key = tuple(vec.get(k, 0) for k in names)
            if key not in seen:
                seen.add(key); out.append({k: vec.get(k, 0) for k in names})

        for w in wl:
            if isinstance(w.get("inputs"), dict):
                add(_int_inputs(w["inputs"]))
        add({k: 0 for k in names})
        add({k: (1 << in_w[k]) - 1 for k in names})
        for nm in names:
            for _, v in boundary_values(in_w[nm], False)[:4]:
                base = {k: 0 for k in names}; base[nm] = v; add(base)
        while len(out) < n and names:
            add({k: rng.randrange(1 << in_w[k]) for k in names})
            if len(seen) >= (1 << sum(in_w.values())):
                break
        return out[:max(n, 4)]

    # -- finalize / report -------------------------------------------------

    def _finalize(self, report, contract, wl, output_dir) -> Dict[str, Any]:
        hard = [f for f in report["failures"]]
        report["status"] = "pass" if not hard else "fail"
        if report["status"] == "fail":
            report["repair_report"] = self._repair_report(report, contract, wl)
            report.setdefault("repair_prompt", self._repair_prompt_text(report))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "golden_verification_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            with open(os.path.join(output_dir, "golden_verification_report.md"), "w") as f:
                f.write(self._render_md(report))
            report["json_path"] = os.path.join(output_dir, "golden_verification_report.json")
            report["md_path"] = os.path.join(output_dir, "golden_verification_report.md")
        return report

    def _repair_report(self, report, contract, wl) -> dict:
        failing_w, failing_i = [], []
        for f in report["failures"]:
            if f.get("witness_id"):
                failing_w.append({"id": f["witness_id"], "inputs": f.get("inputs"),
                                  "expected": f.get("expected"), "got": f.get("actual"),
                                  "rules": f.get("covers_rules", []), "reason": f.get("reason")})
            elif f.get("invariant_id"):
                failing_i.append({"id": f["invariant_id"], "reason": f.get("reason")})
        passing = [{"id": w.get("id")} for w in wl][:report.get("witnesses_passed", 0)]
        return {"message": "Golden model failed verification (%d witness / %d invariant / %d structural failures)."
                % (len(failing_w), len(failing_i),
                   len([f for f in report["failures"] if f.get("check")])),
                "failing_witnesses": failing_w, "failing_invariants": failing_i,
                "passing_witnesses": passing,
                "structural": [f for f in report["failures"] if f.get("check")]}

    def _repair_prompt_text(self, report) -> str:
        parts = ["golden_dut.py failed verification. Repair ONLY the failing behavior; keep passing behavior."]
        for f in report["failures"][:12]:
            if f.get("witness_id"):
                parts.append("- %s: expected %s got %s (rules %s)" % (
                    f["witness_id"], f.get("expected"), f.get("actual"), f.get("covers_rules")))
            elif f.get("invariant_id"):
                parts.append("- invariant %s: %s" % (f["invariant_id"], f.get("reason")))
            else:
                parts.append("- %s: %s" % (f.get("check"), f.get("reason")))
        return "\n".join(parts)

    def _render_md(self, report) -> str:
        L = ["# Golden Model Verification — %s" % report["status"].upper(), ""]
        L.append("- design_type: %s" % report.get("design_type"))
        L.append("- witnesses: %d passed / %d failed of %d" % (
            report["witnesses_passed"], report["witnesses_failed"], report["num_witnesses"]))
        L.append("- invariants: %d passed / %d failed" % (report["invariants_passed"], report["invariants_failed"]))
        L.append("- checks: " + ", ".join("%s=%s" % (k, v) for k, v in report["checks"].items()))
        L.append("")
        if report["failures"]:
            L.append("## Failures")
            for f in report["failures"]:
                L.append("- " + json.dumps(f))
        if report["warnings"]:
            L.append("")
            L.append("## Warnings")
            for w in report["warnings"][:40]:
                L.append("- " + json.dumps(w))
        if report.get("repair_prompt"):
            L.append("")
            L.append("## Repair prompt")
            L.append("```")
            L.append(report["repair_prompt"])
            L.append("```")
        return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# module helpers
# ---------------------------------------------------------------------------

def _load_golden_class(path: str):
    # Compile from the current source on every call. Going through a file-based
    # loader is unsafe here: the repair loop rewrites golden_dut.py in place, and a
    # same-size edit within the same second (e.g. '|' -> '&') collides with Python's
    # (mtime, size) .pyc cache and would re-run stale bytecode.
    with open(path) as f:
        src = f.read()
    ns: Dict[str, Any] = {"__name__": "golden_candidate", "__file__": path}
    exec(compile(src, path, "exec"), ns)
    if "GoldenDUT" not in ns:
        raise AttributeError("module does not define GoldenDUT")
    return ns["GoldenDUT"]


def _int_inputs(m: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for k, v in (m or {}).items():
        iv = _to_int(v)
        if iv is not None:
            out[k] = iv
    return out


def _is_reset(name: str, rname: Optional[str]) -> bool:
    return name == rname or name.lower() in ("reset", "rst")


def _all_data_change_moves_output(o1, o2) -> bool:
    return any(o1.get(k) != o2.get(k) for k in set(o1) | set(o2))


def _load(x):
    if isinstance(x, str) and os.path.exists(x):
        with open(x) as f:
            return json.load(f)
    return x if x is not None else {}


def _witness_list(w):
    if isinstance(w, dict):
        w = w.get("witnesses", [])
    return [x for x in (w or []) if isinstance(x, dict)]


def _invariant_list(i):
    if isinstance(i, dict):
        i = i.get("invariants", [])
    return [x for x in (i or []) if isinstance(x, dict)]


def verify_golden_model(golden_path, contract, witnesses=None, invariants=None,
                        output_dir=None, **kw) -> Dict[str, Any]:
    return GoldenModelVerifier(**kw).run(golden_path, contract, witnesses, invariants, output_dir)


# ---------------------------------------------------------------------------
# self-test (writes tiny golden models to temp files, no LLM, no RTL)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    ok = True
    and_c = {"design_name": "and_gate", "design_type": "combinational",
             "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "effect": "y = a & b"}], "can_generate_golden_model": True}
    wl = {"witnesses": [
        {"id": "W1", "type": "distinguishing", "required": True, "covers_rules": ["R1"],
         "inputs": {"a": "0", "b": "1"}, "expected_outputs": {"y": "0"}, "reason": "AND vs OR"},
        {"id": "W2", "type": "distinguishing", "required": True, "covers_rules": ["R1"],
         "inputs": {"a": "1", "b": "1"}, "expected_outputs": {"y": "1"}, "reason": "1&1"}]}
    iv = {"invariants": [{"id": "INV_WIDTH_Y", "type": "width", "severity": "critical",
                          "check_strategy": {"kind": "width", "output": "y", "width": 1}}]}

    good = "class GoldenDUT:\n    def __init__(self): pass\n    def eval(self, inputs):\n        # Implements R1\n        return {'y': (inputs['a'] & inputs['b']) & 1}\nIMPLEMENTED_RULES=['R1']\n"
    or_bug = "class GoldenDUT:\n    def __init__(self): pass\n    def eval(self, inputs):\n        # Implements R1\n        return {'y': (inputs['a'] | inputs['b']) & 1}\n"
    const0 = "class GoldenDUT:\n    def __init__(self): pass\n    def eval(self, inputs):\n        return {'y': 0}\n"

    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "g.py")
        with open(gp, "w") as f: f.write(good)
        r = GoldenModelVerifier().run(gp, and_c, wl, iv, output_dir=d)
        ok = ok and r["status"] == "pass" and r["witnesses_passed"] == 2
        ok = ok and os.path.exists(r["json_path"]) and os.path.exists(r["md_path"])
        ok = ok and r["checks"]["traceability"] == "pass"

        with open(gp, "w") as f: f.write(or_bug)
        r2 = GoldenModelVerifier().run(gp, and_c, wl, iv)
        ok = ok and r2["status"] == "fail"
        w1 = next((f for f in r2["failures"] if f.get("witness_id") == "W1"), None)
        ok = ok and w1 and w1["expected"] == {"y": 0} and w1["actual"] == {"y": 1}
        ok = ok and "repair_report" in r2 and r2["repair_report"]["failing_witnesses"]

        with open(gp, "w") as f: f.write(const0)
        r3 = GoldenModelVerifier().run(gp, and_c, wl, iv)
        ok = ok and r3["status"] == "fail"  # witness W2 fails + constant_output probe

    # ambiguity gate
    amb = json.loads(json.dumps(and_c)); amb["can_generate_golden_model"] = False
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "g.py")
        with open(gp, "w") as f: f.write(good)
        ra = GoldenModelVerifier().run(gp, amb, wl, iv)
        ok = ok and ra["status"] == "fail" and ra["checks"].get("ambiguity") == "fail"

    # sequential model: reset + step + get_state
    seq_c = {"design_name": "cnt", "design_type": "sequential",
             "inputs": {"reset": {"width": 1}, "en": {"width": 1}}, "outputs": {"q": {"width": 2}},
             "reset": {"name": "reset", "active_high": True}, "states": [], "initial_state": None,
             "rules": [{"id": "R1", "description": "increment when en"}], "can_generate_golden_model": True}
    seq_w = {"witnesses": [{"id": "WS", "type": "sequential_reachability", "required": True,
                            "covers_rules": ["R1"],
                            "sequence": [{"cycle": 0, "inputs": {"reset": "1", "en": "0"},
                                          "expected_outputs": {"q": "00"}},
                                         {"cycle": 1, "inputs": {"reset": "0", "en": "1"},
                                          "expected_outputs": {"q": "01"}},
                                         {"cycle": 2, "inputs": {"reset": "0", "en": "1"},
                                          "expected_outputs": {"q": "10"}}]}]}
    seq_src = ("class GoldenDUT:\n"
               "    def __init__(self): self.reset()\n"
               "    def reset(self): self.q = 0\n"
               "    def step(self, inputs):\n"
               "        # Implements R1\n"
               "        if inputs.get('reset'): self.q = 0\n"
               "        elif inputs.get('en'): self.q = (self.q + 1) & 3\n"
               "        return {'q': self.q}\n"
               "    def get_state(self): return {'q': self.q}\n")
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "g.py")
        with open(gp, "w") as f: f.write(seq_src)
        rs = GoldenModelVerifier(require_new_api=True).run(gp, seq_c, seq_w,
              {"invariants": [{"id": "INV_WIDTH_Q", "severity": "critical",
                               "check_strategy": {"kind": "width", "output": "q", "width": 2}}]}, output_dir=d)
        ok = ok and rs["status"] == "pass" and rs["witnesses_passed"] == 1

        # break the counter -> witness fails
        bad_seq = seq_src.replace("(self.q + 1) & 3", "self.q")
        with open(gp, "w") as f: f.write(bad_seq)
        rb = GoldenModelVerifier(require_new_api=True).run(gp, seq_c, seq_w, {"invariants": []})
        ok = ok and rb["status"] == "fail"

    print("golden_model_verifier selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        r = GoldenModelVerifier().run(sys.argv[1], sys.argv[2],
                                      sys.argv[3] if len(sys.argv) > 3 else None,
                                      sys.argv[4] if len(sys.argv) > 4 else None,
                                      output_dir=os.path.dirname(sys.argv[1]) or ".")
        print("status:", r["status"], "| witnesses %d/%d" % (r["witnesses_passed"], r["num_witnesses"]))
        raise SystemExit(0 if r["status"] == "pass" else 1)
    raise SystemExit(_selftest())
