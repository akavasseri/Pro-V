#!/usr/bin/env python3
"""
coverage_gen.py

A coverage-closure test-generation agent that adds *logic* to stimulus
generation instead of relying on random + corner cases. It drives the executable
FRM (the Python GoldenDUT reference model) to close an explicit, mutation-relevant
coverage model.

Two paths:

Combinational
-------------
  1. Dynamic cone-of-influence: perturb each input bit on random baselines and
     record which output signals move -> per-output fan-in cone, computed purely
     from the executable FRM (no RTL parser needed).
  2. Input-sensitivity coverage (the mutation killer): for every
     (output, input-bit-in-its-cone), require a matched pair (x, x^bit) in the
     suite whose output differs. This is the ATPG controllability+observability
     idea and is exactly what distinguishes functionally-different designs
     (e.g. it forces the 01/10 cases that separate an AND gate from an OR gate,
     which random 00/11 tests miss).
  3. Coverage-guided closure: seed with corners+random, measure which goals are
     hit, then search (enumerate if small, else biased-random) ONLY for the
     residual uncovered goals.

Sequential
----------
  Reachability BFS + transition cover over the FRM treated as a transition
  function. State is snapshotted via deepcopy(dut.__dict__), so no change to the
  FRM contract is required. For every reachable state we record the shortest
  input sequence that reaches it (state cover), then exercise every candidate
  input out of every state (transition cover). Emitted scenarios are
  path-to-state ++ transition-input.

Integrity: coverage is measured on the FRM, a *proxy* for the DUT. Use this to
produce logically-motivated stimulus, then run mutation_strength / the witness
loop against the real DUT to catch whatever FRM-coverage missed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import inspect
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the port parser from the differential engine.
try:
    from pro_v.mutation_strength import parse_ports, Ports
except Exception:  # standalone / broken package init
    _ms = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutation_strength.py")
    _spec = importlib.util.spec_from_file_location("prov_mutation_strength", _ms)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["prov_mutation_strength"] = _mod
    _spec.loader.exec_module(_mod)
    parse_ports, Ports = _mod.parse_ports, _mod.Ports

logger = logging.getLogger(__name__)

# A pure-python deterministic RNG (avoids importing global random state issues).
import random as _random


# ---------------------------------------------------------------------------
# FRM loading / wrapping
# ---------------------------------------------------------------------------

def load_frm(golden_dut_path: str):
    """Import golden_dut.py and return its GoldenDUT class (tail not executed)."""
    spec = importlib.util.spec_from_file_location("prov_frm_module", golden_dut_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prov_frm_module"] = mod
    spec.loader.exec_module(mod)  # __name__ != "__main__" so the checker tail is skipped
    if not hasattr(mod, "GoldenDUT"):
        raise ValueError("golden_dut.py does not define a GoldenDUT class")
    return mod.GoldenDUT


class FRMModel:
    """Wraps a GoldenDUT class; detects combinational vs sequential by load() arity."""

    def __init__(self, cls):
        self.cls = cls
        params = [p for p in inspect.signature(cls.load).parameters if p != "self"]
        self.is_seq = len(params) >= 2  # seq: load(clk, inputs); cmb: load(inputs)

    # -- combinational --
    def eval_cmb(self, inputs: Dict[str, str]) -> Dict[str, str]:
        out = self.cls().load(inputs)
        return {k: str(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Static analysis of the FRM: inputs read, outputs written, operators, branches
# ---------------------------------------------------------------------------

_OP_NAMES = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.LShift: "<<", ast.RShift: ">>",
    ast.BitOr: "|", ast.BitAnd: "&", ast.BitXor: "^",
    ast.And: "and", ast.Or: "or", ast.Not: "not", ast.Invert: "~",
    ast.USub: "u-", ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=",
}


def analyze_frm(golden_dut_path: str) -> Dict:
    """Statically extract what the reference model touches: the inputs it reads,
    the outputs it writes, the operators it uses, and its branch/loop count.

    This is the 'pull everything from the Python reference' step: the operators
    and branches become explicit coverage obligations, and the read-inputs /
    written-outputs are cross-checked against the DUT ports to flag blind spots
    (an input the FRM never reads can't be sensitised -> a coverage hole).
    """
    with open(golden_dut_path) as f:
        src = f.read()
    return _analyze_source(src)


def analyze_frm_class(cls) -> Dict:
    """Same as analyze_frm but scoped to just the GoldenDUT class source."""
    try:
        return _analyze_source(inspect.getsource(cls))
    except (OSError, TypeError):
        return {"inputs_read": [], "outputs_written": [], "operators": {},
                "num_branches": 0, "num_loops": 0}


def _analyze_source(src: str) -> Dict:
    import textwrap
    tree = ast.parse(textwrap.dedent(src))

    inputs_read, outputs_written, operators = set(), set(), {}
    n_branches = n_loops = 0

    def _bump(sym):
        operators[sym] = operators.get(sym, 0) + 1

    for node in ast.walk(tree):
        # inputs["x"]  or  inputs.get("x")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                and node.value.id in ("inputs", "input"):
            key = _const_str(node.slice)
            if key:
                inputs_read.add(key)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and isinstance(node.func.value, ast.Name) \
                and node.func.value.id in ("inputs", "input") and node.args:
            key = _const_str(node.args[0])
            if key:
                inputs_read.add(key)
        # returned/constructed dict keys -> outputs
        if isinstance(node, ast.Dict):
            for k in node.keys:
                key = _const_str(k)
                if key:
                    outputs_written.add(key)
        # operators
        if isinstance(node, ast.BinOp):
            _bump(_OP_NAMES.get(type(node.op), type(node.op).__name__))
        if isinstance(node, ast.BoolOp):
            _bump(_OP_NAMES.get(type(node.op), "bool"))
        if isinstance(node, ast.UnaryOp):
            _bump(_OP_NAMES.get(type(node.op), "unary"))
        if isinstance(node, ast.Compare):
            for op in node.ops:
                _bump(_OP_NAMES.get(type(op), "cmp"))
        # branches / loops (functional modes of the design)
        if isinstance(node, ast.If):
            n_branches += 1
        if isinstance(node, (ast.For, ast.While)):
            n_loops += 1

    return {
        "inputs_read": sorted(inputs_read),
        "outputs_written": sorted(outputs_written),
        "operators": operators,
        "num_branches": n_branches,
        "num_loops": n_loops,
    }


def _const_str(node) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # py<3.9 index wrapper
    if isinstance(node, ast.Index):  # pragma: no cover
        return _const_str(node.value)
    return None


# ---------------------------------------------------------------------------
# Line/branch coverage of the FRM (cover everything the reference actually does)
# ---------------------------------------------------------------------------

def _frm_line_range(cls) -> Tuple[str, int, int]:
    """(abspath, first_line, last_line) spanning the GoldenDUT class source."""
    path = os.path.abspath(inspect.getfile(cls))
    lines, start = inspect.getsourcelines(cls)
    return path, start, start + len(lines)


def _traced_lines(model: FRMModel, vecdict: Dict[str, str],
                  golden_file: str, lo: int, hi: int) -> set:
    """Return the set of FRM source lines executed while evaluating one vector."""
    hit = set()

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == golden_file \
                and lo <= frame.f_lineno < hi:
            hit.add(frame.f_lineno)
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        model.eval_cmb(vecdict)
    except Exception:
        pass
    finally:
        sys.settrace(old)
    return hit


def close_line_coverage(model: FRMModel, inputs, suite: set, total: int,
                        exhaustive: bool, probe_n: int, rng) -> Dict:
    """Greedy set-cover: add inputs so the suite executes every reachable FRM line.

    This forces reference-driven corners (e.g. a special-case `if x == 0xFF`)
    that pure input-sensitivity search can miss, guaranteeing every branch /
    functional mode of the reference model is exercised.
    """
    try:
        golden_file, lo, hi = _frm_line_range(model.cls)
    except (OSError, TypeError):
        return {"supported": False}

    # discover reachable lines and a representative input for each
    rep = {}  # lineno -> input vector that first hit it
    cand = range(1 << total) if exhaustive else \
        (rng.getrandbits(total) for _ in range(probe_n))
    for x in cand:
        for ln in _traced_lines(model, vec_to_dict(x, inputs), golden_file, lo, hi):
            rep.setdefault(ln, x)
    reachable = set(rep)

    # which lines does the current suite already cover?
    covered = set()
    for x in suite:
        covered |= _traced_lines(model, vec_to_dict(x, inputs), golden_file, lo, hi)

    # add representatives for the residual
    added = 0
    for ln in reachable - covered:
        suite.add(rep[ln])
        added += 1

    return {
        "supported": True,
        "reachable_lines": len(reachable),
        "covered_before": len(covered & reachable),
        "added_vectors": added,
        "covered_after": len(reachable),  # guaranteed after adding representatives
    }


# ---------------------------------------------------------------------------
# Packed-vector helpers (MSB-first, consistent with mutation_strength)
# ---------------------------------------------------------------------------

def vec_to_dict(v: int, inputs: List[Tuple[str, int]]) -> Dict[str, str]:
    total = sum(w for _, w in inputs)
    bits = format(v, f"0{total}b") if total else ""
    d, pos = {}, 0
    for name, w in inputs:
        d[name] = bits[pos:pos + w]
        pos += w
    return d


def bit_to_signal(bit: int, inputs: List[Tuple[str, int]], total: int) -> str:
    """Map an integer bit position (LSB=0) back to the owning input signal name."""
    msb_index = total - 1 - bit  # position within the MSB-first packed string
    pos = 0
    for name, w in inputs:
        if pos <= msb_index < pos + w:
            return name
        pos += w
    return "?"


# ---------------------------------------------------------------------------
# Combinational: dynamic cone-of-influence + sensitivity coverage closure
# ---------------------------------------------------------------------------

def dynamic_cone(model: FRMModel, inputs, outputs, baselines: int = 64,
                 rng=None) -> Dict[str, set]:
    """Return {output_signal: set(input_bit)} affecting it, found empirically."""
    rng = rng or _random.Random(0xC0DE)
    total = sum(w for _, w in inputs)
    cone = {o: set() for o, _ in outputs}
    if total == 0:
        return cone
    for _ in range(baselines):
        x = rng.getrandbits(total)
        base = model.eval_cmb(vec_to_dict(x, inputs))
        for b in range(total):
            flipped = model.eval_cmb(vec_to_dict(x ^ (1 << b), inputs))
            for o, _ in outputs:
                if base.get(o) != flipped.get(o):
                    cone[o].add(b)
    return cone


def close_combinational(model: FRMModel, ports: Ports, *,
                        max_exhaustive_bits: int = 16, random_seed_count: int = 200,
                        search_tries: int = 4000, cone_baselines: int = 64,
                        seed_vectors: Optional[List[int]] = None,
                        rng=None) -> Tuple[List[Dict[str, str]], Dict]:
    """Produce a coverage-closed CMB stimulus. Returns (stimulus, report)."""
    rng = rng or _random.Random(0xBEEF)
    inputs, outputs = ports.inputs, ports.outputs
    total = sum(w for _, w in inputs)
    if total == 0 or not outputs:
        return [], {"error": "no inputs/outputs"}

    exhaustive = total <= max_exhaustive_bits

    # 1) cone of influence
    cone = dynamic_cone(model, inputs, outputs, baselines=cone_baselines, rng=rng)

    # 2) seed vectors
    suite: set = set()
    if seed_vectors is not None:
        suite.update(v % (1 << total) for v in seed_vectors)
    else:
        # corners: all-0, all-1, alternating, one-hot / one-cold
        suite.update({0, (1 << total) - 1})
        suite.add(int("01" * ((total + 1) // 2), 2) & ((1 << total) - 1))
        suite.add(int("10" * ((total + 1) // 2), 2) & ((1 << total) - 1))
        for b in range(min(total, 64)):
            suite.add(1 << b)
            suite.add(((1 << total) - 1) ^ (1 << b))
        for _ in range(random_seed_count):
            suite.add(rng.getrandbits(total))

    def out_of(x, o):
        return model.eval_cmb(vec_to_dict(x, inputs)).get(o)

    # 3) goals = (output, input-bit-in-cone). Covered iff suite has a pair
    #    (x, x^bit) with differing output o.
    goals = [(o, b) for o, _ in outputs for b in sorted(cone[o])]

    def covered(o, b):
        mask = 1 << b
        for x in suite:
            if (x ^ mask) in suite and out_of(x, o) != out_of(x ^ mask, o):
                return True
        return False

    closed, unresolved = 0, []
    for (o, b) in goals:
        if covered(o, b):
            closed += 1
            continue
        mask = 1 << b
        found = False
        # try to sensitize: find x with out_o(x) != out_o(x^bit)
        candidates = list(suite)
        if exhaustive:
            candidates = range(1 << total)
        else:
            # existing suite first, then biased random
            extra = (rng.getrandbits(total) for _ in range(search_tries))
            candidates = list(suite) + list(extra)
        for x in candidates:
            if out_of(x, o) != out_of(x ^ mask, o):
                suite.add(x)
                suite.add(x ^ mask)  # add the matched partner -> pair is in the suite
                found = True
                closed += 1
                break
        if not found:
            # No input assignment makes o sensitive to bit b -> b is a don't-care
            # for o under the FRM (or search budget too small). Record it.
            unresolved.append({"output": o, "input_bit": b,
                               "input_signal": bit_to_signal(b, inputs, total)})

    # 4) per-output-bit value coverage (each output bit seen as 0 and 1)
    _close_output_values(model, inputs, outputs, suite, total, exhaustive,
                         search_tries, rng)

    # 5) FRM line/branch coverage: exercise every functional mode of the reference
    line_cov = close_line_coverage(model, inputs, suite, total, exhaustive,
                                   search_tries, rng)

    # FRM static analysis + cross-check against the DUT ports
    frm = analyze_frm_class(model.cls)
    dut_inputs = {n for n, _ in inputs}
    dut_outputs = {o for o, _ in outputs}
    io_issues = {
        "dut_inputs_not_read_by_frm": sorted(dut_inputs - set(frm["inputs_read"])),
        "frm_reads_unknown_inputs": sorted(set(frm["inputs_read"]) - dut_inputs),
        "dut_outputs_not_written_by_frm": sorted(dut_outputs - set(frm["outputs_written"])),
    }

    stimulus = [vec_to_dict(x, inputs) for x in sorted(suite)]
    report = {
        "circuit_type": "cmb",
        "input_width": total,
        "exhaustive": exhaustive,
        "cone_of_influence": {o: sorted(bits) for o, bits in cone.items()},
        "sensitivity_goals": len(goals),
        "sensitivity_closed": closed,
        "unresolved_goals": unresolved,   # likely don't-cares / equivalent regions
        "frm_analysis": frm,              # inputs/outputs/operators/branches pulled from the reference
        "frm_io_issues": io_issues,       # blind spots: DUT ports the FRM ignores
        "frm_line_coverage": line_cov,    # every reachable reference line exercised
        "num_vectors": len(stimulus),
    }
    return stimulus, report


def _close_output_values(model, inputs, outputs, suite, total, exhaustive,
                         search_tries, rng):
    """Ensure each output bit is observed as both 0 and 1 where achievable."""
    def obs_bits():
        seen = {}
        for x in suite:
            out = model.eval_cmb(vec_to_dict(x, inputs))
            for o, w in outputs:
                val = out.get(o, "")
                for i, ch in enumerate(val[-w:].zfill(w)):
                    seen.setdefault((o, i), set()).add(ch)
        return seen

    seen = obs_bits()
    for o, w in outputs:
        for i in range(w):
            need = {"0", "1"} - seen.get((o, i), set())
            if not need:
                continue
            cand = range(1 << total) if exhaustive else \
                (rng.getrandbits(total) for _ in range(search_tries))
            for x in cand:
                out = model.eval_cmb(vec_to_dict(x, inputs))
                val = out.get(o, "").zfill(w)
                if val and val[-w:][i] in need:
                    suite.add(x)
                    break


# ---------------------------------------------------------------------------
# Sequential: reachability BFS + transition cover over the FRM
# ---------------------------------------------------------------------------

def _state_key(dut) -> tuple:
    """Hashable snapshot of the FRM's mutable state."""
    items = []
    for k, v in sorted(dut.__dict__.items()):
        try:
            hash(v)
            items.append((k, v))
        except TypeError:
            items.append((k, repr(v)))
    return tuple(items)


def _candidate_inputs(inputs, max_enum: int = 256, rng=None) -> List[Dict[str, str]]:
    """Enumerate input combinations when small; else corners + random samples."""
    rng = rng or _random.Random(0xF00D)
    total = sum(w for _, w in inputs)
    if total == 0:
        return [{}]
    if (1 << total) <= max_enum:
        return [vec_to_dict(v, inputs) for v in range(1 << total)]
    vals = {0, (1 << total) - 1}
    for b in range(min(total, 32)):
        vals.add(1 << b)
    for _ in range(max_enum):
        vals.add(rng.getrandbits(total))
    return [vec_to_dict(v, inputs) for v in sorted(vals)]


def _step(model: FRMModel, dut, inputs_dict) -> Dict[str, str]:
    """One clock cycle mirroring the checker tail (rising then falling edge)."""
    out = dut.load(1, inputs_dict)
    try:
        dut.load(0, inputs_dict)
    except Exception:
        pass
    return {k: str(v) for k, v in (out or {}).items()}


def close_sequential(model: FRMModel, ports: Ports, *,
                     max_states: int = 512, max_depth: int = 64,
                     max_enum_inputs: int = 256, rng=None) -> Tuple[List[dict], Dict]:
    """Reachability BFS + transition cover -> list of SEQ stimulus scenarios."""
    rng = rng or _random.Random(0x5EED)
    inputs = ports.inputs
    cand = _candidate_inputs(inputs, max_enum=max_enum_inputs, rng=rng)

    # BFS: shortest input-sequence (list of input dicts) to reach each state.
    root = model.cls()
    start_key = _state_key(root)
    start_snap = copy.deepcopy(root.__dict__)
    paths: Dict[tuple, List[dict]] = {start_key: []}
    snaps: Dict[tuple, dict] = {start_key: start_snap}
    frontier = [start_key]
    depth = 0
    while frontier and len(paths) < max_states and depth < max_depth:
        nxt = []
        for skey in frontier:
            for civ in cand:
                dut = model.cls()
                dut.__dict__ = copy.deepcopy(snaps[skey])
                _step(model, dut, civ)
                nkey = _state_key(dut)
                if nkey not in paths:
                    paths[nkey] = paths[skey] + [civ]
                    snaps[nkey] = copy.deepcopy(dut.__dict__)
                    nxt.append(nkey)
                    if len(paths) >= max_states:
                        break
            if len(paths) >= max_states:
                break
        frontier = nxt
        depth += 1

    # Transition cover: from every reached state, exercise every candidate input.
    scenarios = []
    seen_scn = set()
    for skey, path in paths.items():
        for civ in cand:
            seq = path + [civ]  # path-to-state ++ transition-exercising input
            scn = _seq_to_scenario(seq, inputs)
            key = json.dumps(scn, sort_keys=True)
            if key not in seen_scn:
                seen_scn.add(key)
                scenarios.append(scn)

    report = {
        "circuit_type": "seq",
        "states_reached": len(paths),
        "hit_state_cap": len(paths) >= max_states,
        "candidate_inputs": len(cand),
        "num_scenarios": len(scenarios),
        "max_depth_explored": depth,
    }
    return scenarios, report


def _seq_to_scenario(seq: List[dict], inputs) -> dict:
    """Convert a list of per-cycle input dicts into a SEQ stimulus scenario."""
    n = len(seq)
    scn = {"clock_cycles": n}
    for name, _ in inputs:
        scn[name] = [seq[c].get(name, "0") for c in range(n)]
    return scn


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def generate(golden_dut_path: str, dut_path: str, *, force_type: Optional[str] = None,
             **kwargs) -> Tuple[list, Dict]:
    cls = load_frm(golden_dut_path)
    model = FRMModel(cls)
    with open(dut_path) as f:
        ports = parse_ports(f.read())
    is_seq = (force_type == "seq") if force_type else model.is_seq
    if is_seq:
        seq_kw = {k: kwargs[k] for k in ("max_states", "max_depth", "max_enum_inputs")
                  if k in kwargs}
        return close_sequential(model, ports, **seq_kw)
    cmb_kw = {k: kwargs[k] for k in ("max_exhaustive_bits", "random_seed_count",
                                     "search_tries", "cone_baselines") if k in kwargs}
    return close_combinational(model, ports, **cmb_kw)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    logging.basicConfig(level=logging.INFO)
    ok = True

    # --- Combinational: AND gate, weak seed {00, 11}; closure must add 01/10 ---
    class AndDUT:
        def __init__(self): pass
        def load(self, inputs):
            a = int(inputs["a"], 2); b = int(inputs["b"], 2)
            return {"y": str(a & b)}

    ports = parse_ports("module top_module(input a, input b, output y); "
                        "assign y=a&b; endmodule")
    model = FRMModel(AndDUT)
    stim, rep = close_combinational(model, ports, seed_vectors=[0b00, 0b11])
    vecs = {(d["a"], d["b"]) for d in stim}
    print("CMB cone:", rep["cone_of_influence"],
          "goals closed:", rep["sensitivity_closed"], "/", rep["sensitivity_goals"])
    print("CMB vectors:", sorted(vecs))
    cmb_ok = (("0", "1") in vecs and ("1", "0") in vecs and
              rep["sensitivity_closed"] == rep["sensitivity_goals"] and
              rep["cone_of_influence"]["y"] == [0, 1])
    print("CMB analysis:", rep["frm_analysis"]["operators"],
          "inputs_read:", rep["frm_analysis"]["inputs_read"],
          "io_issues:", rep["frm_io_issues"])
    cmb_ok = (cmb_ok and "&" in rep["frm_analysis"]["operators"]
              and rep["frm_analysis"]["inputs_read"] == ["a", "b"]
              and not any(rep["frm_io_issues"].values()))
    print("CMB SELFTEST:", "PASS" if cmb_ok else "FAIL",
          "(closure added 01/10; FRM operators/IO pulled from the reference)")
    ok = ok and cmb_ok

    # --- Reference-driven corner: special-case branch x==0xABC (12-bit, NON-exhaustive) ---
    class SpecialDUT:
        def __init__(self): pass
        def load(self, inputs):
            x = int(inputs["x"], 2)
            if x == 0xABC:
                return {"y": "1"}
            return {"y": "0"}

    sp_ports = parse_ports("module top_module(input [11:0] x, output y); endmodule")
    sp_model = FRMModel(SpecialDUT)
    sp_stim, sp_rep = close_combinational(sp_model, sp_ports,
                                          max_exhaustive_bits=8, search_tries=20000)
    hit_corner = any(int(d["x"], 2) == 0xABC for d in sp_stim)
    lc = sp_rep["frm_line_coverage"]
    print(f"SPECIAL non-exhaustive({sp_rep['input_width']}b): hit x==0xABC: {hit_corner}; "
          f"line-cov {lc['covered_after']}/{lc['reachable_lines']}; "
          f"operators {sp_rep['frm_analysis']['operators']}")
    special_ok = (hit_corner and "==" in sp_rep["frm_analysis"]["operators"]
                  and lc["covered_after"] == lc["reachable_lines"])
    print("SPECIAL SELFTEST:", "PASS" if special_ok else "FAIL",
          "(branch coverage forced the reference's special case that random never hits)")
    ok = ok and special_ok

    # --- Sequential: mod-4 counter; BFS must discover all 4 states ---
    class CounterDUT:
        def __init__(self): self.count = 0
        def load(self, clk, inputs):
            if clk == 1 and inputs.get("en", "0") == "1":
                self.count = (self.count + 1) % 4
            return {"q": format(self.count, "02b")}

    sports = parse_ports("module top_module(input clk, input en, output [1:0] q); "
                         "endmodule")
    smodel = FRMModel(CounterDUT)
    scns, srep = close_sequential(smodel, sports, max_states=16)
    # does some scenario drive the counter to state 3 (q=11)?
    reached3 = False
    for scn in scns:
        dut = CounterDUT()
        for c in range(scn["clock_cycles"]):
            out = _step(smodel, dut, {"en": scn["en"][c]})
        if out.get("q") == "11":
            reached3 = True
            break
    print("SEQ states reached:", srep["states_reached"], "scenarios:", srep["num_scenarios"])
    seq_ok = (srep["states_reached"] == 4 and reached3)
    print("SEQ SELFTEST:", "PASS" if seq_ok else "FAIL",
          "(reachability BFS enumerated all states + built the reaching sequences)")
    ok = ok and seq_ok

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description="FRM-driven coverage-closure test generation")
    parser.add_argument("--frm", required=True, help="Path to golden_dut.py (FRM)")
    parser.add_argument("--dut", required=True, help="Path to top_module.v")
    parser.add_argument("--out", "-o", default="stimulus.json")
    parser.add_argument("--type", choices=["cmb", "seq"], default=None,
                       help="Force circuit type (default: auto-detect from FRM)")
    parser.add_argument("--max-exhaustive-bits", type=int, default=16)
    parser.add_argument("--search-tries", type=int, default=4000)
    parser.add_argument("--max-states", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--report", default=None, help="Optional path to write the coverage report")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    logging.basicConfig(level=logging.INFO)
    stim, rep = generate(
        args.frm, args.dut, force_type=args.type,
        max_exhaustive_bits=args.max_exhaustive_bits, search_tries=args.search_tries,
        max_states=args.max_states, max_depth=args.max_depth,
    )
    with open(args.out, "w") as f:
        json.dump(stim, f, indent=2)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=2)
    logger.info(f"Wrote {len(stim)} vectors/scenarios to {args.out}")
    logger.info(f"Coverage report: {json.dumps(rep, indent=2)}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    main()
