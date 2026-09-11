#!/usr/bin/env python3
"""
functional_coverage_agent.py

Reference-Guided Functional Coverage Agent  --  a PRE-testbench planning stage.

Position in the Pro-V flow
--------------------------
    spec + golden_dut.py (+ optional interface metadata)
        -> functional_coverage_agent.py        (this module)
        -> coverage_plan.json                  (structured test intent)
        -> existing/new testbench generator
        -> RTL simulation against black-box top_module.v

The agent decides *what should be tested*. It does NOT produce HDL. It answers
one question:

    "What behavior does the golden reference model say must be exercised, and
     what inputs / input sequences cause that behavior?"

It never answers "what RTL logic appears inside top_module.v?". `top_module.v`
is the black-box DUT under verification; using its internal operators/branches
to decide functional tests would be circular and would leak the very thing we
are trying to check. The behavioral source of truth is `golden_dut.py` (the
Python reference model, a.k.a. the FRM), plus the spec / interface metadata if
supplied. `top_module.v` is read for ONE purpose only: confirming the module
interface (port names, widths, clk/reset names) and simulation wiring.

Why this beats random stimulus
------------------------------
For an AND gate and an OR gate, testing only 00 and 11 gives "50% input
coverage" and shows output 0 and output 1 -- yet it cannot tell AND from OR.
The distinguishing inputs are 01 and 10, because those are exactly where the
two operations disagree. This agent generalizes that idea: it drives the golden
model to pick *high-information* inputs -- ones that separate similar
operations, cover meaningful output classes, and (for sequential designs) reach
meaningful states and transitions -- and it minimizes to a small set of such
tests rather than a large pile of random ones.

Reuses the FRM loader / Verilog port parser already in the repo
(`coverage_gen.load_frm`, `mutation_strength.parse_ports`).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import itertools
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Robust reuse of existing primitives (works both as a package and standalone)
# ---------------------------------------------------------------------------

def _load_sibling(mod_name: str, file_name: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    from pro_v.mutation_strength import parse_ports, Ports
except Exception:
    _ms = _load_sibling("prov_mutation_strength", "mutation_strength.py")
    parse_ports, Ports = _ms.parse_ports, _ms.Ports

try:
    from pro_v.coverage_gen import load_frm, analyze_frm_class
except Exception:
    _cg = _load_sibling("prov_coverage_gen", "coverage_gen.py")
    load_frm, analyze_frm_class = _cg.load_frm, _cg.analyze_frm_class


# ---------------------------------------------------------------------------
# Budget / runtime control
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Runtime knobs.

    Coverage-completeness policy (combinational): the objective is that nothing
    distinguishing is left untested, NOT that the vector count is minimal.
      * total input bits <= exhaustive_max_bits (11)  -> EXHAUSTIVE (every 2**n;
        n<=11 => at most 2048 vectors, so 8 bits->256, 9->512, 10->1024, 11->2048)
      * total input bits >  exhaustive_max_bits        -> LARGE SAMPLE of
        sample_cap (2048) vectors = structured high-information seeds UNION a
        uniform-random fill. Count scales up with the space, hard-capped at 2048.
    """
    max_tests: int = 4096          # emission cap; must exceed sample_cap so the
                                   # full base set + distinguishing pairs are kept
    max_candidates: int = 5000
    max_sequence_depth: int = 20
    max_bfs_states: int = 256
    max_runtime_seconds: float = 120.0
    exhaustive_max_bits: int = 11  # 2**11 == 2048 == sample_cap
    sample_cap: int = 2048         # hard cap on vectors for large input spaces

    def as_dict(self) -> Dict:
        return {
            "max_tests": self.max_tests,
            "max_candidates": self.max_candidates,
            "max_sequence_depth": self.max_sequence_depth,
            "max_bfs_states": self.max_bfs_states,
            "max_runtime_seconds": self.max_runtime_seconds,
            "exhaustive_max_bits": self.exhaustive_max_bits,
            "sample_cap": self.sample_cap,
        }


class _Clock:
    """Wall-clock budget guard. Injected so tests are deterministic."""

    def __init__(self, limit: float):
        self.limit = limit
        self.start = time.time()

    def expired(self) -> bool:
        return (time.time() - self.start) >= self.limit


# ---------------------------------------------------------------------------
# Control-like inputs: fully enumerated when narrow (they select behavior)
# ---------------------------------------------------------------------------

_CONTROL_HINTS = {
    "sel", "select", "mode", "op", "opcode", "func", "funct", "fn", "cmd",
    "enable", "en", "valid", "ready", "start", "done", "we", "wen", "re", "ren",
    "reset", "rst", "clear", "clr", "load", "ld", "cin", "carry", "sign",
}


def _is_control(name: str, width: int) -> bool:
    low = name.lower()
    if width <= 4 and any(h == low or low.startswith(h) or low.endswith(h)
                          for h in _CONTROL_HINTS):
        return True
    # any very-narrow input behaves like a control/mode selector
    return width <= 2


# ---------------------------------------------------------------------------
# Signedness detection (interface confirmation only, from top_module.v)
# ---------------------------------------------------------------------------

def _detect_signed(verilog: str, meta: Optional[Dict]) -> Set[str]:
    signed: Set[str] = set()
    # per-declaration scan: an "input ... signed ... name" chunk. Reading the
    # DUT here is interface-confirmation only (signedness of a port), never
    # logic extraction.
    for chunk in re.split(r"[;,]", verilog):
        if "input" in chunk and "signed" in chunk:
            ids = re.findall(r"\b([a-zA-Z_]\w*)\b", chunk)
            # last identifier in an input declaration is the port name
            if ids:
                signed.add(ids[-1])
    if meta:
        for name, spec in (meta.get("inputs") or {}).items():
            if isinstance(spec, dict) and spec.get("signed"):
                signed.add(name)
    return signed


# ---------------------------------------------------------------------------
# Bit helpers -- the agent works on {name: int} vectors (readable, per-signal)
# ---------------------------------------------------------------------------

def int_to_bin(v: int, w: int) -> str:
    return format(v & ((1 << w) - 1), f"0{w}b")


def bin_to_int(s: str) -> int:
    s = (s or "").strip()
    return int(s, 2) if s else 0


def maybe_bin_to_int(v) -> Optional[int]:
    """Return None for public don't-care outputs instead of inventing a value."""
    s = str(v or "").strip().lower()
    if not s or any(ch in s for ch in ("x", "z", "d", "?")):
        return None
    return int(s, 2)


# ---------------------------------------------------------------------------
# Golden model adapter -- standardizes how we call golden_dut.py
# ---------------------------------------------------------------------------

class GoldenModelAdapter:
    """A uniform view over a GoldenDUT reference model.

    Exposes, regardless of the exact GoldenDUT API:
      * input_names / input_widths (domains) / output_names
      * is_sequential + reset behavior
      * run(input_vector)   for combinational models   (single evaluation)
      * step(dut, input_vector) for sequential models   (one clocked cycle)
      * reset() -> a fresh state handle for sequential models
      * snapshot()/restore() so the sequential search can explore state space

    Input/output vectors here are {name: int}. We translate to/from the binary
    strings the GoldenDUT.load contract uses at the boundary.
    """

    def __init__(self, cls, ports: Ports, meta: Optional[Dict] = None,
                 signed: Optional[Set[str]] = None):
        self.cls = cls
        self.ports = ports
        self.meta = meta or {}
        self.signed = signed or set()

        params = [p for p in inspect.signature(cls.load).parameters if p != "self"]
        # sequential contract: load(clk, inputs); combinational: load(inputs)
        self.is_sequential = len(params) >= 2

        self.input_names = [n for n, _ in ports.inputs]
        self.input_widths = {n: w for n, w in ports.inputs}
        self.output_names = [o for o, _ in ports.outputs]
        self.output_widths = {o: w for o, w in ports.outputs}
        self.clock_names = [n for n, _ in getattr(ports, "clock_inputs", [])] or (
            [ports.clk_name] if getattr(ports, "clk_name", None) else []
        )

    def clock_arg(self, level: int):
        if len(self.clock_names) > 1:
            return {name: int(level) for name in self.clock_names}
        return int(level)

    # -- domain queries -----------------------------------------------------
    def width(self, name: str) -> int:
        return self.input_widths.get(name, 1)

    def is_signed(self, name: str) -> bool:
        return name in self.signed

    def total_input_bits(self) -> int:
        return sum(self.input_widths.values())

    # -- combinational ------------------------------------------------------
    def run(self, vec: Dict[str, int]) -> Dict[str, int]:
        """One combinational evaluation. vec/{name:int} -> {name:int}."""
        bin_in = {n: int_to_bin(vec.get(n, 0), self.width(n)) for n in self.input_names}
        out = self.cls().load(bin_in) or {}
        converted = {}
        for k, v in out.items():
            iv = maybe_bin_to_int(v)
            if iv is not None:
                converted[k] = iv
        return converted

    # -- sequential ---------------------------------------------------------
    def reset(self):
        """Return a fresh DUT instance -- the reset/initial state."""
        return self.cls()

    def step(self, dut, vec: Dict[str, int]) -> Dict[str, int]:
        """Advance one clock cycle (rising then falling edge, mirroring the
        Pro-V SEQ harness). Returns the rising-edge outputs as {name:int}."""
        bin_in = {n: int_to_bin(vec.get(n, 0), self.width(n)) for n in self.input_names}
        rising = dut.load(self.clock_arg(1), bin_in) or {}
        try:
            dut.load(self.clock_arg(0), bin_in)
        except Exception:
            pass
        converted = {}
        for k, v in rising.items():
            iv = maybe_bin_to_int(v)
            if iv is not None:
                converted[k] = iv
        return converted

    @staticmethod
    def snapshot(dut) -> dict:
        return copy.deepcopy(dut.__dict__)

    def restore(self, snap: dict):
        dut = self.cls()
        dut.__dict__ = copy.deepcopy(snap)
        return dut

    @staticmethod
    def state_key(dut) -> tuple:
        """Hashable snapshot of the model's mutable state (for BFS)."""
        items = []
        for k, v in sorted(dut.__dict__.items()):
            try:
                hash(v)
                items.append((k, v))
            except TypeError:
                items.append((k, repr(v)))
        return tuple(items)


# ---------------------------------------------------------------------------
# Candidate input generation
# ---------------------------------------------------------------------------

def per_input_values(width: int, signed: bool) -> List[int]:
    """The high-information value set for a single input signal (spec section 2):
    0, 1, 2, max, max-1, all-ones, alternating patterns, one-hot, two-hot (small
    widths), and signed boundaries when signedness is known."""
    mask = (1 << width) - 1
    vals: Set[int] = {0, 1, mask}                      # 0, one, all-ones/max
    if width >= 2:
        vals.add(2)
        vals.add(mask - 1)                             # max-1
        vals.add(int("01" * ((width + 1) // 2), 2) & mask)   # alternating
        vals.add(int("10" * ((width + 1) // 2), 2) & mask)   # alternating
        vals.add(1 << (width - 1))                     # MSB one-hot
    for b in range(min(width, 8)):                     # one-hot walk (bounded)
        vals.add(1 << b)
    if width <= 4:                                     # two-hot for small widths
        for i in range(width):
            for j in range(i + 1, width):
                vals.add((1 << i) | (1 << j))
    if signed and width >= 1:
        vals.add(1 << (width - 1))                     # min signed (100..0)
        vals.add((1 << (width - 1)) - 1)               # max signed (011..1)
    return sorted(v & mask for v in vals)


def _all_control(names_widths: List[Tuple[str, int]]) -> List[Tuple[str, List[int]]]:
    """Per-input candidate value lists; control-like inputs are fully enumerated."""
    out = []
    for name, w in names_widths:
        if _is_control(name, w):
            out.append((name, list(range(1 << w))))       # full enumeration
        else:
            out.append((name, per_input_values(w, False)))
    return out


def _onehot_logic_candidates(adapter: GoldenModelAdapter) -> Optional[List[Dict[str, int]]]:
    """Compact but meaningful stimulus for combinational one-hot FSM logic.

    Exhaustively enumerating a 10-bit one-hot state input creates 1024 mostly
    invalid states. Verilog one-hot equations are ORs over state bits, so cover:
    zero-hot, every valid one-hot state under every control value, adjacent
    multi-hot pairs, and all-ones. This preserves the important invalid-state
    checks without exploding compile time.
    """
    names_widths = list(adapter.ports.inputs)
    if len(names_widths) < 2:
        return None
    state_item = next(
        ((n, w) for n, w in names_widths if n.lower() in {"state", "y"} and w >= 5),
        None,
    )
    if not state_item:
        return None
    state_name, state_w = state_item
    controls = [(n, w) for n, w in names_widths if n != state_name]
    if not controls or sum(w for _, w in controls) > 4:
        return None
    out_names = {n.lower() for n, _ in adapter.ports.outputs}
    if not (
        "next_state" in out_names
        or any(re.fullmatch(r"y\d+", n, re.I) for n in out_names)
        or any(re.fullmatch(r"out\d+", n, re.I) for n in out_names)
    ):
        return None

    control_values = []
    for combo in itertools.product(*[range(1 << w) for _, w in controls]):
        control_values.append({n: v for (n, _), v in zip(controls, combo)})

    states = [0]
    states.extend(1 << i for i in range(state_w))
    states.extend((1 << i) | (1 << ((i + 1) % state_w)) for i in range(state_w))
    states.append((1 << state_w) - 1)

    rows: List[Dict[str, int]] = []
    seen: Set[Tuple[int, ...]] = set()
    for st in states:
        for ctrl in control_values:
            vec = {n: 0 for n, _ in names_widths}
            vec[state_name] = st
            vec.update(ctrl)
            key = tuple(vec[n] for n, _ in names_widths)
            if key not in seen:
                seen.add(key)
                rows.append(vec)
    return rows


def generate_candidates(adapter: GoldenModelAdapter, budget: Budget,
                        clock: Optional[_Clock] = None) -> Tuple[List[Dict[str, int]], bool]:
    """Return (candidate vectors, exhaustive?).

    Exhaustive enumeration of the WHOLE input space only when it is small
    (2**total_bits <= max_candidates). Otherwise: the cartesian product of each
    signal's high-information value set (with control-like signals fully
    enumerated), plus equality cases between same-width inputs, capped at
    max_candidates.
    """
    clock = clock or _Clock(budget.max_runtime_seconds)
    names_widths = list(adapter.ports.inputs)
    total = adapter.total_input_bits()

    if total == 0:
        return [{}], True

    exh_bits = getattr(budget, "exhaustive_max_bits", 11)
    cap = getattr(budget, "sample_cap", 2048)

    onehot_rows = _onehot_logic_candidates(adapter)
    if onehot_rows:
        return onehot_rows[:cap], False

    # -- <= exh_bits: EXHAUSTIVE over every input combination (n<=11 => <=2048) --
    if total <= exh_bits:
        cands = []
        for v in range(1 << total):
            vec, pos = {}, 0
            bits = format(v, f"0{total}b")
            for name, w in names_widths:
                vec[name] = int(bits[pos:pos + w], 2)
                pos += w
            cands.append(vec)
        return cands, True

    # -- > exh_bits: LARGE SAMPLE of `cap` vectors -------------------------
    #   Completeness, not minimality: structured high-information seeds first
    #   (every control mode / boundary / equality case), then a uniform-random
    #   fill so the total reaches the cap (>=512 for n>=9, hard-capped at 2048).
    import random as _random
    rng = _random.Random(0xC0FFEE)
    per_input = _all_control(names_widths)
    signed_lookup = {n: adapter.is_signed(n) for n, _ in names_widths}
    per_input = [
        (n, vals if _is_control(n, dict(names_widths)[n])
         else per_input_values(dict(names_widths)[n], signed_lookup[n]))
        for (n, vals) in per_input
    ]

    cands: List[Dict[str, int]] = []
    seen: Set[Tuple[int, ...]] = set()

    def _emit(vec: Dict[str, int]) -> bool:
        key = tuple(vec[n] for n, _ in names_widths)
        if key in seen:
            return True
        seen.add(key)
        cands.append(dict(vec))
        return len(cands) < cap

    # (1) cartesian product of per-signal value sets. itertools.product varies
    # the LAST operand fastest, so control-like signals go last -> every
    # control/mode value is enumerated even when the cap truncates the product.
    per_input = sorted(per_input, key=lambda nv: _is_control(nv[0], dict(names_widths)[nv[0]]))
    value_lists = [vals for _, vals in per_input]
    names = [n for n, _ in per_input]
    for combo in itertools.product(*value_lists):
        if clock.expired() or len(cands) >= cap:
            break
        if not _emit(dict(zip(names, combo))):
            break

    # (2) equality cases between same-width inputs (separate < from <=, == / !=)
    by_width: Dict[int, List[str]] = {}
    for name, w in names_widths:
        by_width.setdefault(w, []).append(name)
    for w, group in by_width.items():
        if len(group) < 2 or len(cands) >= cap:
            continue
        for val in {0, 1, 2, (1 << w) - 1, (1 << (w - 1)) if w else 0}:
            if len(cands) >= cap:
                break
            vec = {n: (val & ((1 << dict(names_widths)[n]) - 1)) for n, _ in names_widths}
            _emit(vec)

    # (3) uniform-random fill up to the cap so nothing is left untested by luck
    guard = 0
    while len(cands) < cap and guard < cap * 40 and not clock.expired():
        guard += 1
        _emit({n: rng.randrange(1 << w) for n, w in names_widths})

    return cands, False


# ---------------------------------------------------------------------------
# Output-class grouping  --  "what input causes this meaningful output?"
# ---------------------------------------------------------------------------

def _output_classes(out_name: str, out_val: int, width: int,
                    in_vec: Dict[str, int], arithmetic: bool) -> List[str]:
    """Interesting behavioral classes for one output under one input vector."""
    classes = []
    maxv = (1 << width) - 1
    in_vals = list(in_vec.values())
    classes.append(f"out[{out_name}]=zero" if out_val == 0 else None)
    classes.append(f"out[{out_name}]=one" if out_val == 1 else None)
    if width > 1 and out_val == maxv:
        classes.append(f"out[{out_name}]=max")
    if width == 1:
        classes.append(f"out[{out_name}]=bool_true" if out_val else f"out[{out_name}]=bool_false")
    if in_vals and out_val in in_vals:
        classes.append(f"out[{out_name}]=eq_input")
    if in_vals and all(out_val != iv for iv in in_vals):
        classes.append(f"out[{out_name}]=neq_all_inputs")
    if arithmetic and in_vals and out_val < max(in_vals) and out_val != maxv:
        # arithmetic present and the result is smaller than an operand -> a
        # wraparound / overflow-like observation worth pinning down.
        classes.append(f"out[{out_name}]=overflow_like")
    return [c for c in classes if c]


# ---------------------------------------------------------------------------
# Greedy weighted set-cover  (shared by combinational + sequential planning)
# ---------------------------------------------------------------------------

@dataclass
class _Item:
    """A candidate unit of stimulus and the coverage goals it satisfies."""
    goals: Set[str]
    cost: int                 # #new vectors (cmb) or #cycles (seq)
    payload: object           # the vector(s) / sequence to materialize
    reason: str
    kind: str


def greedy_cover(items: List[_Item], universe: Set[str], *,
                 max_units: int, unit_of: Callable[[object], list]) -> List[_Item]:
    """Pick items to cover `universe`, maximizing new-goals-per-added-cost, so a
    smaller set of high-information tests wins over many redundant ones.

    `unit_of(payload)` returns the atomic units (vectors / cycles) an item
    introduces; selection stops once the accumulated unique units reach
    max_units. A test is redundant -- and dropped -- when it adds no new goal."""
    covered: Set[str] = set()
    chosen: List[_Item] = []
    used_units: Set = set()
    remaining = list(items)

    while remaining and covered != universe:
        best, best_score = None, 0.0
        for it in remaining:
            gain = len(it.goals - covered)
            if gain == 0:
                continue
            new_units = [u for u in unit_of(it.payload) if _hkey(u) not in used_units]
            denom = max(len(new_units), 1)
            score = gain / denom
            if score > best_score:
                best, best_score = it, score
        if best is None:
            break
        new_units = [u for u in unit_of(best.payload) if _hkey(u) not in used_units]
        if len(used_units) + len(new_units) > max_units:
            # would blow the test budget; drop it and try smaller items
            remaining.remove(best)
            continue
        for u in new_units:
            used_units.add(_hkey(u))
        covered |= best.goals
        chosen.append(best)
        remaining.remove(best)

    return chosen


def _hkey(u) -> str:
    return json.dumps(u, sort_keys=True) if isinstance(u, (dict, list)) else u


# ---------------------------------------------------------------------------
# Combinational planning
# ---------------------------------------------------------------------------

def plan_combinational(adapter: GoldenModelAdapter, budget: Budget,
                       clock: _Clock) -> Dict:
    names_widths = list(adapter.ports.inputs)
    outputs = list(adapter.ports.outputs)
    frm = analyze_frm_class(adapter.cls)
    arithmetic = any(op in frm.get("operators", {}) for op in ("+", "-", "*"))

    candidates, exhaustive = generate_candidates(adapter, budget, clock)

    # Evaluate every candidate once; cache results.
    evals: List[Tuple[Dict[str, int], Dict[str, int]]] = []
    for vec in candidates:
        if clock.expired():
            break
        evals.append((vec, adapter.run(vec)))

    universe: Set[str] = set()
    items: List[_Item] = []

    # ---- (a) single-vector goals: output classes / control modes / boundaries
    for vec, out in evals:
        goals: Set[str] = set()
        for o, w in outputs:
            if o not in out:
                continue
            for g in _output_classes(o, out[o], w, vec, arithmetic):
                goals.add(g)
        for name, w in names_widths:
            if _is_control(name, w):
                goals.add(f"ctrl[{name}]={vec[name]}")
            for lbl, val in _boundaries(name, w, adapter.is_signed(name)):
                if vec[name] == val:
                    goals.add(f"bound[{name}]={lbl}")
        # equality-between-inputs (separates <=/<, ==/!=)
        eq_names = [n for n, _ in names_widths]
        for i in range(len(eq_names)):
            for j in range(i + 1, len(eq_names)):
                a, b = eq_names[i], eq_names[j]
                if adapter.width(a) == adapter.width(b) and vec[a] == vec[b]:
                    goals.add(f"eq[{a}=={b}]")
        if goals:
            universe |= goals
            items.append(_Item(goals=goals, cost=1, payload=vec,
                               reason=_reason_for(vec, out, goals),
                               kind="single_cycle"))

    # ---- (b) distinguishing goals: input-sensitivity witness PAIRS ----------
    #   For each (output o, input signal, bit) find (x, x-with-bit-flipped) whose
    #   output differs. This is the ATPG controllability+observability idea and
    #   is exactly what separates functionally-similar designs (AND vs OR -> 01/10).
    sens_items, sens_goals = _distinguishing_items(adapter, outputs, evals,
                                                   arithmetic, clock)
    universe |= sens_goals
    items.extend(sens_items)

    # ---- emit the FULL set (completeness, NOT minimization) -----------------
    #   The stimulus is the entire candidate set (exhaustive for <=11 input bits,
    #   else a cap-sized sample) UNION both members of every distinguishing pair
    #   (the bit-flip partners are the ATPG vectors that separate similar ops and
    #   may not appear in the base sample for wide inputs). Nothing is pruned:
    #   leaving a distinguishing input untested is exactly the failure mode we
    #   are eliminating.
    pair_goals: Dict[tuple, Set[str]] = {}
    all_vecs: List[Dict[str, int]] = list(candidates)
    for it in items:
        payload = it.payload if isinstance(it.payload, list) else [it.payload]
        if len(payload) > 1:                       # a distinguishing pair
            for vec in payload:
                all_vecs.append(vec)
                pair_goals.setdefault(tuple(sorted(vec.items())), set()).update(it.goals)

    tests, seen_vec, name_ctr = [], {}, {}
    for vec in all_vecs:
        if len(tests) >= budget.max_tests:
            break
        key = tuple(sorted(vec.items()))
        if key in seen_vec:
            continue
        out = adapter.run(vec)
        these = _goals_of_vector(adapter, vec, out, outputs, names_widths, arithmetic)
        these |= pair_goals.get(key, set())
        these &= universe
        seen_vec[key] = len(tests)
        hint = "distinguish" if key in pair_goals else "test"
        tests.append({
            "name": _mk_name(name_ctr, hint, vec, out),
            "kind": "single_cycle",
            "inputs": {n: vec.get(n, 0) for n, _ in names_widths},
            # None (not a fabricated 0) if the FRM never produces this
            # declared output -- see frm_consistency.warnings.
            "expected_outputs": {o: out.get(o) for o, _ in outputs},
            "covers": sorted(these),
            "reason": _reason_for(vec, out, these),
        })
    covered = set()
    for t in tests:
        covered |= set(t["covers"])

    coverage_goals = _describe_goals(sorted(universe))
    return {
        "design_type": "combinational",
        "interface": _interface_dict(adapter),
        "coverage_goals": coverage_goals,
        "tests": tests,
        "sequences": [],
        "exhaustive": exhaustive,
        "frm_analysis": frm,
        "universe": sorted(universe),
        "covered": sorted(covered),
    }


def _boundaries(name: str, w: int, signed: bool) -> List[Tuple[str, int]]:
    mask = (1 << w) - 1
    b = [("zero", 0), ("one", 1 & mask), ("max", mask)]
    if signed and w >= 1:
        b.append(("min_signed", 1 << (w - 1)))
        b.append(("max_signed", (1 << (w - 1)) - 1))
    return b


def _goals_of_vector(adapter, vec, out, outputs, names_widths, arithmetic) -> Set[str]:
    goals: Set[str] = set()
    for o, w in outputs:
        if o in out:
            for g in _output_classes(o, out[o], w, vec, arithmetic):
                goals.add(g)
    for name, w in names_widths:
        if _is_control(name, w):
            goals.add(f"ctrl[{name}]={vec[name]}")
        for lbl, val in _boundaries(name, w, adapter.is_signed(name)):
            if vec[name] == val:
                goals.add(f"bound[{name}]={lbl}")
    en = [n for n, _ in names_widths]
    for i in range(len(en)):
        for j in range(i + 1, len(en)):
            a, b = en[i], en[j]
            if adapter.width(a) == adapter.width(b) and vec[a] == vec[b]:
                goals.add(f"eq[{a}=={b}]")
    return goals


def _distinguishing_items(adapter, outputs, evals, arithmetic, clock
                          ) -> Tuple[List[_Item], Set[str]]:
    """Build witness-pair items that separate similar operations."""
    items: List[_Item] = []
    goals: Set[str] = set()
    names_widths = list(adapter.ports.inputs)
    found: Set[str] = set()

    for vec, out in evals:
        if clock.expired():
            break
        for name, w in names_widths:
            for bit in range(w):
                # search across outputs for a sensitivity witness on this bit
                flipped = dict(vec)
                flipped[name] = vec[name] ^ (1 << bit)
                fout = adapter.run(flipped)
                for o, ow in outputs:
                    if out.get(o) != fout.get(o):
                        gid = f"distinguish[{o}<-{name}[{bit}]]"
                        if gid in found:
                            continue
                        found.add(gid)
                        goals.add(gid)
                        reason = (f"Flipping bit {bit} of '{name}' flips output "
                                  f"'{o}' ({out.get(o)}->{fout.get(o)}); this input "
                                  f"pair distinguishes the reference's behavior from "
                                  f"operations that would be insensitive here "
                                  f"(e.g. AND vs OR need the mixed 01/10 case).")
                        items.append(_Item(
                            goals={gid}, cost=2, payload=[dict(vec), dict(flipped)],
                            reason=reason, kind="distinguish"))
    return items, goals


def _reason_for(vec, out, goals) -> str:
    cls = sorted(g for g in goals if g.startswith("out["))
    if cls:
        return (f"Selected because it produces meaningful output class(es) "
                f"{', '.join(cls)} from the golden model.")
    ctrl = sorted(g for g in goals if g.startswith("ctrl["))
    if ctrl:
        return f"Exercises control mode(s) {', '.join(ctrl)}."
    bnd = sorted(g for g in goals if g.startswith("bound["))
    if bnd:
        return f"Covers input boundary value(s) {', '.join(bnd)}."
    return "Adds new coverage under the golden reference model."


def _mk_name(ctr: Dict[str, int], hint: str, vec, out) -> str:
    base = "test"
    low = hint.lower()
    if "distinguish" in low:
        base = "distinguish"
    elif "control mode" in low or "ctrl" in low:
        base = "control"
    elif "boundary" in low or "bound" in low:
        base = "boundary"
    elif "output class" in low or "out[" in low:
        base = "output_class"
    ctr[base] = ctr.get(base, 0) + 1
    inpart = "_".join(f"{k}{v}" for k, v in sorted(vec.items()))[:40]
    return f"{base}_{ctr[base]:03d}_{inpart}" if inpart else f"{base}_{ctr[base]:03d}"


# ---------------------------------------------------------------------------
# Sequential planning: BFS reachability + transition cover over the FRM
# ---------------------------------------------------------------------------

def _seq_candidate_inputs(adapter: GoldenModelAdapter, budget: Budget
                          ) -> List[Dict[str, int]]:
    """Per-cycle input vectors to try from each state. Control inputs fully
    enumerated; data inputs sampled with high-information values."""
    names_widths = list(adapter.ports.inputs)
    total = adapter.total_input_bits()
    if total == 0:
        return [{}]
    if total <= 10 and (1 << total) <= budget.max_candidates:
        out = []
        for v in range(1 << total):
            vec, pos = {}, 0
            bits = format(v, f"0{total}b")
            for name, w in names_widths:
                vec[name] = int(bits[pos:pos + w], 2)
                pos += w
            out.append(vec)
        return out
    per_input = [
        (n, list(range(1 << w)) if _is_control(n, w)
         else per_input_values(w, adapter.is_signed(n)))
        for n, w in names_widths
    ]
    # control signals vary fastest so their full range survives the cap
    per_input = sorted(per_input, key=lambda nv: _is_control(nv[0], adapter.width(nv[0])))
    names = [n for n, _ in per_input]
    vals = [v for _, v in per_input]
    out, seen = [], set()
    for combo in itertools.product(*vals):
        key = combo
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(zip(names, combo)))
        if len(out) >= budget.max_candidates:
            break
    return out


def plan_sequential(adapter: GoldenModelAdapter, budget: Budget,
                    clock: _Clock, meta: Optional[Dict] = None) -> Dict:
    names_widths = list(adapter.ports.inputs)
    outputs = list(adapter.ports.outputs)
    cand = _seq_candidate_inputs(adapter, budget)

    # BFS from reset; shortest input path per discovered state.
    root = adapter.reset()
    start_key = adapter.state_key(root)
    paths: Dict[tuple, List[Dict[str, int]]] = {start_key: []}
    snaps: Dict[tuple, dict] = {start_key: adapter.snapshot(root)}
    order: List[tuple] = [start_key]          # discovery order -> state labels
    transitions: List[Tuple[tuple, Dict[str, int], tuple]] = []

    frontier = [start_key]
    depth = 0
    while frontier and len(paths) < budget.max_bfs_states \
            and depth < budget.max_sequence_depth and not clock.expired():
        nxt = []
        for skey in frontier:
            for civ in cand:
                dut = adapter.restore(snaps[skey])
                adapter.step(dut, civ)
                nkey = adapter.state_key(dut)
                transitions.append((skey, civ, nkey))
                if nkey not in paths:
                    paths[nkey] = paths[skey] + [civ]
                    snaps[nkey] = adapter.snapshot(dut)
                    order.append(nkey)
                    nxt.append(nkey)
                    if len(paths) >= budget.max_bfs_states:
                        break
            if len(paths) >= budget.max_bfs_states:
                break
        frontier = nxt
        depth += 1

    label = {k: f"S{i}" for i, k in enumerate(order)}

    # optional declared target states from interface metadata
    declared = (meta or {}).get("target_states") or []

    # Build coverage universe: reach every state + cover every distinct
    # state->state transition.
    universe: Set[str] = set()
    for k in order:
        if k != start_key:
            universe.add(f"reach_{label[k]}")
    trans_reps: Dict[Tuple[str, str], Tuple[tuple, Dict[str, int]]] = {}
    for src, civ, dst in transitions:
        if src in label and dst in label:
            tg = (label[src], label[dst])
            universe.add(f"trans_{tg[0]}->{tg[1]}")
            trans_reps.setdefault(tg, (src, civ))

    # Candidate sequences (each covers everything it passes through from reset):
    items: List[_Item] = []

    def _seq_goals(path: List[Dict[str, int]]) -> Tuple[Set[str], List[dict], str]:
        """Simulate a path from reset; return (goals, per-cycle steps, target)."""
        dut = adapter.reset()
        cur = start_key
        goals: Set[str] = set()
        steps: List[dict] = []
        for c, civ in enumerate(path):
            out = adapter.step(dut, civ)
            nxt_key = adapter.state_key(dut)
            steps.append({
                "cycle": c,
                "inputs": {n: civ.get(n, 0) for n, _ in names_widths},
                "expected_outputs": {o: out.get(o) for o, _ in outputs},
            })
            if cur in label and nxt_key in label:
                goals.add(f"trans_{label[cur]}->{label[nxt_key]}")
            if nxt_key in label and nxt_key != start_key:
                goals.add(f"reach_{label[nxt_key]}")
            cur = nxt_key
        target = label.get(cur, "?")
        return goals, steps, target

    # reach-state sequences (shortest paths)
    for k in order:
        if k == start_key:
            continue
        path = paths[k]
        if len(path) > budget.max_sequence_depth:
            continue
        goals, steps, target = _seq_goals(path)
        items.append(_Item(goals=goals, cost=len(path),
                           payload={"target": target, "steps": steps, "path": path},
                           reason=(f"Shortest input sequence ({len(path)} cycle(s)) that "
                                   f"drives the golden model from reset to state {target}."),
                           kind="reach"))

    # transition-exercising sequences (path to src ++ the transition input)
    for (src_lbl, dst_lbl), (src_key, civ) in trans_reps.items():
        path = paths.get(src_key, []) + [civ]
        if len(path) > budget.max_sequence_depth:
            continue
        goals, steps, target = _seq_goals(path)
        items.append(_Item(goals=goals, cost=len(path),
                           payload={"target": target, "steps": steps, "path": path},
                           reason=(f"Reaches {src_lbl} then applies an input that transitions "
                                   f"to {dst_lbl}, exercising that transition in the reference."),
                           kind="transition"))

    # greedy minimization (prefer short, high-coverage sequences)
    def unit_of(payload):
        # cost accounting per-cycle would over-penalize; count each sequence once
        return [payload["path"] and json.dumps(payload["steps"], sort_keys=True)]

    chosen = greedy_cover(items, universe, max_units=budget.max_tests,
                          unit_of=unit_of)

    sequences, ctr = [], {}
    covered: Set[str] = set()
    for it in chosen:
        p = it.payload
        ctr[it.kind] = ctr.get(it.kind, 0) + 1
        nm = f"{it.kind}_{ctr[it.kind]:03d}_reach_{p['target']}"
        sequences.append({
            "name": nm,
            "kind": "multi_cycle",
            "steps": p["steps"],
            "target_state": p["target"],
            "covers": sorted(it.goals & universe),
            "reason": it.reason,
        })
        covered |= it.goals

    return {
        "design_type": "sequential",
        "interface": _interface_dict(adapter),
        "coverage_goals": _describe_goals(sorted(universe)),
        "tests": [],
        "sequences": sequences,
        "states_reached": len(order),
        "declared_targets": declared,
        "universe": sorted(universe),
        "covered": sorted(covered),
        "state_labels": {label[k]: _summarize_state(snaps[k]) for k in order},
    }


def _summarize_state(snap: dict) -> dict:
    out = {}
    for k, v in snap.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = repr(v)
    return out


# ---------------------------------------------------------------------------
# Goal descriptions / interface / assembly
# ---------------------------------------------------------------------------

def _describe_goals(goal_ids: List[str]) -> List[Dict]:
    out = []
    for i, gid in enumerate(goal_ids, 1):
        if gid.startswith("out["):
            typ, desc = "output_class", f"Cover output class {gid}"
        elif gid.startswith("distinguish["):
            typ, desc = "distinguishing", f"Sensitize {gid} (separates similar operations)"
        elif gid.startswith("ctrl["):
            typ, desc = "control_mode", f"Exercise control setting {gid}"
        elif gid.startswith("bound["):
            typ, desc = "boundary", f"Drive input boundary {gid}"
        elif gid.startswith("eq["):
            typ, desc = "equality", f"Equal-operand case {gid} (separates < from <=, == from !=)"
        elif gid.startswith("reach_"):
            typ, desc = "reach_state", f"Reach state {gid[len('reach_'):]}"
        elif gid.startswith("trans_"):
            typ, desc = "transition", f"Exercise transition {gid[len('trans_'):]}"
        else:
            typ, desc = "other", gid
        out.append({"id": f"goal_{i:03d}", "key": gid, "type": typ, "description": desc})
    return out


def _interface_dict(adapter: GoldenModelAdapter) -> Dict:
    return {
        "inputs": {n: {"width": w, "signed": adapter.is_signed(n),
                       "control": _is_control(n, w)}
                   for n, w in adapter.ports.inputs},
        "outputs": {o: {"width": w} for o, w in adapter.ports.outputs},
        "clock": adapter.ports.clk_name,
    }


def _frm_consistency(adapter: GoldenModelAdapter) -> Dict:
    """Cross-check the Python reference model against the AUTHORITATIVE interface
    (the top_module.v header) so a wrong / mis-wired golden_dut.py is surfaced
    instead of silently trusted.

    The header (port names + widths) is the source of truth for the *interface*;
    the FRM is the source of truth for *expected values*. If the FRM disagrees
    with the header -- reads a port that does not exist, ignores a declared
    input, never produces a declared output, or emits a value wider than the
    port -- the plan built on top of it may be wrong at the root, so we report it
    loudly rather than proceeding blindly.
    """
    frm = analyze_frm_class(adapter.cls)
    header_in = {n for n, _ in adapter.ports.inputs}
    header_out = {o for o, _ in adapter.ports.outputs}
    frm_reads = set(frm.get("inputs_read", []))
    frm_writes = set(frm.get("outputs_written", []))

    warnings: List[str] = []

    reads_unknown = sorted(frm_reads - header_in - {adapter.ports.clk_name or ""})
    ignored_inputs = sorted(header_in - frm_reads)
    missing_outputs_static = sorted(header_out - frm_writes)
    extra_outputs = sorted(frm_writes - header_out)

    # runtime probe: does the FRM actually emit every declared output, and does
    # any output exceed its declared width?
    missing_at_runtime, width_overflow = [], []
    runtime_outputs: Set[str] = set()
    try:
        if adapter.is_sequential:
            dut = adapter.reset()
            out = adapter.step(dut, {n: 0 for n in adapter.input_names})
        else:
            out = adapter.run({n: 0 for n in adapter.input_names})
        runtime_outputs = set(out.keys())
        for o, w in adapter.ports.outputs:
            if o not in out:
                missing_at_runtime.append(o)
            elif out[o] >> w:
                width_overflow.append({"output": o, "declared_width": w, "value": out[o]})
    except Exception as e:  # a reference that crashes on a legal input is itself a red flag
        warnings.append(f"golden_dut raised on an all-zero input: {type(e).__name__}: {e}")

    if runtime_outputs:
        missing_outputs_static = sorted(header_out - runtime_outputs)
        extra_outputs = sorted(runtime_outputs - header_out)

    if reads_unknown:
        warnings.append(f"FRM reads inputs not in the header (typo/wrong port?): {reads_unknown}")
    if ignored_inputs:
        warnings.append(f"header inputs the FRM never reads -> cannot be sensitized "
                        f"(coverage blind spot): {ignored_inputs}")
    if missing_outputs_static or missing_at_runtime:
        warnings.append(f"declared outputs the FRM never produces: "
                        f"{sorted(set(missing_outputs_static) | set(missing_at_runtime))}")
    if extra_outputs:
        warnings.append(f"FRM produces outputs not in the header: {extra_outputs}")
    if width_overflow:
        warnings.append(f"FRM output exceeds its declared width: {width_overflow}")

    return {
        "interface_source": "top_module.v header (authoritative)",
        "expected_values_source": "golden_dut.py (behavioral)",
        "status": "ok" if not warnings else "warnings",
        "header_inputs": sorted(header_in),
        "header_outputs": sorted(header_out),
        "frm_inputs_read": sorted(frm_reads),
        "frm_outputs_written": sorted(frm_writes),
        "ignored_inputs": ignored_inputs,
        "reads_unknown_inputs": reads_unknown,
        "missing_outputs": sorted(set(missing_outputs_static) | set(missing_at_runtime)),
        "extra_outputs": extra_outputs,
        "width_overflow": width_overflow,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Top-level agent
# ---------------------------------------------------------------------------

def build_plan(golden_dut_path: str, dut_path: str, *,
               budget: Optional[Budget] = None,
               interface_meta: Optional[Dict] = None,
               design_name: Optional[str] = None,
               force_type: Optional[str] = None) -> Dict:
    """Run the coverage agent end-to-end and return the coverage_plan dict."""
    budget = budget or Budget()
    clock = _Clock(budget.max_runtime_seconds)

    cls = load_frm(golden_dut_path)
    with open(dut_path) as f:
        verilog = f.read()
    # The header is the AUTHORITATIVE interface: port names + widths come from
    # top_module.v, never from the (possibly wrong) golden model. Optional
    # metadata may add signedness / target states but does not override widths.
    ports = parse_ports(verilog)
    signed = _detect_signed(verilog, interface_meta)
    adapter = GoldenModelAdapter(cls, ports, meta=interface_meta, signed=signed)
    consistency = _frm_consistency(adapter)
    if consistency["warnings"]:
        for w in consistency["warnings"]:
            logger.warning("FRM/header consistency: %s", w)

    is_seq = (force_type == "seq") if force_type else adapter.is_sequential
    core = (plan_sequential(adapter, budget, clock, meta=interface_meta)
            if is_seq else plan_combinational(adapter, budget, clock))

    universe = set(core.pop("universe"))
    covered = set(core.pop("covered"))
    dname = design_name or _infer_name(verilog) or os.path.basename(
        os.path.dirname(os.path.abspath(dut_path))) or "design"

    plan = {
        "design_name": dname,
        "design_type": core["design_type"],
        "source_of_truth": os.path.basename(golden_dut_path),
        "black_box_dut": os.path.basename(dut_path),
        "interface_source": "top_module.v header (authoritative)",
        "frm_consistency": consistency,
        "runtime_budget": budget.as_dict(),
        "interface": core["interface"],
        "coverage_goals": core["coverage_goals"],
        "tests": core["tests"],
        "sequences": core["sequences"],
        "summary": {
            "num_tests": len(core["tests"]),
            "num_sequences": len(core["sequences"]),
            "num_goals": len(universe),
            "covered_goals": sorted(covered),
            "uncovered_goals": sorted(universe - covered),
            "exhaustive": core.get("exhaustive", False),
            "runtime_seconds": round(time.time() - clock.start, 4),
        },
    }
    if core["design_type"] == "sequential":
        plan["summary"]["states_reached"] = core.get("states_reached")
        plan["state_labels"] = core.get("state_labels")
    else:
        plan["frm_analysis"] = core.get("frm_analysis")
    return plan


def _infer_name(verilog: str) -> Optional[str]:
    m = re.search(r"module\s+(\w+)", verilog)
    return m.group(1) if m else None


def write_plan(plan: Dict, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(plan, f, indent=2)


# ---------------------------------------------------------------------------
# Integration helper: convert a plan into the existing Pro-V stimulus.json
# (binary-string) schema, so the current golden_dut harness / testbench
# generator can consume it unchanged.
# ---------------------------------------------------------------------------

def plan_to_stimulus(plan: Dict, adapter_widths: Optional[Dict[str, int]] = None
                     ) -> List[dict]:
    """Down-convert a coverage_plan into the current stimulus.json format.

    Combinational -> list of {name: binary_string}.
    Sequential    -> list of scenarios {clock_cycles, <name>: [binary,...]}.
    """
    widths = adapter_widths or {n: v["width"]
                                for n, v in plan["interface"]["inputs"].items()}
    if plan["design_type"] == "combinational":
        stim = []
        for t in plan["tests"]:
            stim.append({n: int_to_bin(v, widths.get(n, 1))
                         for n, v in t["inputs"].items()})
        return stim
    scenarios = []
    for seq in plan["sequences"]:
        steps = seq["steps"]
        scn = {"clock_cycles": len(steps)}
        for n, w in widths.items():
            scn[n] = [int_to_bin(s["inputs"].get(n, 0), w) for s in steps]
        scenarios.append(scn)
    return scenarios


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Reference-guided functional coverage planning (pre-testbench).")
    p.add_argument("--frm", "--golden", dest="frm", required=True,
                   help="Path to golden_dut.py (behavioral source of truth)")
    p.add_argument("--dut", required=True,
                   help="Path to top_module.v (black box; interface only)")
    p.add_argument("--out", "-o", default="coverage_plan.json")
    p.add_argument("--stimulus-out", default=None,
                   help="Also emit a current-format stimulus.json for the existing harness")
    p.add_argument("--meta", default=None, help="Optional interface metadata JSON")
    p.add_argument("--type", choices=["cmb", "seq"], default=None)
    p.add_argument("--max-tests", type=int, default=4096,
                   help="emission cap; keep >= sample-cap so completeness isn't clipped")
    p.add_argument("--max-candidates", type=int, default=5000)
    p.add_argument("--sample-cap", type=int, default=2048,
                   help="hard cap on vectors for >exhaustive-max-bits input spaces")
    p.add_argument("--exhaustive-max-bits", type=int, default=11,
                   help="exhaustive enumeration at or below this many input bits (2**11=2048)")
    p.add_argument("--max-sequence-depth", type=int, default=20)
    p.add_argument("--max-bfs-states", type=int, default=256)
    p.add_argument("--max-runtime-seconds", type=float, default=120.0)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())

    logging.basicConfig(level=logging.INFO)
    meta = json.load(open(args.meta)) if args.meta else None
    budget = Budget(max_tests=args.max_tests, max_candidates=args.max_candidates,
                    max_sequence_depth=args.max_sequence_depth,
                    max_bfs_states=args.max_bfs_states,
                    max_runtime_seconds=args.max_runtime_seconds,
                    exhaustive_max_bits=args.exhaustive_max_bits,
                    sample_cap=args.sample_cap)
    plan = build_plan(args.frm, args.dut, budget=budget, interface_meta=meta,
                      force_type=args.type)  # "cmb" | "seq" | None (auto-detect)
    write_plan(plan, args.out)
    logger.info("Wrote %s (%d tests, %d sequences, %d/%d goals covered)",
                args.out, plan["summary"]["num_tests"],
                plan["summary"]["num_sequences"],
                len(plan["summary"]["covered_goals"]),
                plan["summary"]["num_goals"])
    if args.stimulus_out:
        json.dump(plan_to_stimulus(plan), open(args.stimulus_out, "w"), indent=2)
        logger.info("Wrote %s (current stimulus format)", args.stimulus_out)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    logging.basicConfig(level=logging.WARNING)
    import tempfile
    ok = True
    tmp = tempfile.mkdtemp(prefix="fca_selftest_")

    # ---- AND gate: weak 00/11 is insufficient; agent must select 01 and 10 --
    and_src = (
        "class GoldenDUT:\n"
        "    def load(self, inputs):\n"
        "        a=int(inputs['a'],2); b=int(inputs['b'],2)\n"
        "        return {'y': str(a & b)}\n"
    )
    and_v = "module top_module(input a, input b, output y); assign y=a&b; endmodule"
    gp = os.path.join(tmp, "golden_and.py"); open(gp, "w").write(and_src)
    dp = os.path.join(tmp, "and.v"); open(dp, "w").write(and_v)
    plan = build_plan(gp, dp, budget=Budget(max_tests=20))
    vecs = {(t["inputs"]["a"], t["inputs"]["b"]) for t in plan["tests"]}
    has_dist = any("distinguish" in t["name"] for t in plan["tests"])
    and_ok = (0, 1) in vecs and (1, 0) in vecs and has_dist \
        and plan["design_type"] == "combinational"
    print("AND vectors:", sorted(vecs), "| distinguishing test present:", has_dist)
    print("AND SELFTEST:", "PASS" if and_ok else "FAIL",
          "(agent selects 01 and 10 -- the AND-vs-OR distinguishing inputs)")
    ok = ok and and_ok

    # ---- expected outputs come from the golden model (spot check) -----------
    exp_ok = all(t["expected_outputs"]["y"] == (t["inputs"]["a"] & t["inputs"]["b"])
                 for t in plan["tests"])
    print("EXPECTED-OUTPUTS SELFTEST:", "PASS" if exp_ok else "FAIL",
          "(each expected y equals a&b from golden_dut)")
    ok = ok and exp_ok

    # ---- FSM: BFS must find a sequence to a DONE state ----------------------
    fsm_src = (
        "class GoldenDUT:\n"
        "    def __init__(self):\n"
        "        self.state = 0  # 0=IDLE 1=RUN 2=DONE\n"
        "    def load(self, clk, inputs):\n"
        "        start=int(inputs.get('start','0'),2); rst=int(inputs.get('rst','0'),2)\n"
        "        if clk==1:\n"
        "            if rst: self.state=0\n"
        "            elif self.state==0 and start: self.state=1\n"
        "            elif self.state==1: self.state=2\n"
        "            elif self.state==2: self.state=0\n"
        "        return {'done': '1' if self.state==2 else '0'}\n"
    )
    fsm_v = ("module top_module(input clk, input rst, input start, output done); "
             "endmodule")
    fgp = os.path.join(tmp, "golden_fsm.py"); open(fgp, "w").write(fsm_src)
    fdp = os.path.join(tmp, "fsm.v"); open(fdp, "w").write(fsm_v)
    fplan = build_plan(fgp, fdp, budget=Budget(max_tests=30, max_sequence_depth=8))
    # some planned sequence must drive the model into DONE and observe done=1
    reaches_done = any(step["expected_outputs"].get("done") == 1
                       for seq in fplan["sequences"] for step in seq["steps"])
    fsm_ok = (fplan["design_type"] == "sequential" and reaches_done
              and fplan["summary"]["states_reached"] >= 3)
    print("FSM states reached:", fplan["summary"]["states_reached"],
          "| sequences:", fplan["summary"]["num_sequences"],
          "| reaches done:", reaches_done)
    print("FSM SELFTEST:", "PASS" if fsm_ok else "FAIL",
          "(BFS over the golden model finds a clocked sequence into the DONE state)")
    ok = ok and fsm_ok

    # ---- stimulus down-conversion round-trips into the existing schema ------
    stim = plan_to_stimulus(plan)
    seq_stim = plan_to_stimulus(fplan)
    conv_ok = (all(set(d.keys()) == {"a", "b"} for d in stim)
               and all("clock_cycles" in s for s in seq_stim))
    print("STIMULUS-CONVERSION SELFTEST:", "PASS" if conv_ok else "FAIL",
          "(plan down-converts to the current stimulus.json format)")
    ok = ok and conv_ok

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# NOTE: the reference-guided coverage agent (stage 6) is defined below. Its
# self-test is exposed as `_rg_selftest` and run via import (it is defined after
# this guard, so it is intentionally not wired into the __main__ dispatch here).
if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    main()


# =====================================================================
# SpecKit Reference-Guided Functional Coverage  (pipeline stage 6)
# =====================================================================
# Consumes the APPROVED golden_dut.py + behavior_contract.json + witnesses.json +
# invariants.json and emits coverage_plan.json in the reference-guided schema.
#
# golden_dut.py is the SOLE source of behavioral truth: every expected output here
# is produced by RUNNING the approved model. top_module.v is a black box, used only
# for interface wiring (an optional dut_path is recorded, never parsed for logic).
#
# Reuses (no re-implementation): the verifier's ModelAdapter (API-agnostic golden
# evaluation over eval/step/load), witness_generator (spec-derived candidates), and
# the existing Budget dataclass (runtime controls + completeness policy).


def _rg_w(spec) -> int:
    try:
        return max(1, int(spec.get("width", 1))) if isinstance(spec, dict) else 1
    except Exception:
        return 1


class _RGGoals:
    """Ordered coverage-goal registry with stable CG ids."""

    def __init__(self):
        self.by_key: Dict[Any, Dict] = {}
        self.order: List[Any] = []

    def add(self, key, gtype: str, description: str) -> str:
        if key not in self.by_key:
            gid = "CG%d" % (len(self.order) + 1)
            self.by_key[key] = {"id": gid, "type": gtype, "description": description}
            self.order.append(key)
        return self.by_key[key]["id"]

    def id_of(self, key):
        return self.by_key[key]["id"] if key in self.by_key else None

    def has(self, key) -> bool:
        return key in self.by_key

    def as_list(self) -> List[Dict]:
        return [self.by_key[k] for k in self.order]


def _rg_control_domain(name: str, width: int, budget: "Budget") -> List[int]:
    if width <= 4:
        return list(range(1 << width))
    return [0, (1 << width) - 1]


def _rg_register_goals(goals: _RGGoals, contract: dict, wl: List[dict], seq: bool,
                       in_w: Dict[str, int], budget: "Budget"):
    from pro_v.witness_generator import boundary_values as _bvals, _is_control as _isctrl
    for r in contract.get("rules", []) or []:
        rid = r.get("id")
        if rid:
            goals.add(("rule", rid), "rule",
                      "Exercise %s: %s" % (rid, r.get("effect") or r.get("description") or ""))
    for n, w in in_w.items():
        if _isctrl(n):
            for v in _rg_control_domain(n, w, budget):
                goals.add(("control", n, v), "control_mode", "Control %s = %d" % (n, v))
        for label, _val in _bvals(w, False):
            goals.add(("boundary", n, label), "boundary", "Boundary %s on %s" % (label, n))
    for wit in wl:
        if wit.get("type") == "distinguishing" and wit.get("id"):
            goals.add(("dist", wit["id"]), "distinguishing", wit.get("reason") or wit.get("description", ""))
    if seq:
        for s in contract.get("states", []) or []:
            goals.add(("state", s), "state", "Reach state %s" % s)
        for wit in wl:
            if wit.get("type") == "state_transition" and wit.get("id"):
                goals.add(("trans", wit["id"]), "transition", wit.get("description", ""))
            if wit.get("type") == "reset_hold_enable" and wit.get("id"):
                goals.add(("reset_hold", wit["id"]), "reset_hold", wit.get("description", ""))


def _rg_covers_comb(wit, full_in: Dict[str, int], out: Dict[str, int], goals: _RGGoals,
                    in_w: Dict[str, int], budget: "Budget"):
    """Goal keys a combinational test covers; registers the dynamic output-class goal."""
    from pro_v.witness_generator import boundary_values as _bvals, _is_control as _isctrl
    keys = set()
    for rid in wit.get("covers_rules", []) or []:
        if goals.has(("rule", rid)):
            keys.add(("rule", rid))
    if wit.get("type") == "distinguishing" and goals.has(("dist", wit.get("id"))):
        keys.add(("dist", wit["id"]))
    for n, v in full_in.items():
        if _isctrl(n) and goals.has(("control", n, v)):
            keys.add(("control", n, v))
        for label, bval in _bvals(in_w.get(n, 1), False):
            if v == bval and goals.has(("boundary", n, label)):
                keys.add(("boundary", n, label))
    ockey = ("outclass", tuple(sorted(out.items())))
    goals.add(ockey, "output_class", "Observe output %s" % dict(out))
    keys.add(ockey)
    return keys


def _rg_cover_ids(keys, goals: _RGGoals, wit) -> List[str]:
    ids = [goals.id_of(k) for k in keys if goals.id_of(k)]
    if wit.get("id"):
        ids.append(wit["id"])
    ids += [r for r in (wit.get("covers_rules") or [])]
    # stable, de-duplicated
    seen, out = set(), []
    for x in ids:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out


def _rg_combinational(adapter, contract, wl, goals, in_w, out_names, budget) -> List[Dict]:
    from pro_v.golden_model_verifier import _to_int
    tests: List[Dict] = []
    covered = set()
    namer: Dict[str, int] = {}
    t0 = time.time()
    # required witnesses first (completeness), then the rest (greedy)
    ordered = sorted([w for w in wl if isinstance(w.get("inputs"), dict) and not w.get("relation")
                      and "sequence" not in w],
                     key=lambda w: (0 if w.get("required") else 1,
                                    0 if w.get("type") == "distinguishing" else 1))
    for wit in ordered:
        if len(tests) >= budget.max_tests or (time.time() - t0) > budget.max_runtime_seconds:
            break
        ints = {k: _to_int(v) for k, v in wit["inputs"].items()}
        ints = {k: v for k, v in ints.items() if v is not None}
        full = {n: 0 for n in in_w}
        full.update({k: v for k, v in ints.items() if k in in_w})
        try:
            out = adapter.eval(adapter.cls(), full)
        except Exception:
            continue
        out = {k: v for k, v in out.items() if k in out_names and v is not None}
        if not out:
            continue
        keys = _rg_covers_comb(wit, full, out, goals, in_w, budget)
        new = [k for k in keys if k not in covered]
        if wit.get("required") or new:
            covered.update(keys)
            tests.append({
                "name": _rg_name(namer, wit.get("type", "test")),
                "kind": "single_cycle",
                "inputs": full,
                "expected_outputs": out,
                "covers": _rg_cover_ids(keys, goals, wit),
                "reason": wit.get("reason") or wit.get("description", ""),
            })
    return tests


def _rg_sequential(adapter, contract, wl, goals, in_w, out_names, budget) -> List[Dict]:
    from pro_v.golden_model_verifier import _to_int
    seqs: List[Dict] = []
    covered = set()
    namer: Dict[str, int] = {}
    t0 = time.time()
    ordered = sorted([w for w in wl if "sequence" in w],
                     key=lambda w: (0 if w.get("required") else 1, len(w.get("sequence", []))))
    for wit in ordered:
        if len(seqs) >= budget.max_tests or (time.time() - t0) > budget.max_runtime_seconds:
            break
        if len(wit.get("sequence", [])) > budget.max_sequence_depth:
            continue
        inst = adapter.new()
        cyc_rows, ok = [], True
        for step in wit["sequence"]:
            ints = {k: _to_int(v) for k, v in (step.get("inputs") or {}).items()}
            full = {n: 0 for n in in_w}
            full.update({k: v for k, v in ints.items() if k in in_w and v is not None})
            try:
                out = adapter.step(inst, full)
            except Exception:
                ok = False
                break
            cyc_rows.append({"cycle": step.get("cycle", len(cyc_rows)), "inputs": full,
                             "expected_outputs": {k: v for k, v in out.items()
                                                  if k in out_names and v is not None}})
        if not ok:
            continue
        final = adapter.state(inst)
        keys = set()
        for rid in wit.get("covers_rules", []) or []:
            if goals.has(("rule", rid)):
                keys.add(("rule", rid))
        if goals.has(("trans", wit.get("id"))):
            keys.add(("trans", wit["id"]))
        if goals.has(("reset_hold", wit.get("id"))):
            keys.add(("reset_hold", wit["id"]))
        fstate = (wit.get("expected_final") or {}).get("state")
        if fstate is None and isinstance(final, dict):
            fstate = final.get("state", next(iter(final.values()), None))
        if fstate is not None and goals.has(("state", str(fstate))):
            keys.add(("state", str(fstate)))
        new = [k for k in keys if k not in covered]
        if wit.get("required") or new:
            covered.update(keys)
            seqs.append({
                "name": _rg_name(namer, wit.get("type", "seq")),
                "kind": "sequence",
                "sequence": cyc_rows,
                "expected_final": ({"state": str(fstate)} if fstate is not None else None),
                "covers": _rg_cover_ids(keys, goals, wit),
                "reason": wit.get("reason") or wit.get("description", ""),
            })
    return seqs


def _rg_name(namer: Dict[str, int], hint: str) -> str:
    hint = re.sub(r"[^a-zA-Z0-9]+", "_", hint).strip("_").lower() or "test"
    namer[hint] = namer.get(hint, -1) + 1
    return "%s_%d" % (hint, namer[hint])


def build_coverage_plan(golden_dut_path: str, contract, witnesses=None, invariants=None,
                        dut_path: Optional[str] = None, budget: Optional["Budget"] = None,
                        output_dir: Optional[str] = None) -> Dict:
    """Reference-guided coverage plan (stage 6). golden_dut.py is the source of truth;
    top_module.v (dut_path) is only recorded as the black-box DUT. Returns the plan
    dict and, if output_dir is given, writes coverage_plan.json."""
    from pro_v.golden_model_verifier import ModelAdapter, _load_golden_class, _load, _witness_list
    from pro_v.witness_generator import generate_witnesses

    contract = _load(contract)
    budget = budget or Budget()
    seq = contract.get("design_type") == "sequential"
    in_w = {n: _rg_w(s) for n, s in (contract.get("inputs") or {}).items()}
    out_names = list((contract.get("outputs") or {}).keys())

    cls = _load_golden_class(golden_dut_path)
    adapter = ModelAdapter(cls, contract)

    if witnesses is not None:
        wl = _witness_list(_load(witnesses))
    else:
        wl = generate_witnesses(contract).get("witnesses", [])

    goals = _RGGoals()
    _rg_register_goals(goals, contract, wl, seq, in_w, budget)

    if seq:
        sequences = _rg_sequential(adapter, contract, wl, goals, in_w, out_names, budget)
        tests: List[Dict] = []
    else:
        tests = _rg_combinational(adapter, contract, wl, goals, in_w, out_names, budget)
        sequences = []

    covered_ids = set()
    for t in tests + sequences:
        for c in t.get("covers", []):
            if c.startswith("CG"):
                covered_ids.add(c)
    all_ids = [g["id"] for g in goals.as_list()]
    uncovered = [gid for gid in all_ids if gid not in covered_ids]

    plan = {
        "design_name": contract.get("design_name", _infer_name("") or "design"),
        "source_of_truth": "golden_dut.py",
        "black_box_dut": dut_path or "top_module.v",
        "design_type": contract.get("design_type", "combinational"),
        "coverage_goals": goals.as_list(),
        "tests": tests,
        "sequences": sequences,
        "summary": {
            "num_tests": len(tests),
            "num_sequences": len(sequences),
            "num_goals": len(all_ids),
            "covered_goals": sorted(covered_ids, key=lambda x: int(x[2:])),
            "uncovered_goals": uncovered,
        },
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "coverage_plan.json"), "w") as f:
            json.dump(plan, f, indent=2)
        plan["path"] = os.path.join(output_dir, "coverage_plan.json")
    return plan


class ReferenceGuidedCoverageAgent:
    """Thin OO wrapper around build_coverage_plan (stage 6 of the SpecKit pipeline)."""

    def __init__(self, budget: Optional["Budget"] = None):
        self.budget = budget or Budget()

    def run(self, golden_dut_path: str, contract, witnesses=None, invariants=None,
            dut_path: Optional[str] = None, output_dir: Optional[str] = None) -> Dict:
        return build_coverage_plan(golden_dut_path, contract, witnesses, invariants,
                                   dut_path=dut_path, budget=self.budget, output_dir=output_dir)


def _rg_selftest() -> int:
    import tempfile
    ok = True

    # combinational AND: golden model is the truth; distinguishing 01/10 must appear
    and_c = {"design_name": "and_gate", "design_type": "combinational",
             "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "effect": "y = a & b"}], "can_generate_golden_model": True}
    and_src = ("class GoldenDUT:\n    def __init__(self): pass\n"
               "    def eval(self, inputs):\n        return {'y': (inputs['a'] & inputs['b']) & 1}\n")
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "golden_dut.py")
        with open(gp, "w") as f: f.write(and_src)
        plan = build_coverage_plan(gp, and_c, dut_path="top_module.v", output_dir=d)
        ok = ok and plan["source_of_truth"] == "golden_dut.py" and plan["black_box_dut"] == "top_module.v"
        ins = [(t["inputs"]["a"], t["inputs"]["b"], t["expected_outputs"]["y"]) for t in plan["tests"]]
        # exhaustive small space -> all 4 combos, golden-authoritative outputs
        ok = ok and (0, 1, 0) in ins and (1, 0, 0) in ins and (1, 1, 1) in ins and (0, 0, 0) in ins
        ok = ok and os.path.exists(plan["path"])
        ok = ok and plan["summary"]["num_tests"] >= 4
        # every rule + at least one output class covered, no dangling required goals
        cov = set(plan["summary"]["covered_goals"])
        rule_gid = next(g["id"] for g in plan["coverage_goals"] if g["type"] == "rule")
        ok = ok and rule_gid in cov

    # a WRONG golden model changes the truth: OR model yields different expected outputs
    or_src = and_src.replace("&", "|", 1).replace("& 1", "& 1")
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "golden_dut.py")
        with open(gp, "w") as f: f.write("class GoldenDUT:\n    def __init__(self): pass\n    def eval(self, inputs):\n        return {'y': (inputs['a'] | inputs['b']) & 1}\n")
        plan = build_coverage_plan(gp, and_c, output_dir=d)
        m = {(t["inputs"]["a"], t["inputs"]["b"]): t["expected_outputs"]["y"] for t in plan["tests"]}
        ok = ok and m[(0, 1)] == 1 and m[(1, 0)] == 1   # OR truth, straight from the model

    # sequential counter: sequences reach states, golden gives per-cycle outputs
    seq_c = {"design_name": "cnt", "design_type": "sequential",
             "inputs": {"reset": {"width": 1}, "en": {"width": 1}}, "outputs": {"q": {"width": 2}},
             "reset": {"name": "reset", "active_high": True}, "states": [], "initial_state": None,
             "rules": [{"id": "R1", "description": "increment when en"}], "can_generate_golden_model": True}
    seq_src = ("class GoldenDUT:\n    def __init__(self): self.reset()\n    def reset(self): self.q = 0\n"
               "    def step(self, inputs):\n"
               "        if inputs.get('reset'): self.q = 0\n"
               "        elif inputs.get('en'): self.q = (self.q + 1) & 3\n"
               "        return {'q': self.q}\n    def get_state(self): return {'q': self.q}\n")
    seq_w = {"witnesses": [{"id": "WS", "type": "sequential_reachability", "required": True,
                            "covers_rules": ["R1"],
                            "sequence": [{"cycle": 0, "inputs": {"reset": "1", "en": "0"}},
                                         {"cycle": 1, "inputs": {"reset": "0", "en": "1"}},
                                         {"cycle": 2, "inputs": {"reset": "0", "en": "1"}}]}]}
    with tempfile.TemporaryDirectory() as d:
        gp = os.path.join(d, "golden_dut.py")
        with open(gp, "w") as f: f.write(seq_src)
        plan = build_coverage_plan(gp, seq_c, witnesses=seq_w, output_dir=d)
        ok = ok and plan["summary"]["num_sequences"] == 1 and plan["tests"] == []
        s = plan["sequences"][0]
        ok = ok and s["sequence"][2]["expected_outputs"]["q"] == 2   # golden ran the counter
        ok = ok and s["kind"] == "sequence"

    print("functional_coverage_agent RG selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1
