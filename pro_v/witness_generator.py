#!/usr/bin/env python3
"""
witness_generator.py  --  deterministic witness synthesis (Pro-V stage 1b).

Reads `behavior_contract.json` and produces `witnesses.json`: concrete cases that
FORCE the reference model to implement the intended behavior. These are not random
tests; they are chosen to distinguish likely-wrong implementations.

This is the deterministic, structural complement to the SpecKit helper agent's
LLM-proposed witnesses (it MERGES with any it is given). It never uses
`top_module.v` internals -- only the contract's rules, ports, and states.

Witness-aware abstraction: input coverage alone and output coverage alone are both
insufficient. A test is valuable when it separates plausible wrong functions. AND
and OR agree on 00 and 11; only 01 and 10 distinguish them -- so those are emitted
as `distinguishing` witnesses.

Value convention: all input/output values are BINARY STRINGS ('0'/'1'), because the
downstream verifier calls GoldenDUT.load(inputs) which parses int(inputs[name], 2).

Witness types emitted:
  positive, negative, boundary, distinguishing, control_mode, output_class,
  sequential_reachability, state_transition, reset_hold_enable, metamorphic.
"""
from __future__ import annotations

import ast
import json
import operator
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

CONTROL_KEYWORDS = {
    "sel", "mode", "opcode", "op", "enable", "en", "valid", "ready", "start",
    "done", "we", "re", "rd", "wr", "load", "clear", "clr", "reset", "rst",
    "stall", "flush", "ack", "req", "hold", "shift",
}


def _bits(val: int, width: int) -> str:
    width = max(1, int(width))
    return format(int(val) & ((1 << width) - 1), "0%db" % width)


def _is_control(name: str) -> bool:
    low = name.lower()
    if low in CONTROL_KEYWORDS:
        return True
    parts = [p for p in re.split(r"[_\W]+", low) if p]
    return any(p in CONTROL_KEYWORDS for p in parts)


def _width_of(spec: Any) -> int:
    if isinstance(spec, dict):
        try:
            return max(1, int(spec.get("width", 1)))
        except Exception:
            return 1
    return 1


def _signed_of(spec: Any) -> bool:
    return bool(isinstance(spec, dict) and spec.get("signed"))


def boundary_values(width: int, signed: bool) -> List[Tuple[str, int]]:
    """Boundary classes for a single input width (dedup by value)."""
    w = max(1, width)
    mx = (1 << w) - 1
    out: List[Tuple[str, int]] = [("zero", 0), ("one", 1)]
    if w >= 2:
        out.append(("two", 2))
    out += [("max", mx), ("max_minus_1", mx - 1)]
    if w >= 2:
        alt10 = int("10" * ((w + 1) // 2), 2) & mx
        alt01 = int("01" * ((w + 1) // 2), 2) & mx
        out += [("alt_1010", alt10), ("alt_0101", alt01)]
    out.append(("one_hot_lsb", 1))
    out.append(("one_hot_msb", 1 << (w - 1)))
    if signed:
        out += [("signed_min", 1 << (w - 1)), ("signed_max", (1 << (w - 1)) - 1)]
    seen, uniq = set(), []
    for label, v in out:
        v &= mx
        if v not in seen:
            seen.add(v)
            uniq.append((label, v))
    return uniq


# ---------------------------------------------------------------------------
# safe expression evaluator (compute expected outputs from spec rules only)
# ---------------------------------------------------------------------------

_BIN = {ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.LShift: operator.lshift, ast.RShift: operator.rshift,
        ast.Div: operator.floordiv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_UN = {ast.Invert: operator.invert, ast.USub: operator.neg, ast.UAdd: operator.pos,
       ast.Not: lambda x: int(not x)}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
        ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


def _eval_node(node: ast.AST, env: Dict[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return int(_BIN[type(node.op)](_eval_node(node.left, env), _eval_node(node.right, env)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return int(_UN[type(node.op)](_eval_node(node.operand, env)))
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            return int(all(vals))
        return int(any(vals))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        ok = True
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, env)
            ok = ok and bool(_CMP[type(op)](left, right))
            left = right
        return int(ok)
    if isinstance(node, ast.IfExp):
        return _eval_node(node.body, env) if _eval_node(node.test, env) else _eval_node(node.orelse, env)
    if isinstance(node, ast.Name):
        if node.id in env:
            return int(env[node.id])
        raise ValueError("unbound name %s" % node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return int(node.value)
    if isinstance(node, ast.Num):  # pragma: no cover (py<3.8)
        return int(node.n)
    raise ValueError("unsupported expression node %s" % type(node).__name__)


def _verilog_to_py(expr: str) -> str:
    """Best-effort translate a simple Verilog RHS to a Python expression."""
    e = expr.strip()
    e = e.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"(?<![<>=!])!(?!=)", " not ", e)         # logical not, keep !=
    m = re.match(r"^(.*?)\?(.*?):(.*)$", e)               # single ternary cond?a:b
    if m:
        e = "(%s) if (%s) else (%s)" % (m.group(2), m.group(1), m.group(3))
    return e


def eval_effect(effect: str, env: Dict[str, int], width: int) -> Optional[str]:
    """Evaluate a rule effect ('y = <expr>') for the given inputs; return the
    output as a binary string, or None if it is not a safe evaluable expression."""
    if not effect or "=" not in effect:
        return None
    rhs = effect.split("=", 1)[1] if effect.count("=") else effect
    # avoid matching '==' style effects with no lhs=
    lhs = effect.split("=", 1)[0].strip()
    if not re.match(r"^[A-Za-z_]\w*(\[[^\]]*\])?$", lhs):
        return None
    rhs = _verilog_to_py(rhs)
    if re.search(r"'[bhdo]|\{|\bmemory\b|\bposedge\b", rhs):   # verilog literals/concat/mem -> bail
        return None
    try:
        tree = ast.parse(rhs, mode="eval")
        val = _eval_node(tree, env)
        return _bits(val, width)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# rule / operation inspection
# ---------------------------------------------------------------------------

def _output_exprs(rules: List[dict], out_names: List[str]) -> Dict[str, str]:
    """Map output name -> RHS expression string, from rules whose effect assigns it."""
    exprs: Dict[str, str] = {}
    for r in rules:
        eff = (r.get("effect") or "").strip()
        if "=" not in eff:
            continue
        lhs = eff.split("=", 1)[0].strip()
        base = re.sub(r"\[.*", "", lhs).strip()
        if base in out_names and base not in exprs:
            exprs[base] = eff.split("=", 1)[1].strip()
    return exprs


def _detect_operation(rules: List[dict]) -> Optional[str]:
    """Classify the dominant combinational operation. Operator SYMBOLS in the rule
    effects are the strong signal and are checked first; natural-language words in
    the description are a weaker fallback (so 'sum of a and b' -> add, not and)."""
    effects = " ".join((r.get("effect") or "") for r in rules)
    blob = " ".join(((r.get("effect") or "") + " " + (r.get("description") or "")).lower()
                     for r in rules)
    e = effects
    # 1) symbol-based (from effect expressions)
    if re.search(r"\^", e):
        return "xor"
    if re.search(r"<<|>>", e):
        return "shift"
    if re.search(r"\?", e):
        return "mux"
    if re.search(r"<=|>=", e):
        return "compare_eq"
    if re.search(r"[<>]", e):
        return "compare"
    if re.search(r"\+", e):
        return "add"
    if re.search(r"\*", e):
        return "mul"
    if re.search(r"\s-\s", e):
        return "sub"
    if re.search(r"(?<![&])&(?![&])", e):
        return "and"
    if re.search(r"(?<![|])\|(?![|])", e):
        return "or"
    if re.search(r"~", e):
        return "not"
    # 2) word-based (from full description, weaker)
    def has(*toks): return any(t in blob for t in toks)
    if has(" xor ", "exclusive or", "exclusive-or"):
        return "xor"
    if has(" nand "):
        return "nand"
    if has(" nor "):
        return "nor"
    if has("saturat"):
        return "saturate"
    if has("mux", "multiplex", "select"):
        return "mux"
    if has("shift"):
        return "shift"
    if has("subtract", " minus "):
        return "sub"
    if has(" add ", " sum ", "addition", "adder"):
        return "add"
    if has(" and "):
        return "and"
    if has(" or "):
        return "or"
    if has("invert", "complement", " not "):
        return "not"
    return None


# ---------------------------------------------------------------------------
# FSM graph extraction from contract rules (best effort, spec-only)
# ---------------------------------------------------------------------------

def _extract_fsm(rules: List[dict], states: List[str]) -> Dict[str, Any]:
    """Return {'edges': [(src, guard_inputs:{n:bit}, dst)], 'reset_state': str|None}."""
    edges: List[Tuple[str, Dict[str, str], str]] = []
    reset_state = None
    state_set = set(states or [])
    for r in rules:
        cond = (r.get("condition") or "") + " " + (r.get("description") or "")
        eff = (r.get("effect") or "") + " " + (r.get("description") or "")
        mdst = re.search(r"next_state\s*=\s*([A-Za-z_]\w*)", eff) or \
            re.search(r"(?:goes to|-->|transitions? to|next state is)\s*([A-Za-z_]\w*)", eff, re.I)
        msrc = re.search(r"state\s*==\s*([A-Za-z_]\w*)", cond) or \
            re.search(r"(?:in|from)\s+(?:state\s+)?([A-Za-z_]\w*)", cond, re.I)
        if re.search(r"\breset\b", cond, re.I) and mdst and mdst.group(1) in state_set:
            reset_state = mdst.group(1)
            continue
        if mdst and mdst.group(1) in state_set and msrc and msrc.group(1) in state_set:
            guard = {n: v for n, v in re.findall(r"([A-Za-z_]\w*)\s*==\s*([01]+)", cond)
                     if n.lower() != "state"}
            edges.append((msrc.group(1), guard, mdst.group(1)))
    return {"edges": edges, "reset_state": reset_state}


def _bfs_paths(edges: List[Tuple[str, Dict[str, str], str]], start: str,
               states: List[str]) -> Dict[str, List[Tuple[str, Dict[str, str], str]]]:
    """Shortest edge-path from start to each reachable state."""
    from collections import deque
    adj: Dict[str, List[Tuple[str, Dict[str, str], str]]] = {}
    for e in edges:
        adj.setdefault(e[0], []).append(e)
    paths = {start: []}
    q = deque([start])
    while q:
        cur = q.popleft()
        for e in adj.get(cur, []):
            _, _, dst = e
            if dst not in paths:
                paths[dst] = paths[cur] + [e]
                q.append(dst)
    return paths


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

class WitnessGenerator:
    EXHAUSTIVE_MAX_BITS = 8

    def generate(self, contract: dict, existing: Optional[List[dict]] = None) -> Dict[str, Any]:
        inputs = contract.get("inputs", {}) if isinstance(contract.get("inputs"), dict) else {}
        outputs = contract.get("outputs", {}) if isinstance(contract.get("outputs"), dict) else {}
        rules = contract.get("rules", []) if isinstance(contract.get("rules"), list) else []
        dtype = contract.get("design_type", "combinational")
        in_names = list(inputs.keys())
        out_names = list(outputs.keys())
        in_w = {n: _width_of(inputs[n]) for n in in_names}
        in_signed = {n: _signed_of(inputs[n]) for n in in_names}
        out_w = {n: _width_of(outputs[n]) for n in out_names}
        rule_ids = [r.get("id") for r in rules if r.get("id")]
        exprs = _output_exprs(rules, out_names)

        wl: List[dict] = []
        counters = {"control_total": 0, "control_covered": 0,
                    "boundary_total": 0, "boundary_covered": 0,
                    "distinguishing_total": 0, "distinguishing_covered": 0,
                    "states_total": len(contract.get("states", []) or []), "states_covered": 0,
                    "transitions_total": 0, "transitions_covered": 0}
        self._ctr = counters

        def expected_for(env: Dict[str, int]) -> Dict[str, str]:
            out = {}
            for o in out_names:
                if o in exprs:
                    bits = eval_effect(o + " = " + exprs[o], env, out_w[o])
                    if bits is not None:
                        out[o] = bits
            return out

        if dtype == "sequential":
            self._sequential(contract, wl, in_names, in_w, out_names, rules, rule_ids)
        else:
            self._combinational(wl, in_names, in_w, in_signed, out_names, out_w,
                                 rules, rule_ids, exprs, expected_for)

        # merge externally-provided witnesses (e.g. SpecKit LLM proposals), dedup
        merged = self._merge(existing or [], wl)
        return {"witnesses": merged, "completeness": self._summary(contract, counters)}

    # -- combinational -----------------------------------------------------

    def _combinational(self, wl, in_names, in_w, in_signed, out_names, out_w,
                       rules, rule_ids, exprs, expected_for) -> None:
        total_bits = sum(in_w.values())
        # positive: exhaustive if small, else per-input boundary sweep
        if in_names and total_bits <= self.EXHAUSTIVE_MAX_BITS:
            for combo in range(1 << total_bits):
                env, inp = self._decode_combo(combo, in_names, in_w)
                exp = expected_for(env)
                wl.append(self._w("positive", "Exhaustive input combination.",
                                  rule_ids, inputs=inp, expected=exp,
                                  reason="Small input space fully enumerated to pin behavior.",
                                  required=True))
        else:
            base = {n: 0 for n in in_names}
            for n in in_names:
                for label, val in boundary_values(in_w[n], in_signed[n]):
                    self._ctr["boundary_total"] += 1
                    env = dict(base); env[n] = val
                    inp = {k: _bits(v, in_w[k]) for k, v in env.items()}
                    exp = expected_for(env)
                    self._ctr["boundary_covered"] += 1
                    wl.append(self._w("boundary", "Boundary %s on %s." % (label, n),
                                      rule_ids, inputs=inp, expected=exp,
                                      reason="Boundary value class %s exercises edge behavior of %s." % (label, n),
                                      required=(label in ("zero", "max"))))
            # all-zero / all-one vectors as positive+negative anchors
            for lab, fill in (("all_zero", 0), ("all_one", 1)):
                env = {n: (((1 << in_w[n]) - 1) if fill else 0) for n in in_names}
                inp = {k: _bits(v, in_w[k]) for k, v in env.items()}
                wl.append(self._w("positive" if fill else "negative",
                                  "%s input vector." % lab, rule_ids, inputs=inp,
                                  expected=expected_for(env),
                                  reason="Anchor vector for global behavior.", required=True))

        # distinguishing witnesses from detected operation
        self._distinguishing(wl, in_names, in_w, out_names, out_w, rules, rule_ids, expected_for)
        # control-mode enumeration
        self._control_modes(wl, in_names, in_w, out_names, rules, rule_ids, expected_for)
        # output-class witnesses (each distinct expected output value, if evaluable)
        self._output_classes(wl, in_names, in_w, out_names, out_w, exprs, rule_ids, expected_for)
        # metamorphic: commutativity for symmetric binary ops
        self._metamorphic_comb(wl, in_names, in_w, out_names, rules, rule_ids)

    def _decode_combo(self, combo: int, in_names: List[str], in_w: Dict[str, int]):
        env, inp, shift = {}, {}, 0
        for n in reversed(in_names):
            w = in_w[n]
            v = (combo >> shift) & ((1 << w) - 1)
            env[n] = v
            inp[n] = _bits(v, w)
            shift += w
        return env, {n: inp[n] for n in in_names}

    def _distinguishing(self, wl, in_names, in_w, out_names, out_w, rules, rule_ids, expected_for) -> None:
        op = _detect_operation(rules)
        if not op or len(in_names) < 1:
            return
        two = in_names[:2]
        confusion = {
            "and": [("01", ["and", "or"]), ("10", ["and", "or"])],
            "or": [("01", ["or", "and"]), ("10", ["or", "and"])],
            "xor": [("11", ["xor", "or"]), ("01", ["xor", "and"]), ("10", ["xor", "and"])],
            "nand": [("01", ["nand", "nor"]), ("10", ["nand", "nor"])],
            "nor": [("01", ["nor", "nand"]), ("10", ["nor", "nand"])],
            "not": [("0", ["not", "identity"]), ("1", ["not", "identity"])],
        }
        if op in confusion and len(two) >= (1 if op == "not" else 2):
            for pattern, distinguishes in confusion[op]:
                self._ctr["distinguishing_total"] += 1
                env, inp = {}, {}
                for i, n in enumerate(two[:len(pattern)]):
                    bit = int(pattern[i]) if i < len(pattern) else 0
                    env[n] = bit * ((1 << in_w[n]) - 1) if in_w[n] > 1 else bit
                    inp[n] = _bits(env[n], in_w[n])
                for n in in_names:
                    if n not in inp:
                        env[n] = 0; inp[n] = _bits(0, in_w[n])
                self._ctr["distinguishing_covered"] += 1
                wl.append(self._w("distinguishing",
                                  "Separate %s from %s using pattern %s." % (op, distinguishes[1], pattern),
                                  rule_ids, inputs=inp, expected=expected_for(env),
                                  reason="%s and %s agree on vacuous rows; this row distinguishes them."
                                  % (distinguishes[0], distinguishes[1]), required=True))
        elif op in ("add", "sub"):
            self._ctr["distinguishing_total"] += 1
            env = {n: 0 for n in in_names}
            if len(two) >= 2:
                env[two[0]] = min(3, (1 << in_w[two[0]]) - 1)
                env[two[1]] = 1
            inp = {n: _bits(env[n], in_w[n]) for n in in_names}
            self._ctr["distinguishing_covered"] += 1
            wl.append(self._w("distinguishing", "Asymmetric operands separate + from -.",
                              rule_ids, inputs=inp, expected=expected_for(env),
                              reason="a=3,b=1 gives 4 vs 2; symmetric operands would not distinguish.",
                              required=True))
        elif op in ("compare", "compare_eq") and len(two) >= 2:
            for label, (x, y) in (("equal", (2, 2)), ("less", (1, 2)), ("greater", (2, 1))):
                self._ctr["distinguishing_total"] += 1
                env = {n: 0 for n in in_names}
                env[two[0]] = min(x, (1 << in_w[two[0]]) - 1)
                env[two[1]] = min(y, (1 << in_w[two[1]]) - 1)
                inp = {n: _bits(env[n], in_w[n]) for n in in_names}
                self._ctr["distinguishing_covered"] += 1
                wl.append(self._w("distinguishing", "Compare %s case (separates < from <=)." % label,
                                  rule_ids, inputs=inp, expected=expected_for(env),
                                  reason="Equality case distinguishes strict vs non-strict comparison.",
                                  required=(label == "equal")))
        elif op == "shift" and in_names:
            data = two[0]
            amt = two[1] if len(two) >= 2 else None
            for label, aval in (("shift_0", 0), ("shift_1", 1), ("shift_max", in_w[data] - 1)):
                self._ctr["distinguishing_total"] += 1
                env = {n: 0 for n in in_names}
                env[data] = 1  # one-hot lsb reveals shift direction/amount
                if amt is not None:
                    env[amt] = min(aval, (1 << in_w[amt]) - 1)
                inp = {n: _bits(env[n], in_w[n]) for n in in_names}
                self._ctr["distinguishing_covered"] += 1
                wl.append(self._w("distinguishing", "Shift by %s on one-hot input." % label,
                                  rule_ids, inputs=inp, expected=expected_for(env),
                                  reason="One-hot input under varied shift amount exposes off-by-one/direction bugs.",
                                  required=(label != "shift_max")))

    def _control_modes(self, wl, in_names, in_w, out_names, rules, rule_ids, expected_for) -> None:
        controls = [n for n in in_names if _is_control(n)]
        data = [n for n in in_names if n not in controls]
        for c in controls:
            w = in_w[c]
            if w > 4:      # too wide to enumerate; sample 0 and max only
                vals = [0, (1 << w) - 1]
            else:
                vals = list(range(1 << w))
            self._ctr["control_total"] += (1 << w) if w <= 4 else 2
            for v in vals:
                self._ctr["control_covered"] += 1
                env = {n: 0 for n in in_names}
                env[c] = v
                # make data inputs distinct so a select bug is observable
                for i, d in enumerate(data):
                    env[d] = ((1 << in_w[d]) - 1) if i % 2 else 1
                inp = {n: _bits(env[n], in_w[n]) for n in in_names}
                wl.append(self._w("control_mode", "Control %s = %s." % (c, _bits(v, w)),
                                  rule_ids, inputs=inp, expected=expected_for(env),
                                  reason="Every value of control %s must be exercised (mux/enable/mode bugs)." % c,
                                  required=True))
        # unselected-input sensitivity: for a mux, changing the unselected input must not move output
        if any(_is_control(n) and n.lower().startswith("sel") for n in in_names) and len(data) >= 2:
            wl.append(self._w("metamorphic", "Unselected input must not affect output.",
                              rule_ids, relation="mux_independence",
                              params={"select": next(n for n in in_names if n.lower().startswith("sel")),
                                      "data": data},
                              reason="Changing the unselected data input while sel is fixed must leave output unchanged.",
                              required=True))

    def _output_classes(self, wl, in_names, in_w, out_names, out_w, exprs, rule_ids, expected_for) -> None:
        if not exprs or sum(in_w.values()) > 12:
            return
        seen: Dict[str, dict] = {}
        for combo in range(min(1 << sum(in_w.values()), 4096)):
            env, inp = self._decode_combo(combo, in_names, in_w)
            exp = expected_for(env)
            key = json.dumps(exp, sort_keys=True)
            if exp and key not in seen:
                seen[key] = inp
        for key, inp in list(seen.items())[:64]:
            env = {n: int(inp[n], 2) for n in in_names}
            wl.append(self._w("output_class", "Reach output class %s." % key,
                              rule_ids, inputs=inp, expected=expected_for(env),
                              reason="Each distinct output value class should be reachable.",
                              required=False))

    def _metamorphic_comb(self, wl, in_names, in_w, out_names, rules, rule_ids) -> None:
        op = _detect_operation(rules)
        if op in ("and", "or", "xor", "add") and len(in_names) >= 2:
            a, b = in_names[0], in_names[1]
            sample = {n: _bits(1 if n in (a, b) else 0, in_w[n]) for n in in_names}
            sample[a] = _bits(min(3, (1 << in_w[a]) - 1), in_w[a])
            sample[b] = _bits(min(1, (1 << in_w[b]) - 1), in_w[b])
            wl.append(self._w("metamorphic", "%s is commutative: f(a,b)==f(b,a)." % op,
                              rule_ids, relation="commutative", params={"operands": [a, b], "inputs": sample},
                              reason="Swapping operands must not change output for a commutative op.",
                              required=False))

    # -- sequential --------------------------------------------------------

    def _sequential(self, contract, wl, in_names, in_w, out_names, rules, rule_ids) -> None:
        states = contract.get("states", []) or []
        reset = contract.get("reset", {}) if isinstance(contract.get("reset"), dict) else {}
        reset_name = reset.get("name") or next((n for n in in_names if _is_control(n) and "rst" in n.lower()
                                                or n.lower() in ("reset", "rst")), None)
        active_high = reset.get("active_high", True)
        rlo = "1" if active_high else "0"
        rhi = "0" if active_high else "1"
        init = contract.get("initial_state") or (states[0] if states else None)
        fsm = _extract_fsm(rules, states)
        if fsm["reset_state"]:
            init = fsm["reset_state"]
        self._ctr["transitions_total"] = len(fsm["edges"])

        def cyc(**kw):
            row = {n: _bits(0, in_w.get(n, 1)) for n in in_names}
            for k, v in kw.items():
                if k in in_names:
                    row[k] = _bits(int(v, 2) if isinstance(v, str) else v, in_w.get(k, 1)) \
                        if not (isinstance(v, str) and set(v) <= set("01")) else v
            return row

        # 1) reset witness
        if reset_name:
            seq = [{"cycle": 0, "inputs": cyc(**{reset_name: rlo})},
                   {"cycle": 1, "inputs": cyc(**{reset_name: rhi})}]
            wl.append(self._w("reset_hold_enable", "Assert reset places FSM in initial state.",
                              rule_ids, sequence=seq, expected_final=({"state": init} if init else None),
                              reason="Reset behavior cannot be tested with a single vector.", required=True))

        # 2) reachability + transition witnesses via BFS
        if init and fsm["edges"]:
            paths = _bfs_paths(fsm["edges"], init, states)
            reached = set()
            for st, path in paths.items():
                seq = []
                c = 0
                if reset_name:
                    seq.append({"cycle": c, "inputs": cyc(**{reset_name: rlo})}); c += 1
                    seq.append({"cycle": c, "inputs": cyc(**{reset_name: rhi})}); c += 1
                for (_src, guard, _dst) in path:
                    seq.append({"cycle": c, "inputs": cyc(**guard)}); c += 1
                reached.add(st)
                wl.append(self._w("sequential_reachability", "Reach state %s from reset." % st,
                                  rule_ids, sequence=seq, expected_final={"state": st},
                                  reason="Proves the path to %s is implemented (single vectors cannot)." % st,
                                  required=(st != init)))
            self._ctr["states_covered"] = len(reached)
            # per-edge transition witnesses
            for (src, guard, dst) in fsm["edges"]:
                path = paths.get(src)
                if path is None:
                    continue
                seq, c = [], 0
                if reset_name:
                    seq.append({"cycle": c, "inputs": cyc(**{reset_name: rlo})}); c += 1
                    seq.append({"cycle": c, "inputs": cyc(**{reset_name: rhi})}); c += 1
                for (_s, g, _d) in path:
                    seq.append({"cycle": c, "inputs": cyc(**g)}); c += 1
                seq.append({"cycle": c, "inputs": cyc(**guard)})
                self._ctr["transitions_covered"] += 1
                wl.append(self._w("state_transition", "Transition %s --%s--> %s." % (src, guard, dst),
                                  rule_ids, sequence=seq, expected_final={"state": dst},
                                  reason="Exercises a specific edge with its guard input.", required=True))
        elif init:
            self._ctr["states_covered"] = 1

        # 3) enable-hold and reset-idempotence metamorphic
        en = next((n for n in in_names if n.lower() in ("enable", "en") or n.lower().endswith("_en")), None)
        if en:
            wl.append(self._w("metamorphic", "Enable low holds state/output stable.",
                              rule_ids, relation="enable_hold", params={"enable": en},
                              reason="With enable=0, changing data must not change state/output.", required=True))
        if reset_name:
            wl.append(self._w("metamorphic", "Reset is idempotent.",
                              rule_ids, relation="reset_idempotence", params={"reset": reset_name, "active_low": rlo},
                              reason="Applying reset twice yields the same reset state.", required=False))

    # -- assembly ----------------------------------------------------------

    def _w(self, wtype: str, description: str, covers_rules, inputs=None, expected=None,
           sequence=None, expected_final=None, relation=None, params=None,
           reason="", required=False) -> dict:
        self._n = getattr(self, "_n", 0) + 1
        w = {"id": "W%d" % self._n, "type": wtype, "description": description,
             "covers_rules": list(covers_rules) if covers_rules else [],
             "reason": reason, "required": bool(required)}
        if inputs is not None:
            w["inputs"] = inputs
        if expected:
            w["expected_outputs"] = expected
        if sequence is not None:
            w["sequence"] = sequence
        if expected_final is not None:
            w["expected_final"] = expected_final
        if relation is not None:
            w["relation"] = relation
        if params is not None:
            w["params"] = params
        return w

    def _merge(self, existing: List[dict], generated: List[dict]) -> List[dict]:
        out, seen = [], set()

        def key(w):
            return json.dumps({"t": w.get("type"), "i": w.get("inputs"),
                               "s": w.get("sequence"), "r": w.get("relation")}, sort_keys=True)
        for w in list(existing) + generated:
            if not isinstance(w, dict):
                continue
            k = key(w)
            if k in seen:
                continue
            seen.add(k)
            out.append(w)
        # renumber ids stably
        for i, w in enumerate(out, 1):
            w["id"] = w.get("id") if isinstance(w.get("id"), str) and w["id"].startswith("W") and w not in generated else "W%d" % i
            w["id"] = "W%d" % i
        return out

    def _summary(self, contract: dict, c: dict) -> Dict[str, str]:
        modes = len(contract.get("operation_modes", []) or [])
        def frac(cov, tot): return "%d/%d" % (cov, tot)
        return {
            "operation_modes": frac(modes if c["control_covered"] else 0, modes),
            "control_modes": frac(c["control_covered"], max(c["control_total"], c["control_covered"])),
            "boundary_classes": frac(c["boundary_covered"], max(c["boundary_total"], c["boundary_covered"])),
            "states": frac(c["states_covered"], c["states_total"]),
            "transitions": frac(c["transitions_covered"], c["transitions_total"]),
            "distinguishing_cases": frac(c["distinguishing_covered"],
                                         max(c["distinguishing_total"], c["distinguishing_covered"])),
        }

    # -- io ----------------------------------------------------------------

    def run(self, contract_path: str, out_path: Optional[str] = None,
            existing_witnesses_path: Optional[str] = None) -> Dict[str, Any]:
        with open(contract_path) as f:
            contract = json.load(f)
        existing = None
        if existing_witnesses_path and os.path.exists(existing_witnesses_path):
            try:
                with open(existing_witnesses_path) as f:
                    ex = json.load(f)
                existing = ex.get("witnesses", ex) if isinstance(ex, (dict, list)) else None
                if isinstance(existing, dict):
                    existing = existing.get("witnesses")
            except Exception:
                existing = None
        result = self.generate(contract, existing=existing)
        out_path = out_path or os.path.join(os.path.dirname(contract_path) or ".", "witnesses.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        result["path"] = out_path
        return result


def generate_witnesses(contract: dict, existing: Optional[List[dict]] = None) -> Dict[str, Any]:
    return WitnessGenerator().generate(contract, existing=existing)


# ---------------------------------------------------------------------------
# self-test (no LLM, no RTL)
# ---------------------------------------------------------------------------

def _selftest() -> int:
    ok = True

    # AND gate: expect all 4 combos + 01/10 distinguishing with correct expected
    and_c = {"design_name": "and_gate", "design_type": "combinational",
             "inputs": {"a": {"width": 1}, "b": {"width": 1}}, "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "effect": "y = a & b", "description": "y is a AND b"}]}
    r = generate_witnesses(and_c)
    ws = r["witnesses"]
    pos = {(w["inputs"]["a"], w["inputs"]["b"]): w["expected_outputs"]["y"]
           for w in ws if w["type"] == "positive" and "expected_outputs" in w}
    ok = ok and pos.get(("0", "0")) == "0" and pos.get(("0", "1")) == "0"
    ok = ok and pos.get(("1", "0")) == "0" and pos.get(("1", "1")) == "1"
    dist = [w for w in ws if w["type"] == "distinguishing"]
    ok = ok and any(w.get("inputs") == {"a": "0", "b": "1"} for w in dist)
    ok = ok and any(w.get("inputs") == {"a": "1", "b": "0"} for w in dist)
    # distinguishing expected must match AND (01->0), not OR
    d01 = next((w for w in dist if w.get("inputs") == {"a": "0", "b": "1"}), None)
    ok = ok and d01 and d01["expected_outputs"]["y"] == "0"

    # OR gate: 01 and 10 should be 1
    or_c = json.loads(json.dumps(and_c))
    or_c["rules"] = [{"id": "R1", "effect": "y = a | b", "description": "y is a OR b"}]
    ro = generate_witnesses(or_c)
    poso = {(w["inputs"]["a"], w["inputs"]["b"]): w["expected_outputs"]["y"]
            for w in ro["witnesses"] if w["type"] == "positive" and "expected_outputs" in w}
    ok = ok and poso.get(("0", "1")) == "1" and poso.get(("1", "0")) == "1" and poso.get(("0", "0")) == "0"

    # mux: control enumeration hits every sel value + independence metamorphic
    mux_c = {"design_name": "mux2", "design_type": "combinational",
             "inputs": {"a": {"width": 1}, "b": {"width": 1}, "sel": {"width": 1}},
             "outputs": {"y": {"width": 1}},
             "rules": [{"id": "R1", "condition": "sel == 0", "effect": "y = a"},
                       {"id": "R2", "condition": "sel == 1", "effect": "y = b"}]}
    rm = generate_witnesses(mux_c)
    sels = {w["inputs"]["sel"] for w in rm["witnesses"] if w["type"] == "control_mode"}
    ok = ok and sels == {"0", "1"}
    ok = ok and any(w.get("relation") == "mux_independence" for w in rm["witnesses"])

    # FSM: reach DONE via BFS, transition witnesses, completeness states covered
    fsm_c = {"design_name": "ctrl", "design_type": "sequential",
             "inputs": {"reset": {"width": 1}, "start": {"width": 1}, "valid": {"width": 1}},
             "outputs": {"done": {"width": 1}},
             "reset": {"name": "reset", "active_high": True},
             "states": ["IDLE", "LOAD", "DONE"], "initial_state": "IDLE",
             "rules": [
                 {"id": "R1", "description": "reset goes to IDLE", "condition": "reset", "effect": "next_state = IDLE"},
                 {"id": "R2", "condition": "state == IDLE and start == 1", "effect": "next_state = LOAD"},
                 {"id": "R3", "condition": "state == LOAD and valid == 1", "effect": "next_state = DONE"}]}
    rf = generate_witnesses(fsm_c)
    reach = [w for w in rf["witnesses"] if w["type"] == "sequential_reachability"]
    ok = ok and any(w.get("expected_final", {}).get("state") == "DONE" for w in reach)
    ok = ok and any(w["type"] == "state_transition" for w in rf["witnesses"])
    ok = ok and rf["completeness"]["states"] == "3/3"
    ok = ok and rf["completeness"]["transitions"].endswith("/2")

    # merge dedups
    merged = WitnessGenerator()._merge(
        [{"id": "X", "type": "positive", "inputs": {"a": "0"}}],
        [{"id": "W1", "type": "positive", "inputs": {"a": "0"}}])
    ok = ok and len(merged) == 1

    print("witness_generator selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        res = WitnessGenerator().run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print("wrote", res["path"], "-", len(res["witnesses"]), "witnesses", res["completeness"])
    else:
        raise SystemExit(_selftest())
