#!/usr/bin/env python3
"""Deploy-safe contract extraction for Pro-V tasks.

This module only uses public task text and the module header. It does not read
or derive behavior from benchmark RTL internals.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _strip_comments(text: str) -> str:
    text = re.sub(r"//.*", "", text or "")
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def _parse_ports(header: str) -> Dict[str, Dict[str, int]]:
    try:
        from pro_v.mutation_strength import parse_ports
    except Exception:
        try:
            from mutation_strength import parse_ports  # type: ignore
        except Exception:
            parse_ports = None  # type: ignore
    if parse_ports is not None:
        parsed = parse_ports(header or "")
        return {
            "inputs": {name: int(width) for name, width in parsed.inputs + parsed.clock_inputs},
            "outputs": {name: int(width) for name, width in parsed.outputs},
        }

    ports = {"inputs": {}, "outputs": {}}
    header = _strip_comments(header)
    tokens = re.split(r"\b(input|output)\b", header or "", flags=re.I)
    for i in range(1, len(tokens), 2):
        direction = tokens[i].lower()
        segment = tokens[i + 1] if i + 1 < len(tokens) else ""
        segment = re.split(r"\b(?:input|output|inout|endmodule|module)\b", segment, maxsplit=1, flags=re.I)[0]
        segment = segment.split(";")[0].split(")")[0]
        width_match = re.search(r"\[[^\]]+\]", segment)
        width = _width_from_range(width_match.group(0) if width_match else "")
        segment = re.sub(r"\[[^\]]+\]", " ", segment)
        segment = re.sub(r"\b(?:wire|reg|logic|signed|unsigned)\b", " ", segment, flags=re.I)
        for raw in segment.split(","):
            match = re.search(r"([A-Za-z_][A-Za-z0-9_$]*)", raw)
            if match:
                ports[direction + "s"][match.group(1)] = width
    return ports


def _width_from_range(width_blob: str) -> int:
    if not width_blob:
        return 1
    match = re.search(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", width_blob)
    if not match:
        return 1
    a, b = int(match.group(1)), int(match.group(2))
    return abs(a - b) + 1


def _is_clock_signal(name: str) -> bool:
    lower = (name or "").lower()
    return lower in {"clk", "clock"} or lower.startswith("clk") or lower.endswith("_clk") or "clock" in lower


def _is_reset_signal(name: str) -> bool:
    lower = (name or "").lower()
    explicit = {
        "reset", "rst", "areset", "arst", "sreset", "srst", "clear", "clr",
        "rst_n", "resetn", "aresetn", "arstn", "srstn", "nreset", "nrst",
        "rstb", "reset_b", "reset_l",
    }
    return lower in explicit or "reset" in lower or bool(re.fullmatch(r"[abs]?rstn?", lower))


def classify_public_circuit(description: str, header: str) -> str:
    """Classify from public spec/header without treating every FSM text as seq.

    HDLBits often asks for combinational next-state equations from an FSM table.
    Those specs mention "state machine"/"FSM" but have no clock port and should
    use the combinational harness.
    """
    public = f"{description or ''}\n{header or ''}"
    header_clean = _strip_comments(header)
    lower = public.lower()
    ports = _parse_ports(header)
    has_clock_port = any(_is_clock_signal(name) for name in ports["inputs"])
    explicit_clock_text = bool(re.search(r"\b(posedge|negedge|clocked|on each clock|rising edge|falling edge)\b", lower))
    next_state_equation_only = bool(re.search(
        r"\b(next-state|next state|derive (?:the )?(?:logic|equation)|logic equations?|"
        r"input of (?:state )?flip-flop|implement just the next-state logic|combinational logic)\b",
        lower,
    ))
    if next_state_equation_only and not has_clock_port:
        return "cmb"
    if re.search(r"\b(?:latch|level-sensitive|level sensitive|transparent)\b", lower):
        return "seq" if has_clock_port or explicit_clock_text else "cmb"
    if has_clock_port or explicit_clock_text:
        return "seq"
    if re.search(r"\b(counter|shift register|lfsr|flip-flop|register)\b", lower):
        return "seq"
    return "cmb"


def build_contract(description: str, header: str) -> Dict[str, Any]:
    ports = _parse_ports(header)
    circuit_type = classify_public_circuit(description, header)
    lower = f"{description or ''}\n{header or ''}".lower()

    resets: Dict[str, Dict[str, Any]] = {}
    for name in ports["inputs"]:
        lname = name.lower()
        if _is_reset_signal(name):
            name_pattern = re.escape(lname)
            public_edge_mentions_reset = bool(
                re.search(rf"\b(?:posedge|negedge|rising edge|falling edge|positive edge|negative edge)\b[^\n.;]{{0,120}}\b{name_pattern}\b", lower)
                or re.search(rf"\b{name_pattern}\b[^\n.;]{{0,120}}\b(?:posedge|negedge|rising edge|falling edge|positive edge|negative edge)\b", lower)
            )
            resets[name] = {
                "active_low": lname.endswith("n") or lname.endswith("_n") or "active low" in lower or "active-low" in lower,
                "kind": "async" if lname.startswith("a") or "asynchronous" in lower or public_edge_mentions_reset else "sync",
            }

    families: List[str] = []
    for family, terms in {
        "fsm": ("fsm", "state machine", "one-hot", "one hot", "next-state", "next state"),
        "mux": ("mux", "multiplexer", "select"),
        "arithmetic": ("add", "subtract", "sum", "carry", "signed", "two's complement"),
        "decoder_encoder": ("decoder", "encoder", "priority", "one-hot", "one hot"),
        "truth_table_waveform": ("truth table", "waveform", "scancode"),
        "counter": ("counter", "count"),
        "shift_register": ("shift register", "shift"),
        "lfsr": ("lfsr", "linear feedback"),
        "latch": ("latch", "level-sensitive", "level sensitive", "transparent"),
    }.items():
        if any(term in lower for term in terms):
            families.append(family)

    output_slices = [
        {"output": name, "width": width, "contract_focus": _output_focus(name, description)}
        for name, width in ports["outputs"].items()
    ]

    return {
        "schema": "pro_v.contract.v1",
        "circuit_type": circuit_type,
        "ports": ports,
        "resets": resets,
        "families": sorted(set(families)),
        "output_slices": output_slices,
        "public_examples": extract_public_examples(description, ports),
        "temporal_traces": build_temporal_traces(circuit_type, ports, resets, families),
        "requirements": {
            "no_hidden_rtl": True,
            "all_outputs_every_call": True,
            "no_empty_expected_outputs": True,
            "output_locals_initialized": circuit_type == "seq",
        },
    }


def _output_focus(output: str, description: str) -> str:
    pattern = re.compile(rf"\b{re.escape(output)}\b[^.\n;]*", flags=re.I)
    matches = pattern.findall(description or "")
    return matches[0][:200] if matches else ""


def extract_public_examples(description: str, ports: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    """Extract explicit input/output examples from public prose/tables.

    Keep this intentionally narrow. These are used as public contract evidence,
    not as benchmark-RTL-derived expected outputs.
    """
    examples: List[Dict[str, Any]] = []
    inputs = ports.get("inputs", {})
    outputs = ports.get("outputs", {})
    text = description or ""

    # HDLBits-style scancode table:
    # 16'he06b | left arrow
    if "scancode" in inputs:
        arrow_outputs = {name.lower(): name for name in outputs}
        for literal, label in re.findall(r"16\s*'\s*h\s*([0-9a-fA-F]+)\s*\|\s*([A-Za-z ]+)", text):
            label_lower = label.lower()
            expected = {name: "0" * width for name, width in outputs.items()}
            for key, out_name in arrow_outputs.items():
                if key in label_lower:
                    expected[out_name] = "1".zfill(outputs[out_name])
            value = int(literal, 16)
            examples.append({
                "kind": "explicit_table_row",
                "inputs": {"scancode": format(value, f"0{inputs['scancode']}b")},
                "expected_outputs": expected,
                "source": f"16'h{literal} | {label.strip()}",
            })
        if examples and any("anything else" in line.lower() for line in text.splitlines()):
            expected = {name: "0" * width for name, width in outputs.items()}
            examples.append({
                "kind": "explicit_default_row",
                "inputs": {"scancode": "0" * inputs["scancode"]},
                "expected_outputs": expected,
                "source": "Anything else | none",
            })

    # Publicly specified priority encoder: position of the first high bit, with
    # zero output when no bits are high. Enumerate only small tables from the
    # public header/spec, never from benchmark RTL.
    lower = text.lower()
    if (
        "priority encoder" in lower
        and re.search(r"\bfirst\s+1\b|\bfirst bit\b|\bfirst\s+bit", lower)
        and ("if none" in lower or "input is zero" in lower or "none of the input bits" in lower)
        and len(inputs) == 1
        and len(outputs) == 1
    ):
        in_name, in_width = next(iter(inputs.items()))
        out_name, out_width = next(iter(outputs.items()))
        if in_width <= 8 and out_width <= 8:
            for value in range(1 << in_width):
                pos = 0
                for bit_idx in range(in_width):
                    if value & (1 << bit_idx):
                        pos = bit_idx
                        break
                examples.append({
                    "kind": "derived_priority_encoder_row",
                    "inputs": {in_name: format(value, f"0{in_width}b")},
                    "expected_outputs": {out_name: format(pos, f"0{out_width}b")},
                    "source": "public priority-encoder spec: first high bit index, zero when none",
                })

    # HDLBits next-state equation tasks often give an FSM as public arrow rows
    # and ask for Y<n> inputs to one-hot state flip-flops. Derive those output
    # bits directly from the written transition table.
    if (
        "one-hot" in lower or "one hot" in lower
    ) and any(re.fullmatch(r"Y\d+", name) for name in outputs):
        state_vectors: Dict[str, str] = {}
        assign_match = re.search(
            r"state\s+assignment\s+[^=]*=\s*([^.\n]+)",
            text,
            flags=re.I,
        )
        if assign_match:
            for bits, state in re.findall(
                r"([01]+)\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)",
                assign_match.group(1),
            ):
                state_vectors[state] = bits
        transitions = []
        for src, inp, dst in re.findall(
            r"\b([A-Za-z][A-Za-z0-9_]*)\s*\([^)]*\)\s*--\s*([01])\s*-->\s*([A-Za-z][A-Za-z0-9_]*)",
            text,
        ):
            if src in state_vectors and dst in state_vectors:
                transitions.append((src, inp, dst))
        one_bit_inputs = [name for name, width in inputs.items() if width == 1]
        state_inputs = [(name, width) for name, width in inputs.items() if width == max(inputs.values() or [0])]
        if transitions and one_bit_inputs and state_inputs:
            control_name = one_bit_inputs[0]
            state_name, state_width = state_inputs[0]
            for src, inp, dst in transitions:
                dst_bits = state_vectors[dst]
                expected = {name: "0" * width for name, width in outputs.items()}
                for out_name, out_width in outputs.items():
                    match = re.fullmatch(r"Y(\d+)", out_name)
                    if not match:
                        continue
                    bit_idx = int(match.group(1))
                    if bit_idx < len(dst_bits):
                        expected[out_name] = dst_bits[-1 - bit_idx].zfill(out_width)
                examples.append({
                    "kind": "derived_onehot_fsm_transition",
                    "inputs": {
                        state_name: state_vectors[src].zfill(state_width),
                        control_name: inp,
                    },
                    "expected_outputs": expected,
                    "source": f"public one-hot transition {src} --{inp}--> {dst}",
                })

    return examples[:32]


def build_temporal_traces(circuit_type: str, ports: Dict[str, Dict[str, int]], resets: Dict[str, Dict[str, Any]], families: List[str]) -> List[Dict[str, Any]]:
    if circuit_type != "seq":
        return []
    traces: List[Dict[str, Any]] = []
    traces.append({"name": "steady_hold", "purpose": "hold/reset-free cycles keep outputs defined on rising and falling calls"})
    if resets:
        traces.append({"name": "reset_assert_release", "purpose": "assert reset, observe reset output, release reset, then apply normal input"})
        if any(info.get("kind") == "async" for info in resets.values()):
            traces.append({"name": "async_reset_low_clock", "purpose": "observe reset behavior before a rising clock edge"})
    if "counter" in families:
        traces.append({"name": "count_progression", "purpose": "exercise consecutive cycles and boundary wrap/saturate behavior"})
    if "fsm" in families:
        traces.append({"name": "state_transition_cover", "purpose": "cover named transition rows and output timing"})
    if "shift_register" in families or "lfsr" in families:
        traces.append({"name": "shift_enable_boundary", "purpose": "load/enable/shift behavior including disabled cycles"})
    if "latch" in families:
        traces.append({"name": "transparent_enable", "purpose": "when enable is asserted, output follows data without requiring a clock edge"})
        traces.append({"name": "disabled_hold", "purpose": "when enable is deasserted, output holds its previous value"})
    return traces


def contract_text(contract: Dict[str, Any]) -> str:
    return (
        "<verification_contract>\n"
        + json.dumps(contract, indent=2, sort_keys=True)
        + "\n</verification_contract>\n"
        "This contract is derived only from the public description and module header. "
        "Use it to choose circuit type, timing, reset semantics, output slicing, and "
        "required trace categories. If uncertain, preserve the explicit written spec.\n"
    )
