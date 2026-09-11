#!/usr/bin/env python3
"""Formal-check scaffolding for Pro-V contracts.

The functions here are intentionally conservative: they generate deploy-safe
property/check plans from the public contract, and only run external formal
tools when explicitly requested by callers.
"""
from __future__ import annotations

import shutil
from typing import Any, Dict, List


def tool_status() -> Dict[str, bool]:
    return {
        "yosys": shutil.which("yosys") is not None,
        "sby": shutil.which("sby") is not None,
        "verilator": shutil.which("verilator") is not None,
    }


def build_formal_plan(contract: Dict[str, Any]) -> Dict[str, Any]:
    ports = contract.get("ports", {})
    outputs = ports.get("outputs", {})
    inputs = ports.get("inputs", {})
    circuit_type = contract.get("circuit_type", "cmb")
    properties: List[Dict[str, Any]] = []

    for name, width in outputs.items():
        properties.append({
            "name": f"{name}_known_width",
            "kind": "sanity",
            "text": f"{name} is always a {width}-bit 0/1 value in the generated reference and testbench.",
        })

    if circuit_type == "cmb":
        properties.append({
            "name": "combinational_stability",
            "kind": "sva_template",
            "text": "same inputs imply same outputs; no hidden state is allowed in GoldenDUT.load(inputs)",
        })
    else:
        properties.append({
            "name": "outputs_defined_each_phase",
            "kind": "sva_template",
            "text": "pre-clock, rising-edge, and falling-edge observations all define every output",
        })
        for reset, info in (contract.get("resets") or {}).items():
            properties.append({
                "name": f"{reset}_reset_observable",
                "kind": "sva_template",
                "text": f"{reset} is {'active-low' if info.get('active_low') else 'active-high'} and {info.get('kind', 'sync')} per public contract",
            })

    return {
        "schema": "pro_v.formal_plan.v1",
        "tool_status": tool_status(),
        "properties": properties,
        "input_count": len(inputs),
        "output_count": len(outputs),
    }
