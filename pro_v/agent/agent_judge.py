"""Judge agent for selecting and editing Pro-V generated artifacts."""

import json
import os
import re
import ast
from typing import Any, Dict, List


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _unsafe_allow_hidden_rtl() -> bool:
    return (
        os.getenv("PRO_V_UNSAFE_ALLOW_GOLDEN_OUTPUTS", "0") == "1"
        and os.getenv("PRO_V_ALLOW_RTL_GUIDED_REPAIR", "0") == "1"
    )


def _load_json_file(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _has_real_expected_output(value: Any) -> bool:
    if isinstance(value, dict):
        if value and all(isinstance(v, str) and v and set(v) <= {"0", "1"} for v in value.values()):
            return True
        return any(_has_real_expected_output(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_real_expected_output(item) for item in value)
    return False


def _expected_bit_counts(value: Any) -> Dict[str, int]:
    """Count concrete vs masked expected-output bits in nested TB JSON."""
    counts = {"concrete": 0, "masked": 0, "illegal": 0}
    if isinstance(value, dict):
        for item in value.values():
            sub = _expected_bit_counts(item)
            for key, amount in sub.items():
                counts[key] += amount
    elif isinstance(value, list):
        for item in value:
            sub = _expected_bit_counts(item)
            for key, amount in sub.items():
                counts[key] += amount
    elif isinstance(value, str):
        for ch in value:
            lo = ch.lower()
            if lo in {"0", "1"}:
                counts["concrete"] += 1
            elif lo in {"x", "?"}:
                counts["masked"] += 1
            else:
                counts["illegal"] += 1
    return counts


def _normalize_cmb_stimulus(stimulus_data: Any) -> List[Dict[str, Any]]:
    if not isinstance(stimulus_data, list):
        return []
    normalized = []
    for vec in stimulus_data:
        if not isinstance(vec, dict):
            continue
        list_sigs = {k: v for k, v in vec.items() if k != "clock_cycles" and isinstance(v, list)}
        if list_sigs:
            n = max((len(v) for v in list_sigs.values()), default=0)
            scalars = {k: v for k, v in vec.items() if k != "clock_cycles" and not isinstance(v, list)}
            for i in range(n):
                row = dict(scalars)
                for k, v in list_sigs.items():
                    row[k] = v[i] if i < len(v) else (v[-1] if v else "0")
                normalized.append(row)
        else:
            normalized.append({k: v for k, v in vec.items() if k != "clock_cycles"})
    for row in normalized:
        for k, v in list(row.items()):
            if isinstance(v, bool):
                row[k] = "1" if v else "0"
            elif isinstance(v, int):
                row[k] = format(v, "b")
            else:
                s = str(v).strip()
                if s and all(bit in "01" for bit in s):
                    row[k] = s
                else:
                    try:
                        row[k] = format(int(s, 10) if s.isdigit() else int(s, 0), "b")
                    except Exception:
                        row[k] = "0"
    seen = set()
    unique = []
    for row in normalized:
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _testbench_shape_score(testbench_data: Any, stimulus_data: Any, circuit_type: str) -> Dict[str, Any]:
    score = 0.0
    reasons = []

    if not isinstance(testbench_data, list) or not testbench_data:
        return {"score": -1000.0, "reasons": ["testbench is not a non-empty list"]}

    score += 10.0
    tb_len = len(testbench_data)
    seq_mode = circuit_type.lower() == "seq"
    reference_stimulus = stimulus_data if seq_mode else _normalize_cmb_stimulus(stimulus_data)
    stim_len = len(reference_stimulus) if isinstance(reference_stimulus, list) else None
    if stim_len is not None:
        if (tb_len == stim_len) if seq_mode else (tb_len >= stim_len):
            score += 30.0
            reasons.append("length matches stimulus" if seq_mode else "length covers normalized combinational stimulus")
        else:
            return {
                "score": -1000.0,
                "reasons": [f"invalid shape: insufficient length testbench={tb_len} stimulus={stim_len}"],
                "length": tb_len,
                "total_cycles": 0,
                "invalid_shape": True,
            }

    real_outputs = 0
    total_cycles = 0
    bit_counts = {"concrete": 0, "masked": 0, "illegal": 0}
    for entry in testbench_data:
        if not isinstance(entry, dict):
            return {
                "score": -1000.0,
                "reasons": ["invalid shape: testbench entry is not a dict"],
                "length": tb_len,
                "total_cycles": total_cycles,
                "invalid_shape": True,
            }
        expected_value = entry.get("expected_outputs")
        if _has_real_expected_output(expected_value):
            real_outputs += 1
        entry_counts = _expected_bit_counts(expected_value)
        for key, amount in entry_counts.items():
            bit_counts[key] += amount
        if seq_mode:
            try:
                cycles = int(entry.get("clock_cycles", 0) or 0)
            except Exception:
                cycles = 0
            total_cycles += cycles
            expected = entry.get("expected_outputs")
            if cycles and isinstance(expected, list) and len(expected) != cycles:
                return {
                    "score": -1000.0,
                    "reasons": [
                        "invalid shape: sequential expected_outputs length "
                        f"{len(expected)} does not match clock_cycles {cycles}"
                    ],
                    "length": tb_len,
                    "total_cycles": total_cycles,
                    "invalid_shape": True,
                }

    if real_outputs:
        ratio = real_outputs / max(1, tb_len)
        score += 40.0 * ratio
        reasons.append(f"non-empty expected outputs {real_outputs}/{tb_len}")
    else:
        score -= 25.0
        reasons.append("expected_outputs empty for every entry")

    total_expected_bits = bit_counts["concrete"] + bit_counts["masked"]
    if bit_counts["illegal"]:
        score -= 100.0
        reasons.append(f"illegal expected-output bits {bit_counts['illegal']}")
    if total_expected_bits:
        masked_ratio = bit_counts["masked"] / total_expected_bits
        if masked_ratio > 0.75:
            score -= 60.0
            reasons.append(f"too many don't-care expected bits {masked_ratio:.1%}")
        elif masked_ratio > 0.35:
            score -= 20.0
            reasons.append(f"many don't-care expected bits {masked_ratio:.1%}")
        elif bit_counts["masked"]:
            reasons.append(f"don't-care expected bits {bit_counts['masked']}/{total_expected_bits}")

    seq_cycle_cap = int(os.environ.get("PRO_V_SEQ_TOTAL_CYCLE_HARD_MAX", "1024"))
    if seq_mode and total_cycles > seq_cycle_cap:
        score -= min(100.0, float(total_cycles - seq_cycle_cap) / 4.0)
        reasons.append(f"total cycles over cap: {total_cycles}>{seq_cycle_cap}")

    return {
        "score": score,
        "reasons": reasons,
        "length": tb_len,
        "total_cycles": total_cycles,
        "invalid_shape": False,
    }


def _extract_verilog_hex_literals(text: str) -> List[str]:
    literals = []
    for width, value in re.findall(r"\b(\d+)\s*'\s*[hH]\s*([0-9a-fA-F_xXzZ]+)", text):
        cleaned = value.replace("_", "")
        if re.search(r"[xXzZ]", cleaned):
            continue
        try:
            literals.append(hex(int(cleaned, 16)).lower())
        except ValueError:
            continue
    return literals


def _python_literal_score(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Small tie-breaker for preserving exact constants from the natural-language spec."""
    tb_path = sample.get("testbench_json_path")
    golden_path = sample.get("golden_dut_path")
    if not tb_path or not golden_path:
        return {"score": 0.0, "reasons": []}

    description_path = os.path.join(os.path.dirname(tb_path), "description.txt")
    try:
        with open(description_path, "r") as f:
            description = f.read()
        with open(golden_path, "r") as f:
            code = f.read()
    except OSError:
        return {"score": 0.0, "reasons": []}

    expected_literals = _extract_verilog_hex_literals(description)
    if not expected_literals:
        return {"score": 0.0, "reasons": []}

    code_lower = code.lower()
    present = sum(1 for literal in expected_literals if literal in code_lower)
    score = 3.0 * present / max(1, len(expected_literals))
    reasons = [f"preserves exact hex literals {present}/{len(expected_literals)}"]

    comment_mismatches = 0
    for number, comment in re.findall(r"==\s*(0b[01]+|0x[0-9a-fA-F]+|\d+)\s*:?.*?#\s*([^\n]+)", code):
        for literal in _extract_verilog_hex_literals(comment):
            try:
                actual = int(number, 0)
                expected = int(literal, 16)
            except ValueError:
                continue
            if actual != expected:
                comment_mismatches += 1
    if comment_mismatches:
        score -= 6.0 * comment_mismatches
        reasons.append(f"constant/comment mismatch {comment_mismatches}")

    return {"score": score, "reasons": reasons}


def _contract_score(sample: Dict[str, Any], circuit_type: str) -> Dict[str, Any]:
    """Score candidates against deploy-safe contract artifacts."""
    tb_path = sample.get("testbench_json_path")
    golden_path = sample.get("golden_dut_path")
    if not tb_path or not golden_path:
        return {"score": 0.0, "reasons": []}
    contract_path = os.path.join(os.path.dirname(tb_path), "behavior_contract.json")
    if not os.path.exists(contract_path):
        contract_path = os.path.join(os.path.dirname(tb_path), "verification_contract.json")
    try:
        contract = _load_json_file(contract_path)
        with open(golden_path, "r") as f:
            code = f.read()
        tb_data = _load_json_file(tb_path)
    except Exception:
        return {"score": 0.0, "reasons": []}

    score = 0.0
    reasons: List[str] = []
    outputs = ((contract.get("ports") or {}).get("outputs") or {})
    output_names = list(outputs)
    if output_names:
        missing = []
        for name in output_names:
            if re.search(rf"[\"']{re.escape(name)}[\"']\s*:", code):
                score += 3.0
            else:
                missing.append(name)
        if missing:
            score -= 25.0 * len(missing)
            reasons.append("contract missing returned outputs: " + ",".join(missing[:6]))
        else:
            reasons.append(f"contract returns all outputs {len(output_names)}/{len(output_names)}")

    if circuit_type.lower() == "seq":
        if "getattr(self," in code or re.search(r"self\.[A-Za-z_][A-Za-z0-9_$]*\s*=", code):
            score += 5.0
            reasons.append("sequential state/default pattern present")
        for name in output_names:
            # Penalize obvious local output assignment only inside branches.
            first_assign = re.search(rf"\b{re.escape(name)}\s*=", code)
            load_match = re.search(r"def\s+load\s*\([^)]*\):", code)
            if first_assign and load_match and first_assign.start() > load_match.end():
                prefix = code[load_match.end():first_assign.start()]
                if re.search(r"\bif\b|\bfor\b|\bwhile\b", prefix):
                    score -= 12.0
                    reasons.append(f"contract risk: output local {name} first assigned inside branch")
        traces = contract.get("temporal_traces") or []
        if traces:
            score += min(8.0, len(traces) * 1.5)
            reasons.append(f"contract temporal traces considered: {len(traces)}")

    if isinstance(tb_data, list) and output_names:
        bad_entries = 0
        matched_examples = 0
        public_examples = contract.get("public_examples") or []
        for entry in tb_data[:50]:
            expected = entry.get("expected_outputs") if isinstance(entry, dict) else None
            inputs = entry.get("inputs") if isinstance(entry, dict) else None
            if isinstance(inputs, dict) and isinstance(expected, dict):
                for example in public_examples:
                    if (
                        isinstance(example, dict)
                        and inputs == example.get("inputs")
                        and expected == example.get("expected_outputs")
                    ):
                        matched_examples += 1
            if circuit_type.lower() == "seq" and isinstance(expected, list):
                flat = []
                for cycle in expected:
                    if isinstance(cycle, dict):
                        flat.extend(v for v in cycle.values() if isinstance(v, dict))
                expected_dicts = flat
            else:
                expected_dicts = [expected] if isinstance(expected, dict) else []
            for out_dict in expected_dicts:
                if not all(name in out_dict for name in output_names):
                    bad_entries += 1
                    break
        if bad_entries:
            score -= 10.0 * bad_entries
            reasons.append(f"contract output coverage missing in {bad_entries} sampled entries")
        if public_examples:
            if matched_examples:
                score += 30.0 * matched_examples / max(1, len(public_examples))
                reasons.append(f"public examples matched {matched_examples}/{len(public_examples)}")
            else:
                score -= 12.0
                reasons.append(f"public examples not covered/matched 0/{len(public_examples)}")
    public_run = _run_public_contract_examples(sample, contract, circuit_type)
    score += public_run["score"]
    reasons.extend(public_run["reasons"])
    return {"score": score, "reasons": reasons}


def _run_public_contract_examples(sample: Dict[str, Any], contract: Dict[str, Any], circuit_type: str) -> Dict[str, Any]:
    """Execute candidate GoldenDUT on public examples from the contract."""
    golden_path = sample.get("golden_dut_path")
    examples = contract.get("public_examples") or []
    if not golden_path or not examples:
        return {"score": 0.0, "reasons": []}
    try:
        code = open(golden_path, "r").read()
        namespace: Dict[str, Any] = {}
        exec(compile(code, golden_path, "exec"), namespace, namespace)
        dut_cls = namespace.get("GoldenDUT")
        if dut_cls is None:
            return {"score": -80.0, "reasons": ["public examples failed: missing GoldenDUT class"]}
        dut = dut_cls()
    except Exception as exc:
        return {"score": -80.0, "reasons": [f"public examples failed to load candidate: {exc}"]}

    matches = 0
    checked = 0
    conflicts = []
    for example in examples[:16]:
        inputs = example.get("inputs")
        expected = example.get("expected_outputs")
        if not isinstance(inputs, dict) or not isinstance(expected, dict):
            continue
        try:
            if circuit_type.lower() == "seq":
                actual = dut.load(1, inputs)
            else:
                try:
                    actual = dut.load(inputs)
                except TypeError:
                    actual = dut.load(0, inputs)
        except Exception as exc:
            conflicts.append(f"{example.get('kind', 'example')}: execution error {exc}")
            checked += 1
            continue
        if not isinstance(actual, dict):
            conflicts.append(f"{example.get('kind', 'example')}: returned {type(actual).__name__}")
            checked += 1
            continue
        checked += 1
        if all(str(actual.get(name)) == str(value) for name, value in expected.items()):
            matches += 1
        elif len(conflicts) < 6:
            conflicts.append(
                f"{example.get('kind', 'example')}: expected {expected} got "
                f"{ {name: actual.get(name) for name in expected} }"
            )

    if not checked:
        return {"score": 0.0, "reasons": []}
    ratio = matches / checked
    score = 180.0 * ratio - 260.0 * (1.0 - ratio)
    reasons = [f"public contract examples executed {matches}/{checked}"]
    if conflicts:
        reasons.append("public example conflicts: " + "; ".join(conflicts[:4]))
    return {"score": score, "reasons": reasons}


def _eval_bool_expr(expr: str, values: Dict[str, int]) -> int:
    safe = expr.strip()
    safe = safe.replace("^", " ^ ").replace("&", " & ").replace("|", " | ").replace("~", " ~ ")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|[01]|\s|\^|&|\||~|\(|\)", safe.replace(" ", "")):
        # Fall through to token validation below; this branch mainly keeps the intent obvious.
        pass
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", safe))
    if not tokens <= set(values):
        raise ValueError(f"unsupported tokens in expression: {tokens - set(values)}")
    return int(eval(safe, {"__builtins__": {}}, dict(values))) & 1


def _parse_waveform_truth_table(description: str) -> Dict[tuple, int]:
    """Parse simple comment waveform rows with columns x, y, z into a truth table."""
    rows = []
    for line in description.splitlines():
        cleaned = line.strip().lstrip("/").strip()
        match = re.match(r"\d+\s*ns\s+([01])\s+([01])\s+([01])\s*$", cleaned, flags=re.I)
        if match:
            x, y, z = match.groups()
            rows.append((int(x), int(y), int(z)))

    table = {}
    for x, y, z in rows:
        key = (x, y)
        if key in table and table[key] != z:
            return {}
        table[key] = z
    return table if len(table) == 4 else {}


def _derive_composed_waveform_expected(description: str) -> Dict[tuple, int]:
    """
    Derive expectations for common HDLBits-style specs:
    Module A has a formula, Module B has a waveform table, and the top combines
    two A/B pairs through OR/AND then XOR. This uses only the written spec.
    """
    lower = description.lower()
    if not all(term in lower for term in ("module a", "module b", "waveform", "two a", "two b")):
        return {}
    if not all(term in lower for term in ("or", "and", "xor")):
        return {}

    match = re.search(r"Module\s+A\s+implements.*?\bz\s*=\s*([^.;\n]+)", description, flags=re.I | re.S)
    if not match:
        return {}
    expr = match.group(1).strip()
    expr = re.sub(r"[^A-Za-z0-9_~^&|() ]+$", "", expr)

    b_table = _parse_waveform_truth_table(description)
    if not b_table:
        return {}

    expected = {}
    try:
        for x in (0, 1):
            for y in (0, 1):
                a = _eval_bool_expr(expr, {"x": x, "y": y})
                b = b_table[(x, y)]
                expected[(x, y)] = ((a | b) ^ (a & b)) & 1
    except Exception:
        return {}
    return expected if len(expected) == 4 else {}


def _spec_waveform_score(sample: Dict[str, Any], stimulus_data: Any) -> Dict[str, Any]:
    tb_path = sample.get("testbench_json_path")
    if not tb_path:
        return {"score": 0.0, "reasons": []}

    description_path = os.path.join(os.path.dirname(tb_path), "description.txt")
    try:
        with open(description_path, "r") as f:
            description = f.read()
        tb_data = _load_json_file(tb_path)
    except Exception:
        return {"score": 0.0, "reasons": []}

    expected = _derive_composed_waveform_expected(description)
    if not expected or not isinstance(tb_data, list):
        return {"score": 0.0, "reasons": []}

    matches = 0
    comparable = 0
    conflicts = []
    for entry in tb_data:
        if not isinstance(entry, dict):
            continue
        inputs = entry.get("inputs")
        outputs = entry.get("expected_outputs")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        if set(inputs) >= {"x", "y"} and "z" in outputs:
            try:
                key = (int(str(inputs["x"]), 2), int(str(inputs["y"]), 2))
                actual = int(str(outputs["z"]), 2)
            except Exception:
                continue
            if key not in expected:
                continue
            comparable += 1
            if actual == expected[key]:
                matches += 1
            else:
                conflicts.append(f"x={key[0]} y={key[1]} expected z={expected[key]} got {actual}")

    if comparable == 0:
        return {"score": 0.0, "reasons": ["waveform spec score unavailable: no comparable x/y/z entries"]}

    ratio = matches / comparable
    score = 120.0 * ratio
    if conflicts:
        score -= 80.0 * (1.0 - ratio)
    reasons = [f"waveform-derived spec agreement {matches}/{comparable}"]
    if conflicts:
        reasons.append("waveform conflicts: " + "; ".join(conflicts[:4]))
    return {"score": score, "reasons": reasons}


def _spec_state_table_score(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Score Y0/z candidates against a state table written in the specification."""
    tb_path = sample.get("testbench_json_path")
    if not tb_path:
        return {"score": 0.0, "reasons": []}
    try:
        description = _read_neighbor_file(sample, "description.txt")
        tb_data = _load_json_file(tb_path)
    except Exception:
        return {"score": 0.0, "reasons": []}

    rows = {}
    for line in description.splitlines():
        cleaned = line.strip().lstrip("/").strip()
        match = re.search(
            r"\b([01]+)\s*\|\s*([01]+)\s*,\s*([01]+)\s*\|\s*([01]+)\b",
            cleaned,
        )
        if not match:
            continue
        present, next_zero, next_one, output = match.groups()
        if len(next_zero) != len(present) or len(next_one) != len(present):
            return {"score": 0.0, "reasons": []}
        rows[int(present, 2)] = (next_zero, next_one, int(output, 2))
    if len(rows) < 2 or not isinstance(tb_data, list):
        return {"score": 0.0, "reasons": []}

    matches = 0
    comparable = 0
    conflicts = []
    for entry in tb_data:
        if not isinstance(entry, dict):
            continue
        inputs = entry.get("inputs", {})
        outputs = entry.get("expected_outputs", {})
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        lower_outputs = {str(name).lower(): value for name, value in outputs.items()}
        if "x" not in inputs or "y" not in inputs:
            continue
        try:
            x = int(str(inputs["x"]), 2)
            y = int(str(inputs["y"]), 2)
        except Exception:
            continue
        if x not in (0, 1) or y not in rows:
            continue
        next_zero, next_one, z_expected = rows[y]
        expected = {"y0": int((next_one if x else next_zero)[-1]), "z": z_expected}
        for name, expected_value in expected.items():
            if name not in lower_outputs:
                continue
            try:
                actual_value = int(str(lower_outputs[name]), 2)
            except Exception:
                continue
            comparable += 1
            if actual_value == expected_value:
                matches += 1
            elif len(conflicts) < 6:
                conflicts.append(f"y={y:0b} x={x} {name} expected {expected_value} got {actual_value}")

    if comparable == 0:
        return {"score": 0.0, "reasons": []}
    ratio = matches / comparable
    score = 200.0 * ratio - 400.0 * (1.0 - ratio)
    reasons = [f"state-table spec agreement {matches}/{comparable}"]
    if conflicts:
        reasons.append("adversarial risk: state-table conflicts: " + "; ".join(conflicts))
    return {"score": score, "reasons": reasons}


def _golden_dut_snippet(path: str, limit: int = 3000) -> str:
    try:
        text = open(path, "r").read()
    except OSError as exc:
        return f"<failed to read: {exc}>"
    start = text.find("class GoldenDUT:")
    if start >= 0:
        end = text.find('if __name__ == "__main__"', start)
        text = text[start:end if end >= 0 else len(text)]
    return text[:limit]


def _testbench_summary(testbench_data: Any) -> Any:
    if not isinstance(testbench_data, list):
        return {"shape": type(testbench_data).__name__}
    summary = {
        "length": len(testbench_data),
    }
    first = next((entry for entry in testbench_data if isinstance(entry, dict)), None)
    if first is None:
        return summary
    summary["entry_keys"] = sorted(first)
    inputs = first.get("inputs")
    if isinstance(inputs, dict):
        summary["input_keys"] = sorted(inputs)
        summary["input_widths"] = {
            name: len(value) for name, value in inputs.items() if isinstance(value, str)
        }
    expected = first.get("expected_outputs")
    if isinstance(expected, dict):
        summary["output_keys"] = sorted(expected)
    elif isinstance(expected, list) and expected:
        first_cycle = next((cycle for cycle in expected if isinstance(cycle, dict)), {})
        summary["sequential_edge_keys"] = sorted(first_cycle)
        edge_outputs = next((value for value in first_cycle.values() if isinstance(value, dict)), {})
        summary["output_keys"] = sorted(edge_outputs)
    summary["clock_cycles"] = first.get("clock_cycles")
    return summary


def _read_neighbor_file(sample: Dict[str, Any], filename: str) -> str:
    tb_path = sample.get("testbench_json_path") or sample.get("golden_dut_path")
    if not tb_path:
        return ""
    try:
        return open(os.path.join(os.path.dirname(tb_path), filename), "r").read()
    except OSError:
        return ""


def _adversarial_semantic_score(sample: Dict[str, Any], circuit_type: str) -> Dict[str, Any]:
    """Spec/RTL-derived risk checks that do not execute the benchmark RTL."""
    golden_path = sample.get("golden_dut_path")
    if not golden_path:
        return {"score": 0.0, "reasons": []}
    try:
        code = open(golden_path, "r").read()
    except OSError:
        return {"score": 0.0, "reasons": []}

    description = _read_neighbor_file(sample, "description.txt")
    rtl_code = _read_neighbor_file(sample, "module_code.v") if _unsafe_allow_hidden_rtl() else ""
    desc_lower = description.lower()
    rtl_lower = rtl_code.lower()
    code_lower = code.lower()
    score = 0.0
    reasons: List[str] = []

    if circuit_type.lower() == "seq":
        async_reset = "asynchronous reset" in desc_lower or "areset" in rtl_lower or "async_reset" in rtl_lower
        sync_reset = (
            not async_reset
            and bool(re.search(r"\b(?:synchronous\s+reset|reset.{0,40}synchronous)\b", desc_lower, re.S))
        )
        if sync_reset:
            if "active low" in code_lower or re.search(r"\breset(?:_val)?\s*==\s*0", code):
                score -= 80.0
                reasons.append("adversarial risk: active-high reset modeled as active-low")
            if "reset_rising" in code_lower or "reset_falling" in code_lower:
                score -= 60.0
                reasons.append("adversarial risk: synchronous reset modeled as an edge")
            if "asynchronous reset" in code_lower:
                score -= 120.0
                reasons.append("adversarial risk: synchronous reset modeled outside the rising clock edge")

        if "galois" in desc_lower and "lfsr" in desc_lower:
            if re.search(r"feedback\s*=\s*[^\n]*(?:>>\s*4)[^\n]*\^[^\n]*(?:>>\s*2)", code):
                score -= 120.0
                reasons.append("adversarial risk: Galois LFSR uses Fibonacci-style q[4]^q[2] feedback")
            if "q_next = q[4:1]" in rtl_code and "q_next[2] ^= q[0]" in rtl_code:
                if re.search(r"<<\s*1", code) or "next_q3 = q4 ^ q0" in code_lower:
                    score -= 90.0
                    reasons.append("adversarial risk: Galois LFSR direction/tap index disagrees with RTL q_next[2]^=q[0]")
                if re.search(r">>\s*1", code) and re.search(r"<<\s*4", code) and re.search(r"<<\s*2", code):
                    score += 35.0
                    reasons.append("adversarial evidence: Galois LFSR shift/taps match RTL body")

        if "assign shift_ena" in rtl_lower and "state == b0" in rtl_lower:
            if re.search(r"shift_ena\s*=\s*1\s+if\s+self\.state\s*==\s*1", code):
                score -= 80.0
                reasons.append("adversarial risk: shift_ena tied to invented state 1 instead of RTL B0..B3 state set")

        if "arithmetic" in desc_lower and "shift" in desc_lower:
            if re.search(r"self\.[A-Za-z_][A-Za-z0-9_]*\s*&\s*1\b", code) and re.search(r">>\s*1\b", code):
                score -= 140.0
                reasons.append("adversarial risk: arithmetic right shift extends from LSB instead of MSB sign bit")
            if re.search(r">>\s*55\b", code) and ("64" in desc_lower or "[63:0]" in rtl_lower):
                score -= 120.0
                reasons.append("adversarial risk: 64-bit arithmetic shift-by-8 tests bit 55 instead of sign bit 63")
            one_bit_signs = re.findall(
                r"([A-Za-z_]\w*)\s*=\s*\(?\s*self\.[A-Za-z_]\w*\s*>>\s*63\s*\)?\s*&\s*1",
                code,
            )
            try:
                code_tree = ast.parse(code)
            except SyntaxError:
                code_tree = None

            def has_shift(node: ast.AST, op_type: type, amount: int) -> bool:
                return any(
                    isinstance(child, ast.BinOp)
                    and isinstance(child.op, op_type)
                    and isinstance(child.right, ast.Constant)
                    and child.right.value == amount
                    for child in ast.walk(node)
                )

            bad_shift_eight = False
            if code_tree is not None:
                for node in ast.walk(code_tree):
                    if not (
                        isinstance(node, ast.BinOp)
                        and isinstance(node.op, ast.BitOr)
                        and has_shift(node, ast.RShift, 8)
                    ):
                        continue
                    for child in ast.walk(node):
                        if not (
                            isinstance(child, ast.BinOp)
                            and isinstance(child.op, ast.LShift)
                            and isinstance(child.left, ast.Name)
                            and isinstance(child.right, ast.Constant)
                        ):
                            continue
                        if child.right.value == 63 or child.left.id in one_bit_signs:
                            bad_shift_eight = True
                            break
                    if bad_shift_eight:
                        break
            if bad_shift_eight:
                score -= 160.0
                reasons.append("adversarial risk: arithmetic shift-by-8 uses only one shifted sign bit")

        if (
            "shift register" in desc_lower
            and "right shift" in desc_lower
            and re.search(r"\(\s*self\.[A-Za-z_][A-Za-z0-9_]*\s*>>\s*1[^\n]*\)\s*<<\s*1", code)
        ):
            score -= 140.0
            reasons.append("adversarial risk: right-shift result is shifted left again")
        if "shift register" in desc_lower and "right shift" in desc_lower:
            if (
                re.search(r"new_q3\s*=\s*q2\b", code)
                and re.search(r"new_q2\s*=\s*q1\b", code)
                and re.search(r"new_q0\s*=\s*0\b", code)
            ):
                score -= 140.0
                reasons.append("adversarial risk: explicit bit mapping implements a left shift, not right shift")

        if ("toroid" in desc_lower or "wrap around" in desc_lower) and "neighbor" in code_lower:
            if len(re.findall(r"%\s*16\b", code)) < 2:
                score -= 160.0
                reasons.append("adversarial risk: toroidal neighbor coordinates are discarded instead of modulo-wrapped")
            if re.search(r"neighbors\s*\|=", code):
                score -= 160.0
                reasons.append("adversarial risk: neighbor population uses OR instead of integer addition")
            explicit_neighbors = {
                int(value) for value in re.findall(r"\bneighbor_?(\d+)\s*=", code, flags=re.I)
            }
            if re.search(r"\b8\s+neighbou?rs?\b", desc_lower) and explicit_neighbors and max(explicit_neighbors) < 8:
                score -= 160.0
                reasons.append("adversarial risk: model explicitly counts fewer than eight required neighbors")
            if re.search(r"next_[A-Za-z_]\w*\s*\|=\s*(?:current(?:_val)?|1)\s*(?:\n|$)", code):
                score -= 160.0
                reasons.append("adversarial risk: packed grid cell result is not shifted to bit index")
            axis_loops = len(re.findall(r"for\s+d[ij]\s+in\s+range\(\s*-1\s*,\s*2\s*\)", code))
            if axis_loops >= 4:
                score -= 220.0
                reasons.append("adversarial risk: separate row/column loops double-count toroidal corner neighbors")
            if re.search(r"q_pad\[[^\]]*\bj\b[^\]]*\]\s*=\s*0", code):
                score -= 220.0
                reasons.append("adversarial risk: toroidal top/bottom padding is zero-filled instead of wrapped")

        if "serial" in desc_lower and ("2's complement" in desc_lower or "two's complement" in desc_lower):
            if "assign z = (state == c)" in rtl_lower and "seen_one" in code_lower:
                score -= 120.0
                reasons.append("adversarial risk: serial two's-complement RTL is Moore state output z=(state==C), but GoldenDUT uses seen_one shortcut")
            if (
                "seen_one" in code_lower
                and re.search(r"self\.seen_one\s*=\s*1", code)
                and re.search(r"if\s+self\.seen_one\s*==\s*0\s*:", code)
                and re.search(r"\w+\s*=\s*1\s*-\s*x\b|\w+\s*=\s*~x\b", code)
            ):
                score -= 90.0
                reasons.append("adversarial risk: serial two's-complement first-one output uses updated seen_one state")

    packed_lsb_mux = (
        re.search(r"sel\s*=\s*0.{0,50}\[\s*3\s*:\s*0\s*\]", desc_lower, re.S)
        or re.search(r"\[\s*sel\s*\*\s*4\s*\+\s*3\s*\]", rtl_lower)
    )
    if packed_lsb_mux and (
        re.search(r"\b(?:1020|1023|in_width\s*-\s*1)\s*-\s*(?:start_bit|sel_val\s*\*\s*4)", code)
        or re.search(r"in_width\s*-\s*1\s*-\s*\([^)]*start_bit", code)
        or re.search(r"in_(?:str|bits)\s*\[\s*start_bit\s*:", code)
    ):
        score -= 220.0
        reasons.append(
            "adversarial risk: packed mux reverses Verilog bit numbering instead of treating bit 0 as the integer LSB"
        )
    selector_width = None
    selector_match = re.search(
        r"\binput\b[^;\n]*\[\s*(\d+)\s*:\s*(\d+)\s*\][^;\n]*\bsel\b",
        _read_neighbor_file(sample, "header.v"),
        re.I,
    )
    if selector_match:
        selector_width = abs(int(selector_match.group(1)) - int(selector_match.group(2))) + 1
    truncated_loop = re.search(
        r"for\s+(\w+)\s+in\s+range\(\s*(\d+)\s*\).*?if\s+\1\s*==\s*sel(?:_val)?",
        code,
        re.S,
    )
    if selector_width and truncated_loop and int(truncated_loop.group(2)) < (1 << selector_width):
        score -= 220.0
        reasons.append("adversarial risk: packed mux searches only part of the selector domain")

    registered_edge = re.search(
        r"\b([A-Za-z_]\w*)\s*<=\s*([A-Za-z_]\w*)\s*;.*?"
        r"\b([A-Za-z_]\w*)\s*<=\s*\2\s*\^\s*\1\s*;",
        rtl_code or "",
        re.S,
    )
    if registered_edge:
        state_name = registered_edge.group(1)
        update = re.search(rf"self\.{re.escape(state_name)}\s*=", code)
        xor_use = re.search(rf"[^\n]*\^\s*self\.{re.escape(state_name)}\b", code)
        if update and xor_use and update.start() < xor_use.start():
            score -= 220.0
            reasons.append("adversarial risk: registered edge detector overwrites old state before XOR")

    return {"score": score, "reasons": reasons}


def _has_adversarial_risk(reasons: List[str]) -> bool:
    return any(str(reason).startswith("adversarial risk:") for reason in reasons)


class JudgeAgent:
    """Select the best generated sample without using golden RTL output refresh."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def _llm_review_vote(
        self,
        judged: List[Dict[str, Any]],
        pychecker_results: List[Dict[str, Any]],
        stimulus_json_path: str,
        circuit_type: str,
    ) -> Dict[str, Any]:
        if self.llm_client is None or not judged:
            return {}

        description_path = os.path.join(os.path.dirname(stimulus_json_path), "description.txt")
        header_path = os.path.join(os.path.dirname(stimulus_json_path), "header.v")
        metadata_path = os.path.join(os.path.dirname(stimulus_json_path), "circuit_metadata.json")
        try:
            description = open(description_path, "r").read()
        except OSError:
            description = ""
        try:
            header = open(header_path, "r").read()
        except OSError:
            header = ""
        rtl_code = ""
        if _unsafe_allow_hidden_rtl():
            rtl_path = os.path.join(os.path.dirname(stimulus_json_path), "module_code.v")
            try:
                rtl_code = open(rtl_path, "r").read()
            except OSError:
                rtl_code = ""
        try:
            circuit_metadata = _load_json_file(metadata_path)
        except Exception:
            circuit_metadata = {"circuit_type": circuit_type}

        by_idx = {sample.get("sample_idx"): sample for sample in pychecker_results}
        candidate_payload = []
        for item in judged:
            sample_idx = item.get("sample_idx")
            sample = by_idx.get(sample_idx, {})
            tb_data = None
            try:
                tb_data = _load_json_file(sample.get("testbench_json_path"))
            except Exception:
                pass
            candidate_payload.append({
                "sample_idx": sample_idx,
                "candidate_strategy": sample.get("candidate_strategy", "unknown"),
                "deterministic_score": item.get("score"),
                "deterministic_reasons": item.get("reasons", []),
                "golden_dut_code": _golden_dut_snippet(sample.get("golden_dut_path", "")),
                "testbench_summary": _testbench_summary(tb_data),
            })

        system_prompt = (
            "You are the Pro-V Judge agent. Select the generated Python GoldenDUT/testbench "
            "sample that best matches the RTL specification. Do not invent outputs from hidden RTL. "
            "Use the circuit metadata to distinguish combinational vs sequential behavior, "
            "sync vs async reset timing, reset polarity, and known families such as LFSR/FSM/counter. "
            "Independently derive the relevant truth-table rows, FSM transitions, bit indices, signed-extension source, "
            "and boundary wrapping before comparing candidates. Candidate comments and majority agreement are not evidence. "
            "Reject code whose implementation contradicts its comments or the written transition/table rows. "
            "Return only JSON."
        )
        user_prompt = (
            "<description>\n"
            f"{description[:2000]}\n"
            "</description>\n\n"
            "<module_header>\n"
            f"{header[:600]}\n"
            "</module_header>\n\n"
            + (
                "<rtl_code unsafe_debug_only=\"true\">\n"
                f"{rtl_code[:5000]}\n"
                "</rtl_code>\n\n"
                if rtl_code
                else ""
            )
            +
            f"<circuit_type>{circuit_type}</circuit_type>\n\n"
            "<circuit_metadata_json>\n"
            f"{json.dumps(circuit_metadata, separators=(',', ':'), default=str)[:1200]}\n"
            "</circuit_metadata_json>\n\n"
            "<candidates_json>\n"
            f"{json.dumps(candidate_payload, separators=(',', ':'), default=str)[:18000]}\n"
            "</candidates_json>\n\n"
            "Return ONLY JSON with this schema: "
            "{\"selected_sample_idx\": integer, \"confidence\": number between 0 and 1, "
            "\"reason\": \"short evidence-based reason\", \"semantic_audit\": {"
            "\"port_widths\": \"pass|fail\", \"combinational_or_sequential\": \"pass|fail\", "
            "\"reset_and_edge_timing\": \"pass|fail|na\", \"state_or_truth_table\": \"pass|fail|na\", "
            "\"bit_order_and_signedness\": \"pass|fail|na\", \"candidate_risks\": [\"short risk\"]}}."
        )

        try:
            response = self.llm_client.chat(
                system=system_prompt,
                user=user_prompt,
                max_tokens=int(os.getenv("PRO_V_JUDGE_MAX_TOKENS", "512")),
            )
        except Exception as exc:
            return {"error": f"judge LLM call failed: {exc}"}

        match = re.search(r"\{.*\}", response or "", flags=re.S)
        if not match:
            return {"error": "judge LLM response had no JSON", "raw": (response or "")[:500]}
        try:
            vote = json.loads(match.group(0))
        except Exception as exc:
            return {"error": f"judge LLM JSON parse failed: {exc}", "raw": (response or "")[:500]}

        valid_indices = {item.get("sample_idx") for item in judged if item.get("score", -1000.0) > -1000.0}
        if not valid_indices:
            return {"error": "judge LLM skipped because no structurally valid samples remain"}
        selected = vote.get("selected_sample_idx")
        if selected not in valid_indices:
            return {"error": f"judge LLM selected invalid sample {selected}", "raw_vote": vote}

        try:
            confidence = float(vote.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        vote["confidence"] = max(0.0, min(1.0, confidence))
        return vote

    def run(
        self,
        pychecker_results: List[Dict[str, Any]],
        stimulus_json_path: str,
        circuit_type: str
    ) -> Dict[str, Any]:
        try:
            stimulus_data = _load_json_file(stimulus_json_path)
        except Exception as exc:
            stimulus_data = []
            stimulus_error = str(exc)
        else:
            stimulus_error = None

        judged = []
        outputs_by_index = {}

        for sample in pychecker_results:
            sample_idx = sample.get("sample_idx")
            tb_path = sample.get("testbench_json_path")
            try:
                tb_data = _load_json_file(tb_path)
            except Exception as exc:
                judged.append({
                    "sample_idx": sample_idx,
                    "score": -1000.0,
                    "reasons": [f"failed to load testbench: {exc}"],
                    "testbench_json_path": tb_path,
                })
                continue

            shape = _testbench_shape_score(tb_data, stimulus_data, circuit_type)
            literal_score = _python_literal_score(sample)
            contract_score = _contract_score(sample, circuit_type)
            waveform_score = (
                _spec_waveform_score(sample, stimulus_data)
                if circuit_type.lower() == "cmb"
                else {"score": 0.0, "reasons": []}
            )
            state_table_score = (
                _spec_state_table_score(sample)
                if circuit_type.lower() == "cmb"
                else {"score": 0.0, "reasons": []}
            )
            adversarial_score = _adversarial_semantic_score(sample, circuit_type)
            judged_item = {
                "sample_idx": sample_idx,
                "score": (
                    shape["score"]
                    + literal_score["score"]
                    + contract_score["score"]
                    + waveform_score["score"]
                    + state_table_score["score"]
                    + adversarial_score["score"]
                ),
                "reasons": (
                    shape["reasons"]
                    + literal_score["reasons"]
                    + contract_score["reasons"]
                    + waveform_score["reasons"]
                    + state_table_score["reasons"]
                    + adversarial_score["reasons"]
                ),
                "length": shape.get("length"),
                "total_cycles": shape.get("total_cycles"),
                "testbench_json_path": tb_path,
            }
            judged.append(judged_item)

            if isinstance(tb_data, list):
                for entry_idx, entry in enumerate(tb_data):
                    if isinstance(entry, dict):
                        outputs_by_index.setdefault(entry_idx, []).append(
                            (sample_idx, _stable_json(entry.get("expected_outputs", {})))
                        )

        majority_by_index = {}
        for entry_idx, values in outputs_by_index.items():
            counts = {}
            for _, output_repr in values:
                counts[output_repr] = counts.get(output_repr, 0) + 1
            if counts:
                majority_by_index[entry_idx] = max(counts.items(), key=lambda kv: kv[1])[0]

        for judged_item in judged:
            tb_path = judged_item.get("testbench_json_path")
            if judged_item["score"] <= -1000:
                continue
            try:
                tb_data = _load_json_file(tb_path)
            except Exception:
                continue
            matches = 0
            comparable = 0
            for entry_idx, entry in enumerate(tb_data if isinstance(tb_data, list) else []):
                if entry_idx not in majority_by_index or not isinstance(entry, dict):
                    continue
                comparable += 1
                if _stable_json(entry.get("expected_outputs", {})) == majority_by_index[entry_idx]:
                    matches += 1
            if comparable:
                ratio = matches / comparable
                judged_item["majority_agreement"] = ratio
                majority_bonus = 4.0 * ratio if _has_adversarial_risk(judged_item.get("reasons", [])) else 8.0 * ratio
                judged_item["score"] += majority_bonus
                judged_item["reasons"].append(f"majority agreement {ratio:.2f}")
                judged_item["reasons"].append(f"majority bonus {majority_bonus:.1f}")

        if stimulus_error:
            for judged_item in judged:
                judged_item["reasons"].append(f"stimulus load warning: {stimulus_error}")

        llm_vote = self._llm_review_vote(judged, pychecker_results, stimulus_json_path, circuit_type)
        if llm_vote and "selected_sample_idx" in llm_vote:
            selected_idx = llm_vote["selected_sample_idx"]
            confidence = llm_vote.get("confidence", 0.0)
            bonus = 25.0 * confidence
            semantic_audit = llm_vote.get("semantic_audit", {})
            audit_failures = [
                name for name, value in semantic_audit.items()
                if name != "candidate_risks" and str(value).lower() == "fail"
            ] if isinstance(semantic_audit, dict) else []
            for judged_item in judged:
                if judged_item.get("sample_idx") == selected_idx:
                    majority = float(judged_item.get("majority_agreement", 0.0) or 0.0)
                    adjusted_bonus = bonus
                    if circuit_type.lower() == "seq" and majority < 0.50:
                        adjusted_bonus = 0.0
                    if majority < 0.60:
                        adjusted_bonus = min(adjusted_bonus, 8.0)
                    elif majority < 0.75:
                        adjusted_bonus = min(adjusted_bonus, 18.0)
                    if _has_adversarial_risk(judged_item.get("reasons", [])):
                        adjusted_bonus = min(adjusted_bonus, 5.0)
                    if audit_failures:
                        adjusted_bonus = 0.0
                        judged_item["score"] -= 20.0 * len(audit_failures)
                        judged_item["reasons"].append(
                            "LLM semantic audit failures: " + ", ".join(audit_failures)
                        )
                    judged_item["score"] += adjusted_bonus
                    judged_item["reasons"].append(
                        f"LLM judge selected this sample (confidence {confidence:.2f}, bonus {adjusted_bonus:.1f}): {llm_vote.get('reason', '')}"
                    )
                else:
                    judged_item["reasons"].append(
                        f"LLM judge preferred sample {selected_idx}"
                    )

        best = max(judged, key=lambda item: item["score"]) if judged else None
        if best is None:
            return {"selected_sample_idx": None, "samples": judged, "reason": "no samples to judge"}

        return {
            "selected_sample_idx": best["sample_idx"],
            "selected_score": best["score"],
            "selected_reasons": best["reasons"],
            "samples": judged,
            "llm_vote": llm_vote,
            "agent": "JudgeAgent",
        }
