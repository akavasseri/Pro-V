#!/usr/bin/env python3
"""
Summarize Pro-V generated test vectors, widths, input-space size, and simulation results.

This script reports, per task:

  LLMVecs / LLMCycles:
      Raw vectors/cycles produced directly by the LLM-generated stimulus_gen()
      inside task_X/stimulus_gen.py.

  FinalVecs / FinalCycles:
      Vectors/cycles actually saved in task_X/stimulus.json after Pro-V cleanup.

  TBcases / TBcycles:
      Cases/cycles in the selected or largest testbench_*.json after PyChecker
      generated expected outputs.

  InBits / InputSpace:
      Total non-clock input width and theoretical per-cycle input space.

  Eval0 / Eval1 / Error:
      Simulation metrics from task_result.json, if available.

Usage:
    python scripts/summarize_test_vectors.py \
        --out outputs/verilog_eval_20260603_083822

Optional:
    python scripts/summarize_test_vectors.py \
        --out outputs/verilog_eval_20260603_083822 \
        --csv vector_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Pro-V generated stimulus/testbench vector lengths and port widths."
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output experiment directory, e.g. outputs/verilog_eval_20260603_083822",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--raw-timeout",
        type=int,
        default=20,
        help="Timeout in seconds for calling raw stimulus_gen(). Default: 20",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not execute stimulus_gen.py to measure raw LLM vectors.",
    )
    return parser.parse_args()


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def parse_ports_from_header(header_text: str) -> Dict[str, Dict[str, int]]:
    """
    Parse simple Verilog module headers and return:
        {
            "inputs": {"a": 1, "data": 8, ...},
            "outputs": {"out": 1, ...}
        }

    Handles common HDLBits-style declarations:
        input clk,
        input [7:0] in,
        output reg [7:0] q);
    """
    ports: Dict[str, Dict[str, int]] = {"inputs": {}, "outputs": {}}

    # Remove // comments.
    text = re.sub(r"//.*", "", header_text)

    for raw in text.splitlines():
        line = raw.strip().rstrip(",;)")
        if not line:
            continue

        m = re.match(r"^(input|output)\s+(.*)$", line)
        if not m:
            continue

        direction = m.group(1)
        rest = m.group(2).strip()

        # Remove common Verilog qualifiers.
        rest = re.sub(r"\b(wire|reg|logic|signed|unsigned)\b", "", rest).strip()

        width = 1
        wm = re.search(r"\[(\d+)\s*:\s*(\d+)\]", rest)
        if wm:
            a = int(wm.group(1))
            b = int(wm.group(2))
            width = abs(a - b) + 1
            rest = re.sub(r"\[[^\]]+\]", "", rest).strip()

        # Usually one name per line, but support comma-separated names.
        names = [x.strip() for x in rest.split(",") if x.strip()]
        for name in names:
            name = name.split("=")[0].strip()
            nm = re.match(r"([A-Za-z_][A-Za-z0-9_$]*)", name)
            if not nm:
                continue

            port_name = nm.group(1)
            if direction == "input":
                ports["inputs"][port_name] = width
            else:
                ports["outputs"][port_name] = width

    return ports


def safe_input_space(bits: int) -> str:
    if bits <= 20:
        return str(2**bits)
    return f"2^{bits}"


def short_ports(port_widths: Dict[str, int], max_len: int = 28) -> str:
    if not port_widths:
        return "-"
    s = ",".join(f"{k}:{v}" for k, v in port_widths.items())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def total_cycles(data: Any) -> Optional[int]:
    """
    Count total cycles.

    For CMB:
        List of dictionaries with no clock_cycles -> each case counts as 1.

    For SEQ:
        Each scenario may have clock_cycles -> sum those values.
    """
    if not isinstance(data, list):
        return None

    total = 0
    for item in data:
        if isinstance(item, dict):
            try:
                total += int(item.get("clock_cycles", 1) or 1)
            except Exception:
                total += 1
        else:
            total += 1

    return total


def count_cases(data: Any) -> Optional[int]:
    if isinstance(data, list):
        return len(data)
    return None


def run_raw_llm_stimulus(task_dir: str, timeout_s: int = 20) -> Tuple[Optional[int], Optional[int], str]:
    """
    Execute task_X/stimulus_gen.py without running its __main__ tail.

    This calls:
        stimulus_gen()

    That gives the raw LLM-generated vector list before the Pro-V tail does cleanup,
    fuzzy matching, padding/truncation, and writes stimulus.json.
    """
    stim_py = os.path.join(task_dir, "stimulus_gen.py")
    if not os.path.exists(stim_py):
        return None, None, "no_stimulus_gen.py"

    code = r'''
import json
import runpy

ns = runpy.run_path("stimulus_gen.py", run_name="not_main")
if "stimulus_gen" not in ns:
    raise RuntimeError("stimulus_gen function not found")

data = ns["stimulus_gen"]()

def total_cycles(data):
    if not isinstance(data, list):
        return None
    total = 0
    for item in data:
        if isinstance(item, dict):
            try:
                total += int(item.get("clock_cycles", 1) or 1)
            except Exception:
                total += 1
        else:
            total += 1
    return total

print(json.dumps({
    "cases": len(data) if isinstance(data, list) else None,
    "cycles": total_cycles(data),
}))
'''

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=task_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        if proc.returncode != 0:
            return None, None, "raw_run_error"

        lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
        if not lines:
            return None, None, "raw_no_output"

        info = json.loads(lines[-1])
        return info.get("cases"), info.get("cycles"), "-"

    except subprocess.TimeoutExpired:
        return None, None, "raw_timeout"
    except Exception as e:
        return None, None, f"raw_error:{type(e).__name__}"


def get_task_number(task_dir: str) -> int:
    base = os.path.basename(task_dir.rstrip("/"))
    m = re.match(r"task_(\d+)$", base)
    if not m:
        raise ValueError(f"Not a task directory: {task_dir}")
    return int(m.group(1))


def get_final_stimulus_counts(task_dir: str) -> Tuple[Optional[int], Optional[int]]:
    path = os.path.join(task_dir, "stimulus.json")
    if not os.path.exists(path):
        return None, None

    try:
        data = load_json(path)
        return count_cases(data), total_cycles(data)
    except Exception:
        return None, None


def get_testbench_counts(task_dir: str, selected_sample_idx: Optional[int]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Prefer selected testbench if selected_sample_idx exists.
    Otherwise use the testbench with the largest cycle count.
    """
    tb_paths = sorted(glob.glob(os.path.join(task_dir, "testbench_*.json")))
    if not tb_paths:
        return None, None, None

    preferred_path = None
    if selected_sample_idx is not None:
        candidate = os.path.join(task_dir, f"testbench_{selected_sample_idx}.json")
        if os.path.exists(candidate):
            preferred_path = candidate

    if preferred_path:
        try:
            data = load_json(preferred_path)
            return count_cases(data), total_cycles(data), os.path.basename(preferred_path)
        except Exception:
            return None, None, os.path.basename(preferred_path)

    best_cases = None
    best_cycles = None
    best_name = None

    for path in tb_paths:
        try:
            data = load_json(path)
            cases = count_cases(data)
            cycles = total_cycles(data)
            if cycles is not None and (best_cycles is None or cycles > best_cycles):
                best_cases = cases
                best_cycles = cycles
                best_name = os.path.basename(path)
        except Exception:
            continue

    return best_cases, best_cycles, best_name


def get_result_metrics(task_dir: str) -> Dict[str, Any]:
    path = os.path.join(task_dir, "task_result.json")
    result = {
        "selected_sample_idx": None,
        "eval0": None,
        "eval1": None,
        "agreement_rate": None,
        "agreement_80": None,
        "agreement_90": None,
        "agreement_100": None,
        "overall_success": None,
        "error": "-",
    }

    if not os.path.exists(path):
        return result

    try:
        data = load_json(path)
        sim = data.get("simulation_metrics", {}) or {}
        e2 = sim.get("eval2_mutant_detection", {}) or {}

        result["selected_sample_idx"] = data.get("selected_sample_idx")
        result["eval0"] = sim.get("eval0_compile_success")
        result["eval1"] = sim.get("eval1_module_passes")
        result["agreement_rate"] = e2.get("agreement_rate")
        result["agreement_80"] = e2.get("agreement_80")
        result["agreement_90"] = e2.get("agreement_90")
        result["agreement_100"] = e2.get("agreement_100")
        result["overall_success"] = sim.get("overall_success")
        result["error"] = sim.get("error") or "-"
    except Exception as e:
        result["error"] = f"result_parse_error:{type(e).__name__}"

    return result


def get_width_info(task_dir: str) -> Dict[str, Any]:
    header_path = os.path.join(task_dir, "header.v")
    ports = {"inputs": {}, "outputs": {}}

    if os.path.exists(header_path):
        try:
            ports = parse_ports_from_header(read_text(header_path))
        except Exception:
            ports = {"inputs": {}, "outputs": {}}

    nonclock_inputs = {
        name: width
        for name, width in ports["inputs"].items()
        if name.lower() not in ("clk", "clock")
    }

    input_bits = sum(nonclock_inputs.values())
    output_bits = sum(ports["outputs"].values())

    return {
        "input_bits": input_bits,
        "input_space": safe_input_space(input_bits),
        "output_bits": output_bits,
        "inputs": short_ports(nonclock_inputs, max_len=28),
        "outputs": short_ports(ports["outputs"], max_len=20),
    }


def collect_rows(out_dir: str, raw_timeout: int, no_raw: bool) -> List[Dict[str, Any]]:
    task_dirs = sorted(
        glob.glob(os.path.join(out_dir, "task_*")),
        key=lambda p: get_task_number(p),
    )

    rows: List[Dict[str, Any]] = []

    for task_dir in task_dirs:
        task = get_task_number(task_dir)

        metrics = get_result_metrics(task_dir)
        selected_sample_idx = metrics.get("selected_sample_idx")

        if no_raw:
            llm_vecs, llm_cycles, raw_status = None, None, "skipped"
        else:
            llm_vecs, llm_cycles, raw_status = run_raw_llm_stimulus(task_dir, timeout_s=raw_timeout)

        final_vecs, final_cycles = get_final_stimulus_counts(task_dir)
        tb_cases, tb_cycles, tb_name = get_testbench_counts(task_dir, selected_sample_idx)
        width_info = get_width_info(task_dir)

        row = {
            "task": task,
            "llm_vecs": llm_vecs,
            "llm_cycles": llm_cycles,
            "final_vecs": final_vecs,
            "final_cycles": final_cycles,
            "tb_cases": tb_cases,
            "tb_cycles": tb_cycles,
            "tb_name": tb_name or "-",
            "input_bits": width_info["input_bits"],
            "input_space": width_info["input_space"],
            "output_bits": width_info["output_bits"],
            "inputs": width_info["inputs"],
            "outputs": width_info["outputs"],
            "selected_sample_idx": selected_sample_idx,
            "eval0": metrics["eval0"],
            "eval1": metrics["eval1"],
            "agreement_rate": metrics["agreement_rate"],
            "agreement_80": metrics["agreement_80"],
            "agreement_90": metrics["agreement_90"],
            "agreement_100": metrics["agreement_100"],
            "overall_success": metrics["overall_success"],
            "raw_status": raw_status,
            "error": metrics["error"],
        }
        rows.append(row)

    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def print_table(rows: List[Dict[str, Any]], out_dir: str) -> None:
    print("=" * 190)
    print(f"RAW LLM TEST VECTOR / FINAL STIMULUS / WIDTH SUMMARY FOR: {out_dir}")
    print("=" * 190)

    header = (
        f"{'Task':>4}  "
        f"{'LLMVecs':>8}  "
        f"{'LLMCycles':>10}  "
        f"{'FinalVecs':>9}  "
        f"{'FinalCycles':>11}  "
        f"{'TBcases':>7}  "
        f"{'TBcycles':>9}  "
        f"{'InBits':>6}  "
        f"{'InputSpace':>12}  "
        f"{'OutBits':>7}  "
        f"{'Sel':>4}  "
        f"{'Eval0':>6}  "
        f"{'Eval1':>6}  "
        f"{'Inputs':<28}  "
        f"{'Outputs':<20}  "
        f"{'RawStatus':<14}  "
        f"Error"
    )
    print(header)
    print("-" * 190)

    for r in rows:
        print(
            f"{r['task']:>4}  "
            f"{fmt(r['llm_vecs']):>8}  "
            f"{fmt(r['llm_cycles']):>10}  "
            f"{fmt(r['final_vecs']):>9}  "
            f"{fmt(r['final_cycles']):>11}  "
            f"{fmt(r['tb_cases']):>7}  "
            f"{fmt(r['tb_cycles']):>9}  "
            f"{fmt(r['input_bits']):>6}  "
            f"{fmt(r['input_space']):>12}  "
            f"{fmt(r['output_bits']):>7}  "
            f"{fmt(r['selected_sample_idx']):>4}  "
            f"{fmt(r['eval0']):>6}  "
            f"{fmt(r['eval1']):>6}  "
            f"{fmt(r['inputs']):<28.28}  "
            f"{fmt(r['outputs']):<20.20}  "
            f"{fmt(r['raw_status']):<14.14}  "
            f"{r['error']}"
        )

    print("-" * 190)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    total = len(rows)

    timeout_count = sum(1 for r in rows if r["error"] == "Simulation timeout")
    eval0_count = sum(1 for r in rows if r["eval0"] is True)
    eval1_count = sum(1 for r in rows if r["eval1"] is True)
    overall_count = sum(1 for r in rows if r["overall_success"] is True)

    raw_ge_2k = sum(1 for r in rows if (r["llm_cycles"] or 0) >= 2000)
    raw_ge_10k = sum(1 for r in rows if (r["llm_cycles"] or 0) >= 10000)
    final_ge_2k = sum(1 for r in rows if (r["final_cycles"] or 0) >= 2000)
    final_ge_10k = sum(1 for r in rows if (r["final_cycles"] or 0) >= 10000)

    print()
    print("SUMMARY")
    print("=" * 80)
    print(f"Tasks found:                         {total}")
    print(f"Eval0 compile success:               {eval0_count}/{total}")
    print(f"Eval1 simulation pass:               {eval1_count}/{total}")
    print(f"Overall success:                     {overall_count}/{total}")
    print(f"Simulation timeouts:                 {timeout_count}/{total}")
    print()
    print(f"Raw LLM tasks with >= 2,000 cycles:  {raw_ge_2k}")
    print(f"Raw LLM tasks with >= 10,000 cycles: {raw_ge_10k}")
    print(f"Final tasks with >= 2,000 cycles:    {final_ge_2k}")
    print(f"Final tasks with >= 10,000 cycles:   {final_ge_10k}")
    print()

    print("Largest raw LLM-generated stimulus lengths:")
    for r in sorted(rows, key=lambda x: x["llm_cycles"] or 0, reverse=True)[:15]:
        print(
            f"  Task {r['task']:>4}: "
            f"LLMVecs={r['llm_vecs']}, "
            f"LLMCycles={r['llm_cycles']}, "
            f"FinalVecs={r['final_vecs']}, "
            f"FinalCycles={r['final_cycles']}, "
            f"InBits={r['input_bits']}, "
            f"InputSpace={r['input_space']}, "
            f"Error={r['error']}"
        )

    print()
    print("Largest final stimulus lengths:")
    for r in sorted(rows, key=lambda x: x["final_cycles"] or 0, reverse=True)[:15]:
        print(
            f"  Task {r['task']:>4}: "
            f"FinalVecs={r['final_vecs']}, "
            f"FinalCycles={r['final_cycles']}, "
            f"LLMVecs={r['llm_vecs']}, "
            f"LLMCycles={r['llm_cycles']}, "
            f"InBits={r['input_bits']}, "
            f"InputSpace={r['input_space']}, "
            f"Error={r['error']}"
        )


def write_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    fieldnames = [
        "task",
        "llm_vecs",
        "llm_cycles",
        "final_vecs",
        "final_cycles",
        "tb_cases",
        "tb_cycles",
        "tb_name",
        "input_bits",
        "input_space",
        "output_bits",
        "inputs",
        "outputs",
        "selected_sample_idx",
        "eval0",
        "eval1",
        "agreement_rate",
        "agreement_80",
        "agreement_90",
        "agreement_100",
        "overall_success",
        "raw_status",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print()
    print(f"Wrote CSV: {csv_path}")


def main() -> None:
    args = parse_args()

    out_dir = args.out.rstrip("/")
    if not os.path.isdir(out_dir):
        raise SystemExit(f"ERROR: output directory does not exist: {out_dir}")

    rows = collect_rows(out_dir=out_dir, raw_timeout=args.raw_timeout, no_raw=args.no_raw)

    print_table(rows, out_dir=out_dir)
    print_summary(rows)

    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
