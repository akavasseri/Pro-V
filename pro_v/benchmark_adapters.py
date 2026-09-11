#!/usr/bin/env python3
"""
benchmark_adapters.py -- normalize benchmark inputs into a common task list.

Provides:
  load_benchmark_tasks(path, benchmark_format="auto") -> List[task dict]
  write_normalized_benchmark(tasks, out_path) -> None

A "task dict" carries at least: task_number (int), and typically task_id,
description, header, module_code, testbench, mutants, result. Both the HDLBits
(`test_benchmark_new.json`) and RTLLM (`rtllm_benchmark.json`) benchmarks are JSON:
either a list of task dicts or a dict keyed by task id/number. This adapter accepts
both shapes and guarantees every returned task has an integer task_number.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple


def _to_task_number(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m:
            return int(m.group())
    return fallback


def _normalize_list(raw: List[Any]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        task = dict(item)
        if "task_id" not in task:
            for id_key in ("id", "name", "slug", "problem_id"):
                if task.get(id_key) is not None:
                    task["task_id"] = str(task[id_key])
                    break
        task["task_number"] = _to_task_number(task.get("task_number", task.get("task_id")), i + 1)
        _normalize_task_interface(task)
        tasks.append(task)
    return tasks


def _normalize_dict(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for i, (key, item) in enumerate(raw.items()):
        if not isinstance(item, dict):
            continue
        task = dict(item)
        task["task_number"] = _to_task_number(task.get("task_number", key), i + 1)
        if "task_id" not in task:
            task["task_id"] = str(task.get("id") or task.get("name") or task.get("slug") or key)
        _normalize_task_interface(task)
        tasks.append(task)
    tasks.sort(key=lambda t: t["task_number"])
    return tasks


def load_benchmark_tasks(benchmark_path: str, benchmark_format: str = "auto") -> List[Dict[str, Any]]:
    """Load a benchmark into a list of normalized task dicts (each with task_number)."""
    fmt = (benchmark_format or "auto").lower()

    if os.path.isdir(benchmark_path) or fmt in (
        "rtllm_folder", "folder", "verilog_eval", "verisure",
    ):
        return _load_folder_benchmark(benchmark_path, fmt)

    # JSON file (HDLBits / RTLLM json). auto-detect list vs dict.
    with open(benchmark_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return _normalize_list(data)
    if isinstance(data, dict):
        # a dict may wrap the list under a key, or be keyed by task id
        for wrap_key in ("tasks", "data", "benchmark"):
            if isinstance(data.get(wrap_key), list):
                return _normalize_list(data[wrap_key])
        return _normalize_dict(data)
    raise ValueError("Unsupported benchmark JSON shape: %s" % type(data).__name__)


def _load_folder_benchmark(root: str, fmt: str = "auto") -> List[Dict[str, Any]]:
    """Load a directory benchmark, auto-detecting common RTL task layouts."""
    if not os.path.isdir(root):
        raise FileNotFoundError("benchmark path is not a directory: %s" % root)

    triplets = _find_verilog_eval_triplets(root)
    if fmt in ("verilog_eval", "verisure") or (fmt in ("auto", "folder") and triplets):
        return _load_verilog_eval_triplets(root, triplets)
    return _load_rtllm_folder(root)


def _find_verilog_eval_triplets(root: str) -> List[str]:
    """Return problem ids that have prompt/ref/test file triplets.

    Supports Veri-Sure's VerilogEval-v2-EXT layout:
      Prob001_zero_prompt.txt
      Prob001_zero_ref.sv
      Prob001_zero_test.sv

    Extra files are ignored unless they participate in a full triplet.
    """
    try:
        filenames = os.listdir(root)
    except OSError:
        return []
    prompt_suffix = "_prompt.txt"
    problems = sorted(
        name[: -len(prompt_suffix)]
        for name in filenames
        if name.endswith(prompt_suffix)
    )
    return [
        problem for problem in problems
        if os.path.exists(os.path.join(root, f"{problem}_ref.sv"))
        and os.path.exists(os.path.join(root, f"{problem}_test.sv"))
    ]


def _load_verilog_eval_triplets(root: str, problems: List[str]) -> List[Dict[str, Any]]:
    """Load VerilogEval/Veri-Sure prompt/ref/test triplets.

    The official benchmark compiles generated RTL as ``TopModule`` alongside
    the reference ``RefModule`` and golden ``tb``. Pro-V's internal simulation
    harness expects ``top_module``, so the adapter stores official names as
    metadata and normalizes only the internal ``module_code``/``header`` fields.
    """
    tasks: List[Dict[str, Any]] = []
    for fallback, problem in enumerate(problems, start=1):
        prompt_path = os.path.join(root, f"{problem}_prompt.txt")
        ref_path = os.path.join(root, f"{problem}_ref.sv")
        test_path = os.path.join(root, f"{problem}_test.sv")
        spec = _read_path(prompt_path) or ""
        ref = _read_path(ref_path) or ""
        test = _read_path(test_path) or ""
        task_number = _to_task_number(problem, fallback)
        original_module = extract_module_name(ref)
        task: Dict[str, Any] = {
            "task_number": task_number,
            "task_id": problem,
            "task_path": os.path.relpath(prompt_path, root),
            "benchmark_source": "verilog_eval_triplets",
            "description": _normalize_prompt_module_name(spec),
            "original_description": spec,
            "header": _extract_header(ref),
            "module_code": ref,
            "testbench": test,
            "golden_testbench": test,
            "golden_testbench_path": test_path,
            "golden_ref_path": ref_path,
            "prompt_path": prompt_path,
            "mutants": [],
            "result": [],
            "official_ref_module_name": original_module or "RefModule",
            "official_dut_module_name": _detect_test_dut_module(test) or "TopModule",
            "official_testbench_top": _detect_testbench_top(test) or "tb",
            "internal_dut_module_name": "top_module",
            "eval2_available": False,
        }
        _normalize_task_interface(task)
        tasks.append(task)
    tasks.sort(key=lambda t: t["task_number"])
    return tasks


def _load_rtllm_folder(root: str) -> List[Dict[str, Any]]:
    """Best-effort RTLLM directory loader: each subdir is a task with a spec +
    reference verilog. Supports both flat task folders and category/task trees."""
    task_dirs: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        raise FileNotFoundError("benchmark path is not a directory: %s" % root)

    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        names = set(filenames)
        has_spec = bool(names & {"design_description.txt", "spec.txt", "description.txt"})
        has_ref = bool(names & {"verified.v", "reference.v", "top.v"}) or any(
            fn.startswith("verified") and fn.endswith(".v")
            for fn in filenames
        )
        if has_spec and has_ref:
            rel = os.path.relpath(current, root)
            task_dirs.append((rel, current))

    tasks: List[Dict[str, Any]] = []
    for i, (rel, d) in enumerate(sorted(task_dirs), start=1):
        name = os.path.basename(d)
        spec = _read_first(d, ("design_description.txt", "spec.txt", "description.txt"))
        ref = _read_first_verified(d, name)
        if spec is None and ref is None:
            continue
        task = {
            "task_number": i, "task_id": name, "task_path": rel,
            "description": spec or "", "header": _extract_header(ref or ""),
            "module_code": ref or "", "testbench": "", "mutants": [], "result": "",
        }
        _normalize_task_interface(task)
        tasks.append(task)
    return tasks


def _read_first(d: str, names) -> Any:
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return f.read()
            except Exception:
                pass
    return None


def _read_path(path: str) -> Any:
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return None


def _read_first_verified(d: str, task_name: str) -> Any:
    exact = _read_first(d, ("verified.v", "reference.v", "%s.v" % task_name, "top.v"))
    if exact is not None:
        return exact
    try:
        candidates = sorted(
            n for n in os.listdir(d)
            if n.startswith("verified") and n.endswith(".v")
        )
    except OSError:
        candidates = []
    return _read_first(d, candidates)


def _strip_comments(text: str) -> str:
    text = re.sub(r"//.*", "", text or "")
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def _normalize_prompt_module_name(prompt: str) -> str:
    """Make external benchmark prompts consistent with Pro-V's internal top.

    Original text is preserved in ``original_description``. This rewrite affects
    only the agent-facing internal task spec, avoiding TopModule/top_module
    confusion in prompts generated for Pro-V's Verilator harness.
    """
    if not prompt:
        return prompt
    return re.sub(r"\bTopModule\b", "top_module", prompt)


def _detect_test_dut_module(testbench: str) -> str:
    text = _strip_comments(testbench)
    for candidate in ("TopModule", "top_module"):
        if re.search(rf"\b{candidate}\s+[A-Za-z_]\w*\s*(?:#\s*\(|\()", text):
            return candidate
    return ""


def _detect_testbench_top(testbench: str) -> str:
    text = _strip_comments(testbench)
    if re.search(r"\bmodule\s+tb\b", text):
        return "tb"
    names = re.findall(r"\bmodule\s+([A-Za-z_]\w*)\b", text)
    return names[-1] if names else ""


def _find_module_header_span(verilog: str) -> Tuple[int, int]:
    text = verilog or ""
    matches = list(re.finditer(r"\bmodule\s+([A-Za-z_]\w*)\b", text))
    if not matches:
        return -1, -1
    chosen = next((m for m in matches if m.group(1) == "top_module"), matches[0])
    semi = text.find(";", chosen.end())
    if semi < 0:
        return chosen.start(), len(text)
    return chosen.start(), semi + 1


def _extract_header(verilog: str) -> str:
    clean = _strip_comments(verilog)
    start, end = _find_module_header_span(clean)
    return clean[start:end] if start >= 0 and end > start else ""


def _normalize_task_interface(task: Dict[str, Any]) -> None:
    """Ensure headers carry enough interface info for non-ANSI Verilog tasks.

    RTLLM commonly stores ``header`` as only ``module top_module(a,b,out);`` and
    puts the input/output declarations in ``module_code``. Agents consume the
    public header, so keep the module declaration but append only the following
    port declarations. This exposes the interface without exposing DUT logic.
    """
    _normalize_common_task_fields(task)
    module_code = str(task.get("module_code") or "")
    header = str(task.get("header") or "")
    if not module_code:
        return
    module_name = extract_module_name(module_code)
    if module_name and module_name != "top_module":
        task.setdefault("original_module_name", module_name)
        module_code = _rename_declared_top(module_code, module_name, "top_module")
        task["module_code"] = module_code
        if header:
            header = _rename_declared_top(header, module_name, "top_module")
            task["header"] = header
    if header and re.search(r"\b(input|output|inout)\b", header):
        return
    enriched = _extract_header_with_port_decls(module_code)
    if enriched:
        task["header"] = enriched


def _first_text(task: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _normalize_common_task_fields(task: Dict[str, Any]) -> None:
    """Map common benchmark schemas onto Pro-V's task field names.

    This is intentionally schema-light: it preserves original fields while
    filling only missing Pro-V canonical keys. It avoids benchmark-specific
    assumptions and never derives behavior from hidden RTL.
    """
    if not isinstance(task.get("description"), str) or not task.get("description", "").strip():
        desc = _first_text(task, (
            "prompt", "spec", "specification", "problem", "instruction",
            "description", "design_description", "natural_language_spec",
        ))
        if desc:
            task["description"] = desc

    if not isinstance(task.get("module_code"), str) or not task.get("module_code", "").strip():
        rtl = _first_text(task, (
            "rtl", "verilog", "sv", "solution", "reference_solution",
            "canonical_solution", "golden_rtl", "reference_rtl", "module",
        ))
        if rtl:
            task["module_code"] = rtl

    if not isinstance(task.get("header"), str) or not task.get("header", "").strip():
        header = _first_text(task, (
            "header", "module_header", "interface", "signature",
            "port_declaration", "port_declarations",
        ))
        if header:
            task["header"] = header

    if "mutants" not in task:
        for key in ("mutants", "mutation_list", "mutated_rtls", "mutant_rtls"):
            value = task.get(key)
            if isinstance(value, list):
                task["mutants"] = value
                break


def extract_module_name(verilog: str) -> str:
    m = re.search(r"\bmodule\s+([A-Za-z_]\w*)\b", _strip_comments(verilog))
    return m.group(1) if m else ""


def _is_reset_signal(name: str) -> bool:
    lname = (name or "").lower()
    explicit = {
        "reset", "rst", "areset", "arst", "sreset", "srst", "clear", "clr",
        "rst_n", "resetn", "aresetn", "arstn", "srstn", "nreset", "nrst",
        "rstb", "reset_b", "reset_l",
    }
    return lname in explicit or "reset" in lname or bool(re.fullmatch(r"[abs]?rstn?", lname))


def classify_clocking(verilog: str) -> str:
    """Return a coarse interface-level clocking class for benchmark inspection."""
    text = verilog or ""
    lower = text.lower()
    has_clock_port = bool(
        re.search(r"\b(?:input|inout)\b[^;,\)]*\b(?:clk|clock|[A-Za-z_]\w*_clk|clk[A-Za-z_]\w*)\b", lower)
        or re.search(r"\b(?:posedge|negedge)\s+[A-Za-z_]\w*", lower)
    )
    if not has_clock_port:
        return "cmb"
    edge_resets = [
        sig for sig in re.findall(r"\b(?:posedge|negedge)\s+([A-Za-z_]\w*)", lower)
        if _is_reset_signal(sig)
    ]
    if edge_resets:
        return "seq_async_reset"
    if any(_is_reset_signal(sig) for sig in re.findall(r"\b[A-Za-z_]\w*\b", lower)):
        return "seq_reset"
    return "seq"


def _rename_declared_top(verilog: str, old_name: str, new_name: str) -> str:
    """Rename only the declared top module, preserving RTL behavior.

    Pro-V's Verilator harness instantiates ``top_module``. Benchmark adapters
    own this interface normalization so downstream simulation code does not need
    benchmark-specific module names.
    """
    if not verilog or not old_name or old_name == new_name:
        return verilog
    return re.sub(
        rf"(\bmodule\s+){re.escape(old_name)}\b",
        rf"\1{new_name}",
        verilog,
        count=1,
    )


def _extract_header_with_port_decls(verilog: str) -> str:
    clean_verilog = _strip_comments(verilog)
    header = _extract_header(clean_verilog)
    if not header:
        return ""
    if re.search(r"\b(input|output|inout)\b", header):
        return header

    _, header_end = _find_module_header_span(clean_verilog)
    tail = clean_verilog[header_end:]
    decls: List[str] = []
    for stmt in re.finditer(r"([^;]*;)", tail, flags=re.S):
        text = stmt.group(1).strip()
        if not text:
            continue
        clean = _strip_comments(text).strip()
        if not clean:
            continue
        if re.match(r"^(input|output|inout)\b", clean):
            decls.append(clean)
            continue
        # Stop after the contiguous interface declaration block. Internal wire
        # declarations and logic stay out of the agent-visible header.
        break
    return "\n".join([header] + decls) if decls else header


def write_normalized_benchmark(tasks, out_path: str) -> None:
    """Write the normalized task list to out_path as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(list(tasks), f, indent=2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ts = load_benchmark_tasks(sys.argv[1])
        print("loaded %d tasks; numbers: %s%s" % (
            len(ts), [t["task_number"] for t in ts[:8]], " ..." if len(ts) > 8 else ""))
