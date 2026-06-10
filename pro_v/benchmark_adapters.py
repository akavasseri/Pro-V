"""Benchmark adapters for normalizing external RTL datasets into Pro-V tasks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _extract_module_header(verilog_code: str) -> str:
    text = re.sub(r"//.*", "", verilog_code or "")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    pattern = r"\bmodule\s+{name}\s*(?:#\s*\(.*?\)\s*)?\(.*?\)\s*;"
    match = re.search(pattern.format(name="top_module"), text, flags=re.S)
    if not match:
        match = re.search(
            pattern.format(name=r"[A-Za-z_][A-Za-z0-9_$]*"),
            text,
            flags=re.S,
        )
    return match.group(0).strip() if match else ""


def extract_module_name(verilog_code: str) -> str:
    if re.search(r"\bmodule\s+top_module\b", verilog_code or ""):
        return "top_module"
    match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", verilog_code or "")
    return match.group(1) if match else ""


def _extract_module_names(verilog_code: str) -> List[str]:
    return re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\b", verilog_code or "")


def _rename_module(verilog_code: str, old_name: str, new_name: str = "top_module") -> str:
    if not old_name or old_name == new_name:
        return verilog_code or ""
    return re.sub(
        rf"(\bmodule\s+)({re.escape(old_name)})\b",
        rf"\g<1>{new_name}",
        verilog_code or "",
        count=1,
    )


def _expected_rtllm_module_names(directory: Path, rtl_path: Path) -> List[str]:
    names = []
    stem = rtl_path.stem
    if stem.startswith("verified_"):
        names.append(stem[len("verified_"):])
    names.append(stem)
    names.append(directory.name)
    # RTLLM has a small spelling inconsistency for calendar.
    if directory.name == "calendar":
        names.append("calender")
    return [name for name in names if name]


def _choose_top_module_name(verilog_code: str, directory: Path, rtl_path: Path) -> str:
    modules = _extract_module_names(verilog_code)
    if not modules:
        return ""
    module_set = set(modules)
    for candidate in _expected_rtllm_module_names(directory, rtl_path):
        if candidate in module_set:
            return candidate
    return modules[0]


def classify_clocking(verilog_code: str) -> Dict[str, bool]:
    text = re.sub(r"//.*", "", verilog_code or "")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    has_edge = bool(re.search(r"\b(?:pos|neg)edge\b", text, flags=re.I))
    has_async_reset = False
    for sensitivity in re.findall(r"always\s*@\s*\((.*?)\)", text, flags=re.S | re.I):
        tokens = re.findall(
            r"\b(posedge|negedge)\s*([A-Za-z_][A-Za-z0-9_$]*)",
            sensitivity,
            flags=re.I,
        )
        edge_signals = [sig.lower() for _edge, sig in tokens]
        if len(edge_signals) > 1 and any("reset" in sig or sig in ("rst", "rst_n", "areset", "aresetn") for sig in edge_signals):
            has_async_reset = True
            break
    return {
        "sequential": has_edge,
        "async_reset": has_async_reset,
    }


def _guess_task_id(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = [part for part in rel.parts if part not in (".", "")]
    return "__".join(parts) if parts else path.name


def _normalize_json_task(task: Dict[str, Any], fallback_number: int) -> Dict[str, Any]:
    task_number = task.get("task_number", fallback_number)
    try:
        task_number = int(task_number)
    except (TypeError, ValueError):
        task_number = fallback_number

    module_code = task.get("module_code") or task.get("rtl_code") or task.get("verified_verilog") or ""
    header = task.get("header") or task.get("module_header") or _extract_module_header(module_code)
    return {
        "task_number": task_number,
        "task_id": task.get("task_id") or task.get("id") or f"task_{task_number}",
        "description": task.get("description") or task.get("prompt") or "",
        "header": header,
        "module_code": module_code,
        "testbench": task.get("testbench", ""),
        "mutants": task.get("mutants", []),
        "result": task.get("result", []),
        "benchmark_source": task.get("benchmark_source", "json"),
    }


def load_json_benchmark(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        entries = [raw]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"Unexpected JSON benchmark format: {type(raw).__name__}")

    tasks = []
    for idx, item in enumerate(entries, start=1):
        if isinstance(item, dict):
            tasks.append(_normalize_json_task(item, idx))
    return tasks


def _find_first_existing(directory: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        path = directory / name
        if path.exists() and path.is_file():
            return path
    return None


def _find_first_matching(directory: Path, names: Iterable[str], patterns: Iterable[str]) -> Optional[Path]:
    exact = _find_first_existing(directory, names)
    if exact:
        return exact
    for pattern in patterns:
        matches = sorted(path for path in directory.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def load_rtllm_benchmark(root: Path) -> List[Dict[str, Any]]:
    """Load RTLLM-style design folders.

    A design folder is any directory with a design description and a verified
    Verilog file. Mutants/labels are optional; when absent, Pro-V can still run
    generation, oracle filling, judge, eval0, and eval1.
    """
    root = root.resolve()
    design_dirs = []
    for desc_path in root.rglob("design_description.txt"):
        directory = desc_path.parent
        rtl_path = _find_first_matching(
            directory,
            (
                "verified_verilog.v",
                "verified_Verilog.v",
                "verified_verilog.sv",
                "designer_RTL.v",
                "designer_RTL.sv",
                "design.v",
                "design.sv",
                "rtl.v",
                "rtl.sv",
            ),
            (
                "verified_*.v",
                "verified_*.sv",
            ),
        )
        if rtl_path:
            design_dirs.append((directory, desc_path, rtl_path))

    tasks = []
    for task_number, (directory, desc_path, rtl_path) in enumerate(sorted(design_dirs), start=1):
        original_module_code = _read_text(rtl_path)
        original_module_name = _choose_top_module_name(original_module_code, directory, rtl_path)
        module_code = (
            _rename_module(original_module_code, original_module_name, "top_module")
            if original_module_name and original_module_name != "top_module"
            else original_module_code
        )
        testbench_path = _find_first_existing(directory, ("testbench.v", "testbench.sv", "tb.v", "tb.sv"))
        task = {
            "task_number": task_number,
            "task_id": _guess_task_id(directory, root),
            "description": _read_text(desc_path).strip(),
            "header": _extract_module_header(module_code),
            "module_code": module_code,
            "testbench": _read_text(testbench_path) if testbench_path else "",
            "mutants": [],
            "result": [],
            "benchmark_source": "rtllm_folder",
            "source_dir": str(directory),
            "original_module_name": original_module_name,
        }
        tasks.append(task)
    return tasks


def load_benchmark_tasks(benchmark_path: str, benchmark_format: str = "auto") -> List[Dict[str, Any]]:
    path = Path(benchmark_path)
    fmt = (benchmark_format or "auto").lower()

    if fmt == "auto":
        fmt = "json" if path.is_file() and path.suffix.lower() == ".json" else "rtllm"

    if fmt in ("json", "hdlbits", "hdlbits_json"):
        return load_json_benchmark(path)
    if fmt in ("rtllm", "rtllm_folder", "folder"):
        return load_rtllm_benchmark(path)

    raise ValueError(f"Unsupported benchmark format: {benchmark_format}")


def write_normalized_benchmark(tasks: List[Dict[str, Any]], output_path: str) -> None:
    Path(output_path).write_text(json.dumps(tasks, indent=2))
