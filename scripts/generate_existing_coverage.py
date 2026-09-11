#!/usr/bin/env python3
"""Generate Verilator coverage.dat files for an existing Pro-V run directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pro_v.coverage_closure_loop import parse_coverage_dat  # noqa: E402


def _base_sim_dir(task_dir: Path) -> Path | None:
    for name in ("sim_cmb", "sim_seq"):
        candidate = task_dir / name
        if (candidate / "Makefile").exists() and (candidate / "top_module.v").exists():
            return candidate
    return None


def _fresh_coverage_dir(task_dir: Path) -> tuple[Path | None, str | None]:
    result_path = task_dir / "task_result.json"
    module_path = task_dir / "module_code.v"
    if not result_path.exists():
        return None, "missing task_result.json"
    if not module_path.exists():
        return None, "missing module_code.v"

    with open(result_path) as f:
        result = json.load(f)

    selected_tb = result.get("selected_testbench_path")
    if not selected_tb:
        return None, "missing selected_testbench_path"
    selected_tb_path = Path(selected_tb)
    if not selected_tb_path.exists():
        selected_tb_path = task_dir / selected_tb_path.name
    if not selected_tb_path.exists():
        return None, f"selected testbench not found: {selected_tb}"

    base = _base_sim_dir(task_dir)
    if base is None:
        return None, "no sim_cmb/sim_seq directory with Makefile/top_module.v"

    cov_dir = task_dir / "coverage_verilator"
    if cov_dir.exists():
        shutil.rmtree(cov_dir, ignore_errors=True)
    shutil.copytree(base, cov_dir, ignore=shutil.ignore_patterns("obj_dir", "coverage*.dat", "coverage_status.json"))
    shutil.copy2(module_path, cov_dir / "top_module.v")
    shutil.copy2(selected_tb_path, cov_dir / "testbench.json")
    return cov_dir, None


def _summary(path: Path) -> dict:
    report = parse_coverage_dat(str(path))
    return {
        category: {
            "covered": data["covered"],
            "total": data["total"],
            "pct": data["pct"],
        }
        for category, data in report.percentages().items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--timeout-seq", type=int, default=900)
    parser.add_argument("--timeout-cmb", type=int, default=420)
    parser.add_argument("--task-numbers", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    wanted = None
    if args.task_numbers.strip():
        wanted = {int(x) for x in args.task_numbers.replace(",", " ").split()}

    env = os.environ.copy()
    env["PATH"] = f"/home/abhijna/miniforge3/envs/pro-v/bin:{env.get('PATH', '')}"
    env["PRO_V_VERILATOR_COVERAGE"] = "1"

    rows = []
    for task_dir in sorted(run_dir.glob("task_*"), key=lambda p: int(p.name.split("_")[-1])):
        task_number = int(task_dir.name.split("_")[-1])
        if wanted is not None and task_number not in wanted:
            continue
        sim, setup_error = _fresh_coverage_dir(task_dir)
        row = {
            "task_number": task_number,
            "task_dir": str(task_dir),
            "sim_dir": str(sim) if sim else None,
            "coverage_dat": None,
            "returncode": None,
            "coverage": {},
            "error": None,
        }
        rows.append(row)
        if sim is None:
            row["error"] = setup_error
            continue

        cov = sim / "coverage.dat"
        for stale in [cov, sim / "coverage_status.json"]:
            if stale.exists():
                stale.unlink()
        if (sim / "obj_dir").exists():
            shutil.rmtree(sim / "obj_dir", ignore_errors=True)

        harness = sim / "harness-generator.py"
        if harness.exists():
            harness_proc = subprocess.run(
                [sys.executable, "harness-generator.py"],
                cwd=sim,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            row["harness_returncode"] = harness_proc.returncode
            row["harness_stdout_tail"] = (harness_proc.stdout or "")[-2000:]
            row["harness_stderr_tail"] = (harness_proc.stderr or "")[-2000:]
            if harness_proc.returncode != 0:
                row["error"] = "harness generation failed"
                with open(sim / "coverage_status.json", "w") as f:
                    json.dump(row, f, indent=2)
                print(f"task {task_number}: rc=None coverage=no harness_failed")
                continue

        base_name = _base_sim_dir(task_dir).name if _base_sim_dir(task_dir) else ""
        timeout = args.timeout_seq if base_name == "sim_seq" else args.timeout_cmb
        try:
            proc = subprocess.run(
                ["make", "-j1"],
                cwd=sim,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            row["returncode"] = proc.returncode
            row["stdout_tail"] = (proc.stdout or "")[-2000:]
            row["stderr_tail"] = (proc.stderr or "")[-2000:]
        except subprocess.TimeoutExpired as exc:
            row["returncode"] = "timeout"
            row["stdout_tail"] = (exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""))[-2000:]
            row["stderr_tail"] = (exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""))[-2000:]
            row["error"] = f"coverage make timed out after {timeout}s"

        if cov.exists():
            row["coverage_dat"] = str(cov)
            try:
                row["coverage"] = _summary(cov)
            except Exception as exc:
                row["error"] = f"coverage parse failed: {exc}"
        elif row["error"] is None:
            row["error"] = "coverage.dat not produced"

        with open(sim / "coverage_status.json", "w") as f:
            json.dump(row, f, indent=2)
        print(f"task {task_number}: rc={row['returncode']} coverage={'yes' if row['coverage_dat'] else 'no'}")

    out = run_dir / "verilator_coverage_summary.json"
    with open(out, "w") as f:
        json.dump({"run_dir": str(run_dir), "tasks": rows}, f, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
