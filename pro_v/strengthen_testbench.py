#!/usr/bin/env python3
"""
strengthen_testbench.py

Closed-loop testbench strengthening driver.

Each round:
  1. Evaluate a task's testbench WITH mutation-strength analysis: classify every
     surviving mutant as equivalent vs weak-survivor and extract witnesses (inputs
     that distinguish a weak-survivor mutant from the reference).
  2. If there are no weak-survivors, the testbench is as strong as it can get
     (only equivalent / unknown mutants remain) -> converged, stop.
  3. Otherwise inject the witnesses into stimulus.json, regenerate testbench.json
     by re-running the FRM (golden_dut.py) over the enlarged stimulus, regenerate
     and recompile the harness, and repeat.

It prints the true mutation score per round so you can watch strength climb.

Requires a COMPLETED Pro-V run's task outputs to be present:
    <experiment_dir>/task_<n>/sim_<cmb|seq>/  containing top_module.v,
    harness-generator.py, Makefile, ...  plus golden_dut.py and stimulus.json
    (in the sim dir or the parent task folder), and iverilog + verilator.

This driver never lets the golden RTL seed the checker: witnesses come from an
independent reference-vs-mutant differential search, and expected outputs are
always (re)computed by the FRM program, exactly as in normal operation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the repo root is importable when run as a script (python pro_v/strengthen_testbench.py)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pro_v.simulate_and_evaluate_mutants import (  # noqa: E402
    evaluate_single_task,
    EvaluationMetrics,
    STRENGTH_AVAILABLE,
    augment_stimulus_file,
)
from pro_v.coverage_gen import generate as coverage_generate  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution (mirrors evaluate_single_task)
# ---------------------------------------------------------------------------

def resolve_sim_dir(experiment_dir: Path, task_number: int) -> Tuple[Path, Path, str]:
    """Return (task_dir, sim_dir, circuit_type) for a task, mirroring the evaluator."""
    task_dir = experiment_dir / f"task_{task_number}"
    circuit_type = "cmb"
    tr = task_dir / "task_result.json"
    if tr.exists():
        try:
            with open(tr) as f:
                circuit_type = json.load(f).get("circuit_type", "cmb").lower()
        except Exception:
            pass
    sim_dir = task_dir / f"sim_{circuit_type}"
    return task_dir, sim_dir, circuit_type


def _locate(name: str, dirs: List[Path]) -> Optional[Path]:
    for d in dirs:
        p = d / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Testbench regeneration (FRM -> testbench.json -> harness -> recompile)
# ---------------------------------------------------------------------------

def regenerate_testbench(sim_dir: Path, timeout: int = 180) -> Tuple[bool, str]:
    """Re-run the FRM over the (augmented) stimulus and rebuild the harness.

    Runs, with cwd = sim_dir:
        python golden_dut.py        # stimulus.json -> testbench.json (FRM outputs)
        python harness-generator.py # testbench.json + top_module.v -> rfuzz-harness.cpp
    The subsequent evaluation's `make` recompiles from the regenerated harness.
    """
    parent = sim_dir.parent

    golden = _locate("golden_dut.py", [sim_dir, parent])
    if golden is None:
        return False, "golden_dut.py (FRM program) not found in sim dir or task folder"

    # The FRM reads ./stimulus.json and writes ./testbench.json relative to cwd,
    # so the augmented stimulus must live in sim_dir.
    if not (sim_dir / "stimulus.json").exists():
        src = _locate("stimulus.json", [parent])
        if src is None:
            return False, "stimulus.json not found in sim dir or task folder"
        shutil.copy(src, sim_dir / "stimulus.json")

    # 1) FRM -> testbench.json
    try:
        r = subprocess.run(["python", str(golden)], cwd=sim_dir,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "FRM (golden_dut.py) timed out"
    if not (sim_dir / "testbench.json").exists():
        return False, f"FRM did not produce testbench.json (stderr: {r.stderr[:300]})"

    # 2) harness-generator.py -> rfuzz-harness.cpp
    hg = _locate("harness-generator.py", [sim_dir])
    if hg is None:
        return False, "harness-generator.py not found in sim dir"
    try:
        r2 = subprocess.run(["python", "harness-generator.py"], cwd=sim_dir,
                            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "harness-generator.py timed out"
    if r2.returncode != 0:
        return False, f"harness-generator.py failed: {r2.stderr[:300]}"

    return True, "regenerated testbench.json + harness"


def regenerate_stimulus_with_coverage(sim_dir: Path, timeout: int = 180,
                                      **cov_kwargs) -> Tuple[bool, str, dict]:
    """Replace stimulus.json with a coverage-closed suite generated from the FRM.

    Runs the coverage-closure agent over golden_dut.py + top_module.v to produce
    a logically-motivated stimulus (input-sensitivity coverage for combinational,
    reachability/transition cover for sequential), writes it to sim_dir, and
    regenerates testbench.json + harness. Returns (ok, msg, coverage_report).
    """
    parent = sim_dir.parent
    frm = _locate("golden_dut.py", [sim_dir, parent])
    dut = _locate("top_module.v", [sim_dir])
    if frm is None or dut is None:
        return False, "golden_dut.py or top_module.v not found for coverage-gen", {}
    try:
        stimulus, report = coverage_generate(str(frm), str(dut), **cov_kwargs)
    except Exception as e:
        return False, f"coverage-gen failed: {e}", {}
    if not stimulus:
        return False, f"coverage-gen produced no vectors ({report})", report
    with open(sim_dir / "stimulus.json", "w") as f:
        json.dump(stimulus, f, indent=2)
    ok, msg = regenerate_testbench(sim_dir, timeout=timeout)
    return ok, (f"coverage-gen: {len(stimulus)} vectors; " + msg), report


# ---------------------------------------------------------------------------
# Per-task strengthening loop
# ---------------------------------------------------------------------------

def strengthen_task(task_data: Dict, experiment_dir: Path, *,
                    max_rounds: int = 5, classify_kwargs: Optional[Dict] = None,
                    timeout: int = 180, exclude_noncompiling: bool = False,
                    coverage_seed: bool = False) -> Dict:
    """Iterate find-witnesses -> inject -> regenerate until convergence.

    If coverage_seed is True, first replace the stimulus with a coverage-closed
    suite from the FRM (input-sensitivity / reachability coverage) before the
    witness loop -- "generate smart, then verify strong".

    Returns a summary dict with the per-round history.
    """
    classify_kwargs = classify_kwargs or {}
    task_number = task_data["task_number"]
    task_id = task_data.get("task_id", f"task_{task_number}")
    _, sim_dir, circuit_type = resolve_sim_dir(experiment_dir, task_number)

    coverage_report = None
    if coverage_seed:
        ok, msg, coverage_report = regenerate_stimulus_with_coverage(sim_dir, timeout=timeout)
        logger.info(f"[{task_id}] coverage-seed: {msg}")
        if not ok:
            logger.warning(f"[{task_id}] coverage-seed failed; falling back to existing stimulus")

    history = []
    reason = "max_rounds_reached"
    for rnd in range(max_rounds + 1):
        strength_opts = {"enabled": True, "augment_stimulus": False, **classify_kwargs}
        m: EvaluationMetrics = evaluate_single_task(
            task_data, experiment_dir, mutant_only=False,
            timeout=timeout, strength_opts=strength_opts,
            exclude_noncompiling=exclude_noncompiling,
        )
        history.append({
            "round": rnd,
            "module_passes": m.module_passes,
            "killed": m.mutants_detected,
            "total_mutants": m.total_mutants,
            "equivalent": m.equivalent_mutants,
            "weak_survivors": m.weak_survivors,
            "unknown_survivors": m.unknown_survivors,
            "true_mutation_score": m.true_mutation_score,
        })
        logger.info(f"[{task_id}] round {rnd}: true_score={m.true_mutation_score:.2%} "
                    f"killed={m.mutants_detected}/{m.total_mutants} "
                    f"weak={m.weak_survivors} equiv={m.equivalent_mutants}")

        if not m.module_passes:
            # After injecting new inputs, the reference no longer passes its own
            # testbench: the FRM disagrees with the reference on a new input.
            # That is an FRM bug surfaced by strengthening - stop and report it.
            reason = "reference_no_longer_passes_FRM_may_be_buggy"
            logger.warning(f"[{task_id}] reference module stopped passing after "
                           f"regeneration - FRM likely wrong on a new input; stopping")
            break
        if m.weak_survivors == 0:
            reason = "converged"
            logger.info(f"[{task_id}] converged: no weak-survivors remain")
            break
        if rnd == max_rounds:
            break

        # Inject this round's witnesses and regenerate the testbench.
        if not (sim_dir / "stimulus.json").exists():
            src = _locate("stimulus.json", [sim_dir.parent])
            if src is not None:
                shutil.copy(src, sim_dir / "stimulus.json")
        added = augment_stimulus_file(str(sim_dir / "stimulus.json"), m.witnesses)
        ok, msg = regenerate_testbench(sim_dir, timeout=timeout)
        logger.info(f"[{task_id}] round {rnd}: injected {added} witness(es); {msg}")
        if not ok:
            reason = f"regeneration_failed: {msg}"
            logger.error(f"[{task_id}] {reason}")
            break

    baseline = history[0]["true_mutation_score"] if history else -1.0
    final = history[-1]["true_mutation_score"] if history else -1.0
    return {
        "task_id": task_id,
        "task_number": task_number,
        "circuit_type": circuit_type,
        "stop_reason": reason,
        "rounds_run": len(history) - 1,
        "baseline_true_score": baseline,
        "final_true_score": final,
        "coverage_seed_report": coverage_report,
        "history": history,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop testbench strengthening via witness feedback")
    parser.add_argument("--experiment_dir", "-e", required=True,
                       help="Experiment outputs dir (e.g. outputs/pro_v_eval)")
    parser.add_argument("--benchmark", "-b",
                       default="verilog-eval/HDLBits/merged_benchmark.json",
                       help="Benchmark JSON with module_code/mutants/result")
    parser.add_argument("--task", type=int, default=None,
                       help="Single task_number to strengthen (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Max number of tasks (when --task not given)")
    parser.add_argument("--max-rounds", type=int, default=5,
                       help="Max strengthening rounds per task (default: 5)")
    parser.add_argument("--max-exhaustive-bits", type=int, default=16,
                       help="CMB exhaustive-sweep bit budget (default: 16)")
    parser.add_argument("--random-samples", type=int, default=50000,
                       help="CMB random probe count when not exhaustive (default: 50000)")
    parser.add_argument("--exclude-noncompiling", action="store_true",
                       help="Score excluding non-compiling mutants")
    parser.add_argument("--coverage-seed", action="store_true",
                       help="Before the witness loop, replace stimulus with a "
                            "coverage-closed suite generated from the FRM (adds logic "
                            "vs random+corners)")
    parser.add_argument("--timeout", "-t", type=int, default=180)
    parser.add_argument("--output", "-o", default="strengthen_report.json")
    args = parser.parse_args()

    if not STRENGTH_AVAILABLE:
        logger.error("mutation_strength unavailable (need iverilog). Aborting.")
        sys.exit(1)

    with open(args.benchmark) as f:
        benchmark = json.load(f)

    if args.task is not None:
        tasks = [t for t in benchmark if t.get("task_number") == args.task]
        if not tasks:
            logger.error(f"task_number {args.task} not found in benchmark")
            sys.exit(1)
    else:
        tasks = benchmark[:args.limit] if args.limit else benchmark

    classify_kwargs = {
        "max_exhaustive_bits": args.max_exhaustive_bits,
        "random_samples": args.random_samples,
        "timeout": args.timeout,
    }

    experiment_dir = Path(args.experiment_dir)
    summaries = []
    for t in tasks:
        summary = strengthen_task(
            t, experiment_dir,
            max_rounds=args.max_rounds,
            classify_kwargs=classify_kwargs,
            timeout=args.timeout,
            exclude_noncompiling=args.exclude_noncompiling,
            coverage_seed=args.coverage_seed,
        )
        summaries.append(summary)
        logger.info(f"[{summary['task_id']}] {summary['stop_reason']}: "
                    f"true score {summary['baseline_true_score']:.2%} -> "
                    f"{summary['final_true_score']:.2%} "
                    f"over {summary['rounds_run']} round(s)")

    with open(args.output, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info(f"Wrote strengthening report to {args.output}")

    # Aggregate
    improved = [s for s in summaries
                if s["final_true_score"] > s["baseline_true_score"]]
    converged = [s for s in summaries if s["stop_reason"] == "converged"]
    logger.info("=" * 70)
    logger.info(f"Tasks strengthened: {len(summaries)} | improved: {len(improved)} | "
                f"converged: {len(converged)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
