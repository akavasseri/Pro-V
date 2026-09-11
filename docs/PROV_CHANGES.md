# Pro-V — Change Log & Memory Map

_Last updated: 2026-07-12. Maintainer notes for the testbench-strength / functional-coverage work layered on top of upstream `stable-lab/Pro-V` (arXiv 2506.12200)._

This document is the human-readable companion to the auto-memory notes. It records **what was added, what was modified, and why** — including the earlier work, not just the latest session.

---

## 0. TL;DR — the thesis of this work

Upstream Pro-V verifies RTL with an LLM multi-agent pipeline: generate a testbench (input stimulus) + a Python reference model (FRM / `golden_dut.py`), simulate the black-box DUT (`top_module.v`) on the stimulus, and compare outputs. The paper calls this "coverage-driven," but the original `gen_tb.py` has **no coverage feedback** — stimulus is effectively random/LLM-guessed.

The work here adds the missing rigor:

1. **Measure** testbench strength honestly — not label-agreement, but **true mutation score** = `killed / (total − provably-equivalent)`.
2. **Improve** it — a reference-guided *functional coverage agent* that picks high-information stimulus (the inputs where similar operations disagree), plus witness/closure loops that keep adding stimulus until only equivalent mutants survive.
3. **Keep it leakage-free** — the golden RTL (`module_code`) is used ONLY offline as a measurement oracle, never inside the deployed checker (that would be circular and unavailable at deployment).

Key empirical finding so far: **structural coverage (line/toggle) is necessary but not sufficient** — a suite can read ~100% line coverage yet miss the mutation-distinguishing input. Mutation / true-score is the real adequacy metric.

---

## 1. New modules (all under `pro_v/`)

| File | LOC | Role |
|---|---|---|
| `functional_coverage_agent.py` | 1228 | **Reference-guided functional coverage agent** — the headline deliverable. Pre-testbench planning stage. |
| `coverage_closure_loop.py` | 584 | Iterative coverage-closure driver around the agent. |
| `coverage_gen.py` | 721 | FRM-driven coverage-closure test generation (ATPG-style / cone-of-influence / reachability). |
| `mutation_strength.py` | 424 | Differential witness + equivalence engine (iverilog). Defines *true mutation score*. |
| `strengthen_testbench.py` | 343 | Auto-iterating loop: evaluate → witnesses → augment stimulus → regenerate testbench → re-eval until converged. |
| `verilog_tb_generator.py` | 308 | 4-state self-checking Verilog TB (iverilog) — dual-oracle for X/Z bugs Verilator hides. |
| `connect_test.py` | 208 | GPU-free end-to-end pipeline smoke test with a fake LLM client; set `VLLM_URL` to hit a live model. |

### 1.1 `functional_coverage_agent.py` — the functional coverage agent
- **Position:** `spec + golden_dut.py → functional_coverage_agent → coverage_plan.json → testbench generator → RTL sim`. It decides *what should be tested*; it emits no HDL.
- **Source of truth:** the Python reference model (`golden_dut.py` / FRM) + spec/interface metadata. It reads `top_module.v` for **one purpose only** — confirming the port interface (names/widths/clk/reset). It never inspects the DUT's internal logic (that would leak the thing under test).
- **Core idea:** for an AND vs OR gate, `{00, 11}` gives "50% input coverage" and shows both output values yet cannot distinguish the two — because AND and OR *agree* on `00`/`11`. The distinguishing inputs are `01`/`10`. The agent generalizes this: drive the golden model to pick high-information inputs that separate similar operations, cover output classes, and (for sequential designs) reach meaningful states/transitions — then minimize to a small set.
- **Completeness policy (combinational, added 2026-07-12).** The objective is that nothing distinguishing is left untested — NOT a minimal vector count. `Budget.exhaustive_max_bits=11`, `Budget.sample_cap=2048`:
  - ≤ 8 input bits → exhaustive (2ⁿ, ≤256); 9/10/11 bits → exhaustive 512/1024/2048.
  - ≥ 12 bits → structured high-information seeds + uniform-random fill to a hard cap of 2048.
  - Emitted stimulus = the full set UNION both members of every ATPG distinguishing pair; the old `greedy_cover` minimization is bypassed for emission. This replaced the earlier minimality behavior that caused FCA to miss mutants (e.g. `bugs_mux2` `sel=1,a≠0`). See `docs/FCA_HEADTOHEAD_RESULTS.md`.
- **CLI:** `python -m pro_v.functional_coverage_agent --frm golden_dut.py --dut top_module.v --out coverage_plan.json [--stimulus-out stim.json] [--type cmb|seq] [--sample-cap 2048] [--exhaustive-max-bits 11] ...`
- **Key funcs:** `build_plan(...)`, `GoldenModelAdapter.run(vec)`, `generate_candidates(...)`, `write_plan(...)`. Reuses `coverage_gen.load_frm` and `mutation_strength.parse_ports`.
- **Tests:** `pro_v/tests/test_functional_coverage_agent.py` — 9/9 pass (exhaustive small space, bounded sampling, distinguishing-input for equality, output→input backtracking, redundant-test removal, sequential BFS path discovery, header-authoritative FRM consistency, stimulus downconversion, budget cap).

### 1.2 `coverage_gen.py` — FRM-driven coverage generation
- CMB: dynamic cone-of-influence + input-sensitivity coverage (ATPG-style; forces the `01`/`10` that separate AND from OR).
- SEQ: reachability BFS + transition cover; state snapshotted via `deepcopy(dut.__dict__)` (no FRM contract change).
- `analyze_frm` / `analyze_frm_class`: AST-pulls inputs-read / outputs-written / operators / branches from the Python reference and cross-checks against DUT ports for blind spots.
- `close_line_coverage`: `sys.settrace` greedy set-cover → forces every reachable FRM branch (e.g. a special-case `x==0xABC` that random almost never hits).
- Auto-detects cmb/seq by `GoldenDUT.load` arity. Selftest PASS.

### 1.3 `mutation_strength.py` — the honesty layer
- Differential search: reference RTL (`module_code`) vs mutant RTL over an independent probe set (CMB: exhaustive if input bits ≤ threshold else random; SEQ: random sequences via `$readmemb`, single iverilog compile). Uses `module_code` **purely as an offline oracle** — never enters the deployed checker.
- Classifies each surviving mutant: `equivalent` (no distinguishing input exists) vs `weak_survivor` (a witness input exists that the stimulus missed).
- Emits witnesses in `stimulus.json` format to inject back and re-run → raises the real kill rate. Loop until only equivalent mutants survive.
- **True mutation score = killed / (total − equivalent).** `classify_mutant(module_code, mutant, circuit_type, ...)`. Selftest PASS (AND→OR ⇒ witnesses a≠b; b&a ⇒ proven equivalent).
- **This is why iverilog is a dependency** (differential engine). Verilator handles the main DUT sim; iverilog handles equivalence proofs + 4-state.

### 1.4 `strengthen_testbench.py` — the closure loop
- Chain: `evaluate → witnesses → augment stimulus → regenerate testbench.json (golden_dut.py FRM → harness-generator.py → make) → re-eval` until converged.
- `regenerate_stimulus_with_coverage()`, `strengthen_task(coverage_seed=True)`, CLI `--coverage-seed`.
- E2E proven on AND→OR through real Verilator: weak random `{00,11}` ⇒ OR mutant SURVIVES; coverage_gen stimulus ⇒ OR mutant KILLED in 0 witness rounds.

### 1.5 `verilog_tb_generator.py` — dual-oracle for X/Z
- Self-checking 4-state Verilog TB from `testbench.json`, run via iverilog (`!==` catches X/Z that 2-state Verilator coerces).
- `generate_tb`, `run_iverilog_tb`, `differential_check(dut, sim_dir)` runs BOTH harnesses and flags disagreements.
- Proven: X-bug DUT `y=a?a&b:1'bx` PASSES Verilator (2-state) but FAILS iverilog (4-state). Keep Verilator for speed; Verilog TB for X/Z fidelity + portability.

### 1.6 `connect_test.py` — GPU-free pipeline test
- `FakeLLMClient` (duck-typed `.chat()`) drives the REAL `GenTBAgent` + `PyCheckerAgent` (subprocess worker=None) → testbench → Verilator → mutation. Passes (112 vectors, ref passes, 2/3 killed, 1 equiv, true score 100%). Set `VLLM_URL` to smoke-test a live endpoint.

---

## 2. Modified upstream files

| File | ~Δ | What changed |
|---|---|---|
| `pro_v/simulate_and_evaluate_mutants.py` | +262 | Integrated `--strength` (analyze equivalence/witnesses), `--augment-stimulus` (write witnesses back), `--exclude-noncompiling` (drop non-compiling mutants from agreement + true score). Adds EvaluationMetrics fields: equivalent / weak-survivor counts, `true_mutation_score`, per-mutant compile tracking. |
| `pro_v/sim_cmb/Makefile` | +11 | `--coverage-toggle` wiring for real Verilator coverage. |
| `pro_v/sim_cmb/harness-generator.py` | +10 | Coverage write `top->contextp()->coveragep()->write("coverage.dat")`, guarded `#if VM_COVERAGE`. |
| `pro_v/sim_cmb/sim-main.cpp` | +5 | Coverage hook. |
| `pro_v/sim_seq/Makefile` | +11 | Same coverage-toggle wiring for sequential. |
| `pro_v/sim_seq/harness-generator.py` | +27 | Coverage + sequential harness adjustments. |
| `pro_v/sim_seq/sim-main.cpp` | +5 | Coverage hook. |
| `pro_v/__init__.py` | −? | Removed a dead `back_up` import that broke `import pro_v.*`. |

> Note: `--coverage-line` gives 0 points for pure `assign` (needs procedural `always`/`case`). Toggle coverage is the meaningful structural metric for combinational assign-style designs.

---

## 3. New scripts & fixtures

| Path | Role |
|---|---|
| `scripts/adroit_run.sbatch` | Slurm job for Adroit: launch vLLM (`PRO-V-R1-8B`, A100) → `connect_test` → `run_evaluation_think_simple.sh` on the real benchmark → `simulate_and_evaluate_mutants.py --strength`. |
| `scripts/launch_vllm_local.sh` | vLLM OpenAI server for `./models/PRO-V-R1-8B` (Qwen3ForCausalLM, 4 shards), GPU box only, with preflight. |
| `benchmark/prov_mini_benchmark.json` | 4 real tasks (HDLBits Mux2to1v/Priority4, RTLLM ALU4/Comparator4), correct Pro-V format. GPU-free fixture. |
| `pro_v/examples/` | `and_gate/`, `or_gate/`, `traffic_fsm/` fixtures (golden_dut.py + top_module.v + coverage_plan.json) + `run_examples.py`. |
| `pro_v/tests/` | `test_functional_coverage_agent.py` (9/9), `test_coverage_closure_loop.py` (5/5). |

---

## 4. Leakage audit (important, keep true)

- **Eval flow is already golden-free.** Both eval top-agents (`prompting_top_agent_simple.py`, `prompting_top_agent_ray.py`) use `pychecker.py`, whose prompts are spec-only (`{description}` / `{module_header}`, no golden RTL). Eval numbers do not leak.
- Golden RTL enters ONLY: (a) RL-training env `pychecker_agent.py` `create_prompt(prompt_type="api")`; (b) `prompt.py` generation templates. A toggle (`PROV_GOLDEN_RTL_IN_PROMPT`, default **off**) keeps the RL inference prompt spec-only.
- **Why golden RTL must NOT build the testbench:** circular (mutant-agreement → trivially 100%), data leakage (no golden at deployment), and it kills the independence that gives the checker its power. Using golden RTL *offline to measure* strength is fine (meta-eval); using it to *build the deployed checker* is the leak.
- eval0/1/2 not falsified: the harness has one DUT instance compared to `testbench.json` (FRM outputs), no golden-RTL oracle. Caveats: `--mutant-only` forces eval0/1 True unmeasured; compile-failing mutants otherwise count as "detected" (use `--exclude-noncompiling`).

---

## 5. Pipeline order (how the pieces compose)

```
spec + golden_dut.py (FRM)
   └─> functional_coverage_agent.py  ── coverage_plan.json / stimulus.json   (smart stimulus: WHAT to test)
         └─> coverage_gen.py         ── ATPG / cone-of-influence / reachability closure
               └─> testbench.json (FRM computes expected_outputs)
                     └─> Verilator (fast, 2-state)  +  verilog_tb_generator (iverilog, 4-state X/Z)
                           └─> mutation_strength.py  ── witnesses + equivalence  ⇒  true mutation score
                                 └─> strengthen_testbench.py  ── loop until only equivalent mutants survive
```
Coverage measured on the FRM is a **proxy**; always verify against the DUT + mutants.

---

## 6. Environment map

**Local Mac** (`/Users/akavasseri/Pro-V`): no GPU/torch/vllm. Conda env `pro-v` at `/opt/homebrew/Caskroom/miniforge/base/envs/pro-v/bin/python` has numpy. iverilog + verilator (homebrew) present. Only the 4-task mini benchmark is local — the real 156-task HDLBits set is NOT here.

**Adroit** (`adroit.princeton.edu`, user `ak7587`): the real work box.
- Repo: `/scratch/network/ak7587/Pro-V`.
- Real benchmark: `verilog-eval/HDLBits/test_benchmark_new.json` — **156 tasks, 10 mutants each**, `result` = per-mutant `[true/false]` labels, `circuit_type` null (detect seq via `clk` in header; ~73 of 156 are sequential).
- Conda env `pro-v` at `/scratch/network/ak7587/envs/pro-v` — vllm 0.21.0, torch 2.11.0+cu130, verilator present; **iverilog installed from conda-forge (this session)**.
- conda comes from `module load anaconda3/2024.6`.
- Model: `models/PRO-V-R1-8B` (Qwen3ForCausalLM).
- GPU: partition `gpu`, node `adroit-h11g1` = `nvidia_a100:4` (80 GB). Request `--partition=gpu --gres=gpu:nvidia_a100:1`.
- SSH: `~/.ssh/config` Host `adroit` with `ControlMaster auto` + `ControlPersist 8h` (shared socket → no repeat Duo). Requires Princeton GlobalProtect VPN off-campus.
- Local→remote sync via `rsync` (macOS ships rsync 2.6.9 — no `--info`/`--backup-dir`; use `-az -b --suffix=`). Overwritten remote files backed up with `.prebak` suffix.

---

## 7. Session log (2026-07-12) — full HDLBits benchmark run on Adroit

Goal: run the real HDLBits benchmark on the A100, measure how the functional coverage agent helps, and analyze whether the resulting testbench is **higher quality than the paper's**.

Steps taken:
1. Established SSH/ControlMaster to Adroit (VPN + Duo, user-initiated).
2. Found the real benchmark on remote; confirmed none of the 7 new modules were present remotely → full sync.
3. rsync'd all new modules + tests + examples + mini benchmark to `/scratch/network/ak7587/Pro-V`.
4. Installed `iverilog` into the remote `pro-v` conda env (verilator already present).
5. Verified sync integrity: remote runs `test_functional_coverage_agent.py` 9/9, `test_coverage_closure_loop.py` 5/5.
6. _(in progress)_ Build a subset benchmark (5–10 tasks) + cmb/seq detection; adapt the sbatch for A100; build a head-to-head harness (paper stimulus vs FCA: true mutation score + structural coverage); submit + analyze.

**Comparison design:** same tasks & mutants, two stimulus sources — (a) paper-style Pro-V `gen_tb` stimulus, (b) functional-coverage-agent stimulus — scored on **true mutation score (equivalent-excluded)** AND structural (toggle/line) coverage. Both metrics, because structural coverage alone is known to be necessary-not-sufficient.
