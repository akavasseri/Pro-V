# FCA vs. Paper — Head-to-Head on HDLBits (Testbench Strength)

_Run 2026-07-12 on Adroit (A100 80GB), 15 random HDLBits tasks (seed=1234). No fluff: this records what actually happened, including where the FCA did NOT win._

## What was run

- **Fresh GPU generation** of the real Pro-V paper pipeline (`run_evaluation_think_simple.sh`, `PRO-V-R1-8B` on one A100) over 15 randomly-drawn HDLBits tasks: **2,5,8,9,22,24,26,30,61,89,91,113,119,124,150** (11 combinational, 4 sequential). Output: `outputs/fca_headtohead_15/`. 14/15 generated; **task_8 (`fsm3`, sequential) never finished generating** (model stuck in `<think>`), so it's excluded.
- **Head-to-head scorer** (`pro_v/compare_testbench_quality.py`): holds the model-generated FRM (`golden_dut.py`) fixed and swaps only the *stimulus source* —
  - **paper arm** = the model's own `stimulus.json` (random, `random.getrandbits`),
  - **FCA arm** = `functional_coverage_agent` driving the same FRM.
  Both scored by **true mutation score** = `killed / (mutants − provably-equivalent)`, using the benchmark's `module_code` as an offline differential oracle (iverilog). This isolates exactly what the FCA changes.

## The key correction mid-run

The first FCA run **lost** (0.889 vs paper 1.000) because the agent was optimizing for *minimality* (~7–10 vectors). That was the wrong objective. The FCA now follows a **completeness policy**:

| total input bits | vectors emitted |
|---|---|
| ≤ 8 | **exhaustive** (2ⁿ, ≤ 256) |
| 9 / 10 / 11 | exhaustive = 512 / 1024 / 2048 |
| ≥ 12 | structured seeds + random fill to a **hard cap of 2048** |

…**unioned with** the ATPG distinguishing-pair vectors, and **never pruned**. The point is that nothing distinguishing is left untested — not to produce as few tests as possible.

## Result (combinational, completeness policy)

**9 scored tasks · 72 non-equivalent mutants · 7 equivalent mutants excluded**

| | Paper (random) | FCA (completeness) |
|---|---|---|
| **true mutation score** | **1.000** (72/72) | **1.000** (72/72) |
| per-task wins | 0 | 0 (9 ties) |
| FCA-only kills | — | 0 |
| paper-only kills | 0 | — |
| mean vectors | 520.7 | 512.1 |

Per task (vector counts reveal the policy — exhaustive ≤11 bits, else 2048-sample):

| task | id | eff. mut | equiv | paper vec | FCA vec | paper kill | FCA kill |
|---|---|---|---|---|---|---|---|
| 2 | m2014_q6b | 10 | 0 | 16 | 16 (exh) | 10 | 10 |
| 9 | vector2 | 5 | 0 | 4096 | 2071 (cap) | 5 | 5 |
| 22 | m2014_q6c | 10 | 0 | 128 | 128 (exh) | 10 | 10 |
| 26 | circuit4 | 9 | 1 | 16 | 16 (exh) | 9 | 9 |
| 30 | bugs_mux2 | 6 | **3** | 100 | 2048 (cap) | 6 | 6 |
| 61 | notgate | 3 | 2 | 2 | 2 (exh) | 3 | 3 |
| 91 | kmap1 | 10 | 0 | 8 | 8 (exh) | 10 | 10 |
| 113 | vectorgates | 10 | 0 | 64 | 64 (exh) | 10 | 10 |
| 119 | vector4 | 9 | 1 | 256 | 256 (exh) | 9 | 9 |

(`zero`, `step_one` = 0 effective mutants, unscored.)

## Honest reading of this

1. **On this random HDLBits sample, the FCA does not beat the paper on raw strength — it matches it (100% vs 100%).** These mutants are operator/wiring mutations that many inputs expose, so the paper's random stimulus (100–4096 vectors over small spaces) saturates them by volume. Any adequate testbench maxes out here.

2. **The FCA's advantage is a guarantee, not a bigger number.** For ≤11-bit designs (7 of 9 tasks) the FCA is *exhaustive* — it **cannot** miss a non-equivalent mutant, by construction. The paper's random stimulus reaches the same result by luck and volume (e.g. 4096 vectors on `vector2`, but only 2 on `notgate`), with no guarantee. On a design whose bug triggers on a *rare* input (a `x==0xABC` special case in a wide input space), random under-sampling would miss and the FCA's exhaustive/2048-sample would not. This benchmark simply doesn't contain such adversarial mutants.

3. **The first-cut minimality objective was a real bug**, caught here: `bugs_mux2` mutants 7 & 9 (`out=sel?a:b`) survived a 7-vector FCA suite because it never drove `a≠0` while `sel=1` — it toggled the *unselected* datapath. Input-toggle coverage still read 1.00. The completeness policy fixes it; the witness (`sel=1, a=1, b=0`) is emitted automatically.

## What the paper's own results CANNOT show (the actual contribution)

Testbench strength is not what the paper's pass/detection tables measure. This run demonstrates four things only the strength layer surfaces:

- **Equivalence.** 7 of 79 combinational mutants are *provably equivalent* — no testbench can kill them (`bugs_mux2` alone has 3). The paper's raw metric conflates "survived because equivalent" with "survived because the testbench is weak." True mutation score excludes them; raw detection can't.
- **Witnesses.** When a mutant survives, the differential engine returns the *exact distinguishing input*. The paper reports only pass/fail — no route to "what input would have caught this."
- **Structural coverage is necessary, not sufficient.** The weak 7-vector suite and the complete suite both read **100% input-toggle coverage**, yet only the complete one kills `bugs_mux2`. Coverage % ≠ adequacy; mutation score is the real metric.
- **Harness fragility.** The paper's Verilator-based mutant eval hit **compile failures on 4 of 11 combinational tasks** (2, 22, 26, 30); the iverilog differential scored all of them cleanly. Two independent simulators disagree on what even builds.

## Caveats (kept honest)

- **Sequential is not covered by the completeness fix.** The policy above is combinational (input-space enumeration). Sequential strength is state/transition coverage — a different mechanism (BFS). The preliminary seq numbers (paper 0.947 vs FCA 0.737) used the *old* minimal FCA and a scenario-flattening scorer caveat, so they are **not** a fair FCA measurement yet. Sequential completeness is open work.
- The comparison oracle is `module_code` (ground truth), which measures *stimulus adequacy under a correct oracle* — isolating the FCA's contribution. The paper's deployed testbench also depends on its model-generated FRM being correct; that dimension (FRM oracle errors) is separate and not scored here.
- 14/15 tasks; `fsm3` generation did not complete.

## Per-task eval0 / eval1 / eval2 (paper pipeline, model-FRM oracle)

Computed with iverilog against each task's model-generated `testbench.json` (fast, no hang; matches Pro-V's own tool on eval0/eval1 and on detection counts). `FCA_str` = FCA stimulus strength from the head-to-head (kills / non-equivalent, `module_code` oracle).

- **eval0** = DUT+testbench compile · **eval1** = correct `module_code` passes the testbench · **eval2_det** = mutants the testbench flags · **eval2_agr** = agreement with the benchmark's own labels (True=detectable), corrected convention.

| task | id | type | eval0 | eval1 | eval2_det | eval2_agr | FCA_str | note |
|---|---|---|:--:|:--:|:--:|:--:|:--:|---|
| 2 | m2014_q6b | cmb | Y | **N** | n/a | n/a | 10/10 | model FRM wrong |
| 5 | zero | cmb | Y | Y | 0/5 | 5/5 | – | (all mutants equivalent) |
| 8 | fsm3 | seq | – | – | – | – | – | generation failed |
| 9 | vector2 | cmb | Y | **N** | n/a | n/a | 5/5 | model FRM wrong |
| 22 | m2014_q6c | cmb | Y | **N** | n/a | n/a | 10/10 | model FRM wrong |
| 24 | 2014_q4a | seq | Y | **N** | n/a | n/a | – | model FRM wrong (seq, best-effort TB) |
| 26 | circuit4 | cmb | Y | **N** | n/a | n/a | 9/9 | model FRM wrong |
| 30 | bugs_mux2 | cmb | Y | **N** | n/a | n/a | 6/6 | model FRM wrong (8-bit mux → garbage) |
| 61 | notgate | cmb | Y | Y | 3/5 | 5/5 | 3/3 | |
| 89 | count15 | seq | Y | **N** | n/a | n/a | – | model FRM wrong (seq, best-effort TB) |
| 91 | kmap1 | cmb | Y | Y | 10/10 | 10/10 | 10/10 | |
| 113 | vectorgates | cmb | Y | Y | 10/10 | 10/10 | 10/10 | |
| 119 | vector4 | cmb | Y | Y | 9/10 | 10/10 | 9/9 | |
| 124 | step_one | cmb | Y | Y | 3/10 | 10/10 | – | (7 mutants equivalent) |
| 150 | lemmings4 | seq | Y | **N** | n/a | n/a | – | model FRM wrong (also fails Verilator) |

**Two findings from this table that the paper's headline numbers do not surface:**

1. **The dominant failure is the model FRM, not the stimulus.** 8 of 14 generated tasks have a *wrong* model reference model — the correct `module_code` fails the model's own testbench (verified on `bugs_mux2`: the FRM maps an 8-bit mux to a 1-bit-ish garbage output; corroborated by two independent simulators). On every one of these, the *stimulus* is fine — `FCA_str` = 100% under the correct oracle. A wrong oracle invalidates the testbench regardless of stimulus quality; this is orthogonal to what the FCA improves, and only strength analysis exposes it.
2. **Pro-V's own `calculate_agreement` is label-inverted.** The benchmark's `result[i]=True` means "mutant is detectable" (verified at position level: detected positions == label-True positions exactly on tasks 61/119/124). But `calculate_agreement` computes `detected == (not label)`, reporting ~0% agreement for testbenches that actually agree **100%** with the labels. The corrected `eval2_agr` column is 100% on every task with a valid FRM.

## Repro

```bash
# on Adroit, env: /scratch/network/ak7587/envs/pro-v
export PATH=/scratch/network/ak7587/envs/pro-v/bin:$PATH
export PYTHONPATH=/scratch/network/ak7587/Pro-V
python pro_v/compare_testbench_quality.py \
  --benchmark verilog-eval/HDLBits/test_benchmark_new.json \
  --outputs outputs/fca_headtohead_15 \
  --tasks 2,5,9,22,26,30,61,91,113,119,124 --only-cmb --out compare_cmb_complete.json
```
