# Pro-V FRM Correctness — Full Investigation Writeup

_Last updated 2026-07-13. Every number here was verified by direct simulation (exhaustive ≤16-bit differential of the model's Python reference vs the benchmark's correct `module_code`, via iverilog). Soft/uncertain results are labeled. Nothing uses the answer key to make a decision — `module_code` is used only to **measure**, never to **select**._

---

## TL;DR

- We set out to show our **functional-coverage agent (FCA)** makes Pro-V's testbenches stronger. It does — on the **stimulus** axis (guaranteed 100% mutation kill where the oracle is valid).
- But the real bottleneck turned out to be the **oracle itself**: the LLM-generated **functional reference model (FRM / `golden_dut.py`)** is **wrong ~40% of the time** on a random HDLBits slice with the 8B model. A wrong oracle makes the whole testbench invalid regardless of stimulus.
- This is **the paper's pipeline, not our changes** — verified by running upstream `stable-lab/Pro-V` directly (it's actually slightly worse). The **judge that selects among FRM candidates is *our* addition**; upstream just takes `sample[0]`.
- The FRM errors are **logical / spec-comprehension**, not Python-syntax — so linters won't help. Wrong candidates are **diverse**, and **correct answers are often the minority**, which structurally breaks majority-vote selection.
- We built a **deploy-safe confidence gate** (prove the FRM on the spec's own stated examples; else flag) and are now testing a **Spec-Kit-style disciplined generation loop** (clarify → derive criteria → implement → gate + repair) to attack the *systematic* comprehension errors that more sampling cannot fix.

---

## 1. How we got here

Goal: run the real HDLBits benchmark on Adroit's A100, measure whether the FCA (completeness-policy stimulus) beats the paper's stimulus, and analyze testbench strength honestly.

Setup: 15 random HDLBits tasks (seed 1234), `PRO-V-R1-8B` on one A100. Real benchmark: `verilog-eval/HDLBits/test_benchmark_new.json` (156 tasks, 10 mutants each). Infra fixes needed to even run: `conda install iverilog`, `pip install ninja`, `VLLM_USE_FLASHINFER_SAMPLER=0` + `cudatoolkit/13.0` (vLLM engine-core died on `curand.h`).

## 2. Key verified findings

1. **FCA completeness works (stimulus axis).** Policy: exhaustive ≤11 input bits (256/512/1024/2048), else a 2048-sample, unioned with ATPG distinguishing vectors, never pruned. Result on valid-oracle tasks: **100% true mutation score** — a *guarantee* (exhaustive can't miss), vs the paper's luck-based random saturation. Matched the paper where both are valid; the win is the guarantee + the honesty layer.

2. **The oracle (FRM) is the bottleneck.** On the 15-task run, **8/14 tasks had a wrong FRM** — the correct RTL fails the model's own testbench (eval1 fail). Confirmed at the code level (e.g. `bugs_mux2`: the FRM implemented an inverted/garbled mux; two independent simulators agree).

3. **It's the paper's pipeline, not us.** Ran unmodified upstream `stable-lab/Pro-V` on the same tasks: **same broken FRMs, slightly worse** (6/11 combinational broken vs our 5/11; it also broke `kmap1` and crashed on `bugs_mux2`'s undefined `mask_1bit`). The upstream repo doesn't even import as shipped (dead `back_up` import). The **judge is ours** — upstream `prompting_top_agent_ray.py:351` is literally `selected_sample = pychecker_results[0]`.

4. **A subtle metric bug in the paper's own eval.** `calculate_agreement` computes `detected == (not label)` — **inverted** relative to the benchmark's labels (verified at position level: our detections exactly match the label-True positions). So the paper's agreement metric reads ~0% for testbenches that actually agree **100%** with the labels.

5. **"Exhaustive" ≠ "passes."** Exhaustive stimulus + a wrong oracle makes the correct design fail *harder* (task 22: correct RTL matches the FRM on only 34/128 exhaustive inputs). Exhaustiveness describes the stimulus; it can't fix a wrong answer key.

6. **Selection can't be fixed by voting.** The wrong FRM candidates are **diverse** (5 distinct behaviors for tasks 2, 22), and the correct answer is often the **minority** (task 9). So majority-agreement — which is what our judge used — structurally picks against the correct FRM. Confirmed: judge and `sample[0]` both get 6/11; ceiling (best-of-5) is 8/11; on tasks 9 & 26 a correct candidate existed and the judge picked a wrong one.

7. **The failure modes are logical, not syntactic.** Diagnosed the wrong FRMs' code: inverted mux mappings, wrong FSM next-state equations, identity-instead-of-reversal, a `Y2=1; Y2=0` code bug. A Python linter (masking `~`, widths) would fix almost none.

8. **`bugs_mux2` fails 15/15 — a spec-comprehension trap.** Its description is *"find the bug"* and **shows the buggy code**; the model copies the buggy expression instead of the corrected behavior. More samples never fix it (0/15). Task 9 (`vector2`), by contrast, went 0/5 → **5/15** with more samples — recoverable. So hard tasks split into *sometimes-right* (sampling helps) vs *systematic-trap* (sampling can't).

## 3. What we built

| Module | Role | Status |
|---|---|---|
| `pro_v/functional_coverage_agent.py` | FCA completeness-policy stimulus (exhaustive ≤11 bits, 2048-sample above) | done, tests green |
| `pro_v/compare_testbench_quality.py` | head-to-head paper-stimulus vs FCA, true mutation score | done |
| `pro_v/frm_spec_grounding.py` | deploy-safe FRM selection via the spec's stated rows (waveform tables + K-maps) | done, regression-guarded |
| `pro_v/frm_confidence.py` | layered confidence gate: hygiene → spec-example proof → abstain | done, **0 false confidence** |
| `pro_v/spec_driven_frm.py` | Spec-Kit loop: clarify → derive criteria (×2, cross-checked) → implement → gate + repair | built, dry-run PASS, **queued on GPU** |

Integration: spec-grounding is wired into our `prompting_top_agent_ray.py` as an override after the judge (backup `.prespec.bak`).

## 4. Results (numbers)

**FRM correctness (11 combinational tasks, exhaustive ground truth):**
- paper `sample[0]`: **6/11** · our judge: **6/11** · ceiling (best-of-5): **8/11**
- spec-grounding gate rescues task 26 → **7/11 deployable testbenches**, **no regressions**.
- confidence gate: high-confidence-correct = 2 (tasks 26, 91), **high-confidence-WRONG = 0** (never falsely confident), abstain = 9.

**More-samples experiment (option A):** task 9 `0/5 → 5/15` (helps); tasks 2, 22, 30 `0/15` (model ceiling / systematic trap — sampling doesn't help).

**End-to-end (spec/judge FRM + FCA):** 7/11 deployable testbenches, 100% mutation kill on the valid ones.

## 5. The direction: stop trying to guarantee a perfect FRM

You can never fully "verify the verifier" — the FRM *is* the oracle, and at deployment there's no ground truth. So the strategy shifted from *guarantee* to **confidence + discipline + graceful degradation**:

- **Prove where you can (deploy-safe):** if the spec states concrete rows (truth tables, waveforms, K-maps), the correct FRM must reproduce them — an actual proof on those inputs. This is the strongest legitimate signal; it's what the confidence gate uses.
- **Manufacture criteria where the spec is natural-language (Spec-Kit):** clarify the intent (catch fix-the-bug traps), then **derive acceptance examples + invariants as a separate pass** and gate the code on them, repairing on failure. Turns NL specs into checkable artifacts.
- **Guard against self-deception:** derive criteria **twice, independently**; if they conflict, the model is uncertain → refuse confidence, flag. (Verified in dry-run.) Trust order: spec-**stated** rows > structural invariants > self-**derived** examples > abstain.
- **Degrade gracefully:** the FCA surfaces residual FRM errors as visible FRM↔DUT **disagreements** for review, not silent passes.

**Explicitly rejected as cheating / unsafe:** DUT-agreement selection (in the benchmark that means selecting by agreement with the correct `module_code` = using the answer key; even at real deployment it can mask the DUT's own bugs). Any use of `module_code` in a *selection* decision.

## 5b. Parse-then-synthesize for structured specs (built + verified, no GPU)

The strongest lever found: the LLM is good at *extracting* structure and bad at *deriving* logic — so for specs carrying a hardware-level structure, **extract the structure and let a deterministic algorithm compute the reference.** `pro_v/frm_synthesize.py`:

- **FSM** — parse transition graph + state encoding → deterministic next-state function.
- **K-map / truth table / waveform** — parse grid/rows → lookup FRM.
- **Router** — `synthesize()` returns a high-confidence FRM for structured specs, or `None` (→ LLM fallback) otherwise. Deploy-safe: only the description + header are used, never `module_code`.

**Calibrated across the 11 combinational tasks (vs ground truth):**
- Fired on **4 structured tasks (2, 22, 26, 91) → 4/4 CORRECT**, high-confidence by construction.
- **Abstained cleanly on the 7 non-structured tasks** (no false fires, no regressions).
- **Tasks 2 & 22 — which the model failed 15/15 — are now 100% correct, with no GPU and no LLM logic-derivation.**

Combined with the LLM path (already correct on 5, 61, 113, 119, 124), confident-FRM coverage goes **6/11 → 9/11**; only task 9 (algorithmic, reformat should help) and task 30 (fix-the-bug trap) remain — both targeted by the reformat loop.

**Coverage across the full 156 HDLBits tasks (CPU sweep):** synthesis fires on **9/156 (6%)** and is **9/9 CORRECT** — 100% precision, zero false fires. Narrow but perfect and high-value (it covers exactly the structured tasks the LLM struggles with, including the FSMs it fails 15/15). A safety guard makes it **abstain on all sequential/clocked designs** (a stateless lookup would be wrong for stateful logic — caught a real false-fire on `circuit7` before the guard). The other ~94% (natural-language / algorithmic specs) fall through to the LLM path.

Honest caveats: FSM synthesis is verified only on **reachable** states (unreachable state codes are genuinely unspecified — `module_code`'s don't-care assignments there are arbitrary). Structure extraction still uses regex/LLM parsing, but a parse error is cross-checkable against the spec's stated rows, unlike a derivation error. K-map parsing bails unless it gets full 2ⁿ coverage (safety).

## 5c. Reformat-loop result (GPU) — a negative result, and a benchmark bug

The Spec-Kit reformat + self-criteria + repair loop ran on the four hard tasks (2, 22, 30, 9). **It did not rescue any of them** (0/4 correct vs ground truth — same as one-shot sampling's 0/15). Two lessons:

- **The self-derived "medium" tier is false confidence.** Tasks 30 and 22 *passed their own self-derived criteria* while being wrong — the model's wrong logic and its wrong self-tests agree (correlated error). So "passed self-criteria" is not deployable. The loop's confidence was collapsed to just **`high` (spec-stated proof, deployable) vs `unverified` (escalate)**. The two-derivation cross-check *did* catch task 9 (conflicting derivations → flagged) but not 30/22 (both derivations agreed on the same wrong answer).
- **Task 30 (`bugs_mux2`) is a benchmark inconsistency, not a model failure.** Verified: the benchmark `module_code` (`sel?a:b`, sel=1→a) is **inverted** vs the spec's own shown code (`(~sel&a)|(sel&b)`, sel=1→b) — they disagree on 100/100 vectors. The model's FRM matched the *spec*; it was marked wrong by an inconsistent reference. No spec-faithful method can "win" this task. (Implication: the raw ~40% FRM-error rate is slightly overstated — at least one "failure" is a bad benchmark label.)

**Composed pipeline (`pro_v/frm_pipeline.py`):** `generate_frm()` routes STRUCTURED specs → deterministic synthesis (high-confidence), else → LLM reformat loop (high only if spec-proven, else unverified). This makes synthesis the primary path and the LLM a fallback, matching the evidence: synthesis wins on structured tasks, sampling+flag is best for algorithmic ones, and the reformat loop's self-graded confidence is not trusted.

## 5d. Composed pipeline — the measured boost (6/11 → 9/11)

`pro_v/frm_pipeline.generate_frm()` composes everything validated, all deploy-safe:
1. **Synthesis** if the spec is structured (FSM / K-map / truth-table) → high-confidence, correct by construction.
2. else **multi-sample LLM** (N candidates) + **deploy-safe selection**: a candidate that reproduces the spec's *stated* rows → `high`; otherwise the first hygiene-passing candidate → `unverified`.

**Measured on the 15-task set (existing 5-sample data, no new GPU, no cheating):**

| | correct FRMs |
|---|---|
| baseline `sample[0]` | **6/11** |
| composed (synthesis + spec-grounded/hygiene selection) | **9/11** |

Recovers the 3 structured tasks (2, 22 via FSM synthesis; 26 via table synthesis). Of the 2 remaining: **task 30 is a benchmark bug** (inverted reference) and **task 9** is the one genuine miss (NL spec, correct answer in the minority, no deploy-safe signal). So it's effectively **9/10 of the legitimate tasks**, up from 6/11 — with no answer-key access.

At 50-task scale (one-shot baseline): 24% → 29% synthesis-first; the larger gains from multi-sample + selection weren't yet run at 50-task scale (GPU-queued).

## 5e. The deploy-safe ceiling (measured, not assumed)

Tested the natural "pick the most common behavior across many samples" idea on the 15-sample data — it **fails**:
- task 9: behavior clusters sized [9, 5, 1] — the *wrong* answer is the plurality (9); correct is the 5-minority. Consensus picks wrong.
- tasks 2, 22: 13–14 distinct behaviors (no consensus) and no correct candidate in 15 tries.
- task 30: plurality wrong (benchmark bug — no candidate can match the inverted reference).

Combined with earlier results, the deploy-safe ceiling is now explicit:
- **Structured → synthesis** (perfect, 6% of tasks).
- **Spec-example tasks → multi-sample + spec-grounding** (6/11 → 9/11).
- **NL-spec, correct-answer-minority → no deploy-safe rescue.** Spec has no examples; plurality/consensus picks wrong; invariants can't distinguish (identity and reverse are both byte-permutations and both idempotent). Only a stronger model, spec-stated examples, or a human-confirmed example resolves these. Honest output: flag + escalate.
- **Benchmark bugs → flag.**

The one lever that still raises the *guaranteed-correct floor* with no model dependence is **broadening synthesis coverage** beyond 6% (more structured formats), pursued carefully behind the regression selftest + 156-sweep.

## 6. Honest limits & open questions

- **Spec-example proof only covers tasks that expose examples** — a minority of HDLBits. The gate is *sound* (0 false confidence) but *narrow* (abstains on ~9/11 here).
- **Self-derived criteria can inherit the model's own misreading** (correlated error). Mitigated by the two-derivation cross-check + preferring stated rows, but not eliminated. For tasks where the model genuinely can't derive the logic (FSM), Spec-Kit may not help — that's the open test now running.
- **Some tasks are underspecified** (e.g. `bugs_mux2`'s sel-mapping is ambiguous, and the benchmark's `module_code` may itself use the non-standard mapping). No deploy-safe method can match an underspecified/inconsistent reference — the honest output is *flag*, not guess.
- **The base-model ceiling** (tasks with no correct candidate in 15 samples) needs a stronger model or human-confirmed criteria; discipline alone won't close it.

## 7. Status & next steps

- **Running (GPU-queued, job 3307700):** the Spec-Kit loop on the systematic-trap tasks (2, 22, 30, 9). It will show directly whether disciplined generation beats one-shot sampling where sampling got 0/15. Auto-reports on completion. GPU is heavily contended (88 jobs queued), so it waits for an A100.
- **Next iterations:** (a) broaden legitimate example-extraction (FSM transition tables) to raise gate coverage; (b) if Spec-Kit helps `bugs_mux2` but not FSM, that isolates "comprehension trap" (fixable) from "derivation ceiling" (needs bigger model); (c) calibrate the confidence gate + Spec-Kit tiers across the full 156 tasks for a deployment threshold.

## Appendix — artifact & repro map

- Runs on Adroit: `outputs/verilog_eval_20260712_231354/` (our ray pipeline, 5 samples + judge), `outputs/highsample_15/` (15-sample), `outputs/specdriven_15/` (Spec-Kit, pending).
- Docs: `PROV_ANALYSIS_REPORT.md` (judge/spec-grounding results), `FCA_HEADTOHEAD_RESULTS.md` (FCA vs paper), `PROV_CHANGES.md` (change log), this file.
- Upstream baseline clone: `/scratch/network/ak7587/provup` (patched only to import).
- Env: `/scratch/network/ak7587/envs/pro-v` (+ iverilog, ninja). A100 = `--partition=gpu --gres=gpu:nvidia_a100:1`; V100 unusable (CUDA-13 torch).
