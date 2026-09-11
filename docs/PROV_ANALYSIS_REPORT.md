# Pro-V Analysis Report — FRM reliability, the judge, the FCA, and what to do

_2026-07-12. Self-report. Every number here was verified by direct simulation (iverilog, exhaustive ≤16-bit differential vs the benchmark's correct `module_code`). Where a result is soft, it says so._

## 0. The one-paragraph truth

On a random 15-task HDLBits slice with the 8B model, the paper's real pipeline (`stable-lab/Pro-V`: 5 FRM samples, temperature 0.6, **select `sample[0]`, no judge**) produces a **correct functional reference model (FRM) on 6 of 11 combinational tasks**. The **ceiling** — if you always picked the best of the 5 candidates — is **8 of 11**. **Our judge (the cherry) picks a different sample but lands on the same 6/11** — it does *not* beat `sample[0]` here, and on 2 tasks (9, 26) it had a correct candidate in the pool and chose a wrong one, because its majority-agreement vote is biased toward correlated errors. Separately, **our FCA / completeness policy is a genuine win on the *stimulus* axis** (100% true mutation score where the oracle is valid). Net: the dominant bottleneck is FRM (oracle) correctness, the judge needs a redesign to actually help, and the FCA already helps but on a different axis.

## 1. What was run (accurately, this time)

- **Pipeline:** OUR `prompting_top_agent_ray.py` (full pipeline: GenTB → PyChecker×5 → judge), `sampling_size=5`, temp-sample 0.6, thinking allowed, on one A100. Output: `outputs/verilog_eval_20260712_231354/`.
- **Paper vs ours is isolated cleanly:** the paper selects `pychecker_results[0]`; ours selects via `judge_pychecker_samples` (majority agreement + LLM audit). Same 5-candidate pool → the only variable is selection. (Verified in code: upstream `prompting_top_agent_ray.py:351` `selected_sample = pychecker_results[0]`, no judge; our copy adds the judge. `judge.py` file exists in both but upstream never calls it for selection.)
- **Earlier mistakes corrected:** my first runs used the *reduced* `prompting_top_agent_simple.py` (single FRM, no sampling/judge — `pychecker_results: 0`). Thinking was **not** off (the model's own `chat_template.jinja` is identical to the `nonthinking` one; both only suppress thinking when `enable_thinking` is explicitly false, which the client never sets).

## 2. FRM correctness — the real numbers (combinational, exhaustive ground truth)

| | correct FRMs (of 11 CMB tasks) |
|---|---|
| **Paper `sample[0]`** | **6** (tasks 5, 61, 91, 113, 119, 124) |
| **Our judge pick** | **6** (same set) |
| **Ceiling (best of 5)** | **8** (adds 9, 26 — a correct candidate exists) |
| **Unreachable (no correct candidate in 5)** | tasks 2, 22, 30 |

- **Judge rescued 0 tasks over `sample[0]`.** On task 9 it picked candidate 0 (wrong) while candidate 1 was correct; on task 26 it picked candidate 1 (wrong) while candidate 2 was correct.
- **Sequential (8, 24, 89, 150): all eval1-WRONG, but SOFT** — scored with the best-effort sequential TB (timing assumptions); not exhaustive. Do not treat seq as ground truth yet.
- **Where FRMs are wrong they are genuinely wrong** (verified with witnesses): e.g. `bugs_mux2` reference computed `((~sel)&a)|(sel&b)` = an inverted/garbled mux; two independent simulators agree the correct RTL fails it.

## 3. Causes → effects → side effects

**Cause A — the 8B model writes wrong Python references from the spec.**
- Effect: 3/11 CMB tasks (2, 22, 30) have *no* correct candidate in 5 tries. No selection or judge can fix these.
- Side effect: a wrong FRM = wrong oracle → the testbench rejects even the *correct* RTL (eval1 fail) and its raw "mutant detection rate" is spuriously high (it rejects everything). Only eval1 exposes it. A degenerate FRM can even produce a testbench with **empty `expected_outputs`** (upstream `bugs_mux2`: undefined `mask_1bit`, crashes every vector, writes 100 records with `expected_outputs: {}`), which then **falsely passes** any check that only looks at detection.

**Cause B — the judge selects by majority agreement across candidates.**
- Effect: when 4 of 5 candidates share the same wrong logic, the majority vote picks a wrong candidate even though a correct minority exists (tasks 9, 26). The judge underperforms its own ceiling by 2 tasks.
- Side effect: on this sample the judge adds cost (5× FRM generation + judging) for **zero** correctness gain over `sample[0]`.

**Cause C — stimulus was the *other* axis, and the FCA fixed it.**
- Effect: with the completeness policy (exhaustive ≤11 input bits, 2048-sample above, union ATPG vectors), the FCA reaches **100% true mutation score** on every task whose oracle is valid — matching/again-not-losing to the paper's stimulus, with a *guarantee* (exhaustive can't miss) instead of luck.
- Side effect: none harmful; more vectors by design (the user's explicit goal: completeness, not minimality).

## 4. Answers to the standing questions

**Is our py-ref (FRM) agent better than theirs by a bit?**
No — not measurably on this 15-task sample. Paper `sample[0]` and our judge both get 6/11 CMB FRMs correct. The judge chooses differently but not better, and it has a real flaw (majority-agreement bias) that costs 2 rescuable tasks. Honest verdict: **currently at parity, with a fixable selection defect and clear headroom (ceiling 8).**

**Do we need another agent?**
- **For selection: no** — redesign the judge (Section 5.1). No new LLM agent needed; the fix is a better, spec-grounded criterion + reusing the FCA.
- **For the hard tail (no correct candidate): maybe** — an FRM-repair loop is the only lever for tasks 2/22/30, but it's constrained (no ground truth at deploy) and lower ROI than a stronger model or more samples.

## 5. Solutions — toward the original goals + higher efficiency

### 5.1 Fix the judge (highest leverage, cheap, no new model)
The current criterion (majority agreement) is *wrong* when errors correlate. Replace/augment with an **independent correctness signal**, in priority order:
1. **Spec-example grounding.** Parse concrete rows from the description (truth tables, K-maps, waveforms, explicit `sel=…→…`). Reject any candidate that contradicts a stated row *before* voting. This directly rescues 9 and 26 (their correct candidate matches the spec; the wrong majority doesn't).
2. **FCA-driven disagreement probing.** Run all N candidates on the FCA's exhaustive/high-information stimulus. Cluster by output behavior; the discriminating inputs are exactly where candidates disagree. Break ties by spec-example match at those inputs — not by counting look-alikes.
3. **Hard rejects (do first, trivial):** any candidate that crashes on the FCA stimulus, or yields empty/partial `expected_outputs`, is disqualified. (Catches the upstream `bugs_mux2` degenerate case outright.)
This is a rewrite of `judge_pychecker_samples`, not a new agent.

### 5.2 Raise the ceiling
- **More/among-diverse samples:** the ceiling is bounded by "did any of the 5 get it right." Increasing `sampling_size` and diversity (temp, prompt-variants per sample) raises the odds a correct candidate exists — at linear cost.
- **Stronger model:** the true fix for tasks 2/22/30 (no correct candidate in 5). The 8B is the limiter; the paper's published numbers likely reflect a stronger model and/or golden-RTL filtering we did not run.

### 5.3 FRM repair loop (optional, for the hard tail)
Feed a candidate + a *detected* inconsistency back to the model to fix: a crash trace, a spec-example mismatch, or self-inconsistency across candidates. Deploy-safe signals only (no `module_code`). Worth prototyping for 2/22/30, but expect diminishing returns without a better base model.

### 5.4 Efficiency wins (independent of correctness)
- **Early oracle screening before the expensive mutant sim:** FRM-runs-without-crashing on FCA stimulus + non-empty outputs + spec-example match. Skip/flag tasks with a broken oracle → save the Verilator/iverilog mutant sweep (the slow part) on tasks that can't produce a valid verdict anyway.
- **Never trust eval2 when eval1 fails** — make it explicit and fast-fail (already the intended behavior; enforce it so broken-oracle tasks don't burn compute or inflate detection).
- **Dual-use the FCA:** the exhaustive stimulus we already generate serves *both* mutant killing *and* FRM selection/screening — no extra generation cost.

## 6. Where the FCA (our real win) fits
The FCA solves the **stimulus adequacy** axis: given a *valid* oracle, it guarantees every non-equivalent mutant dies (exhaustive ≤11 bits; 2048-sample above). That's verified and it's the genuine "cherry on top." But it cannot fix a wrong **oracle** — and on this sample the oracle is the bottleneck. The synthesis: **FCA for stimulus completeness + a spec-grounded selector (5.1) that reuses the FCA to screen/select FRMs.** That pairing attacks both axes with what we already built.

## 6b. Implemented + verified: spec-grounded FRM selection (the actual "actively better")

Added `pro_v/frm_spec_grounding.py` and wired it into `prompting_top_agent_ray.py` as an override right after `judge_agent.run`. Deploy-safe (uses ONLY the description text, never `module_code`): if the spec contains a truth table / waveform, force-select the FRM candidate that reproduces every row; if none matches, flag `no_valid_oracle` instead of shipping a broken one; otherwise abstain and keep the existing judge pick.

**End-to-end, 11 combinational tasks, verified against ground truth:**

| | Paper (`sample[0]` + random) | Ours (spec/judge FRM + FCA completeness) |
|---|---|---|
| Deployable testbenches (correct FRM) | **6/11** | **7/11** |
| Rescued | — | task 26 `circuit4` (spec-grounding → cand2; paper's `sample[0]` = cand0 is wrong) |
| Mutation kill on valid-oracle tasks | random (luck) | **100%** true score (26:9/9, 61:3/3, 91:10/10, 113:10/10, 119:9/9) — exhaustive, guaranteed |
| Regressions | — | none (judge abstains where no table → identical 6) |

**Honest magnitude:** +1 deployable testbench on this 15-task slice. Spec-grounding rescues a task only when the description has a parseable table AND at least one candidate is correct (here: only 26). It cannot rescue task 9 (`vector2`, NL-only spec, correct candidate exists but no deploy-safe signal separates it) or tasks 2/22/30 (no correct candidate exists — a base-model limit). So the win is real and verified but modest on this benchmark; the largest remaining lever is FRM base-model quality, which is outside our changes. Efficiency side-benefit: `no_valid_oracle` flagging lets the pipeline skip the expensive mutant sweep on tasks whose oracle can't be built.

## 7. Verified-facts ledger (so nothing here is hand-waved)
- Upstream selects `pychecker_results[0]`, no judge — code line + user confirmation.
- Our pipeline generated 5 candidates/task; judge picks recorded in `task_result.json` (`selected_sample_idx`).
- CMB FRM correctness by exhaustive differential vs `module_code`: paper 6/11, judge 6/11, ceiling 8/11 — per-candidate verdicts in `/tmp/all_candidates.py` output.
- Tasks 9 & 26: correct candidate exists (idx 1, idx 2), judge picked a wrong one (idx 0, idx 1) — per-candidate verified.
- Seq verdicts are best-effort (soft), explicitly not ground truth.
- FCA completeness result (100% true mutation score where oracle valid): `compare_cmb_complete.json`.
