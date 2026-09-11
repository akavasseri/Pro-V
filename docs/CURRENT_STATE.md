# Pro-V — Current State (one page)

_Updated 2026-07-14. Plain-language summary of what we built, why, and what's still weak._

## The big idea
Pro-V verifies a chip design (RTL) by (1) writing a **reference model** (FRM = a Python "answer key" for the spec), (2) generating **test inputs** (stimulus), (3) running the design on those inputs and checking its outputs against the reference. We found the **reference model is the weak link** — the LLM writes it wrong ~45–75% of the time — so most of the new work is about **making the reference correct, or at least knowing when we can't trust it**, without ever peeking at the real answer.

## Walk through the pipeline (ONE path now — no parallel island)
**Spec (plain-English description + module header)**
1. **Make the reference model** → the real pipeline is `prompting_top_agent_ray.py` → `PyCheckerAgent` (in `pro_v/agent/pychecker.py`). For each of its 5 attempts it now does:
   - **Synthesis first** (`frm_synthesize.py`) — if the spec carries a *structure* (state table, Karnaugh map, truth table), extract it and **compute the reference with a real algorithm, no LLM**. Correct by construction; fires on ~7% of HDLBits tasks and **0% of real RTL** (it's a HDLBits-format bonus, not the general engine).
   - **else the LLM** writes the `GoldenDUT` (the paper's original behavior).
2. **Pick the best of the 5** → `frm_spec_grounding.py` (`select_by_spec`) overrides the judge: keep the candidate whose outputs **match the example rows the spec itself states**; else the judge's pick. (`hygiene_ok`, also here, rejects candidates that crash / return empty.)
3. **Turn the reference into expected outputs** → the chosen FRM runs over the stimulus → `testbench.json` (inputs + expected outputs).
4. **Make strong test inputs** → `functional_coverage_agent.py` (the FCA). Rule = **completeness, not fewness**: every input if ≤11 bits (up to 2048), else a big 2048-vector sample, plus the inputs that separate similar logic. Helpers: `coverage_gen.py`, `coverage_closure_loop.py`.
5. **Run the design + score it** → `verilog_tb_generator.py` (4-state iverilog testbench) + Verilator harness. `mutation_strength.py` measures **how strong** the test is (catches injected "mutant" bugs; proves survivors are truly equivalent) and holds the **now-general port parser** (`input [7:0] a, b`). `strengthen_testbench.py` adds missing inputs; `compare_testbench_quality.py` runs the head-to-head vs the paper.

**Cleanup (2026-07-14):** deleted `frm_pipeline.py` (was a parallel re-implementation of the 5-sample LLM loop), `spec_driven_frm.py` (reformat/criteria/repair loop — disproved, gave false confidence), and `frm_confidence.py` (inert confidence "tiers"; its one useful bit, `hygiene_ok`, moved into `frm_spec_grounding.py`). The system now runs on a **single path** — the paper's `PyCheckerAgent` plus our two real wins (**synthesis pre-step** + **spec-grounded selection**).

## Example (what it actually does)
- Spec: *"reverse the byte order of a 32-bit vector."* The LLM often writes **identity** (does nothing). Our pipeline: synthesis can't help (no table) → multi-sample → hygiene-select a running candidate → label `unverified` (we can't prove it, so we flag it).
- Spec: an FSM drawn as `A --0--> B ...`. LLM fails **15/15**. Synthesis **parses the table and computes the next-state logic → 100% correct**, no GPU.
- Spec: `bugs_mux2` "find the bug." The shown code is buggy; the LLM copies it. We flag it — and separately proved the benchmark's own answer is inconsistent (a benchmark bug).

## Key results (measured, honest)
- **Reference-model correctness is the bottleneck:** ~24% one-shot, ~55% with 5-samples+judge, on HDLBits.
- **Composed pipeline lifts it 6/11 → 9/11** on the 15-task set (synthesis + selection), no cheating.
- **Synthesis:** 9–11 of 156 HDLBits tasks (~7%), **100% correct**, but **HDLBits-only** (0/50 on RTLLM).
- **FCA (stimulus):** where the reference is valid, it **guarantees 100% mutant kill** (exhaustive), vs the paper's luck-based random.
- **The judge that picks the best reference is ours** — the original paper just takes the first sample.

## Drawbacks / what's still weak
- **The reference model is still wrong most of the time on hard specs**, and there's often **no safe way to know which one is right** (plain-English specs with no examples, where the correct answer is the minority — "pick the popular one" actively fails).
- **Synthesis is narrow and HDLBits-specific** — it does nothing for real RTL (RTLLM), so it inflated HDLBits numbers. A fair, cross-benchmark run on RTLLM is **queued now** to get the honest number.
- **Confidence is binary and blunt:** only "proven on spec examples" is trustworthy; everything else must be escalated to a human — we can't *guarantee* a reference is correct (you can't fully "verify the verifier").
- **Sequential designs** are only checked best-effort; **>16-bit designs** can't be exhaustively verified (sampled only).
- Some "failures" are actually **benchmark bugs** (e.g. task 30), so raw error rates slightly overstate the model's fault.

## Full detail lives in
`PROV_FRM_INVESTIGATION.md` (the FRM deep-dive), `FCA_HEADTOHEAD_RESULTS.md` (stimulus vs paper), `PROV_ANALYSIS_REPORT.md` (judge/selection), `PROV_CHANGES.md` (change log).
