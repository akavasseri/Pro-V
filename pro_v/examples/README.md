# Reference-Guided Functional Coverage Agent

A **pre-testbench** planning stage for Pro-V. It decides *what should be tested*
from the behavioral reference model (`golden_dut.py`), then hands a structured
plan to the testbench generator.

```
spec + golden_dut.py (+ optional interface metadata)
   -> functional_coverage_agent.py      # this stage
   -> coverage_plan.json                # structured test intent
   -> existing/new testbench generator
   -> RTL simulation against black-box top_module.v
```

## Why

Random/generic stimulus is weak. For an AND gate and an OR gate, testing only
`00` and `11` gives "50% input coverage" and shows both output values — yet it
**cannot tell AND from OR**, because the two agree on `00`/`11`. The
distinguishing inputs are `01` and `10`. This agent generalizes that: it drives
the golden model to pick *high-information* inputs (ones that separate similar
operations, cover output classes, and — for sequential designs — reach
meaningful states and transitions), then greedily minimizes to a small set.

## Architectural rule

`top_module.v` is the **black-box DUT** under verification. The agent never
parses its internal operators/branches/state logic to decide tests — that would
be circular and would leak the very thing we are checking. All test intent comes
from `golden_dut.py` (+ spec/metadata). `top_module.v` is read **only** to
confirm the interface (port names, widths, clock/reset names).

## Run

```bash
# combinational (auto-detected)
python pro_v/functional_coverage_agent.py \
    --frm pro_v/examples/and_gate/golden_dut.py \
    --dut pro_v/examples/and_gate/top_module.v \
    -o coverage_plan.json \
    --stimulus-out stimulus.json     # optional: also emit current-format stimulus

# sequential (auto-detected from the golden model's load(clk, inputs) arity)
python pro_v/functional_coverage_agent.py \
    --frm pro_v/examples/traffic_fsm/golden_dut.py \
    --dut pro_v/examples/traffic_fsm/top_module.v \
    -o coverage_plan.json

# both worked examples end-to-end, with commentary
python -m pro_v.examples.run_examples

# self-test and unit tests
python pro_v/functional_coverage_agent.py --selftest
python pro_v/tests/test_functional_coverage_agent.py
```

## Budget knobs

`--max-tests`, `--max-candidates`, `--max-sequence-depth`, `--max-bfs-states`,
`--max-runtime-seconds`. Exhaustive enumeration is used only when the whole
input space is small (`2**bits <= max_candidates`); otherwise the agent uses
high-information sampling + greedy set-cover minimization. Quality is measured by
meaningful behaviors covered, not by test count — a small set of distinguishing
tests beats a large pile of random ones.

## Integration

`plan_to_stimulus(plan)` down-converts a `coverage_plan.json` into the existing
Pro-V `stimulus.json` schema (binary strings for combinational, per-signal cycle
lists for sequential), so the current `golden_dut.py` harness / testbench
generator consumes it unchanged. The agent is additive — it does not replace the
testbench generator; it decides intent, the generator emits HDL.

## Examples

- `and_gate/`, `or_gate/` — the AND-vs-OR contrast. `run_examples.py` prints the
  truth tables showing `00`/`11` cannot separate the two, and the agent selects
  `01`/`10`.
- `traffic_fsm/` — a request/grant FSM (IDLE→REQ→BUSY→DONE). The agent does BFS
  over the sequential reference model to find the shortest clocked sequence into
  each state (including DONE) plus transition-covering sequences.
