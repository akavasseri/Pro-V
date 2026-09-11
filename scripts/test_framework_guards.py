#!/usr/bin/env python3
"""Focused regression checks for coverage, PyChecker, and judge safeguards."""

import json
import inspect
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules["numpy"] = types.ModuleType("numpy")

try:
    import yaml  # noqa: F401
except ImportError:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: None
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

from pro_v.agent import gen_tb
from pro_v.agent.gen_tb import GenTBAgent
from pro_v.agent.judge import _adversarial_semantic_score, _spec_state_table_score
from pro_v.agent.pychecker import (
    PYCHECKER_CANDIDATE_STRATEGIES,
    PyCheckerAgent,
    _fsm_transition_conflicts,
    _golden_dut_ast_signature,
)
from pro_v.simulate_and_evaluate_mutants import EvaluationMetrics
try:
    from pro_v.prompting_top_agent_ray import (
        _extract_simulation_mismatch_count,
        _calculate_mutant_agreement,
        get_circuit_type,
    )
except ModuleNotFoundError as exc:
    if exc.name != "ray":
        raise
    _extract_simulation_mismatch_count = None
    _calculate_mutant_agreement = None
    get_circuit_type = None


STATE_TABLE_DESCRIPTION = """Present state y[2:0] | Next state Y[2:0] x=0, x=1 | Output z
000 | 000, 001 | 0
001 | 001, 100 | 0
010 | 010, 001 | 0
011 | 001, 010 | 1
100 | 011, 100 | 1
"""


def test_mux_budget_and_coverage():
    agent = GenTBAgent()

    small_header = "module top_module(input [9:0] a, output y); endmodule"
    small_code = agent._build_fallback_stimulus_code("", small_header, "cmb")
    namespace = {"__name__": "framework_guard_test"}
    exec(gen_tb.CMB_PYTHON_HEADER + small_code, namespace)
    small_vectors = namespace["stimulus_gen"]()
    assert len(small_vectors) == 1024
    assert {row["a"] for row in small_vectors} == {format(i, "010b") for i in range(1024)}

    large_header = "module top_module(input [1023:0] in, input [7:0] sel, output y); endmodule"
    large_code = agent._build_fallback_stimulus_code("", large_header, "cmb")
    namespace = {"__name__": "framework_guard_test"}
    exec(gen_tb.CMB_PYTHON_HEADER + large_code, namespace)
    large_vectors = namespace["stimulus_gen"]()
    assert len(large_vectors) == 4096
    assert any(int(row["in"], 2) == 0 and int(row["sel"], 2) == 0 for row in large_vectors)
    assert any(int(row["in"], 2) == (1 << 1024) - 1 and int(row["sel"], 2) == 255 for row in large_vectors)


def test_sequential_operation_bins():
    agent = GenTBAgent()
    header = (
        "module top_module(input clk, input areset, input load, "
        "input ena, input [3:0] data, output [3:0] q); endmodule"
    )
    code = agent._build_fallback_stimulus_code("", header, "seq")
    namespace = {"__name__": "framework_guard_test"}
    exec(gen_tb.SEQ_PYTHON_HEADER + code, namespace)
    scenarios = namespace["stimulus_gen"]()
    assert len(scenarios) == 20
    assert {scenario["clock_cycles"] for scenario in scenarios} == {5, 8, 16, 32}
    assert all(scenario["areset"][0] == "1" for scenario in scenarios)
    assert all(scenario["areset"][1:] == ["0"] * (scenario["clock_cycles"] - 1) for scenario in scenarios)
    assert any(scenario["data"][:4] == ["0000", "1111", "0000", "1111"] for scenario in scenarios)


def test_state_table_domain_filter_and_score():
    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "description.txt"), "w") as f:
            f.write(STATE_TABLE_DESCRIPTION)
        testbench = []
        rows = {
            0: ("000", "001", 0),
            1: ("001", "100", 0),
            2: ("010", "001", 0),
            3: ("001", "010", 1),
            4: ("011", "100", 1),
        }
        for y, (next_zero, next_one, z) in rows.items():
            for x in (0, 1):
                next_state = next_one if x else next_zero
                testbench.append({
                    "inputs": {"x": str(x), "y": format(y, "03b")},
                    "expected_outputs": {"Y0": next_state[-1], "z": str(z)},
                })
        tb_path = os.path.join(directory, "testbench.json")
        golden_path = os.path.join(directory, "golden.py")
        with open(tb_path, "w") as f:
            json.dump(testbench, f)
        with open(golden_path, "w") as f:
            f.write("class GoldenDUT:\n    pass\n")
        score = _spec_state_table_score({
            "testbench_json_path": tb_path,
            "golden_dut_path": golden_path,
        })
        assert score["score"] == 200.0

    checker = PyCheckerAgent()
    generated = checker._synthesize_state_table_golden_from_spec(
        STATE_TABLE_DESCRIPTION,
        "module top_module(input x, input [2:0] y, output Y0, output z);",
    )
    namespace = {}
    exec(generated, namespace)
    dut = namespace["GoldenDUT"]()
    assert dut.load({"x": "1", "y": "001"}) == {"Y0": "0", "z": "0"}
    assert dut.load({"x": "0", "y": "011"}) == {"Y0": "1", "z": "1"}


def test_pychecker_semantic_guards():
    assert len(PYCHECKER_CANDIDATE_STRATEGIES) == 4
    assert "candidate_strategy" in inspect.signature(PyCheckerAgent.run).parameters
    original = "class GoldenDUT:\n    def __init__(self):\n        self.q = 0\n"
    assert _golden_dut_ast_signature(original) == _golden_dut_ast_signature(
        "# comment-only edit\n" + original
    )

    rtl = (
        "parameter A=0, B=1, C=2, S21=6, S22=7; "
        "S21: next = w ? C : B; S22: next = w ? B : C;"
    )
    wrong = (
        "elif self.state == 6:\n    self.state = 1 if w else 0\n"
        "elif self.state == 7:\n    self.state = 0 if w else 2\n"
    )
    assert len(_fsm_transition_conflicts(wrong, rtl)) == 2

    checker = PyCheckerAgent()
    arithmetic = """class GoldenDUT:
    def __init__(self): self.q = 0
    def load(self, clk, inputs):
        if self.q & 1: self.q = (self.q >> 1) | (1 << 63)
        return {"q": format(self.q, "064b")}
"""
    error = checker._validate_golden_dut_contract(
        arithmetic, "seq", "64-bit arithmetic right shift", "output [63:0] q", ""
    )
    assert error and "MSB/sign bit" in error

    shift_eight = arithmetic.replace(
        "if self.q & 1: self.q = (self.q >> 1) | (1 << 63)",
        "sign_bit = (self.q >> 63) & 1\n        self.q = (self.q >> 8) | (sign_bit << 63)",
    )
    error = checker._validate_golden_dut_contract(
        shift_eight, "seq", "64-bit arithmetic right shift", "output [63:0] q", ""
    )
    assert error and "all eight vacated MSBs" in error

    shift_eight_bit56 = arithmetic.replace(
        "if self.q & 1: self.q = (self.q >> 1) | (1 << 63)",
        "msb = (self.q >> 63) & 1\n        self.q = (self.q >> 8) | (msb << 56)",
    )
    error = checker._validate_golden_dut_contract(
        shift_eight_bit56, "seq", "64-bit arithmetic right shift", "output [63:0] q", ""
    )
    assert error and "one-bit sign flag" in error

    shift_eight_bit7 = arithmetic.replace(
        "if self.q & 1: self.q = (self.q >> 1) | (1 << 63)",
        "msb = (self.q >> 63) & 1\n        self.q = (self.q >> 8) | (msb << 7)",
    )
    error = checker._validate_golden_dut_contract(
        shift_eight_bit7, "seq", "64-bit arithmetic right shift", "output [63:0] q", ""
    )
    assert error and "one-bit sign flag" in error

    shift_eight_low_mask = arithmetic.replace(
        "if self.q & 1: self.q = (self.q >> 1) | (1 << 63)",
        "msb = (self.q >> 63) & 1\n"
        "        sign_ext = msb * ((1 << 56) - 1)\n"
        "        self.q = (self.q >> 8) | sign_ext",
    )
    error = checker._validate_golden_dut_contract(
        shift_eight_low_mask, "seq", "64-bit arithmetic right shift", "output [63:0] q", ""
    )
    assert error and "low-56-bit mask" in error

    valid_multi_amount_shift = arithmetic.replace(
        "if self.q & 1: self.q = (self.q >> 1) | (1 << 63)",
        "sign_bit = (self.q >> 63) & 1\n"
        "        if inputs['amount'] == '10':\n"
        "            self.q = (self.q >> 1) | (sign_bit << 63)\n"
        "        else:\n"
        "            self.q = (self.q >> 8) | ((0xFF << 56) if sign_bit else 0)",
    )
    assert checker._validate_golden_dut_contract(
        valid_multi_amount_shift, "seq", "64-bit arithmetic right shift", "output [63:0] q", ""
    ) is None
    with tempfile.TemporaryDirectory() as directory:
        golden_path = os.path.join(directory, "golden.py")
        with open(golden_path, "w") as f:
            f.write(valid_multi_amount_shift)
        with open(os.path.join(directory, "description.txt"), "w") as f:
            f.write("64-bit arithmetic shift register with right shifts by 1 and 8")
        with open(os.path.join(directory, "module_code.v"), "w") as f:
            f.write("module top_module(output [63:0] q); endmodule")
        judged = _adversarial_semantic_score(
            {"golden_dut_path": golden_path}, "seq"
        )
        assert not [reason for reason in judged["reasons"] if "shift-by-8" in reason]

    packed_mux = """class GoldenDUT:
    def __init__(self): pass
    def load(self, inputs):
        sel_val = int(inputs["sel"], 2)
        sel_lower = sel_val & 0xF
        return {"out": "0000"}
"""
    error = checker._validate_golden_dut_contract(
        packed_mux, "cmb", "256-to-1 mux",
        "module top_module(input [7:0] sel, output [3:0] out);", ""
    )
    assert error and "aliases valid selector values" in error

    reversed_packed_mux = """class GoldenDUT:
    def __init__(self): pass
    def load(self, inputs):
        in_val = int(inputs["in"], 2)
        sel_val = int(inputs["sel"], 2)
        start_bit = sel_val * 4
        out_val = (in_val >> (1023 - start_bit - 3)) & 0xF
        return {"out": format(out_val, "04b")}
"""
    packed_description = "sel=0 selects bits in[3:0], sel=1 selects bits in[7:4]"
    packed_rtl = "assign out = {in[sel*4+3], in[sel*4+2], in[sel*4+1], in[sel*4+0]};"
    error = checker._validate_golden_dut_contract(
        reversed_packed_mux,
        "cmb",
        packed_description,
        "module top_module(input [1023:0] in, input [7:0] sel, output [3:0] out);",
        packed_rtl,
    )
    assert error and "integer LSB" in error

    truncated_packed_mux = """class GoldenDUT:
    def __init__(self): pass
    def load(self, inputs):
        sel_val = int(inputs["sel"], 2)
        out_val = 0
        for i in range(16):
            if i == sel_val: out_val = i
        return {"out": format(out_val & 15, "04b")}
"""
    error = checker._validate_golden_dut_contract(
        truncated_packed_mux,
        "cmb",
        packed_description,
        "module top_module(input [1023:0] in, input [7:0] sel, output [3:0] out);",
        packed_rtl,
    )
    assert error and "256 valid values" in error

    with tempfile.TemporaryDirectory() as directory:
        golden_path = os.path.join(directory, "golden.py")
        with open(golden_path, "w") as f:
            f.write(reversed_packed_mux)
        with open(os.path.join(directory, "description.txt"), "w") as f:
            f.write(packed_description)
        with open(os.path.join(directory, "header.v"), "w") as f:
            f.write("module top_module(input [1023:0] in, input [7:0] sel, output [3:0] out);")
        with open(os.path.join(directory, "module_code.v"), "w") as f:
            f.write(packed_rtl)
        judged = _adversarial_semantic_score({"golden_dut_path": golden_path}, "cmb")
        assert any("packed mux reverses" in reason for reason in judged["reasons"])

    life = """class GoldenDUT:
    def __init__(self): self.q = 0
    def load(self, clk, inputs):
        neighbors = 0
        row = col = n_row = n_col = 0
        neighbor_idx = (n_row % 16) * 16 + (n_col % 16)
        neighbors |= 1
        return {"q": format(self.q, "0256b")}
"""
    error = checker._validate_golden_dut_contract(
        life, "seq", "16x16 toroidal grid with neighbors", "output [255:0] q", ""
    )
    assert error and "integer count" in error

    doubled_corners = """class GoldenDUT:
    def __init__(self): self.q = 0
    def load(self, clk, inputs):
        neighbors = 0
        for dj in range(-1, 2): neighbors += 0
        for dj in range(-1, 2): neighbors += 0
        for di in range(-1, 2): neighbors += 0
        for di in range(-1, 2): neighbors += 0
        row = col = 0
        index = ((row + 1) % 16) * 16 + ((col + 1) % 16)
        return {"q": format(self.q, "0256b")}
"""
    error = checker._validate_golden_dut_contract(
        doubled_corners,
        "seq",
        "16x16 toroidal grid where each cell has 8 neighbours",
        "output [255:0] q",
        "",
    )
    assert error and "double-count" in error

    wrong_edge = """class GoldenDUT:
    def __init__(self): self.d_last = 0; self.anyedge = 0
    def load(self, clk, inputs):
        in_val = int(inputs["in"], 2)
        if clk == 1:
            self.d_last = in_val
            self.anyedge = in_val ^ self.d_last
        return {"anyedge": format(self.anyedge, "08b")}
"""
    edge_rtl = "always @(posedge clk) begin d_last <= in; anyedge <= in ^ d_last; end"
    error = checker._validate_golden_dut_contract(
        wrong_edge,
        "seq",
        "detect each changed bit in an 8-bit vector",
        "module top_module(input clk, input [7:0] in, output [7:0] anyedge);",
        edge_rtl,
    )
    assert error and ("OLD d_last" in error or "old-state snapshots" in error)
    edge_model = checker._synthesize_registered_edge_golden(
        "module top_module(input clk, input [7:0] in, output [7:0] anyedge);", edge_rtl
    )
    namespace = {}
    exec(edge_model, namespace)
    dut = namespace["GoldenDUT"]()
    assert dut.load(1, {"in": "01010101"})["anyedge"] == "01010101"
    dut.load(0, {"in": "01010101"})
    assert dut.load(1, {"in": "01011111"})["anyedge"] == "00001010"

    grid = checker._synthesize_toroidal_grid_golden(
        "16x16 toroidal grid; a cell survives with 2 neighbors and is born with 3 neighbors",
        "module top_module(input clk, input load, input [255:0] data, output [255:0] q);",
        "",
    )
    namespace = {}
    exec(grid, namespace)
    dut = namespace["GoldenDUT"]()
    horizontal = (1 << 0) | (1 << 1) | (1 << 2)
    dut.load(1, {"load": "1", "data": format(horizontal, "0256b")})
    dut.load(0, {"load": "0", "data": "0" * 256})
    result = int(dut.load(1, {"load": "0", "data": "0" * 256})["q"], 2)
    expected_vertical = (1 << (15 * 16 + 1)) | (1 << 1) | (1 << (1 * 16 + 1))
    assert result == expected_vertical
    four_neighbors = """class GoldenDUT:
    def __init__(self): self.q = 0
    def load(self, clk, inputs):
        row = col = n_row = n_col = 0
        neighbor_idx = (n_row % 16) * 16 + (n_col % 16)
        neighbor1 = neighbor2 = neighbor3 = neighbor4 = 0
        neighbors = neighbor1 + neighbor2 + neighbor3 + neighbor4
        return {"q": format(self.q, "0256b")}
"""
    error = checker._validate_golden_dut_contract(
        four_neighbors,
        "seq",
        "16x16 toroidal grid where each cell has 8 neighbours",
        "output [255:0] q",
        "",
    )
    assert error and "explicitly models only 4" in error

    async_reset = """class GoldenDUT:
    def __init__(self): self.q = 0
    def load(self, clk, inputs):
        # asynchronous reset, synchronous load and enable
        if int(inputs["areset"], 2): self.q = 0
        elif clk == 1 and int(inputs["ena"], 2): self.q >>= 1
        return {"q": format(self.q, "04b")}
"""
    error = checker._validate_golden_dut_contract(
        async_reset,
        "seq",
        "asynchronous positive edge areset, synchronous active high signals load and enable; right shift register",
        "module top_module(input clk, input areset, input load, input ena, output [3:0] q);",
        "",
    )
    assert error is None


def test_invalid_mutants_never_earn_agreement():
    if _calculate_mutant_agreement is None:
        return
    scored = _calculate_mutant_agreement(
        [False, True, False],
        [False, True, True],
        [
            {"mutant_idx": 0, "status": "timeout"},
            {"mutant_idx": 1, "status": "simulated"},
            {"mutant_idx": 2, "status": "compile_or_runtime_failed"},
        ],
        3,
    )
    assert scored["agreement_count"] == 1
    assert scored["agreement_rate"] == 1 / 3
    assert scored["valid_agreement_rate"] == 1.0

    legacy = EvaluationMetrics(task_id="guard", task_number=0)
    legacy.compile_success = True
    legacy.module_passes = True
    legacy.total_mutants = 2
    legacy.mutant_results = [None, True]
    legacy.expected_mutant_results = [True, False]
    legacy.calculate_agreement()
    assert legacy.mutant_agreement_80 is False
    assert legacy.mutant_agreement_90 is False


def test_mismatch_progress_parser():
    if _extract_simulation_mismatch_count is None:
        return
    assert _extract_simulation_mismatch_count(
        "Total mismatches reported by simulator: 30"
    ) == 30
    assert _extract_simulation_mismatch_count("Unpass: 11") == 11


def test_circuit_and_async_edge_detection():
    if get_circuit_type is not None:
        assert get_circuit_type("// mention posedge only\nassign y = a & b;") == "cmb"
        assert get_circuit_type("always_ff @(posedge clk) q <= d;") == "seq"
    compile(gen_tb.SEQ_PYTHON_HEADER + gen_tb.SEQ_TAIL, "<seq-tail>", "exec")
    from pro_v.agent.pychecker import SEQ_CHECKER_TAIL
    compile(SEQ_CHECKER_TAIL, "<seq-checker-tail>", "exec")
    assert 'cycle_output["pre_clock"]' in SEQ_CHECKER_TAIL
    assert "always(?:_ff)?" in SEQ_CHECKER_TAIL


def test_harness_rejects_empty_mapped_outputs():
    root = Path(__file__).resolve().parents[1]
    cases = [
        (
            root / "pro_v/sim_cmb/harness-generator.py",
            "module top_module(input a, output y); assign y = a; endmodule\n",
            [{"inputs": {"a": "0"}, "expected_outputs": {"wrong": "0"}}],
        ),
        (
            root / "pro_v/sim_seq/harness-generator.py",
            "module top_module(input clk, input d, output reg q); always @(posedge clk) q <= d; endmodule\n",
            [{
                "clock_cycles": 1,
                "d": ["0"],
                "expected_outputs": [{
                    "rising_edge": {"wrong": "0"},
                    "falling_edge": {"wrong": "0"},
                }],
            }],
        ),
    ]
    for generator, rtl, testbench in cases:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "top_module.v").write_text(rtl)
            Path(directory, "testbench.json").write_text(json.dumps(testbench))
            proc = subprocess.run(
                [sys.executable, str(generator)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert proc.returncode != 0
            assert "No test vectors" in proc.stderr or "No scenarios" in proc.stderr


def test_harness_mapping_is_literal_and_deterministic():
    root = Path(__file__).resolve().parents[1]
    for index, generator in enumerate((
        root / "pro_v/sim_cmb/harness-generator.py",
        root / "pro_v/sim_seq/harness-generator.py",
    )):
        spec = importlib.util.spec_from_file_location(f"harness_generator_{index}", generator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.fuzzy_match_signal("spin", {"sp": 1}) is None
        try:
            module.sanitize_value("")
        except ValueError:
            pass
        else:
            raise AssertionError("empty binary value was accepted")


def test_harness_accepts_valid_edge_shapes():
    root = Path(__file__).resolve().parents[1]
    cases = [
        (
            root / "pro_v/sim_cmb/harness-generator.py",
            "module top_module(output one); assign one = 1'b1; endmodule\n",
            [{"inputs": {}, "expected_outputs": {"one": "1"}}],
        ),
        (
            root / "pro_v/sim_cmb/harness-generator.py",
            "module top_module #(parameter int WIDTH=64) "
            "(input [WIDTH-1:0] a, output [WIDTH-1:0] y); assign y = a; endmodule\n",
            [{"inputs": {"a": "0" * 64}, "expected_outputs": {"y": "0" * 64}}],
        ),
        (
            root / "pro_v/sim_cmb/harness-generator.py",
            "module top_module #(parameter DEPTH=256, AW=$clog2(DEPTH)) "
            "(input [AW-1:0] a, output [AW-1:0] y); assign y = a; endmodule\n",
            [{"inputs": {"a": "0" * 8}, "expected_outputs": {"y": "0" * 8}}],
        ),
        (
            root / "pro_v/sim_seq/harness-generator.py",
            "module top_module(input clk, output reg [3:0] q); "
            "always @(posedge clk) q <= q + 1'b1; endmodule\n",
            [{
                "clock_cycles": 1,
                "expected_outputs": [{
                    "rising_edge": {"q": "0001"},
                    "falling_edge": {"q": "0001"},
                }],
            }],
        ),
    ]
    for generator, rtl, testbench in cases:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "top_module.v").write_text(rtl)
            Path(directory, "testbench.json").write_text(json.dumps(testbench))
            proc = subprocess.run(
                [sys.executable, str(generator)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert proc.returncode == 0, proc.stderr
            harness = Path(directory, "rfuzz-harness.cpp")
            assert harness.exists() and harness.stat().st_size > 500


def main():
    test_mux_budget_and_coverage()
    test_sequential_operation_bins()
    test_state_table_domain_filter_and_score()
    test_pychecker_semantic_guards()
    test_invalid_mutants_never_earn_agreement()
    test_mismatch_progress_parser()
    test_circuit_and_async_edge_detection()
    test_harness_rejects_empty_mapped_outputs()
    test_harness_mapping_is_literal_and_deterministic()
    test_harness_accepts_valid_edge_shapes()
    print("framework guard tests passed")


if __name__ == "__main__":
    main()
