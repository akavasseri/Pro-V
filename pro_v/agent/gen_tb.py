"""
GenTB Agent - Testbench Generation Agent (Simplified Architecture)

This agent generates Python stimulus code for RTL designs.
Architecture: Only __init__ and run methods.
"""

import json
import logging
import os
import subprocess
from typing import Dict, Any, Optional

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

logger = logging.getLogger(__name__)

CMB_SYSTEM_PROMPT = """
You are an expert in RTL design. 
"""

CMB_GENERATION_PROMPT = """
Your task is to generate a Python function named "stimulus_gen" that produces test vectors for a combinational circuit.

<description>
{description}
</description>

<module_header>
{module_header}
</module_header>

**CRITICAL**: Use the EXACT signal names from the module_header. DO NOT use generic placeholder names.

**IMPORTANT - Input Signals Only**: When generating test stimulus, you should ONLY include INPUT signals. Even though the module_header may contain both inputs and outputs, your stimulus should IGNORE all output signals.

**Example**: For this module header:
```verilog
module top_module (
    input [5:0] y,
    input w,
    output Y1,
    output Y3
);
```
Your stimulus_gen() should ONLY generate test vectors for the **input signals**: `y`, `w`
Do NOT include output signals `Y1`, `Y3` in your test vectors.

## Output Format

**CRITICAL**: Your function MUST return a **list of dictionaries** (NOT tuples, NOT lists).

Each dictionary represents one test vector:
- Dictionary keys = **INPUT** signal names from module_header (DO NOT include output signals)
- Dictionary values = binary strings (e.g., "0", "1", "1010")

**CORRECT format** (dictionary):
```python
[
  {{"signal1": "0", "signal2": "101"}},
  {{"signal1": "1", "signal2": "010"}},
  ...
]
```

**WRONG formats** (DO NOT USE):
```python
# Wrong: tuple
[("0", "101"), ("1", "010")]  # ❌ NO!

# Wrong: list
[["0", "101"], ["1", "010"]]  # ❌ NO!

# Wrong: no signal names
[{{"0": "0", "1": "101"}}]  # ❌ NO! Use actual signal names
```

## Requirements

1. **INPUT signals only**: Only include INPUT signals in your test vectors. DO NOT include output signals.
2. **Binary strings only**: Use "0", "1", "101", etc. No 'X' or 'Z' values.
3. **All inputs included**: Every test vector must include all INPUT signals from the module header (excluding clk).
4. **Variable names**: Must exactly match the INPUT signal names from module header.
5. **Dictionary format**: Each test vector MUST be a dictionary with INPUT signal names as keys.
6. **IMPORTANT**: DO NOT hard-code stimulus as JSON arrays directly. Use Python programming techniques to generate them dynamically.
7. **Hard cap**: Generate at most 100 test vectors. Exhaustive testing is only allowed when the total input space is 64 combinations or fewer.
8. **Coverage mix**: Prefer a compact mix of zeros, ones, one-hot/walking-bit, alternating patterns, boundary values, and random cases. Wide buses must be sampled; never loop over `2**width` for large widths.

## Programming Techniques Required

You can use Python programming constructs to generate test vectors:
- Use `random` module for random value generation
- Use `numpy` for array operations and batch generation
- Use list comprehensions and loops for dynamic generation
- Use format strings to convert numbers to binary
- Use bounded loops. Do not create unbounded or huge exhaustive products.




## Key Techniques

**Small exhaustive testing with bit manipulation only when width is small:**
```python
limit = min(2**width, 64)
for i in range(limit):
    signal = format(i, f'0{{width}}b')
```

**Random values with specific bit width:**
```python
val = random.getrandbits(8)  # 8-bit random value
test_vectors.append({{"data": format(val, '08b')}})
```

**Batch generation with numpy:**
```python
values = np.random.randint(0, 256, 32)  # bounded random 8-bit values
signals = [format(v, '08b') for v in values]
```

**List comprehension for signal generation:**
```python
random_signals = [format(random.getrandbits(8), '08b') for _ in range(32)]
```

## Complete Example

For a module with this header:
```verilog
module example(input [3:0] data, input enable, output out);
```

Your stimulus_gen() should generate test vectors with keys: `"data"`, `"enable"` only.
Do NOT include the output signal `"out"` in your test vectors.

Now implement the stimulus_gen() function using Python programming techniques.

**IMPORTANT**: You MUST return your code in this exact format:
```python
def stimulus_gen():
    test_vectors = []
    # your implementation here
    # Each test vector MUST be a dictionary with signal names as keys
    test_vectors.append({{"signal_name": "binary_value"}})
    return test_vectors  # MUST return list of dictionaries
```

**REMINDER**:
- Use `{{"key": "value"}}` (dictionary) NOT `("value1", "value2")` (tuple)
- Use actual INPUT signal names from module_header as dictionary keys (NO output signals)
"""

CMB_INSTRUCTIONS = """
Instructions for stimulus_gen():
1. Return a list of dictionaries
2. Each dictionary has INPUT signal names as keys, binary strings as values (NO output signals)
3. Include all INPUT signals (except clk) from the module header
4. DO NOT include any output signals in the stimulus
5. Generate comprehensive test cases covering corners, edges, and random cases
"""

CMB_PYTHON_HEADER = """
import json
import random
import numpy as np
import re
import os
"""

CMB_TAIL = """
def extract_module_ports(verilog_file="top_module.v"):
    \"\"\"
    Extract module input port names from Verilog file.
    Excludes ONLY 'clk' and 'clock' signals - reset signals ARE included!
    Also extracts output ports for validation purposes.
    Returns dict with 'inputs' and 'outputs' sets.
    \"\"\"
    try:
        with open(verilog_file, 'r') as f:
            content = f.read()

        # Extract module declaration
        module_match = re.search(r'module\\s+\\w+\\s*\\((.*?)\\);', content, re.DOTALL)
        if not module_match:
            return {"inputs": set(), "outputs": set()}

        port_list = module_match.group(1)

        # Extract input signals
        # IMPORTANT: Only exclude 'clk' and 'clock' - keep ALL other inputs including reset signals!
        input_pattern = r'input\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*\\d+\\s*:\\s*\\d+\\s*\\])?\\s*(\\w+)'
        inputs = set()
        for match in re.findall(input_pattern, port_list):
            # Only exclude clock signals - reset signals should be included in stimulus
            if match.lower() not in ['clk', 'clock']:
                inputs.add(match)

        # Extract output signals (for filtering purposes only - NOT for stimulus generation)
        output_pattern = r'output\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*\\d+\\s*:\\s*\\d+\\s*\\])?\\s*(\\w+)'
        outputs = set(re.findall(output_pattern, port_list))

        widths = {}
        width_pattern = r'\\b(input|output)\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*(\\d+)\\s*:\\s*(\\d+)\\s*\\]\\s*)?(\\w+)'
        for _direction, msb, lsb, name in re.findall(width_pattern, port_list):
            if msb and lsb:
                widths[name] = abs(int(msb) - int(lsb)) + 1
            else:
                widths[name] = 1

        return {"inputs": inputs, "outputs": outputs, "widths": widths}
    except Exception as e:
        print(f"Error extracting ports: {e}")
        return {"inputs": set(), "outputs": set(), "widths": {}}

def resolve_verilog_file(preferred="module_code.v"):
    for candidate in [preferred, "module_code.v", "top_module.v"]:
        if candidate and os.path.exists(candidate):
            return candidate
    return preferred

def fuzzy_match_signal(test_name, actual_signals):
    \"\"\"Fuzzy match test signal name to actual signal name.\"\"\"
    if test_name in actual_signals:
        return test_name
    for actual in actual_signals:
        if test_name.lower() == actual.lower():
            print(f"  Fuzzy match: '{test_name}' -> '{actual}' (case-insensitive)")
            return actual
    test_variations = [test_name.replace('_', ''), test_name + '_in', test_name + '_out',
                      test_name.rstrip('_in'), test_name.rstrip('_out')]
    for variation in test_variations:
        for actual in actual_signals:
            if variation.lower() == actual.lower():
                print(f"  Fuzzy match: '{test_name}' -> '{actual}' (via '{variation}')")
                return actual
    abbreviation_map = {
        'data_in': ['data', 'din', 'd', 'in'], 'data_out': ['data', 'dout', 'q', 'out'],
        'din': ['data_in', 'data', 'd', 'in'], 'dout': ['data_out', 'data', 'q', 'out'],
        'load': ['load', 'ld', 'en', 'enable'], 'ena': ['enable', 'en', 'ena'],
        'enable': ['ena', 'en', 'enable'], 'data': ['data_in', 'din', 'data_out', 'dout'],
        'q': ['data_out', 'dout', 'out'], 'd': ['data_in', 'din', 'data'],
        'in': ['data_in', 'din'], 'out': ['data_out', 'dout'],
    }
    if test_name.lower() in abbreviation_map:
        for candidate in abbreviation_map[test_name.lower()]:
            for actual in actual_signals:
                if candidate.lower() == actual.lower():
                    print(f"  Fuzzy match: '{test_name}' -> '{actual}' (abbreviation)")
                    return actual
    for actual in actual_signals:
        if actual.lower() in abbreviation_map:
            for candidate in abbreviation_map[actual.lower()]:
                if candidate.lower() == test_name.lower():
                    print(f"  Fuzzy match: '{test_name}' -> '{actual}' (reverse abbreviation)")
                    return actual
    for actual in actual_signals:
        test_lower, actual_lower = test_name.lower(), actual.lower()
        if test_lower in actual_lower or actual_lower in test_lower:
            min_len, max_len = min(len(test_lower), len(actual_lower)), max(len(test_lower), len(actual_lower))
            if min_len / max_len >= 0.6:
                print(f"  Fuzzy match: '{test_name}' -> '{actual}' (substring)")
                return actual
    return None

def fix_stimulus(stimulus_data, verilog_file="top_module.v"):
    \"\"\"
    Fix stimulus by ensuring it only contains INPUT signals.
    - Fuzzy matches signal names to actual input signals
    - Filters out any OUTPUT signals (they should NOT be in stimulus)
    - Adds missing input signals with random values
    \"\"\"
    ports = extract_module_ports(verilog_file)
    expected_inputs = ports["inputs"]  # Only INPUT signals should be in stimulus
    expected_outputs = ports["outputs"]  # OUTPUT signals used for filtering only
    widths = ports.get("widths", {})
    if not expected_inputs:
        if not expected_outputs:
            return stimulus_data, ["Could not extract input or output ports"]
        # A no-input combinational module still needs one vector so the RTL
        # oracle can observe the constant output.
        if not isinstance(stimulus_data, list) or not stimulus_data:
            return [{}], ["No input ports; using one empty stimulus vector"]
        return [{} for _ in stimulus_data], ["No input ports; cleared stimulus inputs"]
    if not stimulus_data or not isinstance(stimulus_data, list):
        return stimulus_data, ["Stimulus is empty or invalid"]

    def normalize_value(value, width):
        text = str(value).strip().replace("_", "")
        if text.startswith(("0b", "0B")):
            text = text[2:]
        if not text or any(ch not in "01" for ch in text):
            try:
                text = format(int(str(value), 0), f"0{width}b")
            except Exception:
                text = "0" * width
        if len(text) > width:
            text = text[-width:]
        elif len(text) < width:
            text = text.zfill(width)
        return text

    corrected_data, warnings = [], []
    for idx, test_vector in enumerate(stimulus_data):
        if not isinstance(test_vector, dict):
            warnings.append(f"Vector {idx} not a dict, skipping")
            continue
        corrected_vector = {}
        for test_signal, value in test_vector.items():
            # Check if it's an output signal (OUTPUT signals should NOT be in stimulus)
            if test_signal in expected_outputs or fuzzy_match_signal(test_signal, expected_outputs):
                warnings.append(f"Signal '{test_signal}' is an OUTPUT, filtering out (stimulus should only have inputs)")
                continue
            # Try to match to input signals (stimulus should only contain inputs)
            matched = fuzzy_match_signal(test_signal, expected_inputs)
            if matched:
                corrected_vector[matched] = normalize_value(value, widths.get(matched, 1))
            else:
                warnings.append(f"Signal '{test_signal}' not matched to any input, skipping")
        for expected_signal in expected_inputs:
            if expected_signal not in corrected_vector:
                width = widths.get(expected_signal, 1)
                random_val = format(random.getrandbits(width), f"0{width}b")
                corrected_vector[expected_signal] = random_val
                warnings.append(f"Added missing '{expected_signal}': {random_val}")
        corrected_data.append(corrected_vector)
    return corrected_data, warnings

def cap_stimulus(stimulus_data, verilog_file="top_module.v"):
    max_vectors = int(os.environ.get("PRO_V_MAX_CMB_VECTORS", "100"))
    ports = extract_module_ports(verilog_file)
    inputs = ports.get("inputs", set())
    widths = ports.get("widths", {})
    total_bits = sum(int(widths.get(name, 1) or 1) for name in inputs)
    exhaustive_bits = int(os.environ.get("PRO_V_EXHAUSTIVE_CMB_BITS", "10"))
    if inputs and total_bits <= exhaustive_bits:
        max_vectors = max(max_vectors, 1 << total_bits)
    elif any(name.lower() in ("sel", "select", "selector") for name in inputs):
        max_vectors = max(max_vectors, int(os.environ.get("PRO_V_MAX_MUX_VECTORS", "256")))
    if not isinstance(stimulus_data, list) or len(stimulus_data) <= max_vectors:
        return stimulus_data, []

    selected = []
    seen = set()
    for idx in list(range(min(16, len(stimulus_data)))) + list(range(len(stimulus_data) - 16, len(stimulus_data))):
        if 0 <= idx < len(stimulus_data) and idx not in seen:
            selected.append(stimulus_data[idx])
            seen.add(idx)

    remaining = [i for i in range(len(stimulus_data)) if i not in seen]
    budget = max_vectors - len(selected)
    if budget > 0:
        if len(remaining) <= budget:
            sampled = remaining
        else:
            random.seed(0)
            sampled = sorted(random.sample(remaining, budget))
        selected.extend(stimulus_data[i] for i in sampled)

    return selected[:max_vectors], [f"Capped combinational stimulus from {len(stimulus_data)} to {max_vectors} vectors"]

def enrich_small_comb_stimulus(stimulus_data, verilog_file="top_module.v"):
    # Use exhaustive vectors for small combinational input spaces.
    if not isinstance(stimulus_data, list):
        return stimulus_data, []

    ports = extract_module_ports(verilog_file)
    inputs = sorted(ports.get("inputs", []))
    widths = ports.get("widths", {})
    if not inputs:
        return stimulus_data, []

    total_bits = sum(int(widths.get(name, 1) or 1) for name in inputs)
    exhaustive_bits = int(os.environ.get("PRO_V_EXHAUSTIVE_CMB_BITS", "10"))
    if total_bits > exhaustive_bits:
        return stimulus_data, []

    vectors = []
    for value in range(1 << total_bits):
        cursor = total_bits
        vector = {}
        for name in inputs:
            width = int(widths.get(name, 1) or 1)
            cursor -= width
            vector[name] = format((value >> cursor) & ((1 << width) - 1), f"0{width}b")
        vectors.append(vector)

    return vectors, [f"Using exhaustive combinational stimulus: {len(vectors)} vectors for {total_bits} input bits"]

def enrich_mux_comb_stimulus(stimulus_data, verilog_file="top_module.v"):
    # Add selector-focused coverage for larger mux-like combinational modules.
    if not isinstance(stimulus_data, list):
        return stimulus_data, []

    ports = extract_module_ports(verilog_file)
    inputs = sorted(ports.get("inputs", []))
    widths = ports.get("widths", {})
    selectors = [name for name in inputs if name.lower() in ("sel", "select", "selector")]
    data_inputs = [
        name for name in inputs
        if name not in selectors and (
            name.lower().startswith("data") or name.lower() in {"a", "b", "c", "d", "e", "f", "g", "h", "i"}
        )
    ]
    if not selectors or not data_inputs:
        return stimulus_data, []

    enriched = list(stimulus_data)
    seen = {json.dumps(v, sort_keys=True) for v in enriched if isinstance(v, dict)}

    def bits(value, width):
        return format(value & ((1 << width) - 1), f"0{width}b")

    for sel in selectors:
        sel_width = int(widths.get(sel, 1) or 1)
        for sel_value in range(1 << sel_width):
            for pattern_idx in range(4):
                vector = {}
                for name in inputs:
                    width = int(widths.get(name, 1) or 1)
                    if name == sel:
                        vector[name] = bits(sel_value, width)
                    elif name in data_inputs:
                        if pattern_idx == 0:
                            value = 0
                        elif pattern_idx == 1:
                            value = (1 << width) - 1
                        elif pattern_idx == 2:
                            value = (sel_value + 1) * (data_inputs.index(name) + 1)
                        else:
                            value = int(("10" * ((width + 1) // 2))[:width], 2)
                        vector[name] = bits(value, width)
                    else:
                        vector[name] = "0" * width
                key = json.dumps(vector, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    enriched.append(vector)

    return enriched, [f"Added selector-focused mux stimulus; total vectors now {len(enriched)}"]

if __name__ == "__main__":
    import os
    import sys
    result = stimulus_gen()
    print("\\n=== Fixing and verifying stimulus ===")
    verilog_file = resolve_verilog_file()
    fixed_result, warnings = fix_stimulus(result, verilog_file=verilog_file)
    fixed_result, exhaustive_warnings = enrich_small_comb_stimulus(fixed_result, verilog_file=verilog_file)
    warnings.extend(exhaustive_warnings)
    fixed_result, mux_warnings = enrich_mux_comb_stimulus(fixed_result, verilog_file=verilog_file)
    warnings.extend(mux_warnings)
    capped_result, cap_warnings = cap_stimulus(fixed_result, verilog_file=verilog_file)
    fixed_result = capped_result
    warnings.extend(cap_warnings)
    if warnings:
        print("⚠ Warnings:")
        for w in warnings: print(f"  - {w}")
    with open("stimulus.json", "w") as f:
        json.dump(fixed_result, f, indent=2)
    print("✓ Saved corrected stimulus.json")
    print("==============================================\\n")
"""



# ============================================================================
# SEQ (Sequential) Circuit Configuration
# ============================================================================

SEQ_SYSTEM_PROMPT = """
You are an expert in RTL design. You can always write correct testbenches for RTL designs.
"""
SEQ_GENERATION_PROMPT = """
Your task is to generate a Python function named "stimulus_gen" that produces test scenarios for a sequential circuit.

<description>
{description}
</description>

<module_header>
{module_header}
</module_header>

## CRITICAL REQUIREMENTS - READ CAREFULLY

**1. EXACT Signal Names**: You MUST use ONLY the signal names that appear in the module_header above.
   - ✓ CORRECT: If module has "reset", use "reset"
   - ✓ CORRECT: If module has "areset", use "areset"
   - ✓ CORRECT: If module has "resetn", use "resetn"
   - ❌ WRONG: If module has "reset", do NOT use "rst" or "areset"
   - ❌ WRONG: Do NOT invent signal names like "load", "data_in" if they don't exist in module_header

**2. Input Signals Only (excluding clk)**:
   - ONLY include INPUT signals from module_header (NOT outputs, NOT clk)
   - If module has "input clk, input reset, output q", ONLY use "reset" in stimulus
   - Do NOT include "clk" or "q" in your stimulus

**3. FLAT List Structure**:
   - Return a FLAT list of dictionaries
   - Do NOT create nested lists
   - Use `scenarios.append({{"clock_cycles": N, ...}})` NOT `scenarios.append([...])`

## Output Format

Return a **FLAT list** of test scenarios. Each scenario is a dictionary specifying clock cycles and input signal sequences.

**WRONG** (nested list):
```python
scenarios = []
reset_sequence = [  # ❌ WRONG: Creating a list of dicts
    {{"signal": "0"}},
    {{"signal": "1"}}
]
scenarios.append(reset_sequence)  # ❌ WRONG: Appending a list!
```

**CORRECT** (flat list):
```python
scenarios = []
# ✓ CORRECT: Append each scenario directly
scenarios.append({{"clock_cycles": 3, "signal": ["0", "1", "0"]}})
scenarios.append({{"clock_cycles": 5, "signal": ["1", "0", "1", "0", "1"]}})
# Or use extend for multiple scenarios
scenarios.extend([
    {{"clock_cycles": 3, "signal": ["0", "1", "0"]}},
    {{"clock_cycles": 5, "signal": ["1", "0", "1", "0", "1"]}}
])
```

**Example**: For this module header:
```verilog
module top_module (
    input clk,
    input [5:0] y,
    input w,
    output Y1,
    output Y3
);
```
Your stimulus_gen() should ONLY generate test scenarios for the **input signals** (excluding clk): `y`, `w`
Do NOT include `clk` or output signals `Y1`, `Y3` in your test scenarios.


## Requirements

1. **INPUT signals only (excluding clk)**: Only include INPUT signals in your test scenarios. DO NOT include clk or output signals.
2. **Clock cycles**: Each scenario must have a "clock_cycles" field (integer)
3. **Signal sequences**: Each INPUT signal is a list of binary strings with length = clock_cycles
4. **Binary strings only**: Use "0", "1", etc. No 'X' or 'Z' values.
5. **CRITICAL - Signal Names**: You MUST use EXACTLY the same INPUT signal names as in the module_header above (excluding clk). DO NOT use generic names like "reset", "enable", etc. if the actual signal names are different (e.g., "areset", "en").
6. **All inputs included**: Include all INPUT signals from module header (except clk). DO NOT include output signals.
7. **Bounded test coverage**: Generate 8-24 test scenarios total. Each scenario should normally use 4-32 cycles. Keep total cycles across all scenarios under 512.
   - Reset/initialization sequences: 2-4 scenarios
   - Normal operation patterns: 3-8 scenarios
   - Edge cases and state transitions: 3-8 scenarios
   - Random test cases: 4-12 scenarios
   - Do not generate 50-100 scenarios; that causes avoidable simulator timeouts.
8. **Comprehensive testing**: Include:
   - Reset sequences (active and release). Distinguish synchronous reset, which changes on clock edges, from asynchronous reset, which can be asserted before or outside a clock edge.
   - Normal operation with different input patterns
   - Edge cases (state transitions, wraparounds, corner cases)
   - Exhaustive or near-exhaustive patterns only for very small input spaces
   - Random scenarios for large input spaces
9. **IMPORTANT**: DO NOT hard-code stimulus sequences as literal arrays. Use Python programming techniques to generate them dynamically.

## Programming Techniques Required

You MUST use Python programming constructs to generate test scenarios:
- Use `random` module for random value generation and patterns
- Use `numpy` for array operations and sequence generation
- Use list comprehensions to generate signal sequences
- Use loops and conditionals for intelligent pattern generation
- Use format strings to convert numbers to binary

## Complete Example

For a shift register with this header:
```verilog
module shift_reg(input clk, input rst, input [7:0] data_in, input load, output [7:0] data_out);
```

**IMPORTANT**: Your stimulus should ONLY include the input signals (excluding clk): `rst`, `data_in`, `load`
Do NOT include `clk` or the output signal `data_out` in your stimulus.

```python
import random
import numpy as np

def stimulus_gen():
    scenarios = []

    # Reset scenarios - NOTE: using "rst" because that's the signal name in module_header
    for _ in range(2):
        cycles = 5
        scenarios.append({{
            "clock_cycles": cycles,
            "rst": ["1"] + ["0"] * (cycles - 1),  # Use actual signal name from module_header
            "load": ["0"] * cycles,
            "data_in": ["00000000"] * cycles
        }})

    # Normal operation patterns - test different load patterns
    for _ in range(4):
        cycles = random.randint(10, 20)
        scenarios.append({{
            "clock_cycles": cycles,
            "rst": ["0"] * cycles,
            "load": [random.choice(["0", "1"]) for _ in range(cycles)],
            "data_in": [format(random.getrandbits(8), '08b') for _ in range(cycles)]
        }})

    # Edge cases - specific patterns
    for load_pattern in ["0011", "1100", "0101", "1010"]:
        cycles = len(load_pattern) * 3
        load_seq = list(load_pattern) * 3
        scenarios.append({{
            "clock_cycles": cycles,
            "rst": ["0"] * cycles,
            "load": load_seq,
            "data_in": [format(i, '08b') for i in range(cycles)]
        }})

    # Random scenarios - IMPORTANT: Generate many random cases for comprehensive testing
    for _ in range(8):
        cycles = random.randint(15, 30)
        scenarios.append({{
            "clock_cycles": cycles,
            "rst": ["0"] * cycles,
            "load": [random.choice(["0", "1"]) for _ in range(cycles)],
            "data_in": [format(random.getrandbits(8), '08b') for _ in range(cycles)]
        }})

    # Total: 2 + 4 + 4 + 8 = 18 scenarios
    return scenarios
```

**ANTI-PATTERN WARNING - Common Mistakes to Avoid**:

❌ **WRONG Example 1 - Insufficient Test Cases**:
```python
def stimulus_gen():
    scenarios = []
    # ❌ WRONG: Only 1-2 test cases is NOT sufficient!
    scenarios.append({{"clock_cycles": 5, "reset": ["1", "0", "0", "0", "0"]}})
    return scenarios  # ❌ Only 1 scenario - this will fail testing!
```

✓ **CORRECT** - Generate a bounded but diverse set:
```python
def stimulus_gen():
    scenarios = []
    # Reset cases (2-4)
    for _ in range(2):
        scenarios.append({{"clock_cycles": 5, "reset": ["1", "0", "0", "0", "0"]}})
    # Normal operation (3-8)
    for _ in range(4):
        scenarios.append({{"clock_cycles": 10, "reset": ["0"] * 10}})
    # Random cases (4-12)
    for _ in range(8):
        cycles = random.randint(10, 20)
        reset_val = random.choice(["0", "1"])
        scenarios.append({{"clock_cycles": cycles, "reset": [reset_val] * cycles}})
    return scenarios  # ✓ Total: 60 scenarios
```

❌ **WRONG Example 2 - Nested List**:
```python
def stimulus_gen():
    scenarios = []
    reset_seq = [
        {{"rst": "1"}},  # This creates a list!
        {{"rst": "0"}}
    ]
    scenarios.append(reset_seq)  # ❌ Appending a list creates nested structure!
    return scenarios
```

❌ **WRONG Example 2 - Using Non-Existent Signals**:
```python
# If module only has: input clk, input reset, output q
def stimulus_gen():
    scenarios = []
    scenarios.append({{
        "clock_cycles": 5,
        "reset": ["1", "0", "0", "0", "0"],  # ✓ OK: reset exists
        "load": ["0"] * 5,    # ❌ WRONG: load doesn't exist in module!
        "data_in": ["0"] * 5  # ❌ WRONG: data_in doesn't exist!
    }})
    return scenarios
```

✓ **CORRECT** - Check module_header first, use only existing signals:
```python
# If module only has: input clk, input reset, output q
def stimulus_gen():
    scenarios = []
    scenarios.append({{
        "clock_cycles": 5,
        "reset": ["1", "0", "0", "0", "0"]  # ✓ Only include existing input signals
    }})
    return scenarios
```

## Key Techniques

**Reset pattern generation (use actual reset signal name from module_header):**
```python
# If module has "areset": use areset
areset_signal = ["1"] + ["0"] * (cycles - 1)  # Assert reset on first cycle
# If module has "rst": use rst
# If module has "reset": use reset
```

**Random single-bit sequences (use actual signal names from module_header):**
```python
# Example: if module has "enable" signal
enable = [random.choice(["0", "1"]) for _ in range(cycles)]
```

**Random multi-bit sequences:**
```python
data = [format(random.getrandbits(8), '08b') for _ in range(cycles)]
```

**Numpy for probability-based patterns:**
```python
pattern = (np.random.random(cycles) > 0.3).astype(int)  # 70% ones, 30% zeros
signal = [str(v) for v in pattern]
```

**Burst operation patterns:**
```python
signal = ["0"] * cycles
for i in range(start_pos, end_pos):
    signal[i] = "1"
```

**State transition sequences:**
```python
test_patterns = ["0011", "1010", "1111"]
for pattern in test_patterns:
    signal_seq = list(pattern) + ["0"] * extra_cycles
```

Now implement the stimulus_gen() function using Python programming techniques.

**IMPORTANT**: You MUST return your code in this exact format:
```python
def stimulus_gen():
    # your implementation here
```
"""

SEQ_INSTRUCTIONS = """
Instructions for stimulus_gen():
1. Return a **FLAT list** of dictionaries (scenarios) - DO NOT create nested lists!
2. Each scenario must have "clock_cycles" (integer) and INPUT signal lists
3. All INPUT signal lists must have length equal to clock_cycles
4. Include all INPUT signals (except clk) from the module header
5. DO NOT include clk or output signals in the stimulus
6. **CRITICAL - Generate AT LEAST 50-100 test scenarios total**:
   - Reset sequences: 5-10 scenarios
   - Normal operation: 10-20 scenarios
   - Edge cases: 10-20 scenarios
   - Random tests: 30-50 scenarios
7. Generate comprehensive scenarios: reset, normal operation, edge cases, and random tests
8. **CRITICAL**: Use scenarios.append() to add individual scenarios, NOT scenarios.append([...])
"""

SEQ_PYTHON_HEADER = """
import re
import json
import random
import numpy as np
import os
"""

SEQ_TAIL = """
def extract_module_ports(verilog_file="top_module.v"):
    \"\"\"
    Extract module input port names from Verilog file.
    Excludes ONLY 'clk' and 'clock' signals - reset signals ARE included!
    Also extracts output ports for validation purposes.
    Returns dict with 'inputs' and 'outputs' sets.
    \"\"\"
    try:
        with open(verilog_file, 'r') as f:
            content = f.read()

        # Extract module declaration
        module_match = re.search(r'module\\s+\\w+\\s*\\((.*?)\\);', content, re.DOTALL)
        if not module_match:
            return {"inputs": set(), "outputs": set()}

        port_list = module_match.group(1)

        # Extract input signals
        # IMPORTANT: Only exclude 'clk' and 'clock' - keep ALL other inputs including reset signals!
        input_pattern = r'input\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*\\d+\\s*:\\s*\\d+\\s*\\])?\\s*(\\w+)'
        inputs = set()
        for match in re.findall(input_pattern, port_list):
            # Only exclude clock signals - reset signals should be included in stimulus
            if match.lower() not in ['clk', 'clock']:
                inputs.add(match)

        # Extract output signals (for filtering purposes only - NOT for stimulus generation)
        output_pattern = r'output\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*\\d+\\s*:\\s*\\d+\\s*\\])?\\s*(\\w+)'
        outputs = set(re.findall(output_pattern, port_list))

        widths = {}
        width_pattern = r'\\b(input|output)\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*(\\d+)\\s*:\\s*(\\d+)\\s*\\]\\s*)?(\\w+)'
        for _direction, msb, lsb, name in re.findall(width_pattern, port_list):
            if msb and lsb:
                widths[name] = abs(int(msb) - int(lsb)) + 1
            else:
                widths[name] = 1

        return {"inputs": inputs, "outputs": outputs, "widths": widths}
    except Exception as e:
        print(f"Error extracting ports: {e}")
        return {"inputs": set(), "outputs": set(), "widths": {}}

def resolve_verilog_file(preferred="module_code.v"):
    for candidate in [preferred, "module_code.v", "top_module.v"]:
        if candidate and os.path.exists(candidate):
            return candidate
    return preferred

def fuzzy_match_signal(test_name, actual_signals):
    \"\"\"Fuzzy match test signal name to actual signal name.\"\"\"
    if test_name in actual_signals:
        return test_name
    for actual in actual_signals:
        if test_name.lower() == actual.lower():
            print(f"  Fuzzy match: '{test_name}' -> '{actual}' (case-insensitive)")
            return actual
    test_variations = [test_name.replace('_', ''), test_name + '_in', test_name + '_out',
                      test_name.rstrip('_in'), test_name.rstrip('_out')]
    for variation in test_variations:
        for actual in actual_signals:
            if variation.lower() == actual.lower():
                print(f"  Fuzzy match: '{test_name}' -> '{actual}' (via '{variation}')")
                return actual
    abbreviation_map = {
        'data_in': ['data', 'din', 'd', 'in'], 'data_out': ['data', 'dout', 'q', 'out'],
        'din': ['data_in', 'data', 'd', 'in'], 'dout': ['data_out', 'data', 'q', 'out'],
        'load': ['load', 'ld', 'en', 'enable'], 'ena': ['enable', 'en', 'ena'],
        'enable': ['ena', 'en', 'enable'], 'data': ['data_in', 'din', 'data_out', 'dout'],
        'q': ['data_out', 'dout', 'out'], 'd': ['data_in', 'din', 'data'],
        'in': ['data_in', 'din'], 'out': ['data_out', 'dout'],
    }
    if test_name.lower() in abbreviation_map:
        for candidate in abbreviation_map[test_name.lower()]:
            for actual in actual_signals:
                if candidate.lower() == actual.lower():
                    print(f"  Fuzzy match: '{test_name}' -> '{actual}' (abbreviation)")
                    return actual
    for actual in actual_signals:
        if actual.lower() in abbreviation_map:
            for candidate in abbreviation_map[actual.lower()]:
                if candidate.lower() == test_name.lower():
                    print(f"  Fuzzy match: '{test_name}' -> '{actual}' (reverse abbreviation)")
                    return actual
    for actual in actual_signals:
        test_lower, actual_lower = test_name.lower(), actual.lower()
        if test_lower in actual_lower or actual_lower in test_lower:
            min_len, max_len = min(len(test_lower), len(actual_lower)), max(len(test_lower), len(actual_lower))
            if min_len / max_len >= 0.6:
                print(f"  Fuzzy match: '{test_name}' -> '{actual}' (substring)")
                return actual
    return None

def fix_stimulus_seq(stimulus_data, verilog_file="top_module.v"):
    \"\"\"
    Fix sequential stimulus by ensuring it only contains INPUT signals (excluding clk).
    - Flattens nested lists in stimulus data
    - Filters out any OUTPUT signals (they should NOT be in stimulus)
    - Fuzzy matches signal names to actual input signals
    - Fixes cycle counts, pads/truncates sequences
    - Adds missing input signals with random sequences
    \"\"\"
    if not stimulus_data or not isinstance(stimulus_data, list):
        return stimulus_data, ["Stimulus is empty or invalid"]

    # Flatten nested lists (handle cases where stimulus_gen returns nested structures)
    flattened_data = []
    for item in stimulus_data:
        if isinstance(item, list):
            # If item is a list, extend (flatten) it into the main list
            flattened_data.extend(item)
        else:
            # If item is not a list, append it directly
            flattened_data.append(item)

    stimulus_data = flattened_data

    ports = extract_module_ports(verilog_file)
    expected_inputs = ports["inputs"]  # Only INPUT signals (excluding clk) should be in stimulus
    expected_outputs = ports["outputs"]  # OUTPUT signals used for filtering only
    widths = ports.get("widths", {})
    if not expected_inputs:
        return stimulus_data, ["Could not extract input ports"]

    def normalize_value(value, width):
        text = str(value).strip().replace("_", "")
        if text.startswith(("0b", "0B")):
            text = text[2:]
        if not text or any(ch not in "01" for ch in text):
            try:
                text = format(int(str(value), 0), f"0{width}b")
            except Exception:
                text = "0" * width
        if len(text) > width:
            text = text[-width:]
        elif len(text) < width:
            text = text.zfill(width)
        return text

    corrected_data, warnings = [], []
    for idx, scenario in enumerate(stimulus_data):
        if not isinstance(scenario, dict):
            warnings.append(f"Scenario {idx} not a dict after flattening, skipping")
            continue

        # Infer clock_cycles if missing: assume single-cycle test case (all values are single strings)
        if "clock_cycles" not in scenario:
            # Check if all values are single strings (single cycle) or lists (multi-cycle)
            has_lists = any(isinstance(v, list) for k, v in scenario.items() if k != "clock_cycles")
            if has_lists:
                # If any value is a list, infer clock_cycles from the longest list
                max_len = max((len(v) if isinstance(v, list) else 1) for v in scenario.values())
                clock_cycles = max_len
                warnings.append(f"Scenario {idx} missing 'clock_cycles', inferred as {clock_cycles} from signal lengths")
            else:
                # All values are single strings, so it's a single cycle test
                clock_cycles = 1
                warnings.append(f"Scenario {idx} missing 'clock_cycles', inferred as 1 (single cycle)")
        else:
            clock_cycles = scenario["clock_cycles"]

        corrected_scenario = {"clock_cycles": clock_cycles}
        # Fuzzy match and fix signal sequences
        for test_signal, values in scenario.items():
            if test_signal == "clock_cycles":
                continue
            # Check if it's an output signal (OUTPUT signals should NOT be in stimulus)
            if test_signal in expected_outputs or fuzzy_match_signal(test_signal, expected_outputs):
                warnings.append(f"Scenario {idx}: Signal '{test_signal}' is an OUTPUT, filtering out (stimulus should only have inputs)")
                continue
            # Try to match to input signals (stimulus should only contain inputs, excluding clk)
            matched = fuzzy_match_signal(test_signal, expected_inputs)
            if matched:
                if isinstance(values, list):
                    values = [normalize_value(v, widths.get(matched, 1)) for v in values]
                    if len(values) < clock_cycles:
                        # Pad with zeros
                        padded = values + [normalize_value("0", widths.get(matched, 1))] * (clock_cycles - len(values))
                        corrected_scenario[matched] = padded
                        warnings.append(f"Scenario {idx}: Padded '{matched}' from {len(values)} to {clock_cycles} cycles")
                    elif len(values) > clock_cycles:
                        # Truncate
                        truncated = values[:clock_cycles]
                        corrected_scenario[matched] = truncated
                        warnings.append(f"Scenario {idx}: Truncated '{matched}' from {len(values)} to {clock_cycles} cycles")
                    else:
                        corrected_scenario[matched] = values
                else:
                    # Single value, convert to list
                    corrected_scenario[matched] = [normalize_value(values, widths.get(matched, 1))] * clock_cycles
                    warnings.append(f"Scenario {idx}: Converted single value '{matched}' to sequence")
            else:
                warnings.append(f"Scenario {idx}: Signal '{test_signal}' not matched to any input, skipping")

        # IMPORTANT: Only keep signals that exist in the module header
        # Do NOT add missing signals - if LLM didn't generate them, don't force-add them
        # This prevents adding signals that don't exist in the actual module

        # Check if we have at least one signal from the module (besides clock_cycles)
        has_valid_signals = any(sig in corrected_scenario for sig in expected_inputs)

        if not has_valid_signals and expected_inputs:
            # Only add missing signals if we have ZERO valid signals
            # This handles cases where LLM completely missed the required signals
            warnings.append(f"Scenario {idx}: No valid signals found, adding all expected inputs with random sequences")
            for expected_signal in expected_inputs:
                width = widths.get(expected_signal, 1)
                random_seq = [format(random.getrandbits(width), f"0{width}b") for _ in range(clock_cycles)]
                corrected_scenario[expected_signal] = random_seq

        corrected_data.append(corrected_scenario)
    return corrected_data, warnings

def cap_stimulus_seq(stimulus_data):
    max_scenarios = int(os.environ.get("PRO_V_MAX_SEQ_SCENARIOS", "24"))
    max_cycles = int(os.environ.get("PRO_V_MAX_SEQ_CYCLES", "32"))
    max_total_cycles = int(os.environ.get("PRO_V_MAX_SEQ_TOTAL_CYCLES", "512"))
    if not isinstance(stimulus_data, list):
        return stimulus_data, []

    capped, warnings, total_cycles = [], [], 0
    for idx, scenario in enumerate(stimulus_data):
        if len(capped) >= max_scenarios:
            warnings.append(f"Dropped scenario {idx}: scenario cap {max_scenarios} reached")
            continue
        if not isinstance(scenario, dict):
            continue

        item = dict(scenario)
        cycles = int(item.get("clock_cycles", 1) or 1)
        if cycles > max_cycles:
            warnings.append(f"Scenario {idx}: capped clock_cycles from {cycles} to {max_cycles}")
            cycles = max_cycles
            item["clock_cycles"] = cycles
            for key, values in list(item.items()):
                if key != "clock_cycles" and isinstance(values, list):
                    item[key] = values[:cycles]

        if total_cycles + cycles > max_total_cycles:
            warnings.append(f"Dropped scenario {idx}: total cycle cap {max_total_cycles} reached")
            continue

        capped.append(item)
        total_cycles += cycles

    if len(capped) < len(stimulus_data):
        warnings.append(f"Capped sequential stimulus from {len(stimulus_data)} to {len(capped)} scenarios")
    return capped, warnings

if __name__ == "__main__":
    import os
    import sys
    result = stimulus_gen()
    print("\\n=== Fixing and verifying stimulus ===")
    verilog_file = resolve_verilog_file()
    fixed_result, warnings = fix_stimulus_seq(result, verilog_file=verilog_file)
    capped_result, cap_warnings = cap_stimulus_seq(fixed_result)
    fixed_result = capped_result
    warnings.extend(cap_warnings)
    if warnings:
        print("⚠ Warnings:")
        for w in warnings: print(f"  - {w}")
    with open("stimulus.json", "w") as f:
        json.dump(fixed_result, f, indent=2)
    print("✓ Saved corrected stimulus.json")
    print("==============================================\\n")
"""

# ============================================================================
# GenTBAgent Class
# ============================================================================

class GenTBAgent:
    """Agent for generating stimulus files"""
    
    def __init__(self, llm_client=None, max_retries: int = 3, worker=None):
        """Initialize the GenTB Agent
        
        Args:
            llm_client: LLM client for generating code
            max_retries: Maximum number of retry attempts
            worker: Ray worker actor for executing Python code (optional)
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.worker = worker
        logger.info(f"GenTBAgent initialized with max_retries={max_retries}, worker={'provided' if worker else 'None'}")

    def _generate_stimulus_prompt(self, problem_input: str, spec: str, circuit_type: str) -> str:
        """Generate prompt for stimulus generation"""

        circuit_type = (circuit_type or "").strip().upper()

        if circuit_type == "CMB":
            prompt = CMB_SYSTEM_PROMPT + "\n\n" + CMB_GENERATION_PROMPT.format(
                description=problem_input,
                module_header=spec
            )
        else:  # SEQ
            prompt = SEQ_SYSTEM_PROMPT + "\n\n" + SEQ_GENERATION_PROMPT.format(
                description=problem_input,
                module_header=spec
            )

        return prompt
        
    def run(self,  description: str, header: str, circuit_type: str, output_dir: str) -> Dict[str, Any]:
        """Generate stimulus.json file

        Args:
            rtl_code: RTL source code
            specification: Circuit specification
            circuit_type: "cmb" or "seq"
            output_dir: Directory to save outputs

        Returns:
            Dictionary with success status and file paths
        """
        circuit_type = (circuit_type or "").strip().upper()
        logger.info(f"GenTBAgent.run started for circuit_type={circuit_type}")
        
        # Track previous attempts for error feedback
        previous_code = None
        previous_error = None

        for attempt in range(self.max_retries):
            try:
                system_prompt = ""
                user_prompt = self._generate_stimulus_prompt(description, header, circuit_type)

                # For retry attempts, append previous code and error information
                if attempt > 0 and previous_code and previous_error:
                    user_prompt += f"\n\n---\n\n**RETRY ATTEMPT {attempt + 1}**\n\n"
                    user_prompt += "The previous code generated has errors. Please fix the issues.\n\n"
                    user_prompt += f"<previous_code>\n```python\n{previous_code}\n```\n</previous_code>\n\n"
                    user_prompt += f"<error_message>\n{previous_error}\n</error_message>\n\n"
                    user_prompt += "Please analyze the error and provide corrected code."

                # Call LLM to generate code
                logger.info(f"Attempt {attempt + 1}/{self.max_retries}: Sending LLM request")
                response = self._call_llm(system_prompt, user_prompt)

                if not response:
                    logger.warning(f"Attempt {attempt + 1} failed: Empty LLM response")
                    previous_error = "Empty LLM response"
                    continue

                # Extract Python code
                python_code = (CMB_PYTHON_HEADER if circuit_type == "CMB" else SEQ_PYTHON_HEADER) + "\n\n" + self._extract_code(response)+"\n\n"+(CMB_TAIL if circuit_type == "CMB" else SEQ_TAIL)

                if not python_code:
                    logger.warning(f"Attempt {attempt + 1} failed: No Python code extracted")
                    previous_error = "No Python code extracted from LLM response"
                    continue

                # Store current code for potential retry
                previous_code = python_code

                # Validate Python syntax before saving
                logger.info(f"Validating Python syntax for attempt {attempt + 1}")
                syntax_error = self._validate_python_syntax(python_code)
                if syntax_error:
                    error_msg = f"Python syntax error:\n{syntax_error}"
                    logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                    previous_error = error_msg
                    continue

                # Save Python code
                os.makedirs(output_dir, exist_ok=True)
                stimulus_py_path = os.path.join(output_dir, "stimulus_gen.py")

                with open(stimulus_py_path, 'w') as f:
                    f.write(python_code)

                # Execute Python code
                logger.info(f"Executing Python code: {stimulus_py_path}")
                
                if self.worker is not None:
                    # Use Ray worker for execution
                    try:
                        if RAY_AVAILABLE:
                            # Increased timeouts to handle complex stimulus generation
                            # Worker timeout: 120s for execution
                            # Ray.get timeout: 300s (5 min) to allow for Ray overhead and queueing
                            obj_ref = self.worker.run_stimulus_generation.remote(
                                stimulus_py_path=stimulus_py_path,
                                task_folder=output_dir,
                                timeout=120.0
                            )
                            success, error_msg = ray.get(obj_ref, timeout=300.0)
                            
                            if not success:
                                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                                previous_error = error_msg
                                continue
                        else:
                            raise RuntimeError("Ray is not available but worker was provided")
                    except Exception as e:
                        error_msg = f"Worker execution error: {str(e)}"
                        logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                        previous_error = error_msg
                        continue
                else:
                    # Fallback to subprocess if no worker provided
                    result = subprocess.run(
                        ["python", "stimulus_gen.py"],
                        cwd=output_dir,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    if result.returncode != 0:
                        error_msg = f"Python execution error:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                        logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                        previous_error = error_msg
                        continue

                # Verify stimulus.json was created
                stimulus_json_path = os.path.join(output_dir, "stimulus.json")

                if not os.path.exists(stimulus_json_path):
                    # Detailed diagnostics
                    logger.warning(f"Attempt {attempt + 1} failed: stimulus.json not created")
                    logger.warning(f"Python execution returncode: {result.returncode}")
                    logger.warning(f"Python execution STDOUT:\n{result.stdout}")
                    logger.warning(f"Python execution STDERR:\n{result.stderr}")
                    logger.warning(f"Expected stimulus.json path: {stimulus_json_path}")
                    logger.warning(f"Output directory exists: {os.path.exists(output_dir)}")
                    logger.warning(f"Output directory path: {output_dir}")
                    
                    # List files in output directory
                    try:
                        files_in_dir = os.listdir(output_dir)
                        logger.warning(f"Files in output directory: {files_in_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to list files in output directory: {e}")
                    
                    # Check if stimulus_gen.py exists and show its content
                    stimulus_py_path = os.path.join(output_dir, "stimulus_gen.py")
                    if os.path.exists(stimulus_py_path):
                        try:
                            with open(stimulus_py_path, 'r') as f:
                                py_content = f.read()
                                logger.warning(f"stimulus_gen.py content (first 500 chars):\n{py_content[:500]}")
                        except Exception as e:
                            logger.warning(f"Failed to read stimulus_gen.py: {e}")
                    else:
                        logger.warning(f"stimulus_gen.py not found at: {stimulus_py_path}")
                    
                    error_msg = f"stimulus.json was not created after execution. Returncode: {result.returncode}, STDOUT: {result.stdout}, STDERR: {result.stderr}"
                    previous_error = error_msg
                    continue

                # Validate JSON format
                with open(stimulus_json_path, 'r') as f:
                    stimulus_data = json.load(f)

                logger.info(f"GenTBAgent.run succeeded on attempt {attempt + 1}")
                return {
                    "success": True,
                    "stimulus_json_path": stimulus_json_path,
                    "stimulus_py_path": stimulus_py_path,
                    "num_test_cases": len(stimulus_data),
                    "attempt": attempt + 1
                }

            except subprocess.TimeoutExpired:
                error_msg = "Python execution timeout (exceeded 60 seconds)"
                logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                previous_error = error_msg
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON format: {str(e)}"
                logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                previous_error = error_msg
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"Attempt {attempt + 1} failed: {error_msg}", exc_info=True)
                previous_error = error_msg

        # All retries failed
        logger.error(f"GenTBAgent.run failed after {self.max_retries} attempts")
        if previous_error:
            logger.error(f"Last error: {previous_error}")
        return {
            "success": False,
            "error": f"Failed after {self.max_retries} attempts. Last error: {previous_error if previous_error else 'Unknown error'}",
            "stimulus_json_path": None
        }
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM to generate Python code
        
        Args:
            system_prompt: System message for LLM
            user_prompt: User message for LLM
            
        Returns:
            LLM response string
        """
        if self.llm_client is None:
            logger.warning("No LLM client configured, returning placeholder code")
            return """```python
import json
import random

def stimulus_gen():
    test_vectors = []
    for i in range(10):
        test_vectors.append({"input": format(i, '04b')})
    return test_vectors

if __name__ == "__main__":
    result = stimulus_gen()
    with open("stimulus.json", "w") as f:
        json.dump(result, f, indent=2)
```"""
        
        try:
            response = self.llm_client.chat(system=system_prompt, user=user_prompt)
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted Python code
        """
        import re
        
        # Try to find code in ```python``` blocks
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Try to find code in ``` blocks
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Use entire response if no code blocks found
        logger.warning("No code blocks found in response, using entire response")
        return response.strip()
    
    def _validate_python_syntax(self, code: str) -> Optional[str]:
        """Validate Python code syntax
        
        Args:
            code: Python code string
            
        Returns:
            Error message if syntax is invalid, None if valid
        """
        import ast
        
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}\n{e.text}"
        except Exception as e:
            return f"Parse error: {str(e)}"
