#!/usr/bin/env python3
"""
mutation_strength.py

Measure the TRUE strength of a generated testbench via mutation analysis, and
extract counterexample *witnesses* that can be fed back to strengthen a weak
testbench.

Why this exists
---------------
A mutant is "killed" only if the stimulus (a) *activates* the mutated logic AND
(b) makes that mutation change an *observed* output. So a mutant that survives
the testbench is one of two very different things:

  * EQUIVALENT      - no input distinguishes it from the reference (a redundant
                      term, a dead branch, ...). It *should not* count against
                      the testbench.
  * WEAK-SURVIVOR   - a distinguishing input EXISTS, the stimulus just never
                      exercised it (e.g. an AND->OR mutant when the stimulus
                      never drives the two operands to differing values). This
                      *is* a hole in the testbench.

We separate the two with an INDEPENDENT differential search: simulate the
reference RTL and the mutant RTL side by side over a large/exhaustive probe
input set and look for any input where their outputs differ.

    found a difference  -> WEAK-SURVIVOR, and the differing input is a witness
    none over full/large probe -> EQUIVALENT

Integrity note: the reference RTL (`module_code`) is used here ONLY as an
offline oracle for *meta-evaluation*. It never enters the deployed checker /
testbench, so this does not leak the golden into the verification claim. Using
the golden to *build* the checker would be circular; using it to *measure* how
strong an independently-built checker is, is legitimate.

True mutation score = killed / (total - equivalent).

Each witness is emitted in the same schema as `stimulus.json` so it can be
injected back into the stimulus and the testbench re-run. Iterating
(find witnesses -> inject -> re-run) drives the real kill rate up until only
equivalent mutants survive.

The differential engine is Icarus Verilog (`iverilog`/`vvp`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TOP_NAME = "top_module"


# ---------------------------------------------------------------------------
# Verilog port parsing
# ---------------------------------------------------------------------------

_INPUT_RE = re.compile(
    r"input\s+(?:(?:wire|reg|logic)\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)"
)
_OUTPUT_RE = re.compile(
    r"output\s+(?:(?:wire|reg|logic)\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)"
)
_DECL_RE = re.compile(
    r"\b(input|output|inout)\b\s+"
    r"(?:(?:wire|reg|logic|signed|unsigned)\s+)*"
    r"(?:\[\s*([^:\]]+)\s*:\s*([^\]]+)\s*\]\s*)?"
    r"([^;()]+);",
    re.S,
)
_CLKS = {"clk", "clock"}


def _is_clock_signal(name: str) -> bool:
    lower = (name or "").lower()
    return lower in _CLKS or lower.startswith("clk") or lower.endswith("_clk") or "clock" in lower


def _strip_verilog_comments(text: str) -> str:
    text = re.sub(r"//.*", "", text or "")
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _split_top_level_commas(blob: str) -> List[str]:
    parts, cur = [], []
    paren = bracket = brace = 0
    for ch in blob or "":
        if ch == "(":
            paren += 1
        elif ch == ")" and paren:
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]" and bracket:
            bracket -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}" and brace:
            brace -= 1
        if ch == "," and paren == bracket == brace == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _safe_int_expr(expr: str, params: Dict[str, int]) -> Optional[int]:
    safe = str(expr or "").strip()
    if not safe:
        return None
    safe = re.sub(r"\b\d+\s*'\s*[dD]\s*([0-9_]+)", lambda m: m.group(1).replace("_", ""), safe)
    safe = re.sub(r"\b\d+\s*'\s*[hH]\s*([0-9a-fA-F_]+)", lambda m: str(int(m.group(1).replace("_", ""), 16)), safe)
    safe = re.sub(r"\b\d+\s*'\s*[bB]\s*([01_xXzZ]+)", lambda m: str(int(re.sub(r"[xXzZ]", "0", m.group(1)).replace("_", ""), 2)), safe)
    safe = safe.replace("_", "")
    for pname, value in sorted((params or {}).items(), key=lambda kv: len(kv[0]), reverse=True):
        safe = re.sub(rf"\b{re.escape(pname)}\b", str(value), safe)
    previous = None
    while previous != safe:
        previous = safe
        safe = re.sub(
            r"\$clog2\s*\(\s*([0-9+\-*/% ()]+)\s*\)",
            lambda m: str(max(1, (int(eval(m.group(1), {"__builtins__": {}}, {})) - 1).bit_length())),
            safe,
        )
    if not re.fullmatch(r"[0-9+\-*/% ()]+", safe):
        return None
    try:
        return int(eval(safe, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _collect_params(text: str) -> Dict[str, int]:
    params: Dict[str, int] = {}
    pattern = re.compile(
        r"\b(?:localparam|parameter)\b\s+"
        r"(?:(?:integer|int|logic|bit|reg|signed|unsigned)\s+)*"
        r"(?:\[[^\]]+\]\s*)?"
        r"([A-Za-z_]\w*)\s*=",
        re.S,
    )
    for match in pattern.finditer(text or ""):
        name = match.group(1)
        i = match.end()
        cur = []
        paren = bracket = brace = 0
        while i < len(text):
            ch = text[i]
            if ch == "(":
                paren += 1
            elif ch == ")" and paren:
                paren -= 1
            elif ch == "[":
                bracket += 1
            elif ch == "]" and bracket:
                bracket -= 1
            elif ch == "{":
                brace += 1
            elif ch == "}" and brace:
                brace -= 1
            if paren == bracket == brace == 0 and ch in ",;":
                break
            if paren == bracket == brace == 0 and ch == ")" and re.match(r"\s*\(", text[i + 1:]):
                break
            cur.append(ch)
            i += 1
        expr = "".join(cur).strip()
        value = _safe_int_expr(expr, params)
        if value is not None:
            params[name] = value
    return params


def _width(msb: str, lsb: str, params: Optional[Dict[str, int]] = None) -> int:
    if msb and lsb:
        hi = _safe_int_expr(msb, params or {})
        lo = _safe_int_expr(lsb, params or {})
        if hi is not None and lo is not None:
            return abs(hi - lo) + 1
    return 1


def _find_top_module(verilog: str) -> Tuple[str, str]:
    """Return (module_header_text, module_body_through_endmodule) for top_module
    when present, otherwise for the first module. Handles parameter lists better
    than a single regex, and never crosses into helper modules after endmodule."""
    text = verilog or ""
    matches = list(re.finditer(r"\bmodule\s+([A-Za-z_]\w*)\b", text))
    if not matches:
        return "", text
    chosen = next((m for m in matches if m.group(1) == TOP_NAME), matches[0])
    start = chosen.start()
    semi = text.find(";", chosen.end())
    if semi < 0:
        return text[start:], text[start:]
    end = re.search(r"\bendmodule\b", text[semi:], re.I)
    body_end = semi + (end.end() if end else len(text[semi:]))
    return text[start:semi + 1], text[start:body_end]


def _module_port_blob(module_header: str) -> str:
    """Extract the text inside the module port list from a single module header.

    Uses the final parenthesized group before the declaration semicolon, which
    avoids being confused by a preceding parameter list.
    """
    header = (module_header or "").rstrip()
    if header.endswith(";"):
        header = header[:-1]
    close = header.rfind(")")
    if close < 0:
        return ""
    depth = 0
    for idx in range(close, -1, -1):
        ch = header[idx]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                return header[idx + 1:close]
    return ""


@dataclass
class Ports:
    inputs: List[Tuple[str, int]]        # probe-driven inputs (clk excluded), ordered
    outputs: List[Tuple[str, int]]       # ordered
    clk_name: Optional[str] = None
    clock_inputs: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def input_width(self) -> int:
        return sum(w for _, w in self.inputs)

    @property
    def output_width(self) -> int:
        return sum(w for _, w in self.outputs)


def parse_ports(verilog: str) -> Ports:
    """Extract ordered input/output ports (with widths) from the top module."""
    verilog = _strip_verilog_comments(verilog)
    module_header, module_text = _find_top_module(verilog)
    params = _collect_params(module_text)
    port_blob = _module_port_blob(module_header) or verilog

    # General ANSI port parser: handles comma-shared declarations like
    # "input [7:0] a, b" (a name with no direction keyword inherits the previous
    # declaration's direction and width). Falls back to the old regex if this
    # yields nothing (e.g. non-ANSI style).
    inputs: List[Tuple[str, int]] = []
    outputs: List[Tuple[str, int]] = []
    clk_name: Optional[str] = None
    clock_inputs: List[Tuple[str, int]] = []
    cur_dir = None; cur_w = 1
    for item in _split_top_level_commas(port_blob):
        item = item.strip()
        if not item:
            continue
        dm = re.search(r"\b(input|output|inout)\b", item)
        if dm:
            cur_dir = dm.group(1)
            wm = re.search(r"\[\s*([^:\]]+)\s*:\s*([^\]]+)\s*\]", item)
            cur_w = _width(wm.group(1), wm.group(2), params) if wm else 1
        cleaned = re.sub(r"\b(input|output|inout|wire|reg|logic|bit|signed|unsigned|tri|supply0|supply1)\b", "", item)
        cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
        cleaned = re.sub(r"=.*", "", cleaned)
        names = re.findall(r"[A-Za-z_]\w*", cleaned)
        if not names:
            continue
        name = names[-1]
        if name in params:
            continue
        if cur_dir == "input":
            if _is_clock_signal(name):
                if name not in {n for n, _ in clock_inputs}:
                    clock_inputs.append((name, cur_w))
                if clk_name is None:
                    clk_name = name
            else:
                inputs.append((name, cur_w))
        elif cur_dir == "output":
            outputs.append((name, cur_w))
        elif cur_dir == "inout":
            # The current harness cannot drive/observe bidirectional ports
            # faithfully. Keep parsing conservative: do not misclassify inout
            # as a normal input or output.
            continue
    if not inputs and not outputs:   # fallback: non-ANSI / odd formatting
        # Non-ANSI declarations appear after the module header, e.g.
        #   module top_module(A,B,S); input [32:1] A; output S; ...
        # Scan the module text, not only the parenthesized port list.
        decl_blob = module_text or verilog
        for direction, msb, lsb, names_blob in _DECL_RE.findall(decl_blob):
            width = _width(msb, lsb, params)
            for raw_name in _split_top_level_commas(names_blob):
                match = re.search(r"\b([A-Za-z_]\w*)\b", raw_name)
                if not match:
                    continue
                name = match.group(1)
                if name.lower() in {"wire", "reg", "logic", "bit", "signed", "unsigned"} or name in params:
                    continue
                if direction == "input":
                    if _is_clock_signal(name):
                        if name not in {n for n, _ in clock_inputs}:
                            clock_inputs.append((name, width))
                        if clk_name is None:
                            clk_name = name
                    else:
                        inputs.append((name, width))
                elif direction == "output":
                    outputs.append((name, width))
                elif direction == "inout":
                    continue

    # de-dup preserving order (regex can double-match on odd formatting)
    inputs = _dedup(inputs)
    outputs = _dedup(outputs)
    return Ports(inputs=inputs, outputs=outputs, clk_name=clk_name, clock_inputs=_dedup(clock_inputs))


def _dedup(pairs: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    seen, out = set(), []
    for name, w in pairs:
        if name not in seen:
            seen.add(name)
            out.append((name, w))
    return out


def _rename_top(verilog: str, new_name: str) -> str:
    """Rename the top module and namespace helper modules.

    Differential simulation compiles the reference RTL and mutant RTL together.
    RTLLM-style designs often include helper modules, so keeping helper names
    unchanged causes duplicate-module compile failures. Prefix every module name
    found in this source while mapping ``top_module`` to the requested wrapper
    name; references/instantiations are updated by identifier replacement.
    """
    module_names = []
    for name in re.findall(r"\bmodule\s+([A-Za-z_]\w*)\b", verilog or ""):
        if name not in module_names:
            module_names.append(name)
    if not module_names:
        return verilog
    mapping = {
        name: (new_name if name == TOP_NAME or i == 0 else f"{new_name}__{name}")
        for i, name in enumerate(module_names)
    }
    out = verilog
    for old in sorted(mapping, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(old)}\b", mapping[old], out)
    return out


# ---------------------------------------------------------------------------
# iverilog runner
# ---------------------------------------------------------------------------

def _run_iverilog(sources: Dict[str, str], workdir: str, timeout: int) -> Tuple[bool, str]:
    """Compile *.v sources with iverilog and run with vvp. Returns (ok, stdout)."""
    for fname, text in sources.items():
        with open(os.path.join(workdir, fname), "w") as f:
            f.write(text)
    out_vvp = os.path.join(workdir, "a.out")
    compile_cmd = ["iverilog", "-g2012", "-o", out_vvp] + [
        f for f in sources if f.endswith(".v")
    ]
    try:
        c = subprocess.run(compile_cmd, cwd=workdir, capture_output=True,
                           text=True, timeout=timeout)
        if c.returncode != 0:
            return False, f"[compile] {c.stderr.strip()}"
        r = subprocess.run(["vvp", out_vvp], cwd=workdir, capture_output=True,
                           text=True, timeout=timeout)
        return True, r.stdout
    except subprocess.TimeoutExpired:
        return False, "[timeout]"
    except FileNotFoundError as e:
        return False, f"[missing tool] {e}"


# ---------------------------------------------------------------------------
# Combinational differential search (exact)
# ---------------------------------------------------------------------------

def _decode(bits: str, inputs: List[Tuple[str, int]]) -> Dict[str, str]:
    """Slice an MSB-first packed bit string into per-signal binary strings."""
    d, pos = {}, 0
    for name, w in inputs:
        d[name] = bits[pos:pos + w]
        pos += w
    return d


def _cmb_tb(ports: Ports, exhaustive: bool, count: int, max_witnesses: int) -> str:
    tw, ow = ports.input_width, ports.output_width
    in_decls = "\n  ".join(f"reg [{w-1}:0] {n};" for n, w in ports.inputs)
    ref_outs = "\n  ".join(f"wire [{w-1}:0] ref_{n};" for n, w in ports.outputs)
    mut_outs = "\n  ".join(f"wire [{w-1}:0] mut_{n};" for n, w in ports.outputs)
    ref_conn = ", ".join([f".{n}({n})" for n, _ in ports.inputs] +
                         [f".{n}(ref_{n})" for n, _ in ports.outputs])
    mut_conn = ", ".join([f".{n}({n})" for n, _ in ports.inputs] +
                         [f".{n}(mut_{n})" for n, _ in ports.outputs])
    lhs = "{" + ", ".join(n for n, _ in ports.inputs) + "}"
    refcat = "{" + ", ".join(f"ref_{n}" for n, _ in ports.outputs) + "}"
    mutcat = "{" + ", ".join(f"mut_{n}" for n, _ in ports.outputs) + "}"

    if exhaustive:
        drive = "probe = i;"
        loop_count = f"({1 << tw})"
    else:
        nwords = max(1, (tw + 31) // 32)
        drive = "probe = {" + ", ".join(["$random"] * nwords) + "};"
        loop_count = str(count)

    return f"""`timescale 1ns/1ps
module diff_tb;
  reg [{tw-1}:0] probe;
  {in_decls}
  {ref_outs}
  {mut_outs}
  wire [{ow-1}:0] refcat = {refcat};
  wire [{ow-1}:0] mutcat = {mutcat};
  ref_module refi({ref_conn});
  mut_module muti({mut_conn});
  integer i; integer nmm;
  initial begin
    nmm = 0;
    for (i = 0; i < {loop_count}; i = i + 1) begin
      {drive}
      {lhs} = probe;
      #1;
      if (refcat !== mutcat) begin
        if (nmm < {max_witnesses}) $display("MM %b", probe);
        nmm = nmm + 1;
        if (nmm >= {max_witnesses}) begin $display("DONE %0d", nmm); $finish; end
      end
    end
    $display("DONE %0d", nmm);
    $finish;
  end
endmodule
"""


# ---------------------------------------------------------------------------
# Sequential differential search (best-effort)
# ---------------------------------------------------------------------------

def _seq_tb(ports: Ports, ncells: int) -> str:
    tw, ow = ports.input_width, ports.output_width
    in_decls = "\n  ".join(f"reg [{w-1}:0] {n};" for n, w in ports.inputs)
    ref_outs = "\n  ".join(f"wire [{w-1}:0] ref_{n};" for n, w in ports.outputs)
    mut_outs = "\n  ".join(f"wire [{w-1}:0] mut_{n};" for n, w in ports.outputs)
    clocks = ports.clock_inputs or ([(ports.clk_name, 1)] if ports.clk_name else [("clk", 1)])
    clock_names = [name for name, _ in clocks]
    clk_decls = "\n  ".join(f"reg {n};" for n in clock_names)
    clock_ref_conn = [f".{n}({n})" for n in clock_names]
    ref_conn = ", ".join(clock_ref_conn +
                         [f".{n}({n})" for n, _ in ports.inputs] +
                         [f".{n}(ref_{n})" for n, _ in ports.outputs])
    mut_conn = ", ".join(clock_ref_conn +
                         [f".{n}({n})" for n, _ in ports.inputs] +
                         [f".{n}(mut_{n})" for n, _ in ports.outputs])
    lhs = "{" + ", ".join(n for n, _ in ports.inputs) + "}"
    refcat = "{" + ", ".join(f"ref_{n}" for n, _ in ports.outputs) + "}"
    mutcat = "{" + ", ".join(f"mut_{n}" for n, _ in ports.outputs) + "}"
    clk_low = "\n      ".join(f"{n} = 0;" for n in clock_names)
    if len(clock_names) == 1:
        clk_high = f"{clock_names[0]} = 1;"
    else:
        # Rotate independent clock domains instead of toggling every clock
        # together. Repeating non-primary clocks gives CDC/synchronizer paths
        # time to observe delayed enables without assuming hidden internals.
        pattern = [clock_names[0]]
        for name in clock_names[1:]:
            pattern.extend([name, name, name])
        cases = []
        for idx, active in enumerate(pattern):
            assigns = " ".join(f"{name} = {1 if name == active else 0};" for name in clock_names)
            cases.append(f"{idx}: begin {assigns} end")
        clk_high = f"case (g % {len(pattern)}) " + " ".join(cases) + f" default: begin {clk_low} end endcase"

    return f"""`timescale 1ns/1ps
module diff_tb;
  {clk_decls}
  reg [{tw-1}:0] probe;
  {in_decls}
  {ref_outs}
  {mut_outs}
  reg [{tw-1}:0] mem [0:{ncells-1}];
  wire [{ow-1}:0] refcat = {refcat};
  wire [{ow-1}:0] mutcat = {mutcat};
  ref_module refi({ref_conn});
  mut_module muti({mut_conn});
  integer g;
  initial begin
    $readmemb("probe.mem", mem);
    {clk_low}
    for (g = 0; g < {ncells}; g = g + 1) begin
      probe = mem[g];
      {lhs} = probe;
      {clk_low} #1; {clk_high} #1;
      if (refcat !== mutcat) begin
        $display("MM %0d", g);
        $finish;
      end
      {clk_low} #1;
    end
    $display("DONE 0");
    $finish;
  end
endmodule
"""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class MutantVerdict:
    equivalent: bool = False        # no distinguishing input found
    certain: bool = False           # equivalence proven by exhaustive sweep
    witnesses: List[dict] = field(default_factory=list)  # stimulus-format witnesses
    error: str = ""                 # non-empty => classification unknown

    @property
    def unknown(self) -> bool:
        return bool(self.error)


def classify_cmb(ref_v: str, mut_v: str, ports: Ports, *,
                 max_exhaustive_bits: int = 16, random_samples: int = 50000,
                 max_witnesses: int = 16, timeout: int = 120) -> MutantVerdict:
    if not ports.inputs or not ports.outputs:
        return MutantVerdict(error="no inputs/outputs to differentiate")

    exhaustive = ports.input_width <= max_exhaustive_bits
    tb = _cmb_tb(ports, exhaustive, random_samples, max_witnesses)
    sources = {
        "ref.v": _rename_top(ref_v, "ref_module"),
        "mut.v": _rename_top(mut_v, "mut_module"),
        "diff_tb.v": tb,
    }
    with tempfile.TemporaryDirectory() as wd:
        ok, out = _run_iverilog(sources, wd, timeout)
    if not ok:
        return MutantVerdict(error=out)

    witnesses = [_decode(line.split(" ", 1)[1].strip(), ports.inputs)
                 for line in out.splitlines() if line.startswith("MM ")]
    if witnesses:
        return MutantVerdict(equivalent=False, certain=exhaustive, witnesses=witnesses)
    # no difference found
    return MutantVerdict(equivalent=True, certain=exhaustive)


def classify_seq(ref_v: str, mut_v: str, ports: Ports, *,
                 seq_trials: int = 80, seq_cycles: int = 128,
                 timeout: int = 180, rng=None) -> MutantVerdict:
    if not ports.inputs or not ports.outputs:
        return MutantVerdict(error="no inputs/outputs to differentiate")
    import random as _random
    rng = rng or _random.Random(0xC0FFEE)

    tw = ports.input_width
    ncells = seq_trials * seq_cycles
    mem_lines = ["".join(rng.choice("01") for _ in range(tw)) for _ in range(ncells)]
    tb = _seq_tb(ports, ncells)
    sources = {
        "ref.v": _rename_top(ref_v, "ref_module"),
        "mut.v": _rename_top(mut_v, "mut_module"),
        "diff_tb.v": tb,
        "probe.mem": "\n".join(mem_lines) + "\n",
    }
    with tempfile.TemporaryDirectory() as wd:
        ok, out = _run_iverilog(sources, wd, timeout)
    if not ok:
        return MutantVerdict(error=out)

    hit = next((int(l.split()[1]) for l in out.splitlines() if l.startswith("MM ")), None)
    if hit is None:
        # Sequential search is bounded and cannot prove equivalence. Treat this
        # as unknown so benchmark generation does not create fake "equivalent"
        # labels that punish stronger later testbenches.
        return MutantVerdict(error="no divergence found in bounded sequential search")

    # Emit the full prefix [0..hit] as one scenario so the witness reproduces
    # exactly (state carries across cycles; there is no reset assumption).
    cells = [_decode(mem_lines[g], ports.inputs) for g in range(hit + 1)]
    scenario = {"clock_cycles": hit + 1}
    for name, _ in ports.inputs:
        scenario[name] = [cells[g][name] for g in range(hit + 1)]
    return MutantVerdict(equivalent=False, certain=False, witnesses=[scenario])


def classify_mutant(ref_v: str, mut_v: str, circuit_type: str, **kw) -> MutantVerdict:
    ports = parse_ports(ref_v)
    if circuit_type.lower() == "seq":
        seq_kw = {k: kw[k] for k in ("seq_trials", "seq_cycles", "timeout") if k in kw}
        return classify_seq(ref_v, mut_v, ports, **seq_kw)
    cmb_kw = {k: kw[k] for k in
              ("max_exhaustive_bits", "random_samples", "max_witnesses", "timeout")
              if k in kw}
    return classify_cmb(ref_v, mut_v, ports, **cmb_kw)


# ---------------------------------------------------------------------------
# Stimulus augmentation
# ---------------------------------------------------------------------------

def augment_stimulus_file(stimulus_json_path: str, witnesses: List[dict]) -> int:
    """Append witnesses to an existing stimulus.json (de-duplicated). Returns #added."""
    try:
        with open(stimulus_json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    existing = {json.dumps(v, sort_keys=True) for v in data}
    added = 0
    for w in witnesses:
        key = json.dumps(w, sort_keys=True)
        if key not in existing:
            data.append(w)
            existing.add(key)
            added += 1
    with open(stimulus_json_path, "w") as f:
        json.dump(data, f, indent=2)
    return added


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest() -> int:
    logging.basicConfig(level=logging.INFO)
    ref = "module top_module(input a, input b, output y); assign y = a & b; endmodule"
    weak = "module top_module(input a, input b, output y); assign y = a | b; endmodule"   # AND->OR: non-equivalent
    equiv = "module top_module(input a, input b, output y); assign y = b & a; endmodule"  # reordered: equivalent

    v1 = classify_mutant(ref, weak, "CMB", max_exhaustive_bits=8)
    v2 = classify_mutant(ref, equiv, "CMB", max_exhaustive_bits=8)
    print("AND->OR  :", "unknown" if v1.unknown else
          f"equivalent={v1.equivalent} certain={v1.certain} witnesses={v1.witnesses}")
    print("reordered:", "unknown" if v2.unknown else
          f"equivalent={v2.equivalent} certain={v2.certain}")

    ok = (not v1.unknown and not v1.equivalent and v1.witnesses
          and not v2.unknown and v2.equivalent and v2.certain)
    # witness for AND->OR must be an input where a!=b
    if ok:
        w = v1.witnesses[0]
        ok = w.get("a") != w.get("b")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
