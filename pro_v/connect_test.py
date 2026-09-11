#!/usr/bin/env python3
"""
connect_test.py — end-to-end connectivity test for the Pro-V pipeline that runs
WITHOUT a GPU / vLLM / llama_index.

The Pro-V agents (GenTBAgent, PyCheckerAgent) take any object with a `.chat()`
method as their LLM client and fall back to plain subprocess execution when no
Ray worker is given. So we can drive the REAL agent code paths (prompt build ->
code extract -> execute -> stimulus.json / golden_dut.py / testbench.json) with a
canned "fake" model, then push the result through the real Verilator harness,
mutation analysis, coverage and 2/4-state differential.

What this proves locally:  the whole pipeline is wired correctly and our tooling
connects to the stock Pro-V agents on real task data.
What still needs a GPU box: replacing FakeLLMClient with the actual PRO-V-R1-8B
served by scripts/launch_vllm_local.sh (set VLLM_URL to smoke-test that link).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pro_v.agent.gen_tb import GenTBAgent
from pro_v.agent.pychecker import PyCheckerAgent
from pro_v import strengthen_testbench as st
from pro_v.simulate_and_evaluate_mutants import run_simulation
from pro_v.mutation_strength import classify_mutant

SCRATCH = Path(os.environ.get(
    "PROV_CONNECT_SCRATCH",
    "/private/tmp/claude-501/-Users-akavasseri/9d396b70-2286-45e8-9386-c49b82d53cc8/scratchpad/connect"))


# --- a real HDLBits-style task (Mux2to1v) as a correctly-formatted benchmark entry ---
TASK = {
    "task_number": 1,
    "task_id": "Mux2to1v",
    "description": "8-bit wide 2-to-1 multiplexer. When sel=0, out=a; when sel=1, out=b.",
    "header": "module top_module(input sel, input [7:0] a, input [7:0] b, output [7:0] out);",
    "module_code": ("module top_module(input sel, input [7:0] a, input [7:0] b, output reg [7:0] out);\n"
                    "  always @(*) begin\n    case(sel)\n      1'b0: out = a;\n      1'b1: out = b;\n"
                    "    endcase\n  end\nendmodule\n"),
    "mutants": [
        "module top_module(input sel, input [7:0] a, input [7:0] b, output reg [7:0] out);\n always @(*) case(sel) 1'b0: out=b; 1'b1: out=a; endcase\nendmodule\n",
        "module top_module(input sel, input [7:0] a, input [7:0] b, output reg [7:0] out);\n always @(*) out=a;\nendmodule\n",
        "module top_module(input sel, input [7:0] a, input [7:0] b, output reg [7:0] out);\n always @(*) if(sel) out=b; else out=a;\nendmodule\n",
    ],
    # result: True = mutant SHOULD pass (equivalent); False = should be detected
    "result": [False, False, True],
}


class FakeLLMClient:
    """Duck-typed stand-in for the vLLM client: returns canned valid code so we
    exercise the real agents without a model. Replace with the real client
    (LLMClient over the vLLM endpoint) on a GPU box."""
    def chat(self, system: str = "", user: str = "", **kw) -> str:
        if "stimulus_gen" in user:
            return (
                "```python\n"
                "def stimulus_gen():\n"
                "    import random\n"
                "    tv = []\n"
                "    for sel in [0, 1]:\n"
                "        for a in [0, 255, 170, 85]:\n"
                "            for b in [0, 255, 85, 170]:\n"
                "                tv.append({\"sel\": str(sel), \"a\": format(a, '08b'), \"b\": format(b, '08b')})\n"
                "    for _ in range(80):\n"
                "        tv.append({\"sel\": str(random.getrandbits(1)),\n"
                "                   \"a\": format(random.getrandbits(8), '08b'),\n"
                "                   \"b\": format(random.getrandbits(8), '08b')})\n"
                "    return tv\n"
                "```")
        if "GoldenDUT" in user:
            return (
                "```python\n"
                "class GoldenDUT:\n"
                "    def __init__(self):\n        pass\n"
                "    def load(self, inputs):\n"
                "        sel = int(inputs[\"sel\"], 2); a = int(inputs[\"a\"], 2); b = int(inputs[\"b\"], 2)\n"
                "        return {\"out\": format(a if sel == 0 else b, '08b')}\n"
                "```")
        return "```python\n# unrecognized prompt\n```"


def check_model_integrity(model_dir: Path) -> dict:
    ok = model_dir.exists()
    cfg = {}
    if (model_dir / "config.json").exists():
        cfg = json.loads((model_dir / "config.json").read_text())
    shards = len(list(model_dir.glob("model-*.safetensors")))
    return {"exists": ok, "arch": cfg.get("architectures"),
            "hidden_size": cfg.get("hidden_size"), "shards": shards,
            "has_index": (model_dir / "model.safetensors.index.json").exists(),
            "has_tokenizer": (model_dir / "tokenizer.json").exists()}


def maybe_live_smoke(url: str) -> dict:
    """If VLLM_URL is set, send one real completion to test the live endpoint."""
    import urllib.request
    payload = json.dumps({
        "model": os.environ.get("VLLM_MODEL", "PRO-V-R1-8B"),
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 5, "temperature": 0,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
        return {"ok": True, "reply": body["choices"][0]["message"]["content"][:40]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run():
    print("=" * 78)
    print("Pro-V connectivity test (GPU-free: real agents + fake model)")
    print("=" * 78)

    # 1) model integrity
    model = check_model_integrity(Path(_REPO_ROOT) / "models" / "PRO-V-R1-8B")
    print(f"[1] model weights: {model}")

    # 2) benchmark schema
    required = {"task_number", "module_code", "mutants", "result"}
    print(f"[2] benchmark entry has required fields "
          f"{required}: {required.issubset(TASK)}; "
          f"{len(TASK['mutants'])} mutants, labels={TASK['result']}")

    # 3) drive the REAL agents with the fake client
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    out_dir = SCRATCH / "task_1"
    out_dir.mkdir(parents=True)
    fake = FakeLLMClient()

    gtb = GenTBAgent(llm_client=fake, worker=None)
    g = gtb.run(description=TASK["description"], header=TASK["header"],
                circuit_type="CMB", output_dir=str(out_dir))
    print(f"[3] GenTBAgent -> success={g.get('success')} "
          f"vectors={g.get('num_test_cases')} (real agent, subprocess exec)")
    assert g.get("success"), g

    pyc = PyCheckerAgent(llm_client=fake, worker=None)
    p = pyc.run(description=TASK["description"], header=TASK["header"], circuit_type="CMB",
                stimulus_json_path=str(out_dir / "stimulus.json"), output_dir=str(out_dir))
    print(f"[4] PyCheckerAgent -> success={p.get('success')} "
          f"testbench.json written (real agent, subprocess exec)")
    assert p.get("success"), p

    # 4) push through the real Verilator harness + mutation eval
    sim = out_dir / "sim_cmb"
    shutil.copytree(Path(_REPO_ROOT) / "pro_v" / "sim_cmb", sim)
    (sim / "top_module.v").write_text(TASK["module_code"])
    shutil.copy(out_dir / "golden_dut.py", sim / "golden_dut.py")
    shutil.copy(out_dir / "stimulus.json", sim / "stimulus.json")
    ok, msg = st.regenerate_testbench(sim, timeout=180)
    assert ok, msg
    base = run_simulation(TASK["module_code"], sim, timeout=180)
    print(f"[5] Verilator: reference module passes = {base.passed}")

    killed = equiv = 0
    detail = []
    for i, m in enumerate(TASK["mutants"]):
        res = run_simulation(m, sim, timeout=180)
        det = not res.passed
        v = classify_mutant(TASK["module_code"], m, "cmb", max_exhaustive_bits=17)
        cls = "unknown" if v.unknown else ("equivalent" if v.equivalent else "nonequiv")
        killed += det
        equiv += (not det and cls == "equivalent")
        # agreement vs benchmark label
        should_detect = not TASK["result"][i]
        agree = (det == should_detect)
        detail.append((i, "KILLED" if det else "survived", cls, "OK" if agree else "MISMATCH"))
    non_eq = len(TASK["mutants"]) - equiv
    print(f"[6] mutants: {killed}/{len(TASK['mutants'])} killed; equivalent={equiv}; "
          f"true_mutation_score={killed/non_eq:.0%}")
    for d in detail:
        print(f"      mutant {d[0]}: {d[1]:8s} class={d[2]:11s} vs-label:{d[3]}")

    agree_all = all(d[3] == "OK" for d in detail)

    # 5) optional live endpoint smoke test
    url = os.environ.get("VLLM_URL")
    if url:
        print(f"[7] live vLLM smoke test @ {url}: {maybe_live_smoke(url)}")
    else:
        print("[7] live vLLM smoke test: skipped (set VLLM_URL=http://host:8020 on a GPU box)")

    ok_all = (g.get("success") and p.get("success") and base.passed and agree_all)
    print("=" * 78)
    print("CONNECT TEST:", "PASS — real Pro-V agents + eval pipeline fully wired"
          if ok_all else "FAIL")
    print("=" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(run())
