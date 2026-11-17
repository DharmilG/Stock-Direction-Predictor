"""
Colab Q0 parity check — vLLM (candidate fast backend) vs the existing local
Ollama ratings, on the SAME Pool C cases.

Purpose
-------
Before switching the Q0-D1 gate from Ollama to a batched vLLM path (Option B),
prove statistical equivalence: identical prompts, matched sampling settings,
and the same go/no-go statistic (Qwen agreement vs FinBERT agreement).

Run in a Google Colab notebook with an L4/A100 GPU:
  1. Create a Drive folder `StockQParity` containing:
       pool_c.parquet, qwen_rating_ledger.parquet  (and optionally traces/)
  2. Mount Drive and run this script (see COLAB_SETUP below).

The Ollama side is read from the ledger (no Ollama needed in Colab).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# COLAB_SETUP (paste into the first notebook cell, then Runtime -> Restart,
# then run this script as the second cell):
#
# !pip install -q "vllm>=0.8.5" pandas pyarrow
# from google.colab import drive
# drive.mount("/content/drive")
# !git clone -q https://github.com/DharmilG/Stock-Direction-Predictor.git /content/repo
# ---------------------------------------------------------------------------

DATA = Path("/content/q0_parity")
DRIVE_SRC = Path("/content/drive/MyDrive/StockQParity")
REPO = Path("/content/repo")


def prepare_data() -> None:
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir()
    for name in ("pool_c.parquet", "qwen_rating_ledger.parquet", "traces"):
        src = DRIVE_SRC / name
        if src.exists():
            dst = DATA / name
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
        elif name != "traces":
            raise FileNotFoundError(f"Missing required file in Drive: {src}")


def main() -> None:
    prepare_data()
    sys.path.insert(0, str(REPO / "src"))
    import pandas as pd
    from stockml.data.qwen_harness import (
        DEFAULT_THINKING_BUDGET, build_prompt, validate_output,
    )
    from stockml.data.q0_d1 import LABEL_TO_VOTE

    led = pd.read_parquet(DATA / "qwen_rating_ledger.parquet")
    pool = pd.read_parquet(DATA / "pool_c.parquet").set_index("case_id")
    traces = DATA / "traces"

    good = led[
        (~led["harness_fail"].fillna(False).astype(bool))
        & led["qwen_vote"].isin(["UP", "DOWN", "FLAT"])
    ].copy()
    if len(good) < 30:
        raise RuntimeError(f"Need >=30 clean Ollama rows, got {len(good)}")

    # Stratified sample (seeded) so all three votes are represented.
    sample = (
        good.groupby("qwen_vote", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 34), random_state=43))
        .reset_index(drop=True)
    )
    print(f"Parity sample: {len(sample)} cases", sample.qwen_vote.value_counts().to_dict())

    prompts = []
    for _, r in sample.iterrows():
        tp = r.get("trace_path")
        fname = os.path.basename(str(tp)) if pd.notna(tp) else ""
        pf = traces / fname
        if pf.exists():
            prompts.append(json.loads(pf.read_text(encoding="utf-8"))["prompt"])
            continue
        row = pool.loc[r["case_id"]]
        ctx = (
            f"5-day trend: {row['trend_5d']:.2%}, volume shock (z): {row['volume_z']:.2f}, "
            f"FinBERT prior: {row['finbert_score']:.2f}"
            if pd.notna(row["finbert_score"])
            else f"5-day trend: {row['trend_5d']:.2%}, NO_NEWS FinBERT prior"
        )
        p, _ = build_prompt(
            row["symbol"], row["sector"] or "unknown", ctx, row["text"],
            thinking_budget=DEFAULT_THINKING_BUDGET,
        )
        prompts.append(p)

    # ---- vLLM generation ---------------------------------------------------
    from vllm import LLM, SamplingParams

    # T4 has 16 GB: 8B in bf16 won't fit. Use 4-bit (matches Ollama's Q4_K_M
    # reference and leaves room for the KV cache).
    llm = LLM(
        model="Qwen/Qwen3-8B", max_model_len=2048,
        quantization="bitsandbytes", dtype="float16",
        gpu_memory_utilization=0.85,
    )
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=1024, seed=42)
    try:
        outs = llm.chat(
            [{"role": "user", "content": p} for p in prompts], sp,
            chat_template_kwargs={"enable_thinking": True},
        )
    except TypeError:
        outs = llm.generate(prompts, sp)

    votes, think_lens, valids = [], [], []
    for o in outs:
        req = o.outputs[0] if hasattr(o, "outputs") else o
        txt = req.text
        think = getattr(req, "reasoning_content", "") or ""
        if "<think>" in txt:
            think = txt.split("<think>", 1)[1].split("</think>", 1)[0]
            txt = txt.split("</think>", 1)[-1]
        parsed, _ = validate_output(txt, think)
        votes.append(parsed["vote"] if parsed else None)
        think_lens.append(len(think))
        valids.append(parsed is not None)

    # ---- Comparison --------------------------------------------------------
    sample["vllm_vote"] = votes
    sample["vllm_think_len"] = think_lens
    sample["vllm_valid"] = valids
    sample["label_vote"] = sample["historical_label"].map(LABEL_TO_VOTE)

    agree = (sample.vllm_vote == sample.qwen_vote).mean()
    oll_acc = sample.vote_matches_label.mean()
    vll_acc = (sample.vllm_vote == sample.label_vote).mean()
    fin_acc = sample.finbert_matches_label.mean()
    report = {
        "n": len(sample),
        "vote_agreement_vllm_vs_ollama": round(float(agree), 4),
        "json_valid_rate_vllm": round(float(sample.vllm_valid.mean()), 4),
        "harness_fail_rate_ollama_sample": round(float(sample.harness_fail.mean()), 4),
        "qwen_agreement_ollama": round(float(oll_acc), 4),
        "qwen_agreement_vllm": round(float(vll_acc), 4),
        "finbert_agreement_same_cases": round(float(fin_acc), 4),
        "delta_vs_finbert_ollama": round(float(oll_acc - fin_acc), 4),
        "delta_vs_finbert_vllm": round(float(vll_acc - fin_acc), 4),
        "mean_think_len": int(sample.vllm_think_len.mean()),
    }
    print(json.dumps(report, indent=2))
    try:
        (DRIVE_SRC / "parity_report.json").write_text(json.dumps(report, indent=2))
    except Exception as exc:
        print("Could not save report to Drive:", exc)
    ok = (
        report["vote_agreement_vllm_vs_ollama"] >= 0.90
        and (report["delta_vs_finbert_ollama"] > 0) == (report["delta_vs_finbert_vllm"] > 0)
    )
    print("PARITY:", "PASS" if ok else "FAIL", "- see DECISION RULE in docstring")


if __name__ == "__main__":
    main()
