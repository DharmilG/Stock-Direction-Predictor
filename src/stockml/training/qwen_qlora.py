from __future__ import annotations

"""Phase 6 — QLoRA training on the STaR kept corpus.

CODE READY — UNTESTED

Run from the WSL2 venv:
    python -m stockml.training.qwen_qlora --probe
    python -m stockml.training.qwen_qlora --train
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ROOT
from ..qpipeline import state as q_state
from ..qpipeline.contracts import load_q_track_config, phase_artifacts
from ..utils.io import atomic_write_json


@dataclass
class TrainingConfig:
    seq_len: int = 4096
    fallback_seq_len: int = 3072
    rank: int = 16
    fallback_rank: int = 8
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 2e-4
    micro_batch_size: int = 1
    grad_accum: int = 16
    seed: int = 42
    max_steps: int | None = None
    output_dir: Path = ROOT / "artifacts" / "models" / "qwen_lora_r1"
    checkpoint_dir: Path = ROOT / "checkpoints" / "qwen_lora"
    train_jsonl: Path = ROOT / "data" / "llm" / "star_kept.jsonl"
    eval_jsonl: Path = ROOT / "data" / "llm" / "star_val.jsonl"
    results_path: Path = ROOT / "results" / "qwen_qlora.json"
    probe_results_path: Path = ROOT / "results" / "qwen_10step_probe.json"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Training corpus not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _to_sft_example(row: dict) -> dict:
    """Convert a STaR candidate into an SFT prompt/completion pair."""
    prompt = row.get("prompt") or ""
    thinking = row.get("thinking") or ""
    content = row.get("content") or ""
    vote = row.get("vote") or ""
    return {
        "prompt": prompt,
        "completion": f"<think>{thinking}\n{content}\nVote: {vote}",
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _build_dataset(rows: list[dict]):
    from datasets import Dataset

    return Dataset.from_list([_to_sft_example(row) for row in rows])


def _train_once(cfg: TrainingConfig, rank: int, seq_len: int, probe: bool = False) -> dict:
    """One training attempt. Raises on failure so the caller can fall back."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    _set_seed(cfg.seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
        max_seq_length=seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    train_rows = _load_jsonl(cfg.train_jsonl)
    eval_rows = _load_jsonl(cfg.eval_jsonl) if cfg.eval_jsonl.exists() else []
    train_ds = _build_dataset(train_rows)
    eval_ds = _build_dataset(eval_rows) if eval_rows else None

    output_dir = cfg.output_dir
    checkpoint_dir = cfg.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=1 if probe else 3,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1 if probe else 10,
        save_steps=10 if probe else 50,
        save_total_limit=3,
        optim="adamw_torch_fused",
        bf16=True,
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        report_to=[],
        load_best_model_at_end=False,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=10 if probe else 50,
        per_device_eval_batch_size=1,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=seq_len,
        args=args,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return {
        "rank": rank,
        "seq_len": seq_len,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "output_dir": str(output_dir),
    }


def run_phase6(probe: bool = False, max_steps: int | None = None, **_kwargs) -> dict:
    """Run the Phase 6 training (or 10-step probe) with fallbacks."""
    cfg_map = load_q_track_config().get("phase6", {})
    artifacts = phase_artifacts()
    cfg = TrainingConfig(
        seq_len=int(cfg_map.get("seq_len", 4096)),
        fallback_seq_len=int(cfg_map.get("fallback_seq_len", 3072)),
        rank=int(cfg_map.get("rank", 16)),
        fallback_rank=int(cfg_map.get("fallback_rank", 8)),
        alpha=int(cfg_map.get("alpha", 32)),
        dropout=float(cfg_map.get("dropout", 0.05)),
        learning_rate=float(cfg_map.get("learning_rate", 2e-4)),
        micro_batch_size=int(cfg_map.get("micro_batch_size", 1)),
        grad_accum=int(cfg_map.get("grad_accum", 16)),
        seed=int(cfg_map.get("seed", 42)),
        max_steps=int(max_steps if max_steps else cfg_map.get("max_steps_probe", 10)) if probe else None,
        output_dir=Path(artifacts["phase6_train"]["adapter_dir"]) if not probe else Path(ROOT / "checkpoints" / "qwen_lora_probe"),
        checkpoint_dir=Path(ROOT / "checkpoints" / "qwen_lora"),
        train_jsonl=Path(cfg_map.get("train_jsonl", "data/llm/star_kept.jsonl")),
        eval_jsonl=Path(cfg_map.get("eval_jsonl", "data/llm/star_val.jsonl")),
        results_path=Path(artifacts["phase6_train"]["results"]),
        probe_results_path=Path(cfg_map.get("probe_results_path", "results/qwen_10step_probe.json")),
    )
    attempts: list[dict] = []
    error: str | None = None
    for seq_len in (cfg.seq_len, cfg.fallback_seq_len):
        for rank in (cfg.rank, cfg.fallback_rank):
            attempt = {"seq_len": seq_len, "rank": rank, "status": "trying"}
            attempts.append(attempt)
            try:
                info = _train_once(cfg, rank=rank, seq_len=seq_len, probe=probe)
                attempt.update(info)
                attempt["status"] = "success"
                report = {
                    "phase": "phase6_probe" if probe else "phase6_train",
                    "pass": True,
                    "attempts": attempts,
                    **info,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                target = cfg.probe_results_path if probe else cfg.results_path
                atomic_write_json(target, report)
                if not probe:
                    q_state.set_phase_status("phase6_train", "passed", result=report)
                return report
            except Exception as exc:
                attempt["status"] = "failed"
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                error = str(exc)
                continue
    report = {
        "phase": "phase6_probe" if probe else "phase6_train",
        "pass": False,
        "attempts": attempts,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    target = cfg.probe_results_path if probe else cfg.results_path
    atomic_write_json(target, report)
    if not probe:
        q_state.set_phase_status("phase6_train", "failed", result=report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 QLoRA training.")
    parser.add_argument("--probe", action="store_true", help="Run 10-step dummy pre-probe.")
    parser.add_argument("--train", action="store_true", help="Run full training.")
    args = parser.parse_args()
    if not args.probe and not args.train:
        parser.print_help()
        return
    print(run_phase6(probe=args.probe))


if __name__ == "__main__":
    main()
