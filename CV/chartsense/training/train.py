"""Resumable two-stage training for the CV lane.

Stage 1 pretrains both heads on synthetic pattern charts. Stage 2 freezes the
backbone and pattern head and fine-tunes the direction head on this platform's
triple-barrier-labeled real charts. Every epoch writes an atomic checkpoint
containing optimizer, scheduler, RNG, epoch and fingerprint state.
"""

from __future__ import annotations

import io
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from ..config import (
    BEST_MODEL_PATH,
    CHECKPOINT_VERSION,
    CV_ROOT,
    DIRECTIONS,
    EVALUATION_DIR,
    LABELS_CSV,
    LOGS_DIR,
    METRICS_DIR,
    MODEL_ARCH,
    PATTERNS,
    QUARANTINE_DIR,
    REAL_DIR,
    REAL_LABELS_CSV,
    STAGE1_BEST_PATH,
    STAGE1_LAST_PATH,
    STAGE2_LAST_PATH,
    SYNTHETIC_DIR,
    ensure_dirs,
)
from ..data.dataset import ChartDataset, RealChartDataset
from ..fault import (
    GracefulStop,
    Lock,
    code_fingerprint,
    dict_fingerprint,
    file_fingerprint,
    initial_state,
    read_state,
    require_match,
    update_state,
    atomic_save_torch,
)
from ..model.chartsense import ChartSenseModel

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def _log(name: str, payload: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (LOGS_DIR / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _write_metrics(name: str, payload: dict) -> None:
    """Append one fsynced per-epoch record; the API reads this file live."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _class_metrics(y_true: list[int], y_pred: list[int], scores: list[list[float]] | None, class_names: list[str]) -> dict:
    """Per-class precision/recall/F1 plus a macro F1 and one-vs-rest AUC."""
    if not y_true:
        return {}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    result = {
        "f1_macro": float(np.mean(f1)),
        "per_class": {
            class_names[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(len(class_names))
        },
    }
    if scores and len(set(y_true)) > 1:
        try:
            result["auc_ovr"] = float(
                roc_auc_score(np.eye(len(class_names))[y_true], np.asarray(scores), multi_class="ovr", average="macro")
            )
        except Exception:
            pass
    return result


def _system_metrics() -> dict:
    try:
        return {
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "ram_available_mb": round(float(psutil.virtual_memory().available / 1024 / 1024), 1),
        }
    except Exception:
        return {}


def _quarantine(path: Path, reason: str) -> str:
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    dest = QUARANTINE_DIR / f"{path.name}.{reason}-{int(time.time())}"
    try:
        os.replace(path, dest)
    except OSError:
        return ""
    return str(dest)


def load_checkpoint(path: Path | str) -> dict:
    """Accept both the new payload format and the original raw state_dict."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        return payload
    return {"model": payload}


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _restore_rng(state: dict | None) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
    except Exception:
        # A partially written RNG state must not prevent resume.
        pass


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fingerprint(stage: int, hyper: dict, dataset_fp: str) -> str:
    return dict_fingerprint(
        {
            "arch": MODEL_ARCH,
            "checkpoint_version": CHECKPOINT_VERSION,
            "stage": stage,
            "dataset": dataset_fp,
            "hyper": hyper,
            "code": code_fingerprint(CV_ROOT / "chartsense"),
        }
    )


def _chronological_split(frame: pd.DataFrame, validation_fraction: float = 0.2):
    """Chronological holdout: all validation dates are after the train cutoff."""
    dates = pd.to_datetime(frame["date"])
    unique = np.sort(dates.unique())
    if len(unique) < 2:
        raise ValueError("need at least two distinct dates for a chronological split")
    cutoff = unique[max(1, int(len(unique) * (1 - validation_fraction)))]
    train_idx = np.where(dates < cutoff)[0].tolist()
    val_idx = np.where(dates >= cutoff)[0].tolist()
    if not train_idx or not val_idx:
        raise ValueError("chronological split produced an empty side")
    return train_idx, val_idx


def _split_dataset(dataset, seed: int, train_fraction: float = 0.8):
    n = len(dataset)
    if n < 2:
        raise ValueError("need at least two charts to split train/validation")
    train_size = max(1, int(n * train_fraction))
    val_size = n - train_size
    indices = list(range(n))
    train_subset, val_subset = random_split(indices, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    return Subset(dataset, train_subset.indices), Subset(dataset, val_subset.indices)


def _run_epoch(
    model,
    loader,
    criterion_stage: str,
    optimizer,
    device,
    training: bool,
    class_weight=None,
    compute_aux: bool = True,
    label_smoothing: float = 0.0,
):
    model.train(training)
    totals = {"pattern_correct": 0, "direction_correct": 0, "n": 0, "loss": 0.0}
    pattern_truth: list[int] = []
    pattern_pred: list[int] = []
    pattern_scores: list[list[float]] = []
    direction_truth: list[int] = []
    direction_pred: list[int] = []
    direction_scores: list[list[float]] = []
    direction_idx = 0 if criterion_stage == "direction" else 1

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, *labels in loader:
            images = images.to(device)
            labels = [label.to(device) for label in labels]
            if training:
                optimizer.zero_grad()
            pattern_out, direction_out = model(images)
            loss = torch.zeros((), device=device)
            if criterion_stage in ("both", "pattern"):
                loss = loss + nn.functional.cross_entropy(pattern_out, labels[0], weight=class_weight, label_smoothing=label_smoothing)
            if criterion_stage in ("both", "direction"):
                loss = loss + nn.functional.cross_entropy(direction_out, labels[direction_idx], weight=class_weight, label_smoothing=label_smoothing)
            if training:
                loss.backward()
                optimizer.step()

            batch_n = len(images)
            totals["loss"] += float(loss.item()) * batch_n
            totals["n"] += batch_n
            direction_labels = labels[direction_idx]
            direction_pred.extend(direction_out.argmax(1).cpu().tolist())
            direction_truth.extend(direction_labels.cpu().tolist())
            if compute_aux:
                direction_scores.extend(torch.softmax(direction_out, dim=1).cpu().tolist())
            totals["direction_correct"] += int((direction_out.argmax(1) == direction_labels).sum().item())
            if len(labels) == 2:
                totals["pattern_correct"] += int((pattern_out.argmax(1) == labels[0]).sum().item())
                pattern_pred.extend(pattern_out.argmax(1).cpu().tolist())
                pattern_truth.extend(labels[0].cpu().tolist())
                if compute_aux:
                    pattern_scores.extend(torch.softmax(pattern_out, dim=1).cpu().tolist())

    n = max(1, totals["n"])
    totals["loss"] /= n
    totals["pattern_acc"] = totals["pattern_correct"] / n
    totals["direction_acc"] = totals["direction_correct"] / n
    if compute_aux:
        totals["direction_metrics"] = _class_metrics(direction_truth, direction_pred, direction_scores, DIRECTIONS)
        if pattern_truth:
            totals["pattern_metrics"] = _class_metrics(pattern_truth, pattern_pred, pattern_scores, PATTERNS)
    return totals


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
def train_stage1(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    resume: bool = True,
    seed: int = 42,
    stop: GracefulStop | None = None,
) -> dict:
    ensure_dirs()
    _set_seed(seed)
    hyper = {"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed}
    dataset_fp = file_fingerprint([LABELS_CSV])
    fp = _fingerprint(1, hyper, dataset_fp)

    with Lock("train") as lock:
        state = initial_state("train_stage1", status="initialized", fingerprint=fp, epoch=-1, best_metric=0.0)
        if resume:
            require_match(state.get("fingerprint"), fp, "stage-1 training")

        payload: dict | None = None
        start_epoch, best, history = 0, 0.0, []
        if resume and STAGE1_LAST_PATH.exists():
            try:
                candidate = load_checkpoint(STAGE1_LAST_PATH)
                require_match(candidate.get("fingerprint"), fp, "stage-1 checkpoint")
                require_match(candidate.get("arch"), MODEL_ARCH, "stage-1 architecture")
                payload = candidate
                start_epoch = int(candidate.get("epoch", -1)) + 1
                best = float(candidate.get("best_metric", 0.0))
                history = list(candidate.get("history", []))
                _log("train_stage1", {"event": "resumed", "epoch": start_epoch})
            except Exception as exc:
                _quarantine(STAGE1_LAST_PATH, "stage1-corrupt")
                _log("train_stage1", {"event": "last-checkpoint-invalid", "error": str(exc)})
                payload = None

        model = ChartSenseModel()
        if payload and "model" in payload:
            model.load_state_dict(payload["model"])
        elif STAGE1_BEST_PATH.exists():
            try:
                fallback = load_checkpoint(STAGE1_BEST_PATH)
                model.load_state_dict(fallback["model"])
                _log("train_stage1", {"event": "restored-from-best"})
            except Exception as exc:
                _log("train_stage1", {"event": "best-checkpoint-invalid", "error": str(exc)})

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        if payload and "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception:
                pass
        if payload and "scheduler" in payload:
            try:
                scheduler.load_state_dict(payload["scheduler"])
            except Exception:
                pass
        _restore_rng(payload.get("rng") if payload else None)

        train_full = ChartDataset(LABELS_CSV, SYNTHETIC_DIR, transform=TRAIN_TRANSFORM)
        val_full = ChartDataset(LABELS_CSV, SYNTHETIC_DIR, transform=VAL_TRANSFORM)
        train_ds, val_ds = _split_dataset(train_full, seed)
        _, val_idx = _split_dataset(val_full, seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_idx, batch_size=batch_size, shuffle=False)
        device = torch.device("cpu")

        summary = {"stage": 1, "epochs_run": 0, "best_metric": best, "stopped": False}
        try:
            for epoch in range(start_epoch, epochs):
                if stop is not None:
                    stop.check()
                epoch_started = time.perf_counter()
                train_metrics = _run_epoch(model, train_loader, "both", optimizer, device, True)
                val_metrics = _run_epoch(model, val_loader, "both", optimizer, device, False)
                scheduler.step(val_metrics["pattern_acc"])
                duration = round(time.perf_counter() - epoch_started, 2)
                history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
                _write_metrics(
                    "train_stage1",
                    {
                        "epoch": epoch,
                        "fingerprint": fp,
                        "train": train_metrics,
                        "val": val_metrics,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch_seconds": duration,
                        "images_per_second": round(train_metrics["n"] / max(duration, 0.01), 1),
                        **_system_metrics(),
                    },
                )
                improved = val_metrics["pattern_acc"] > best
                if improved:
                    best = val_metrics["pattern_acc"]
                    best_payload = {
                        "version": CHECKPOINT_VERSION,
                        "arch": MODEL_ARCH,
                        "stage": 1,
                        "model": model.state_dict(),
                        "fingerprint": fp,
                    }
                    atomic_save_torch(best_payload, STAGE1_BEST_PATH)
                    atomic_save_torch(best_payload, BEST_MODEL_PATH)
                checkpoint = {
                    "version": CHECKPOINT_VERSION,
                    "arch": MODEL_ARCH,
                    "stage": 1,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metric": best,
                    "seed": seed,
                    "rng": _rng_state(),
                    "history": history,
                    "dataset_fingerprint": dataset_fp,
                    "fingerprint": fp,
                }
                atomic_save_torch(checkpoint, STAGE1_LAST_PATH)
                update_state(
                    "train_stage1",
                    status="running",
                    epoch=epoch,
                    best_metric=best,
                    train_pattern_acc=train_metrics["pattern_acc"],
                    val_pattern_acc=val_metrics["pattern_acc"],
                )
                _log("train_stage1", {"epoch": epoch, "train": train_metrics, "val": val_metrics})
                summary["epochs_run"] += 1
                lock.heartbeat()
        except KeyboardInterrupt:
            summary["stopped"] = True
            update_state("train_stage1", status="paused", epoch=max(start_epoch, summary["epochs_run"] - 1 + start_epoch))
        else:
            update_state("train_stage1", status="done", epoch=epochs - 1, best_metric=best)
        summary["best_metric"] = best
    return summary


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def train_stage2(
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    resume: bool = True,
    seed: int = 42,
    stop: GracefulStop | None = None,
    reinit_head: bool = False,
    unfreeze_layer4: bool = False,
    label_smoothing: float = 0.0,
) -> dict:
    ensure_dirs()
    if not STAGE1_BEST_PATH.exists():
        raise FileNotFoundError(
            f"stage-1 checkpoint missing: {STAGE1_BEST_PATH}. "
            "Run: python -m chartsense.cli train --stage 1 --resume"
        )
    _set_seed(seed)
    hyper = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "class_weighting": True,
        "reinit_head": reinit_head,
        "unfreeze_layer4": unfreeze_layer4,
        "label_smoothing": label_smoothing,
    }
    dataset_fp = dict_fingerprint(
        {
            "real_labels": file_fingerprint([REAL_LABELS_CSV]),
            "stage1": file_fingerprint([STAGE1_BEST_PATH]),
        }
    )
    fp = _fingerprint(2, hyper, dataset_fp)

    labels = pd.read_csv(REAL_LABELS_CSV)
    directions = set(labels["direction"].unique())
    expected = {"neutral", "bullish", "bearish"}
    if not expected.issubset(directions):
        raise ValueError(f"real labels must contain all 3 directions, found {sorted(directions)}")
    counts = labels["direction"].map(DIRECTIONS.index).value_counts().reindex(range(len(DIRECTIONS)), fill_value=0).to_numpy()
    class_weights = torch.tensor(
        [len(labels) / (len(DIRECTIONS) * max(int(count), 1)) for count in counts],
        dtype=torch.float32,
    )

    with Lock("train") as lock:
        state = initial_state("train_stage2", status="initialized", fingerprint=fp, epoch=-1, best_metric=0.0)
        if resume:
            require_match(state.get("fingerprint"), fp, "stage-2 training")

        payload: dict | None = None
        start_epoch, best, history = 0, 0.0, []
        if resume and STAGE2_LAST_PATH.exists():
            try:
                candidate = load_checkpoint(STAGE2_LAST_PATH)
                require_match(candidate.get("fingerprint"), fp, "stage-2 checkpoint")
                require_match(candidate.get("arch"), MODEL_ARCH, "stage-2 architecture")
                payload = candidate
                start_epoch = int(candidate.get("epoch", -1)) + 1
                best = float(candidate.get("best_metric", 0.0))
                history = list(candidate.get("history", []))
                _log("train_stage2", {"event": "resumed", "epoch": start_epoch})
            except Exception as exc:
                _quarantine(STAGE2_LAST_PATH, "stage2-corrupt")
                _log("train_stage2", {"event": "last-checkpoint-invalid", "error": str(exc)})
                payload = None

        model = ChartSenseModel()
        try:
            stage1 = load_checkpoint(STAGE1_BEST_PATH)
            stage1_state = stage1["model"]
            if reinit_head:
                stage1_state = {k: v for k, v in stage1_state.items() if not k.startswith("direction_head.")}
            model.load_state_dict(stage1_state, strict=not reinit_head)
        except Exception as exc:
            raise RuntimeError(f"cannot load stage-1 weights: {exc}") from exc
        if payload and "model" in payload:
            model.load_state_dict(payload["model"])

        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        for parameter in model.pattern_head.parameters():
            parameter.requires_grad = False
        for parameter in model.direction_head.parameters():
            parameter.requires_grad = True
        if unfreeze_layer4:
            for parameter in model.backbone.layer4.parameters():
                parameter.requires_grad = True

        if unfreeze_layer4:
            optimizer = torch.optim.Adam(
                [
                    {"params": list(model.direction_head.parameters()), "lr": lr},
                    {"params": list(model.backbone.layer4.parameters()), "lr": lr * 0.1},
                ]
            )
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        if payload and "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except Exception:
                pass
        if payload and "scheduler" in payload:
            try:
                scheduler.load_state_dict(payload["scheduler"])
            except Exception:
                pass
        _restore_rng(payload.get("rng") if payload else None)

        train_full = RealChartDataset(REAL_LABELS_CSV, REAL_DIR, transform=TRAIN_TRANSFORM)
        val_full = RealChartDataset(REAL_LABELS_CSV, REAL_DIR, transform=VAL_TRANSFORM)
        train_idx, val_idx = _chronological_split(labels)
        train_ds = Subset(train_full, train_idx)
        val_ds = Subset(val_full, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        device = torch.device("cpu")

        summary = {"stage": 2, "epochs_run": 0, "best_metric": best, "stopped": False}
        try:
            for epoch in range(start_epoch, epochs):
                if stop is not None:
                    stop.check()
                epoch_started = time.perf_counter()
                train_metrics = _run_epoch(model, train_loader, "direction", optimizer, device, True, class_weight=class_weights, label_smoothing=label_smoothing)
                val_metrics = _run_epoch(model, val_loader, "direction", optimizer, device, False, class_weight=class_weights, label_smoothing=label_smoothing)
                scheduler.step(val_metrics["direction_acc"])
                duration = round(time.perf_counter() - epoch_started, 2)
                history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
                _write_metrics(
                    "train_stage2",
                    {
                        "epoch": epoch,
                        "fingerprint": fp,
                        "train": train_metrics,
                        "val": val_metrics,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch_seconds": duration,
                        "images_per_second": round(train_metrics["n"] / max(duration, 0.01), 1),
                        "class_weights": [round(float(w), 3) for w in class_weights],
                        **_system_metrics(),
                    },
                )
                improved = val_metrics["direction_acc"] > best
                if improved:
                    best = val_metrics["direction_acc"]
                    best_payload = {
                        "version": CHECKPOINT_VERSION,
                        "arch": MODEL_ARCH,
                        "stage": 2,
                        "model": model.state_dict(),
                        "fingerprint": fp,
                    }
                    atomic_save_torch(best_payload, BEST_MODEL_PATH)
                checkpoint = {
                    "version": CHECKPOINT_VERSION,
                    "arch": MODEL_ARCH,
                    "stage": 2,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metric": best,
                    "seed": seed,
                    "rng": _rng_state(),
                    "history": history,
                    "dataset_fingerprint": dataset_fp,
                    "fingerprint": fp,
                }
                atomic_save_torch(checkpoint, STAGE2_LAST_PATH)
                update_state(
                    "train_stage2",
                    status="running",
                    epoch=epoch,
                    best_metric=best,
                    val_direction_acc=val_metrics["direction_acc"],
                )
                _log("train_stage2", {"epoch": epoch, "train": train_metrics, "val": val_metrics})
                summary["epochs_run"] += 1
                lock.heartbeat()
        except KeyboardInterrupt:
            summary["stopped"] = True
            update_state("train_stage2", status="paused", epoch=start_epoch + max(0, summary["epochs_run"] - 1))
        else:
            update_state("train_stage2", status="done", epoch=epochs - 1, best_metric=best)
        summary["best_metric"] = best
    return summary


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(resume: bool = True) -> dict:
    ensure_dirs()
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"no CV checkpoint to evaluate: {BEST_MODEL_PATH}")
    dataset_fp = file_fingerprint([REAL_LABELS_CSV if REAL_LABELS_CSV.exists() else LABELS_CSV, BEST_MODEL_PATH])
    fp = dict_fingerprint({"eval": "v1", "dataset": dataset_fp})
    state = read_state("evaluate", {})
    if resume and state.get("status") == "done" and state.get("fingerprint") == fp:
        return {"status": "already-done", "fingerprint": fp}

    from ..api.inference import load_cv_model

    model = load_cv_model()
    use_real = REAL_LABELS_CSV.exists()
    if use_real:
        frame = pd.read_csv(REAL_LABELS_CSV)
        _, val_idx = _chronological_split(frame)
        full = RealChartDataset(REAL_LABELS_CSV, REAL_DIR, transform=VAL_TRANSFORM)
        dataset = Subset(full, val_idx)
        split_name = "chronological-holdout-20pct"
    else:
        full = ChartDataset(LABELS_CSV, SYNTHETIC_DIR, transform=VAL_TRANSFORM)
        _, dataset = _split_dataset(full, 42)
        split_name = "random-20pct"
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    ys, yp, scores = [], [], []
    with torch.no_grad():
        for images, *labels in loader:
            pattern_out, direction_out = model(images)
            probabilities = torch.softmax(direction_out, dim=1)
            yp.extend(direction_out.argmax(1).tolist())
            scores.extend(probabilities.tolist())
            ys.append(labels[1] if len(labels) == 2 else labels[0])
    y_true = torch.cat(ys).tolist()
    labels_order = ["neutral", "bullish", "bearish"]
    if len(set(y_true)) > 1:
        auc = float(
            roc_auc_score(
                np.eye(3)[y_true],
                np.array(scores),
                multi_class="ovr",
                average="macro",
            )
        )
    else:
        auc = 0.0
    report = classification_report(y_true, yp, target_names=labels_order, zero_division=0)
    report_dict = classification_report(y_true, yp, target_names=labels_order, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, yp, labels=[0, 1, 2])

    from datetime import datetime, timezone

    from ..fault import atomic_write_json

    (EVALUATION_DIR / "direction_report.txt").write_text(report, encoding="utf-8")
    atomic_write_json(
        EVALUATION_DIR / "cv_metrics.json",
        {
            "accuracy": float(accuracy_score(y_true, yp)),
            "auc_ovr": auc,
            "dataset": "real" if use_real else "synthetic",
            "split": split_name,
            "n_evaluated": len(y_true),
            "labels": labels_order,
            "matrix": matrix.tolist(),
            "report": report_dict,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    figure, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix)
    ax.set_xticks(range(3), labels_order)
    ax.set_yticks(range(3), labels_order)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    from ..fault import atomic_write_bytes

    atomic_write_bytes(EVALUATION_DIR / "cv_confusion_matrix.png", buffer.getvalue())
    update_state("evaluate", status="done", fingerprint=fp, accuracy=accuracy_score(y_true, yp), auc_ovr=auc, split=split_name, n_evaluated=len(y_true))
    return {"status": "done", "accuracy": accuracy_score(y_true, yp), "auc_ovr": auc}


__all__ = ["train_stage1", "train_stage2", "evaluate", "load_checkpoint", "TRAIN_TRANSFORM", "VAL_TRANSFORM"]
