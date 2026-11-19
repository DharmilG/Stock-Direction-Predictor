"""Per-epoch metric helpers. No training run is started."""

from __future__ import annotations

import json

from CV.chartsense.training import train


def test_class_metrics_computes_precision_recall_f1_and_auc():
    metrics = train._class_metrics(
        y_true=[0, 1, 2, 2],
        y_pred=[0, 1, 1, 2],
        scores=[
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.0, 0.2, 0.8],
        ],
        class_names=["neutral", "bullish", "bearish"],
    )
    assert 0.0 < metrics["f1_macro"] <= 1.0
    assert set(metrics["per_class"]) == {"neutral", "bullish", "bearish"}
    assert 0.0 <= metrics["auc_ovr"] <= 1.0
    assert metrics["per_class"]["neutral"]["support"] == 1


def test_write_metrics_appends_fsynced_json_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "METRICS_DIR", tmp_path)
    train._write_metrics("stage_test", {"epoch": 0, "learning_rate": 0.001})
    train._write_metrics("stage_test", {"epoch": 1, "learning_rate": 0.0005})
    lines = (tmp_path / "stage_test.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["epoch"] == 1
