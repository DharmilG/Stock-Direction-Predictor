"""The model registry, the paired statistics and the plumbing between them.

Three things are worth testing here and nothing else is: that the registry can
only ever describe models it can find, that a paired interval resamples dates
rather than rows, and that the per-date sufficient statistics reproduce the
reference implementation of every metric they claim to compute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from stockml.api import models as models_module
from stockml.api.compare import ECE_BINS, _Daily, _axes, _bin_index, _paired, _verdict
from stockml.evaluation.metrics import expected_calibration_error
from stockml.evaluation.paired import clustered_delta, mcnemar
from stockml.v2.workspace import V2_ROOT


# --- registry --------------------------------------------------------------- #
def test_registry_is_one_served_incumbent_and_one_unservable_twin():
    specs = models_module.registry()
    assert set(specs) == {"v1", "v2"}

    assert specs["v1"].servable is True
    assert specs["v1"].status == "serving"

    # The whole point of registering v2 is that it can be measured, not served.
    assert specs["v2"].servable is False
    assert specs["v2"].status == "research"

    assert specs["v2"].paths.root.resolve() == V2_ROOT.resolve()
    assert specs["v1"].paths.root.resolve() != specs["v2"].paths.root.resolve()


def test_resolve_is_case_and_whitespace_insensitive():
    specs = models_module.registry()
    assert models_module.resolve(None) is specs[models_module.DEFAULT_MODEL]
    assert models_module.resolve(" V2 ") is specs["v2"]


def test_unknown_model_is_a_404_not_a_silent_fallback():
    with pytest.raises(HTTPException) as excinfo:
        models_module.resolve("v9")
    assert excinfo.value.status_code == 404
    assert "v1" in excinfo.value.detail


def test_v2_is_scored_under_its_own_flat_band():
    """Serving v2 under the incumbent's 0.4 band would describe a model never fitted."""
    spec = models_module.registry()["v2"]
    selection = models_module.json_file(V2_ROOT / "artifacts" / "v2_selection.json")
    if not selection:
        pytest.skip("v2 has not been trained in this checkout")
    tau = (selection.get("policy") or {}).get("tau_flat")
    assert tau is not None
    assert spec.settings.flat_threshold == pytest.approx(float(tau))


def _fake_metadata(monkeypatch, incumbent: dict, candidate: dict) -> None:
    """Point the registry's file reader at two in-memory metadata blobs."""

    def fake_json_file(path):
        path = Path(path)
        if path.name != "model_metadata.json":
            return None
        # v2's workspace lives under a directory component called "v2"; v1's
        # artefacts sit directly under the platform root.
        return candidate if "v2" in path.parts else incumbent

    monkeypatch.setattr(models_module, "json_file", fake_json_file)


def test_registry_warns_only_when_the_incumbent_moved_since_v2_was_fitted(monkeypatch):
    incumbent = {"dataset_signature": "a" * 64}

    _fake_metadata(monkeypatch, incumbent, {"dataset_signature": "b" * 64, "incumbent_signature": "a" * 64})
    assert models_module.registry_warnings() == []

    _fake_metadata(monkeypatch, incumbent, {"dataset_signature": "b" * 64, "incumbent_signature": "c" * 64})
    warnings = models_module.registry_warnings()
    assert len(warnings) == 1
    assert "refit" in warnings[0]


def test_registry_warns_when_v2_never_recorded_the_incumbent_signature(monkeypatch):
    _fake_metadata(monkeypatch, {"dataset_signature": "a" * 64}, {"dataset_signature": "b" * 64})
    warnings = models_module.registry_warnings()
    assert len(warnings) == 1
    assert "cannot be proven comparable" in warnings[0]


# --- the date-clustered interval -------------------------------------------- #
def _dates(per_day: int):
    return np.repeat(np.array(["2024-01-0%d" % day for day in range(1, len(per_day) + 1)], dtype="datetime64[ns]"), per_day)


def test_identical_models_produce_a_zero_delta_with_a_zero_interval():
    scores = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    result = clustered_delta(_dates([2, 2, 2]), scores, scores, reps=200)
    assert result["mean_delta"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["dates"] == 3
    assert result["rows"] == 6


def test_a_clear_winner_gets_an_interval_that_excludes_zero():
    dates = _dates([4, 4, 4, 4])
    a = np.zeros(16)
    b = np.ones(16)
    result = clustered_delta(dates, a, b, reps=200)
    assert result["mean_delta"] == pytest.approx(1.0)
    assert result["ci_low"] > 0
    assert result["t"] is None or result["t"] > 0


def test_a_date_weighs_the_same_whether_it_carries_one_row_or_a_hundred():
    """The reason rows are not resampled: fifty names share one day's move."""
    # Date one: 100 rows, no difference. Date two: 1 row, b right and a wrong.
    dates = _dates([100, 1])
    a = np.zeros(101)
    b = np.concatenate([np.zeros(100), np.ones(1)])
    result = clustered_delta(dates, a, b, reps=200)
    assert result["mean_delta"] == pytest.approx(0.5)


def test_mcnemar_counts_only_the_discordant_rows():
    a = np.array([True] * 40 + [False] * 10 + [True] * 5 + [False] * 5)
    b = np.array([False] * 40 + [True] * 10 + [True] * 5 + [False] * 5)
    result = mcnemar(a, b)
    assert result["a_only"] == 40
    assert result["b_only"] == 10
    assert result["discordant"] == 50
    assert result["p_value"] < 0.05


def test_mcnemar_is_silent_when_the_models_never_disagree_differently():
    scores = np.array([True, False, True, False])
    assert mcnemar(scores, scores)["p_value"] == 1.0

    # Equal discordance in both directions is no evidence either way.
    a = np.array([True, False])
    b = np.array([False, True])
    assert mcnemar(a, b)["p_value"] == 1.0


# --- per-date sufficient statistics ------------------------------------------ #
ACTUAL = np.array([0, 0, 1, 2, 2, 2])
PREDICTED = np.array([0, 2, 1, 2, 0, 2])
PROBA = np.array(
    [
        [0.9, 0.05, 0.05],
        [0.2, 0.3, 0.5],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
        [0.7, 0.1, 0.2],
        [0.1, 0.3, 0.6],
    ]
)
# Two dates, so the matrix has more than one row to aggregate over.
DATES = np.array(["2024-03-01", "2024-03-01", "2024-03-01", "2024-03-04", "2024-03-04", "2024-03-04"], dtype="datetime64[ns]")


def test_bin_index_matches_the_reference_binning():
    assert _bin_index(np.array([0.0]))[0] == 0
    assert _bin_index(np.array([1.0]))[0] == ECE_BINS - 1
    # (lo, hi] with 0 inclusive: a confidence of 0.30 belongs to the 0.2-0.3 bin.
    assert _bin_index(np.array([0.3]))[0] == 2
    assert _bin_index(np.array([0.31]))[0] == 3
    assert _bin_index(np.linspace(0, 1, 50)).max() == ECE_BINS - 1


def test_axes_reproduce_the_reference_metrics():
    daily = _Daily(DATES, ACTUAL, PREDICTED, PROBA)
    assert daily.dates.size == 2
    axes = _axes(daily.stats.sum(axis=0))

    correct = (PREDICTED == ACTUAL).astype(float)
    assert axes["accuracy_all"] == pytest.approx(correct.mean())
    assert axes["coverage"] == pytest.approx(1.0)

    # Per-class recall, over the rows whose truth is that class.
    assert axes["recall_DOWN"] == pytest.approx(1 / 2)
    assert axes["recall_FLAT"] == pytest.approx(1.0)
    assert axes["recall_UP"] == pytest.approx(2 / 3)

    log_loss = -np.log(np.take_along_axis(PROBA, ACTUAL.reshape(-1, 1), axis=1)).mean()
    assert axes["log_loss"] == pytest.approx(float(log_loss))

    # The strongest check available: the same number as the served implementation.
    assert axes["ece"] == pytest.approx(expected_calibration_error(ACTUAL, PROBA)["ece"])

    assert -1.0 <= axes["mcc"] <= 1.0
    assert 1 / 3 <= axes["balanced_accuracy"] <= 1.0


def test_a_bootstrap_replicate_of_one_date_matches_the_point_estimate():
    """A replicate that draws the same date repeatedly must collapse to it."""
    daily = _Daily(DATES, ACTUAL, PREDICTED, PROBA)
    for date_index in range(daily.dates.size):
        draws = np.full((3, 1), date_index, dtype=int)
        batch = _axes(daily.stats[draws].sum(axis=1))
        assert batch["accuracy_all"].shape == (3,)
        assert np.allclose(batch["accuracy_all"], batch["accuracy_all"][0])
        assert np.allclose(batch["log_loss"], batch["log_loss"][0])


def test_a_model_that_never_calls_a_class_scores_zero_recall_there():
    predicted = np.array([2, 2, 2, 2, 2, 2])
    axes = _axes(_Daily(DATES, ACTUAL, predicted, PROBA).stats.sum(axis=0))
    assert axes["recall_FLAT"] == pytest.approx(0.0)
    assert axes["recall_DOWN"] == pytest.approx(0.0)
    # Every UP row in this sample is called UP, so the conditional recall is one.
    assert axes["recall_UP"] == pytest.approx(1.0)


def test_verdict_never_names_a_single_winner():
    rows = [
        {"axis": "recall_UP", "label": "UP recall (called)", "better": "v2", "significant": True},
        {"axis": "recall_DOWN", "label": "DOWN recall (called)", "better": "v1", "significant": True},
        {"axis": "ece", "label": "Calibration error (ECE)", "better": None, "significant": False},
    ]
    verdict = _verdict(rows, "v1", "v2", [{"label": "AUC(UP)", "pass": False}])
    assert "v2 is ahead on 1 of 2" in verdict
    assert "not promotable" in verdict
    # The gates come from v2's own report on its own test window. Attributing
    # them to the shared rows would credit the challenger with a floor it was
    # never held to.
    assert "on its own test window" in verdict
    assert "shared test rows" not in verdict


# --- the paired join --------------------------------------------------------- #
def test_the_paired_join_never_awards_a_row_whose_label_moved():
    try:
        spec_a, spec_b, *_rest, accounting = _paired("v1", "v2")
    except HTTPException as exc:
        pytest.skip(f"needs both evaluations on disk: {exc.detail}")

    # A row whose true label differs between the two snapshots is scoreable for
    # neither model, so it must be counted as discarded rather than handed to one.
    assert accounting["paired_rows"] + accounting["label_mismatch"] == accounting["overlap_rows"]
    assert accounting["discarded"] == (
        accounting[f"{spec_a.id}_only"] + accounting[f"{spec_b.id}_only"] + accounting["label_mismatch"]
    )
    if accounting["label_mismatch"]:
        assert sum(row["n"] for row in accounting["label_drift"]) == accounting["label_mismatch"]
