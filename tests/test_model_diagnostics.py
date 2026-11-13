"""Unit tests for the behaviour-plate analysis.

The point of these tests is that the page must describe *any* output
distribution, not just the one this checkout happens to produce. So the fixtures
construct known shares and known probability vectors, and the mode taxonomy is
exercised in every branch — including a tilt toward UP and a balanced model that
is actually informative. If a future change hard-wires the page to DOWN, one of
these fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockml.api import model_diagnostics as diag


def _arrays(actual, predicted):
    return np.asarray(actual, dtype=int), np.asarray(predicted, dtype=int)


def _plate_frame(rows: int = 12) -> pd.DataFrame:
    """A frame with the columns ``plate()`` consumes."""
    dates = pd.date_range("2026-01-05", periods=rows, freq="D")
    actual = np.tile([0, 1, 2], rows // 3 + 1)[:rows]
    predicted = np.tile([0, 0, 1], rows // 3 + 1)[:rows]
    return pd.DataFrame(
        {
            "date": dates,
            "actual_label": actual,
            "predicted_label": predicted,
            "p_down": np.tile([0.45, 0.20, 0.30], rows // 3 + 1)[:rows],
            "p_flat": np.tile([0.15, 0.50, 0.20], rows // 3 + 1)[:rows],
            "p_up": np.tile([0.40, 0.30, 0.50], rows // 3 + 1)[:rows],
        }
    )


# --------------------------------------------------------------------------- #
# Panel arithmetic
# --------------------------------------------------------------------------- #
def test_distribution_reports_shares_and_bias_in_percentage_points():
    # 100 rows: the data is 50/30/20, the model produces 80/15/5.
    actual = np.array([0] * 50 + [1] * 30 + [2] * 20)
    predicted = np.array([0] * 80 + [1] * 15 + [2] * 5)
    dist = diag.distribution(actual, predicted)

    assert dist["rows"] == 100
    assert dist["actual"] == {"DOWN": 0.5, "FLAT": 0.3, "UP": 0.2}
    assert dist["predicted"] == {"DOWN": 0.8, "FLAT": 0.15, "UP": 0.05}
    assert dist["bias_pp"] == {"DOWN": 30.0, "FLAT": -15.0, "UP": -15.0}
    assert dist["dominant"] == "DOWN"
    assert dist["dominant_share"] == 0.8
    assert dist["majority_class"] == "DOWN"
    assert dist["majority_share"] == 0.5


def test_distribution_of_an_empty_frame_is_empty_not_zero():
    dist = diag.distribution(np.array([]), np.array([]))
    assert dist["rows"] == 0
    assert dist["actual"] == {}
    assert dist["dominant"] is None


def test_confusion_keeps_raw_counts_and_row_normalised_recall():
    actual, predicted = _arrays([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 1, 2])
    block = diag.confusion(actual, predicted)

    assert block["labels"] == ["DOWN", "FLAT", "UP"]
    assert block["counts"] == [[1, 1, 0], [0, 2, 0], [0, 1, 1]]
    assert block["recall"] == {"DOWN": 0.5, "FLAT": 1.0, "UP": 0.5}
    assert block["accuracy"] == pytest.approx(4 / 6, abs=1e-4)


def test_separation_is_zero_when_the_model_says_the_same_thing_whatever_is_true():
    actual = np.array([0, 0, 1, 1, 2, 2])
    proba = np.tile([0.5, 0.2, 0.3], (6, 1))
    profile = diag.class_profile(actual, proba)

    assert profile["separation"] == 0.0
    assert profile["mean_probability"] == {"DOWN": 0.5, "FLAT": 0.2, "UP": 0.3}


def test_separation_matches_the_pairwise_total_variation_distance():
    # Each class sits on its own corner: every pairwise distance is 0.4.
    actual = np.array([0, 0, 1, 1, 2, 2])
    proba = np.array(
        [[0.6, 0.2, 0.2], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.2, 0.2, 0.6]]
    )
    profile = diag.class_profile(actual, proba)
    assert profile["separation"] == pytest.approx(0.4, abs=1e-4)


def test_conviction_counts_calls_below_a_coin_flip():
    proba = np.array([[0.40, 0.35, 0.25], [0.20, 0.60, 0.20], [0.70, 0.20, 0.10]])
    predicted = proba.argmax(axis=1)
    block = diag.conviction(proba, predicted)

    # max(p) = 0.40 → [0.34,0.45), 0.60 → [0.55,0.65), 0.70 → [0.65,1.0]
    assert [b["n"] for b in block["bins"]] == [0, 1, 0, 1, 1]
    assert block["share_below_half"] == pytest.approx(1 / 3, abs=1e-4)
    assert block["by_class"]["DOWN"]["calls"] == 2
    assert block["by_class"]["FLAT"]["calls"] == 1
    assert block["by_class"]["UP"]["calls"] == 0


def test_threshold_sweep_uses_the_deployed_rule():
    actual = np.array([0, 1, 2, 0])
    proba = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.2, 0.2, 0.6], [0.4, 0.35, 0.25]])
    sweep = diag.threshold_sweep(actual, proba, taus=(0.4, 0.6))

    low, high = sweep
    assert low["tau_flat"] == 0.4
    # apply_flat_threshold() predicts FLAT only where p_flat >= tau, so a higher
    # tau can only shrink the FLAT share, never grow it.
    assert low["flat_share"] == 0.25  # only the row with p_flat = 0.5 clears 0.4
    assert high["flat_share"] == 0.0
    assert low["flat_share"] > high["flat_share"]
    # Coverage of the directional classes is the complement of the FLAT share.
    assert low["coverage_direction"] == pytest.approx(0.75, abs=1e-4)
    assert low["recall"]["FLAT"] == 1.0


# --------------------------------------------------------------------------- #
# Mode taxonomy — every branch, both directions
# --------------------------------------------------------------------------- #
def _dist(bias, *, predicted, actual, accuracy_prior=(0.45, 0.5)):
    accuracy, majority = accuracy_prior
    return {
        "bias_pp": bias,
        "predicted": predicted,
        "actual": actual,
        "dominant": max(predicted, key=lambda name: predicted[name]),
        "majority_share": majority,
    }, accuracy


def test_mode_collapsed_prior_when_the_output_piles_onto_one_class():
    dist, accuracy = _dist(
        {"DOWN": 40.0, "FLAT": -30.0, "UP": -10.0},
        predicted={"DOWN": 0.8, "FLAT": 0.1, "UP": 0.1},
        actual={"DOWN": 0.4, "FLAT": 0.4, "UP": 0.2},
    )
    mode = diag.mode_of(dist, {"separation": 0.02}, {"accuracy": accuracy})

    assert mode["mode"] == "collapsed_prior"
    assert "collapsed onto DOWN" in mode["headline"]
    assert "starved:FLAT" in mode["flags"]
    assert "over_produced:DOWN" in mode["flags"]


def test_mode_tilted_works_toward_up_as_well_as_down():
    """The taxonomy is symmetric: a tilt toward UP is the same function."""
    dist, accuracy = _dist(
        {"UP": 25.0, "DOWN": -20.0, "FLAT": -5.0},
        predicted={"UP": 0.65, "DOWN": 0.10, "FLAT": 0.25},
        actual={"UP": 0.40, "DOWN": 0.30, "FLAT": 0.30},
    )
    mode = diag.mode_of(dist, {"separation": 0.22}, {"accuracy": accuracy})

    assert mode["mode"] == "tilted"
    assert "UP" in mode["headline"]
    assert "starved:DOWN" in mode["flags"]


def test_mode_balanced_but_uninformative():
    dist, accuracy = _dist(
        {"DOWN": 1.0, "FLAT": -1.0, "UP": 0.0},
        predicted={"DOWN": 0.50, "FLAT": 0.30, "UP": 0.20},
        actual={"DOWN": 0.49, "FLAT": 0.31, "UP": 0.20},
        accuracy_prior=(0.49, 0.49),
    )
    mode = diag.mode_of(dist, {"separation": 0.01}, {"accuracy": accuracy})

    assert mode["mode"] == "balanced_undiscriminating"
    assert mode["flags"] == []


def test_mode_balanced_and_discriminating():
    dist, accuracy = _dist(
        {"DOWN": 2.0, "FLAT": -1.0, "UP": -1.0},
        predicted={"DOWN": 0.50, "FLAT": 0.30, "UP": 0.20},
        actual={"DOWN": 0.48, "FLAT": 0.31, "UP": 0.21},
        accuracy_prior=(0.62, 0.48),
    )
    mode = diag.mode_of(dist, {"separation": 0.18}, {"accuracy": accuracy})

    assert mode["mode"] == "balanced_discriminating"
    assert "beats" in mode["headline"]


def test_mode_mild_bias_is_between_a_tilt_and_a_collapse():
    dist, accuracy = _dist(
        {"DOWN": 8.0, "FLAT": -4.0, "UP": -4.0},
        predicted={"DOWN": 0.55, "FLAT": 0.25, "UP": 0.20},
        actual={"DOWN": 0.47, "FLAT": 0.29, "UP": 0.24},
    )
    mode = diag.mode_of(dist, {"separation": 0.02}, {"accuracy": accuracy})

    assert mode["mode"] == "mild_bias"
    assert mode["evidence"]


def test_mode_without_rows_says_so_instead_of_inventing_a_verdict():
    mode = diag.mode_of({"bias_pp": {}}, {}, {})
    assert mode["mode"] == "unknown"


# --------------------------------------------------------------------------- #
# Hypotheses
# --------------------------------------------------------------------------- #
def _plates(**overrides):
    test = {
        "profile": {"separation": 0.02},
        "conviction": {"share_below_half": 0.6},
        "distribution": {"majority_share": 0.5},
        "rule": {
            "fallback_down_share": 0.9,
            "below_tau_share": 0.7,
            "tau_flat": 0.4,
            "accuracy_argmax": 0.47,
        },
    }
    test.update(overrides.pop("test", {}))
    train = {"rule": {"accuracy_argmax": 0.75}}
    train.update(overrides.pop("train", {}))
    return {"test": test, "train": train}


def _verdicts(rows):
    return {row["id"]: row["verdict"] for row in rows}


def test_every_hypothesis_supports_when_the_evidence_is_there():
    sweep = [{"split": "test", "tau_flat": 0.30, "accuracy": 0.52}]
    importance = [{"in_top15": True, "psi": 0.30}] * 5 + [{"in_top15": True, "psi": 0.02}] * 5
    rows = diag.hypotheses(
        plates=_plates(),
        sweep=sweep,
        importance=importance,
        tau_flat=0.40,
        psi_threshold=0.20,
    )
    verdicts = _verdicts(rows)

    assert verdicts["class_overlap"] == diag.VERDICT_SUPPORTED
    assert verdicts["decision_rule_tiebreak"] == diag.VERDICT_SUPPORTED
    assert verdicts["flat_band_misset"] == diag.VERDICT_SUPPORTED  # 0.30 vs configured 0.40
    assert verdicts["overfit"] == diag.VERDICT_SUPPORTED  # 75% train vs 47% test
    assert verdicts["drift"] == diag.VERDICT_SUPPORTED  # 5/10 of the top-15 drifted
    assert verdicts["label_prior"] == diag.VERDICT_INFO


def test_a_rejected_hypothesis_is_reported_as_rejected_not_omitted():
    plates = _plates(
        test={
            "profile": {"separation": 0.30},
            "conviction": {"share_below_half": 0.05},
            "rule": {
                "fallback_down_share": 0.5,
                "below_tau_share": 0.2,
                "tau_flat": 0.40,
                "accuracy_argmax": 0.46,
            },
        },
        train={"rule": {"accuracy_argmax": 0.48}},
    )
    rows = diag.hypotheses(
        plates=plates,
        sweep=[{"split": "test", "tau_flat": 0.42, "accuracy": 0.50}],
        importance=[],
        tau_flat=0.40,
        psi_threshold=0.20,
    )
    verdicts = _verdicts(rows)

    assert verdicts["class_overlap"] == diag.VERDICT_REJECTED
    assert verdicts["decision_rule_tiebreak"] == diag.VERDICT_REJECTED
    assert verdicts["flat_band_misset"] == diag.VERDICT_REJECTED  # 0.02 apart, below the 0.05 bar
    assert verdicts["overfit"] == diag.VERDICT_REJECTED
    # No importance rows means the drift question cannot be answered, which is
    # INCONCLUSIVE rather than a silent REJECTED.
    assert verdicts["drift"] == diag.VERDICT_INCONCLUSIVE
    assert len(rows) == 6


# --------------------------------------------------------------------------- #
# The plate and the waterfall
# --------------------------------------------------------------------------- #
def test_plate_of_an_empty_split_degrades_without_raising():
    plate = diag.plate(pd.DataFrame(), "test")
    assert plate["rows"] == 0
    assert plate["mode"]["mode"] == "unavailable"
    assert plate["confusion"]["counts"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_plate_exposes_the_five_panels_and_the_rule():
    plate = diag.plate(_plate_frame(12), "test")

    assert plate["rows"] == 12
    assert set(plate["distribution"]) >= {"actual", "predicted", "bias_pp", "dominant"}
    assert plate["confusion"]["labels"] == ["DOWN", "FLAT", "UP"]
    assert len(plate["conviction"]["bins"]) == len(diag.CONVICTION_EDGES) - 1
    assert len(plate["margins"]["bins"]) == 5
    assert plate["profile"]["separation"] is not None
    assert plate["mode"]["mode"] in {
        "collapsed_prior",
        "tilted",
        "balanced_undiscriminating",
        "balanced_discriminating",
        "mild_bias",
    }
    assert "accuracy" in plate["rule"]
    assert plate["rolling"], "rolling series is what the drift line plots"


def test_plate_reproduces_the_always_down_story_from_its_numbers():
    """A collapsed fixture must reach the collapsed mode without being told to."""
    rows = 40
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-05", periods=rows, freq="D"),
            "actual_label": np.array([0] * 20 + [1] * 12 + [2] * 8),
            "predicted_label": np.zeros(rows, dtype=int),
            "p_down": np.full(rows, 0.45),
            "p_flat": np.full(rows, 0.15),
            "p_up": np.full(rows, 0.40),
        }
    )
    plate = diag.plate(frame, "test")

    assert plate["distribution"]["predicted"]["DOWN"] == 1.0
    assert plate["profile"]["separation"] == 0.0
    assert plate["mode"]["mode"] == "collapsed_prior"
    # Every row falls below tau, and the p_down vs p_up comparison is one-sided.
    assert plate["rule"]["fallback_down_share"] == 1.0


def test_waterfall_reconstructs_the_class_score():
    base = 0.20
    contributions = [
        {"feature": "news_sentiment_3d", "contribution": 0.50},
        {"feature": "volume_zscore_20", "contribution": -1.20},
        {"feature": "rsi_14", "contribution": 0.30},
    ]
    flow = diag.waterfall(contributions, base)

    assert flow["base_value"] == pytest.approx(base)
    # Ordered by |contribution| so the biggest lever is drawn first.
    assert [step["feature"] for step in flow["steps"]] == ["volume_zscore_20", "news_sentiment_3d", "rsi_14"]
    assert flow["final"] == pytest.approx(base + sum(row["contribution"] for row in contributions), abs=1e-6)
    assert flow["steps"][-1]["cumulative"] == flow["final"]


def test_driver_rows_only_keeps_features_that_push_toward_a_class():
    shap = pd.DataFrame(
        {
            "class": ["DOWN", "DOWN", "UP"],
            "feature": ["a", "b", "c"],
            "mean_signed": [0.4, -0.9, 0.2],
            "mean_abs": [0.5, 0.9, 0.2],
        }
    )
    rows = diag.driver_rows(shap)
    assert [(row["class"], row["feature"]) for row in rows] == [("DOWN", "a"), ("UP", "c")]

    only_up = diag.driver_rows(shap, dominant="up")
    assert [row["feature"] for row in only_up] == ["c"]
