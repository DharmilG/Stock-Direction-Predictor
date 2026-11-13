"""Unit tests for the v2 decision policy.

The policy is where a number can lie quietly: an implicit denominator, a branch
evaluated twice, or an infeasible policy that crashes instead of being recorded.
Each test below pins one of those.
"""

from __future__ import annotations

import numpy as np

from stockml.v2 import policy as P

FLAT = P.FLAT
UP = P.UP
DOWN = P.DOWN
ABSTAIN = P.ABSTAIN


def test_abstains_on_low_confidence():
    # Max probability 0.45 is below the 0.60 floor: no class is called.
    label, branch = P.decide(0.30, 0.10, 0.45, P.Policy(tau_flat=1.01, delta=0.0, min_conf=0.60))
    assert label == ABSTAIN
    assert branch == "low_confidence"


def test_flat_rule_beats_margin_when_both_could_fire():
    # p_flat clears tau_flat and the UP/DOWN gap is also inside delta.
    # Precedence is part of the contract: this must be attributed to flat_rule.
    label, branch = P.decide(0.20, 0.45, 0.35, P.Policy(tau_flat=0.40, delta=0.30, min_conf=0.0))
    assert label == FLAT
    assert branch == "flat_rule"


def test_flat_rule_respects_its_threshold():
    below, below_branch = P.decide(0.32, 0.39, 0.29, P.Policy(tau_flat=0.40))
    at, at_branch = P.decide(0.32, 0.40, 0.28, P.Policy(tau_flat=0.40))
    assert (below, below_branch) == (DOWN, "direction")
    assert (at, at_branch) == (FLAT, "flat_rule")


def test_margin_abstains_inside_delta():
    # UP leads by 0.10, which is thinner than the 0.15 the policy requires.
    label, branch = P.decide(0.40, 0.10, 0.50, P.Policy(delta=0.15))
    assert label == ABSTAIN
    assert branch == "margin"
    # The same probabilities with no margin requirement are called.
    label, branch = P.decide(0.40, 0.10, 0.50, P.Policy(delta=0.05))
    assert (label, branch) == (UP, "direction")


def test_coverage_is_direction_calls_over_all_rows():
    y = np.array([UP, DOWN, DOWN, DOWN])
    labels = np.array([UP, ABSTAIN, DOWN, DOWN])
    scored = P.score(y, labels)
    assert scored["rows"] == 4
    assert scored["direction_calls"] == 3
    assert scored["coverage"] == 0.75


def test_accuracy_all_penalised_counts_every_abstention_wrong():
    y = np.array([UP, UP, DOWN, DOWN])
    labels = np.array([UP, ABSTAIN, DOWN, DOWN])
    scored = P.score(y, labels)
    # Three right out of four rows: the abstention is not a free skip.
    assert scored["accuracy_all_penalised"] == 0.75
    assert scored["correct_all"] == 3
    silent = P.score(y, np.full(4, ABSTAIN))
    assert silent["accuracy_all_penalised"] == 0.0
    assert silent["coverage"] == 0.0


def test_accuracy_called_excludes_silence_from_its_denominator():
    y = np.array([UP, UP, DOWN, DOWN])
    labels = np.array([UP, ABSTAIN, DOWN, DOWN])
    scored = P.score(y, labels)
    # Three calls, all correct: this is the number that looks best and is not
    # comparable to accuracy_all_penalised.
    assert scored["accuracy_called"] == 1.0
    assert scored["abstentions"] == 1


def test_recall_floor_marks_a_policy_infeasible_instead_of_raising():
    # Probabilities that never favour UP, so no threshold set can recall it.
    n = 200
    y = np.array([UP] * (n // 2) + [DOWN] * (n // 2))
    proba = np.tile([0.60, 0.35, 0.05], (n, 1))
    result = P.search(y, proba)
    assert result["grid_size"] == 245
    assert result["selected"] is None
    assert result["feasible_count"] == 0
    assert all(entry["feasible"] is False for entry in result["entries"])
    assert "UP recall" in result["entries"][0]["infeasible_reason"]


def test_selection_score_prefers_balance_over_raw_coverage():
    balanced = {"balanced_called": 0.55, "mcc_called": 0.05, "accuracy_all_penalised": 0.48}
    degenerate = {"balanced_called": 0.50, "mcc_called": 0.00, "accuracy_all_penalised": 0.52}
    assert P.selection_score(balanced) > P.selection_score(degenerate)
    # A policy with no direction calls has no balanced accuracy and cannot rank.
    assert P.selection_score({"balanced_called": None, "mcc_called": 0.0, "accuracy_all_penalised": 0.9}) == float("-inf")


def test_direction_only_reports_both_denominators():
    y = np.array([UP, UP, FLAT, DOWN])
    labels = np.array([UP, FLAT, FLAT, DOWN])
    result = P.direction_only(y, labels)
    # all rows: three correct of four, because the second UP row was called FLAT
    assert result["rows_all"] == 4
    assert result["accuracy_all_rows"] == 0.75
    # movers only: two correct of three; the FLAT truth row is not in this set
    assert result["rows_movers"] == 3
    assert abs(result["accuracy_movers"] - 2 / 3) < 1e-12
    assert result["accuracy_all_rows"] != result["accuracy_movers"]


def test_vectorised_and_scalar_decisions_agree():
    proba = np.array(
        [
            [0.30, 0.10, 0.60],
            [0.55, 0.40, 0.05],
            [0.33, 0.34, 0.33],
            [0.20, 0.70, 0.10],
            [0.45, 0.10, 0.45],
        ]
    )
    for policy in (P.Policy(), P.Policy(tau_flat=0.40, delta=0.10, min_conf=0.50), P.Policy(delta=0.30)):
        labels, branch = P.apply(policy, proba)
        for index, row in enumerate(proba):
            scalar = P.decide(row[DOWN], row[FLAT], row[UP], policy)
            assert scalar == (int(labels[index]), P.BRANCHES[int(branch[index])])
