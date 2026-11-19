"""Unit tests for the Q0 analytics module.

The ledger, the traces and the GPU logs are the only sources; every number these
tests pin down is recomputed from a synthetic copy of them. Two properties matter
most: the counts agree with a hand-computed answer, and a field the run never
wrote is reported as absent instead of being plotted as zero.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from stockml.api import q0_analytics as qa


def _paths(tmp_path):
    """The directory handles the module reaches for, nothing else."""
    base = tmp_path / "track_q" / "phase_0"
    paths = SimpleNamespace(
        root=tmp_path,
        logs=base / "logs",
        results=tmp_path / "results",
        state=base / "state",
        checkpoints=tmp_path / "checkpoints",
        artifacts=tmp_path / "artifacts",
    )
    for directory in (paths.logs, paths.results, paths.state, paths.checkpoints,
                      base / "traces", base / "verdict", base / "data"):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def _ledger(trace_path: str | None = None) -> pd.DataFrame:
    """Four rows: two hits, one miss, one harness failure."""
    return pd.DataFrame(
        {
            "case_id": ["AAA|2026-01-06|a", "BBB|2026-01-05|b", "CCC|2026-01-07|c", "DDD|2026-01-08|d"],
            "symbol": ["AAA.NS", "BBB.NS", "CCC.NS", "AAA.NS"],
            "decision_date": ["2026-01-06", "2026-01-05", "2026-01-07", "2026-01-08"],
            "qwen_vote": ["FLAT", "UP", "DOWN", None],
            "historical_label": ["FLAT", "DOWN", "DOWN", "FLAT"],
            "vote_matches_label": [True, False, True, None],
            "finbert_vote": ["FLAT", "UP", "DOWN", "FLAT"],
            "finbert_score": [0.2, 0.5, -0.4, 0.1],
            "qwen_conf": [0.8, 0.6, 0.9, None],
            "harness_fail": [False, False, False, True],
            "kept_for_training": [False, False, False, False],
            "no_news": [False, True, False, False],
            "thermal_state": ["green", "green", "amber", "green"],
            "trace_path": [trace_path, None, None, None],
        }
    )


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def test_analytics_funnel_separates_rated_from_judged(tmp_path):
    payload = qa.analytics(_paths(tmp_path), _ledger())

    funnel = {step["step"]: step["n"] for step in payload["funnel"]}
    # A harness failure is attempted but not judged; nothing was kept.
    assert funnel == {"attempted": 4, "rated": 4, "judged": 3, "kept": 0}
    assert payload["judged_rate"] == pytest.approx(0.75, abs=1e-4)
    assert payload["harness_fail_rate"] == pytest.approx(0.25, abs=1e-4)
    assert payload["funnel"][2]["retention"] is not None


def test_analytics_vote_by_label_matrix_matches_the_rows(tmp_path):
    payload = qa.analytics(_paths(tmp_path), _ledger())
    block = payload["votes_by_label"]

    assert block["rows"] == ["DOWN", "FLAT", "UP"]
    assert block["cols"] == ["DOWN", "FLAT", "UP"]
    # Rows are the realised label, columns the call, over judged rows only:
    # two DOWN labels (called UP and DOWN) and one FLAT label (called FLAT).
    assert block["counts"] == [[1, 0, 1], [0, 1, 0], [0, 0, 0]]
    assert block["row_totals"] == {"DOWN": 2, "FLAT": 1, "UP": 0}
    assert sum(sum(row) for row in block["counts"]) == 3


def test_analytics_calibration_and_disagreement(tmp_path):
    payload = qa.analytics(_paths(tmp_path), _ledger())

    calibration = payload["calibration"]
    assert calibration["n"] == 3
    # conf 0.6 missed, 0.8 and 0.9 hit → ECE = (0.6 + 0.2 + 0.1) / 3
    assert calibration["ece"] == pytest.approx(0.3, abs=1e-3)
    assert [bin["lo"] for bin in calibration["bins"]] == [0.6, 0.8, 0.9]

    disagreement = payload["disagreement"]
    assert disagreement["qwen_matches"] == pytest.approx(2 / 3, abs=1e-4)
    assert disagreement["both_match"] + disagreement["qwen_only"] + disagreement["finbert_only"] + disagreement["neither"] == 3
    assert disagreement["neither"] == 1  # the UP call on a DOWN label


def test_analytics_confidence_histogram_covers_every_judged_row(tmp_path):
    payload = qa.analytics(_paths(tmp_path), _ledger())
    histogram = payload["confidence_hist"]

    assert len(histogram) == 3  # 0.6, 0.8, 0.9 — the NaN row is absent, not zero-filled
    assert sum(bin["n"] for bin in histogram) == 3
    assert all(0.0 <= bin["share"] <= 1.0 for bin in histogram)


def test_latency_is_absent_rather_than_estimated_without_traces(tmp_path):
    payload = qa.analytics(_paths(tmp_path), _ledger())
    assert payload["latency"]["available"] is False
    assert payload["latency"]["overall"]["n"] == 0
    assert payload["latency"]["overall"]["p50"] is None

    # With a trace that records seconds, the duration becomes measurable.
    paths = _paths(tmp_path)
    trace = paths.root / "track_q" / "phase_0" / "traces" / "a.json"
    trace.write_text(json.dumps({"seconds": 12.5, "vote": "FLAT"}), encoding="utf-8")
    measured = qa.analytics(paths, _ledger(str(trace)))
    assert measured["latency"]["available"] is True
    assert measured["latency"]["overall"]["n"] == 1
    assert measured["latency"]["overall"]["p50"] == 12.5


def test_cohorts_and_per_symbol_group_the_same_judged_rows(tmp_path):
    payload = qa.analytics(_paths(tmp_path), _ledger())

    cohorts = {row["cohort"]: row for row in payload["cohorts"]}
    assert cohorts["news"]["n"] + cohorts["no_news"]["n"] == 3
    assert cohorts["no_news"]["n"] == 1

    per_symbol = {row["symbol"]: row for row in payload["per_symbol"]}
    assert per_symbol["AAA.NS"]["n"] == 1
    assert sum(row["n"] for row in payload["per_symbol"]) == 3


def test_analytics_of_an_empty_ledger_reports_unavailable():
    assert qa.analytics(SimpleNamespace(), None) == {"available": False}
    assert qa.analytics(SimpleNamespace(), pd.DataFrame()) == {"available": False}


# --------------------------------------------------------------------------- #
# Traces and cases
# --------------------------------------------------------------------------- #
def test_case_rows_filter_and_order_like_the_route(tmp_path):
    frame = _ledger()

    everything = qa.case_rows(frame)
    assert everything["total"] == 4
    assert [row["decision_date"] for row in everything["rows"]] == [
        "2026-01-08",
        "2026-01-07",
        "2026-01-06",
        "2026-01-05",
    ]
    assert everything["rows"][0]["case_id"] == "DDD|2026-01-08|d"

    just_up = qa.case_rows(frame, vote="UP")
    assert just_up["total"] == 1
    assert just_up["rows"][0]["symbol"] == "BBB.NS"

    hits = qa.case_rows(frame, outcome="match")
    assert hits["total"] == 2
    misses = qa.case_rows(frame, outcome="mismatch")
    assert misses["total"] == 1

    by_symbol = qa.case_rows(frame, symbol="aaa")
    assert by_symbol["total"] == 2

    page = qa.case_rows(frame, limit=2, offset=1)
    assert page["total"] == 4
    assert len(page["rows"]) == 2
    assert page["rows"][0]["case_id"] == "CCC|2026-01-07|c"


def test_case_rows_of_a_missing_ledger_is_an_empty_page():
    payload = qa.case_rows(None)
    assert payload == {"total": 0, "offset": 0, "limit": 50, "rows": []}


@pytest.mark.parametrize("case_id", ["../../etc/passwd", "a/b", "a\\b", "C:\\Windows", "", "x" * 201])
def test_case_ids_that_could_escape_the_trace_directory_are_refused(case_id):
    assert qa.safe_case_id(case_id) is False


def test_case_id_in_the_normal_form_is_accepted():
    assert qa.safe_case_id("TMPV.NS|2023-04-11|179f49c9") is True


def test_case_detail_returns_the_ledger_row_and_its_trace(tmp_path):
    paths = _paths(tmp_path)
    trace = paths.root / "track_q" / "phase_0" / "traces" / "a.json"
    trace.write_text(
        json.dumps({"seconds": 12.5, "vote": "FLAT", "thinking": "because", "content": "FLAT"}),
        encoding="utf-8",
    )
    payload = qa.case_detail(paths, _ledger(str(trace)), "AAA|2026-01-06|a")

    assert payload["trace_available"] is True
    assert payload["ledger"]["symbol"] == "AAA.NS"
    assert payload["trace"]["seconds"] == 12.5
    assert payload["trace_chars"] == len("becauseFLAT")


def test_case_detail_does_not_read_outside_the_trace_directory(tmp_path):
    paths = _paths(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"vote": "UP"}), encoding="utf-8")
    frame = _ledger(str(outside))

    # The row exists but points outside track_q/phase_0/traces: refuse rather than serve it.
    assert qa.case_detail(paths, frame, "AAA|2026-01-06|a") is None
    # And a traversal id never reaches the filesystem at all.
    assert qa.case_detail(paths, frame, "../../etc/passwd") is None
    # An id nobody knows is a miss, not an exception.
    assert qa.case_detail(paths, frame, "ZZZ|2026-01-09|z") is None


def test_trace_summary_is_cached_until_the_file_changes(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"seconds": 3.0, "vote": "UP"}), encoding="utf-8")
    first = qa.trace_summary(path)
    assert first["seconds"] == 3.0

    path.write_text(json.dumps({"seconds": 9.0, "vote": "UP", "thinking": "t"}), encoding="utf-8")
    second = qa.trace_summary(path)
    assert second["seconds"] == 9.0
    assert second["thinking_chars"] == 1


def test_trace_summary_of_a_missing_file_is_none(tmp_path):
    assert qa.trace_summary(tmp_path / "nope.json") is None


# --------------------------------------------------------------------------- #
# Timeline: what was measured, and nothing else
# --------------------------------------------------------------------------- #
def test_resource_series_with_only_the_legacy_temp_file_reports_temp_only(tmp_path):
    paths = _paths(tmp_path)
    pd.DataFrame(
        {"timestamp": ["2026-01-05 10:00:00", "2026-01-05 10:00:05"], "temp_c": [55.0, 56.0]}
    ).to_csv(paths.logs / "gpu_5s.csv", index=False)

    payload = qa.timeline(paths, None, window="2h", step=5)

    assert payload["resource"]["fields"] == ["temp_c"]
    assert payload["resource"]["supplementary"] is False
    assert len(payload["resource"]["series"]) == 2
    # The absence is stated, not filled with a synthetic utilisation line.
    assert any("q0_resource_5s.csv is missing" in warning for warning in payload["warnings"])


def test_resource_series_prefers_the_superset_file(tmp_path):
    paths = _paths(tmp_path)
    pd.DataFrame(
        {
            "timestamp": ["2026-01-05 10:00:00"],
            "temp_c": [61.0],
            "util_pct": [91.0],
            "mem_used_mb": [9000],
            "mem_total_mb": [12288],
            "power_w": [170],
            "fan_pct": [""],  # the card answered but does not report a fan
            "clock_sm_mhz": [1950],
            "clock_max_sm_mhz": [2100],
        }
    ).to_csv(paths.logs / "q0_resource_5s.csv", index=False)

    payload = qa.timeline(paths, None, window="2h", step=5)
    resource = payload["resource"]
    row = resource["series"][0]

    assert resource["fields"] == ["temp_c", "util_pct", "mem_used_mb", "mem_total_mb", "power_w", "clock_sm_mhz", "clock_max_sm_mhz"]
    assert resource["supplementary"] is True
    assert resource["source"] == "logs/q0_resource_5s.csv"
    assert row["temp_c"] == 61.0
    # A blank cell stays a gap: the field is reported as absent, never as 0.
    assert "fan_pct" not in resource["fields"]
    assert "fan_pct" not in row


def test_long_series_is_strided_to_stay_under_the_payload_cap(tmp_path):
    paths = _paths(tmp_path)
    rows = qa.MAX_SERIES_POINTS * 3
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05", periods=rows, freq="5s"),
            "temp_c": 55.0,
        }
    ).to_csv(paths.logs / "q0_resource_5s.csv", index=False)

    payload = qa.timeline(paths, None, window="all", step=5)

    assert len(payload["resource"]["series"]) <= qa.MAX_SERIES_POINTS
    assert payload["resource"]["stride_seconds"] > qa.SAMPLE_SECONDS


def test_window_bounds_are_parsed_or_ignored():
    assert qa._window_seconds("30m") == 1800.0
    assert qa._window_seconds("2h") == 7200.0
    assert qa._window_seconds(" 45 m ") == 2700.0
    assert qa._window_seconds("all") is None
    assert qa._window_seconds("7d") is None


def test_thin_keeps_the_last_point_and_respects_the_limit():
    rows = [{"seq": index} for index in range(1000)]
    thinned = qa._thin(rows, 100)

    assert len(thinned) <= 101
    assert thinned[-1] is rows[-1]
    assert qa._thin(rows[:10], 100) == rows[:10]


def test_progress_rows_are_cumulative_and_never_invent_durations():
    frame = _ledger()
    rows = qa._progress_rows(frame)

    assert [row["seq"] for row in rows] == [1, 2, 3, 4]
    assert rows[-1]["rated"] == 4
    assert rows[-1]["judged"] == 3
    assert rows[-1]["kept"] == 0
    # No trace was supplied, so every duration is a gap.
    assert all(row["case_seconds_p50"] is None for row in rows)
    assert all(row["agreement_lo"] <= row["qwen_matches"] <= row["agreement_hi"] for row in rows)


def test_progress_rows_carry_durations_when_traces_provide_them():
    seconds = pd.Series([10.0, 20.0, 30.0, None], index=_ledger().index)
    rows = qa._progress_rows(_ledger(), seconds)
    assert rows[-1]["case_seconds_p50"] is not None
    assert rows[-1]["case_seconds_p95"] is not None
