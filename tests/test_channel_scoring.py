"""Unit tests for blink_evaluation.channel_scoring."""

from __future__ import annotations

import mne
import pandas as pd
import pytest

from blink_evaluation.channel_scoring import ChannelEvaluationResult, evaluate_channels


def _make_annotations(onsets, durations, description="blink") -> mne.Annotations:
    return mne.Annotations(
        onset=list(onsets),
        duration=list(durations),
        description=[description] * len(onsets),
    )


def _make_channel_result(
    channel: str,
    candidates: list[tuple[int, float, float]],
) -> dict:
    """Build a channel_result dict.

    candidates: list of (epoch_index, blink_onset, blink_duration) tuples.
    df_positions mirrors mapped_candidates for count purposes.
    """
    mapped = pd.DataFrame(
        candidates, columns=["epoch_index", "blink_onset", "blink_duration"]
    )
    return {
        "channel": channel,
        "df_positions": mapped,
        "mapped_candidates": mapped,
    }


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_evaluate_channels_perfect_match():
    # GT: one blink at t=5.0 for 0.3s
    gt = _make_annotations([5.0], [0.3])
    # Prediction: one channel with exact match (epoch 0, onset 5.0, duration 0.3)
    channel_results = [_make_channel_result("ch1", [(0, 5.0, 0.3)])]

    result = evaluate_channels(channel_results, gt, epoch_duration=60.0, iou_threshold=0.1)

    assert result.best_channel == "ch1"
    em = result.best_eval_result.event_metrics
    assert em.tp == 1
    assert em.fp == 0
    assert em.fn == 0
    assert em.f1 == pytest.approx(1.0)


def test_evaluate_channels_no_predictions():
    gt = _make_annotations([5.0], [0.3])
    channel_results = [_make_channel_result("ch1", [])]

    result = evaluate_channels(channel_results, gt, epoch_duration=60.0, iou_threshold=0.1)

    em = result.best_eval_result.event_metrics
    assert em.tp == 0
    assert em.fn == 1
    assert em.recall == pytest.approx(0.0)


def test_evaluate_channels_best_channel_selected_by_f1():
    gt = _make_annotations([5.0, 65.0], [0.3, 0.3])

    # ch1: matches only first blink (recall=0.5, precision=1.0)
    ch1 = _make_channel_result("ch1", [(0, 5.0, 0.3)])
    # ch2: matches both blinks (recall=1.0, precision=1.0)
    ch2 = _make_channel_result("ch2", [(0, 5.0, 0.3), (1, 5.0, 0.3)])

    result = evaluate_channels([ch1, ch2], gt, epoch_duration=60.0, iou_threshold=0.1)

    assert result.best_channel == "ch2"
    assert result.best_eval_result.event_metrics.f1 == pytest.approx(1.0)


def test_evaluate_channels_lane_summary_sorted_by_f1_descending():
    gt = _make_annotations([5.0, 65.0], [0.3, 0.3])

    ch_none = _make_channel_result("ch_none", [])
    ch_one = _make_channel_result("ch_one", [(0, 5.0, 0.3)])
    ch_both = _make_channel_result("ch_both", [(0, 5.0, 0.3), (1, 5.0, 0.3)])

    result = evaluate_channels(
        [ch_none, ch_one, ch_both], gt, epoch_duration=60.0, iou_threshold=0.1
    )

    f1_values = result.lane_summary["f1"].tolist()
    assert f1_values == sorted(f1_values, reverse=True)


def test_evaluate_channels_returns_dataclass():
    gt = _make_annotations([5.0], [0.3])
    result = evaluate_channels(
        [_make_channel_result("ch1", [(0, 5.0, 0.3)])],
        gt,
        epoch_duration=60.0,
    )
    assert isinstance(result, ChannelEvaluationResult)
    assert isinstance(result.lane_summary, pd.DataFrame)


def test_evaluate_channels_lane_summary_columns():
    gt = _make_annotations([5.0], [0.3])
    result = evaluate_channels(
        [_make_channel_result("ch1", [(0, 5.0, 0.3)])],
        gt,
        epoch_duration=60.0,
    )
    expected_cols = {
        "channel",
        "raw_candidate_count",
        "mapped_candidate_count",
        "tp", "fp", "fn",
        "precision", "recall", "f1",
    }
    assert expected_cols == set(result.lane_summary.columns)


def test_evaluate_channels_tiebreak_prefers_lower_fp():
    # Two channels with equal F1 and TP — the one with fewer FP wins.
    gt = _make_annotations([5.0], [0.3])

    # ch_clean: TP=1, FP=0
    ch_clean = _make_channel_result("ch_clean", [(0, 5.0, 0.3)])
    # ch_noisy: TP=1, FP=1 (extra spurious prediction far from GT)
    ch_noisy = _make_channel_result("ch_noisy", [(0, 5.0, 0.3), (0, 50.0, 0.3)])

    result = evaluate_channels([ch_clean, ch_noisy], gt, epoch_duration=60.0, iou_threshold=0.1)

    assert result.best_channel == "ch_clean"
