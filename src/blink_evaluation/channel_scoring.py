"""Per-channel blink evaluation and lane scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import mne

from blink_evaluation.api import evaluate_annotations
from blink_evaluation.epoch_utils import enrich_absolute_times
from blink_evaluation.io import annotations_to_events, dataframe_to_annotations
from blink_evaluation.matching import match_events
from blink_evaluation.metrics import compute_event_metrics, compute_sample_metrics
from blink_evaluation.types import AnnotationEvent, EvaluationResult, Match


def _compute_peak_time_abs(
    onset_abs: float,
    duration: float,
    time_series: np.ndarray,
    sfreq: float,
) -> float | None:
    """Return the absolute time of the signal peak within an annotation window.

    Finds the sample with maximum absolute amplitude in the continuous
    time series that corresponds to [onset_abs, onset_abs + duration].
    """
    start_s = max(0, int(onset_abs * sfreq))
    end_s = min(len(time_series), int((onset_abs + duration) * sfreq))
    if start_s >= end_s:
        return None
    window = time_series[start_s:end_s]
    peak_offset = int(np.argmax(np.abs(window)))
    return (start_s + peak_offset) / sfreq


def _augment_peak_times(
    events: list[AnnotationEvent],
    time_series: np.ndarray,
    sfreq: float,
) -> None:
    """Set ``peak_time`` in-place for each event using the continuous time series."""
    for ev in events:
        ev.peak_time = _compute_peak_time_abs(ev.onset, ev.duration, time_series, sfreq)


def _combine_scored_events(
    tp_events: list[Match],
    fp_events: list[AnnotationEvent],
    fn_events: list[AnnotationEvent],
    pred_events: list[AnnotationEvent],
) -> list[AnnotationEvent]:
    """Combine already-enriched tp/fp/fn events into a flat list sorted by onset.

    ``match_events`` fills all status and timing fields in-place, so this
    function only assembles the final sorted list.  TP entries are the
    prediction-side ``AnnotationEvent`` objects looked up from ``pred_events``.
    """
    pred_by_idx = {ev.index: ev for ev in pred_events}
    all_events: list[AnnotationEvent] = [pred_by_idx[m.pred_index] for m in tp_events]
    all_events.extend(fp_events)
    all_events.extend(fn_events)
    all_events.sort(key=lambda e: e.onset)
    return all_events


@dataclass
class ChannelEvaluationResult:
    """Summary of per-channel blink evaluation."""

    lane_summary: pd.DataFrame
    best_channel: str | None
    best_eval_result: EvaluationResult | None
    best_channel_result: dict | None
    """Full channel-result dict for the best channel (includes ``mapped_candidates``,
    ``df_positions``, and any strategy-specific keys)."""
    best_predicted: pd.DataFrame | None
    """Enriched predictions DataFrame for the best channel
    (has ``absolute_onset_s`` / ``absolute_offset_s`` columns)."""
    best_scored_events: list[AnnotationEvent] | None = None
    """All scored events for the best channel, sorted by onset.

    Each ``AnnotationEvent`` has ``status`` ('tp'/'fp'/'fn') and both-sided
    timing fields (``onset_pred``, ``onset_gt``, ``duration_pred``,
    ``duration_gt``) for plotting.
    """


def evaluate_channels(
    channel_results: list[dict],
    gt_annotations: mne.Annotations,
    epoch_duration: float,
    iou_threshold: float = 0.1,
    peak_required: bool = False,
    peak_tolerance: float | None = None,
    raw: mne.io.BaseRaw | None = None,
) -> ChannelEvaluationResult:
    """Score each channel against ground-truth annotations and return a ranked summary.

    Parameters
    ----------
    channel_results:
        List of dicts, each with keys ``channel``, ``df_positions``, and
        ``mapped_candidates`` (epoch-relative DataFrames).
    gt_annotations:
        Ground-truth ``mne.Annotations`` expressed in absolute time.
    epoch_duration:
        Duration of each epoch in seconds; used to convert epoch-relative
        timings to absolute time via ``enrich_absolute_times``.
    iou_threshold:
        IoU threshold for a prediction to count as a true positive.
    peak_required:
        When ``True``, the matching also requires that the signal peak within
        each annotation window aligns between GT and prediction (within
        ``peak_tolerance`` seconds).  Peaks are computed from the continuous
        time series extracted from ``raw``; falls back to IoU-only matching
        when the channel is not present in ``raw`` or ``raw`` is ``None``.
    peak_tolerance:
        Maximum allowed peak-time difference in seconds when ``peak_required``
        is ``True``.  ``None`` means peaks must coincide exactly.
    raw:
        Full continuous recording as an ``mne.Raw`` object.  Required for
        ``peak_required=True``; the channel signal is extracted per channel
        result using the channel name.

    Returns
    -------
    ChannelEvaluationResult
        Contains ``lane_summary`` (DataFrame sorted by F1 descending),
        ``best_channel``, ``best_eval_result``, and ``best_scored_events``
        (enriched tp/fp/fn events for the best channel).
    """
    lane_rows: list[dict] = []
    best_channel: str | None = None
    best_eval_result: EvaluationResult | None = None
    best_channel_result: dict | None = None
    best_predicted: pd.DataFrame | None = None
    best_scored_events: list[AnnotationEvent] | None = None

    for cr in channel_results:
        predicted_df = enrich_absolute_times(cr["mapped_candidates"], epoch_duration)
        pred_annotations = dataframe_to_annotations(predicted_df)

        time_series: np.ndarray | None = None
        sfreq: float | None = None
        if peak_required and raw is not None:
            try:
                time_series = raw.get_data(picks=[cr["channel"]])[0]
                sfreq = raw.info["sfreq"]
            except Exception:
                time_series = None

        if peak_required and time_series is not None:
            gt_events = annotations_to_events(gt_annotations, "blink")
            pred_events = annotations_to_events(pred_annotations, "blink")
            _augment_peak_times(gt_events, time_series, sfreq)
            _augment_peak_times(pred_events, time_series, sfreq)
            recording_duration = float(raw.times[-1]) if raw is not None else None
            sample_metrics, sample_confusion = compute_sample_metrics(
                gt_events, pred_events, sample_rate=sfreq, recording_duration=recording_duration
            )
            tp_events, fp_events, fn_events = match_events(
                gt_events,
                pred_events,
                iou_threshold=iou_threshold,
                peak_required=True,
                peak_tolerance=peak_tolerance,
            )
            combined_events = _combine_scored_events(tp_events, fp_events, fn_events, pred_events)
            event_metrics = compute_event_metrics(tp_events, len(gt_events), len(pred_events))
            result = EvaluationResult(
                event_metrics=event_metrics,
                sample_metrics=sample_metrics,
                matches=tp_events,
                true_positives=tp_events,
                false_positives=fp_events,
                false_negatives=fn_events,
                sample_confusion=sample_confusion,
            )
        else:
            gt_events = annotations_to_events(gt_annotations, "blink")
            pred_events = annotations_to_events(pred_annotations, "blink")
            sample_metrics, sample_confusion = compute_sample_metrics(
                gt_events, pred_events, sample_rate=100.0, recording_duration=None
            )
            tp_events, fp_events, fn_events = match_events(
                gt_events,
                pred_events,
                iou_threshold=iou_threshold,
                peak_required=False,
                peak_tolerance=None,
            )
            combined_events = _combine_scored_events(tp_events, fp_events, fn_events, pred_events)
            event_metrics = compute_event_metrics(tp_events, len(gt_events), len(pred_events))
            result = EvaluationResult(
                event_metrics=event_metrics,
                sample_metrics=sample_metrics,
                matches=tp_events,
                true_positives=tp_events,
                false_positives=fp_events,
                false_negatives=fn_events,
                sample_confusion=sample_confusion,
            )

        em = result.event_metrics
        lane_rows.append(
            {
                "channel": cr["channel"],
                "raw_candidate_count": int(len(cr["df_positions"])),
                "mapped_candidate_count": int(len(cr["mapped_candidates"])),
                "tp": em.tp,
                "fp": em.fp,
                "fn": em.fn,
                "precision": em.precision,
                "recall": em.recall,
                "f1": em.f1,
            }
        )

        if best_eval_result is None or (
            em.f1,
            em.tp,
            -em.fp,
            cr["channel"],
        ) > (
            best_eval_result.event_metrics.f1,
            best_eval_result.event_metrics.tp,
            -best_eval_result.event_metrics.fp,
            best_channel,
        ):
            best_channel = cr["channel"]
            best_eval_result = result
            best_channel_result = cr
            best_predicted = predicted_df
            combined_events_all = combined_events

    lane_summary = (
        pd.DataFrame(lane_rows)
        .sort_values(["f1", "tp", "fp", "channel"], ascending=[False, False, True, True])
        .reset_index(drop=True)
    )

    return ChannelEvaluationResult(
        lane_summary=lane_summary,
        best_channel=best_channel,
        best_eval_result=best_eval_result,
        best_channel_result=best_channel_result,
        best_predicted=best_predicted,
        best_scored_events=combined_events_all,
    )


__all__ = ["ChannelEvaluationResult", "evaluate_channels"]
