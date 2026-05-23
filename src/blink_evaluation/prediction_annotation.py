"""Build tp/fp/fn-labeled annotations and masterlist CSVs from matched events.

After calling ``scored = evaluate_channels(...)``, extract tp/fp/fn events
from ``scored.best_eval_result`` and pass them to:

- :func:`build_events_masterlist_df` — flat DataFrame with one row per event
- :func:`build_annotations_from_events` — ``mne.Annotations`` for visual replay
- :func:`save_scored_annotations_csv` — write annotations to CSV
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from blink_evaluation.channel_scoring import ChannelEvaluationResult
from blink_evaluation.io import annotations_to_events, dataframe_to_annotations
from blink_evaluation.types import AnnotationEvent, EvaluationResult, Match


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        prev_s, prev_e = merged[-1]
        if start <= prev_e + 1e-9:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    return merged


def _complement_intervals(
    covered: list[tuple[float, float]],
    total_duration: float,
) -> list[tuple[float, float]]:
    """Return intervals within [0, total_duration] not covered by *covered*."""
    merged = _merge_intervals(covered)
    result = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor + 1e-9:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_duration - 1e-9:
        result.append((cursor, total_duration))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_scored_annotations(
    result: EvaluationResult,
    gt_annotations: mne.Annotations,
    pred_annotations: mne.Annotations,
    *,
    target_label: str = "blink",
    recording_duration: float | None = None,
    include_tn: bool = True,
) -> mne.Annotations:
    """Build MNE annotations labeled ``tp``/``fp``/``fn``/``tn`` from an evaluation result.

    TP timings use the predicted event window.  FP and FN use their respective
    event timings (predicted and ground-truth respectively).  TN annotations
    cover every interval not occupied by a tp/fp/fn event.

    Parameters
    ----------
    result:
        Output of :func:`~blink_evaluation.api.evaluate_annotations`.
    gt_annotations:
        Ground-truth annotations in absolute time (used to derive
        *recording_duration* when not supplied explicitly).
    pred_annotations:
        Predicted annotations in absolute time — the same object that was
        passed to ``evaluate_annotations``.
    target_label:
        Annotation description to filter on.
    recording_duration:
        Total recording length in seconds.  Inferred from events when ``None``.
    include_tn:
        When ``True`` add TN annotations that cover all time not occupied by
        tp/fp/fn events, giving a complete non-overlapping partition of the
        recording.

    Returns
    -------
    mne.Annotations
        Sorted by onset; description values are in ``{tp, fp, fn, tn}``.
    """
    pred_events = annotations_to_events(pred_annotations, target_label)
    pred_by_idx = {e.index: e for e in pred_events}

    rows: list[dict] = []
    covered: list[tuple[float, float]] = []

    for match in result.true_positives:
        ev = pred_by_idx.get(match.pred_index)
        if ev is not None:
            rows.append({"onset": ev.onset_pred, "duration": ev.duration_pred, "description": "tp"})
            covered.append((ev.onset_pred, ev.onset_pred + ev.duration_pred))

    for fp_ev in result.false_positives:
        rows.append({"onset": fp_ev.onset_pred, "duration": fp_ev.duration_pred, "description": "fp"})
        covered.append((fp_ev.onset_pred, fp_ev.onset_pred + fp_ev.duration_pred))

    for fn_ev in result.false_negatives:
        rows.append({"onset": fn_ev.onset_gt, "duration": fn_ev.duration_gt, "description": "fn"})
        covered.append((fn_ev.onset_gt, fn_ev.onset_gt + fn_ev.duration_gt))

    # if include_tn:
    #     if recording_duration is None:
    #         gt_events = annotations_to_events(gt_annotations, target_label)
    #         all_ends = (
    #             [e.onset + e.duration for e in gt_events]
    #             + [e.onset + e.duration for e in pred_events]
    #         )
    #         recording_duration = max(all_ends) if all_ends else 1.0
    #     for start, end in _complement_intervals(covered, recording_duration):
    #         rows.append({"onset": start, "duration": end - start, "description": "tn"})

    rows.sort(key=lambda r: r["onset"])

    if not rows:
        return mne.Annotations(onset=[], duration=[], description=[])
    return mne.Annotations(
        onset=[r["onset"] for r in rows],
        duration=[r["duration"] for r in rows],
        description=[r["description"] for r in rows],
    )


def save_scored_annotations_csv(
    annotations: mne.Annotations,
    csv_path: Path | str,
) -> Path:
    """Save tp/fp/fn/tn annotations to a CSV with onset/duration/description columns.

    Creates parent directories if they do not exist.  On Windows, if the target
    file is locked by another process, writes to a timestamped sibling file
    instead and returns that path.

    Returns
    -------
    Path
        The path that was actually written (may differ from *csv_path* if the
        target was locked).
    """
    from datetime import datetime

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "onset": annotations.onset,
            "duration": annotations.duration,
            "description": annotations.description,
        }
    )
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        # Target is locked (e.g. open in Excel/Explorer).  Fall back to a
        # timestamped sibling file so the caller is never blocked.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_stem(f"{path.stem}_{ts}")
        df.to_csv(fallback, index=False)
        import warnings
        warnings.warn(
            f"'{path}' is locked by another process. "
            f"Annotation CSV written to '{fallback}' instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback


def export_scored_prediction_csv(
    scored: ChannelEvaluationResult,
    gt_annotations: mne.Annotations,
    strategy: str,
    csv_path_template: Path | str,
    *,
    recording_duration: float | None = None,
    target_label: str = "blink",
) -> Path:
    """Build tp/fp/fn/tn annotations for the best channel and save to CSV.

    Typical usage after ``scored = evaluate_channels(...)``::

        CSV_PATH = Path(
            r"D:\\dataset\\...\\annotation_prediction\\ear_eog_predicted_{strategy}.csv"
        )
        out = export_scored_prediction_csv(
            scored, gt_annotations, strategy="strategy_dbo_drop", csv_path_template=CSV_PATH
        )

    Parameters
    ----------
    scored:
        Result from :func:`~blink_evaluation.channel_scoring.evaluate_channels`.
    gt_annotations:
        Ground-truth annotations in absolute time.
    strategy:
        Strategy name substituted into the ``{strategy}`` placeholder in
        *csv_path_template*.
    csv_path_template:
        Path string containing a ``{strategy}`` placeholder.
    recording_duration:
        Full recording duration in seconds; inferred from events when ``None``.
    include_tn:
        Whether to include TN complement annotations.
    target_label:
        Annotation label used during evaluation.

    Returns
    -------
    Path
        Path to the saved CSV file.

    Raises
    ------
    ValueError
        When *scored* has no best-channel result.
    """
    if scored.best_eval_result is None or scored.best_predicted is None:
        raise ValueError("scored has no best channel result — cannot export annotations")

    pred_annotations = dataframe_to_annotations(scored.best_predicted)

    scored_ann = build_scored_annotations(
        scored.best_eval_result,
        gt_annotations,
        pred_annotations,
        target_label=target_label,
        recording_duration=recording_duration,
    )

    out_path = Path(str(csv_path_template).replace("{strategy}", strategy))
    return save_scored_annotations_csv(scored_ann, out_path)


_MASTERLIST_COLS = [
    "description",
    "duration_gt",
    "duration_pred",
    "idx",
    "onset_gt",
    "onset_pred",
    "peak_time",
    "status",
]


def build_events_masterlist_df(
    tp_events: list[Match],
    fp_events: list[AnnotationEvent],
    fn_events: list[AnnotationEvent],
) -> pd.DataFrame:
    """Build a flat masterlist DataFrame from already-enriched tp/fp/fn events.

    Returns one row per event sorted by ``onset_gt`` (FP rows, which have no
    GT onset, appear at the end).  Columns: description, duration_gt,
    duration_pred, idx, onset_gt, onset_pred, peak_time, status.
    """
    rows: list[dict] = []

    for m in tp_events:
        rows.append(
            {
                "idx": m.gt_index,
                "status": "tp",
                "description": "tp",
                "onset_gt": m.onset_gt,
                "onset_pred": m.onset_pred,
                "duration_gt": m.duration_gt,
                "duration_pred": m.duration_pred,
                "peak_time": m.peak_time,
            }
        )

    for ev in fp_events:
        rows.append(
            {
                "idx": ev.index,
                "status": "fp",
                "description": "fp",
                "onset_gt": ev.onset_gt,
                "onset_pred": ev.onset_pred,
                "duration_gt": ev.duration_gt,
                "duration_pred": ev.duration_pred,
                "peak_time": ev.peak_time,
            }
        )

    for ev in fn_events:
        rows.append(
            {
                "idx": ev.index,
                "status": "fn",
                "description": "fn",
                "onset_gt": ev.onset_gt,
                "onset_pred": ev.onset_pred,
                "duration_gt": ev.duration_gt,
                "duration_pred": ev.duration_pred,
                "peak_time": ev.peak_time,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_MASTERLIST_COLS)

    df = pd.DataFrame(rows)
    df = df.sort_values("onset_gt", na_position="last").reset_index(drop=True)
    return df[_MASTERLIST_COLS]


def build_annotations_from_events(
    tp_events: list[Match],
    fp_events: list[AnnotationEvent],
    fn_events: list[AnnotationEvent],
) -> mne.Annotations:
    """Build ``mne.Annotations`` labeled tp/fp/fn directly from matched events.

    TP entries use the predicted window; FP use the predicted window; FN use
    the ground-truth window.  Sorted by onset.
    """
    rows: list[dict] = []

    for m in tp_events:
        rows.append({"onset": m.onset_pred, "duration": m.duration_pred, "description": "tp"})

    for ev in fp_events:
        rows.append({"onset": ev.onset_pred, "duration": ev.duration_pred, "description": "fp"})

    for ev in fn_events:
        rows.append({"onset": ev.onset_gt, "duration": ev.duration_gt, "description": "fn"})

    rows.sort(key=lambda r: r["onset"])

    if not rows:
        return mne.Annotations(onset=[], duration=[], description=[])

    return mne.Annotations(
        onset=[r["onset"] for r in rows],
        duration=[r["duration"] for r in rows],
        description=[r["description"] for r in rows],
    )


__all__ = [
    "build_annotations_from_events",
    "build_events_masterlist_df",
    "build_scored_annotations",
    "export_scored_prediction_csv",
    "save_scored_annotations_csv",
]
