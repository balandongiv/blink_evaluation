"""Per-channel blink evaluation and lane scoring."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import mne

from blink_evaluation.api import evaluate_annotations
from blink_evaluation.epoch_utils import enrich_absolute_times
from blink_evaluation.io import dataframe_to_annotations
from blink_evaluation.types import EvaluationResult


@dataclass
class ChannelEvaluationResult:
    """Summary of per-channel blink evaluation."""

    lane_summary: pd.DataFrame
    best_channel: str | None
    best_eval_result: EvaluationResult | None


def evaluate_channels(
    channel_results: list[dict],
    gt_annotations: mne.Annotations,
    epoch_duration: float,
    iou_threshold: float = 0.1,
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

    Returns
    -------
    ChannelEvaluationResult
        Contains ``lane_summary`` (DataFrame sorted by F1 descending),
        ``best_channel``, and ``best_eval_result``.
    """
    lane_rows: list[dict] = []
    best_channel: str | None = None
    best_eval_result: EvaluationResult | None = None

    for cr in channel_results:
        predicted_df = enrich_absolute_times(cr["mapped_candidates"], epoch_duration)
        pred_annotations = dataframe_to_annotations(predicted_df)

        result = evaluate_annotations(
            gt_annotations,
            pred_annotations,
            target_label="blink",
            iou_threshold=iou_threshold,
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

    lane_summary = (
        pd.DataFrame(lane_rows)
        .sort_values(["f1", "tp", "fp", "channel"], ascending=[False, False, True, True])
        .reset_index(drop=True)
    )

    return ChannelEvaluationResult(
        lane_summary=lane_summary,
        best_channel=best_channel,
        best_eval_result=best_eval_result,
    )


__all__ = ["ChannelEvaluationResult", "evaluate_channels"]
