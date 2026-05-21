from __future__ import annotations

import mne

from blink_evaluation.io import annotations_to_events
from blink_evaluation.matching import match_events
from blink_evaluation.metrics import compute_event_metrics, compute_sample_metrics
from blink_evaluation.types import EvaluationResult


def evaluate_annotations(
    ground_truth: mne.Annotations,
    predicted: mne.Annotations,
    target_label: str = "blink",
    iou_threshold: float = 0.5,
    pad: float = 0.0,
    sample_rate: float = 100.0,
    recording_duration: float | None = None,
    peak_required: bool = False,
    peak_tolerance: float | None = None,
) -> EvaluationResult:
    gt_events = annotations_to_events(ground_truth, target_label)
    pred_events = annotations_to_events(predicted, target_label)

    sample_metrics, sample_confusion = compute_sample_metrics(
        gt_events,
        pred_events,
        sample_rate=sample_rate,
        recording_duration=recording_duration,
    )

    matches, false_positives, false_negatives = match_events(
        gt_events,
        pred_events,
        iou_threshold=iou_threshold,
        pad=pad,
        peak_required=peak_required,
        peak_tolerance=peak_tolerance,
    )

    event_metrics = compute_event_metrics(matches, len(gt_events), len(pred_events))

    return EvaluationResult(
        event_metrics=event_metrics,
        sample_metrics=sample_metrics,
        matches=matches,
        true_positives=matches,
        false_positives=false_positives,
        false_negatives=false_negatives,
        sample_confusion=sample_confusion,
    )
