from __future__ import annotations

import numpy as np

from blink_evaluation.intervals import annotations_to_binary_array
from blink_evaluation.types import AnnotationEvent, EventMetrics, Match, SampleMetrics


def compute_event_metrics(
    matches: list[Match],
    n_gt: int,
    n_pred: int,
) -> EventMetrics:
    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0.0 else 0.0

    if matches:
        mean_iou_raw = sum(m.iou_raw for m in matches) / len(matches)
        mean_iou_expanded = sum(m.iou_expanded for m in matches) / len(matches)
    else:
        mean_iou_raw = 0.0
        mean_iou_expanded = 0.0

    return EventMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou_raw=mean_iou_raw,
        mean_iou_expanded=mean_iou_expanded,
    )


def _class_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = prec + rec
    f1 = 2.0 * prec * rec / denom if denom > 0.0 else 0.0
    return prec, rec, f1


def compute_sample_metrics(
    gt_events: list[AnnotationEvent],
    pred_events: list[AnnotationEvent],
    sample_rate: float,
    recording_duration: float | None,
) -> tuple[SampleMetrics, dict]:
    if recording_duration is None:
        all_ends = [e.onset + e.duration for e in gt_events] + [e.onset + e.duration for e in pred_events]
        recording_duration = max(all_ends) if all_ends else 1.0

    n_samples = int(np.ceil(recording_duration * sample_rate))

    gt_mask = annotations_to_binary_array(gt_events, sample_rate, n_samples)
    pred_mask = annotations_to_binary_array(pred_events, sample_rate, n_samples)

    tp = int(np.sum(gt_mask & pred_mask))
    fp = int(np.sum(~gt_mask & pred_mask))
    fn = int(np.sum(gt_mask & ~pred_mask))
    tn = int(np.sum(~gt_mask & ~pred_mask))

    total = n_samples
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Blink class metrics
    blink_prec, blink_rec, blink_f1 = _class_f1(tp, fp, fn)

    # Non-blink class: TN is the "correct non-blink", FN_nb=fp, FP_nb=fn
    nonblink_prec, nonblink_rec, nonblink_f1 = _class_f1(tn, fn, fp)

    micro_precision = blink_prec
    micro_recall = blink_rec
    micro_f1 = blink_f1

    macro_precision = (blink_prec + nonblink_prec) / 2.0
    macro_recall = (blink_rec + nonblink_rec) / 2.0
    macro_f1 = (blink_f1 + nonblink_f1) / 2.0

    sample_metrics = SampleMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        accuracy=accuracy,
        precision=blink_prec,
        recall=blink_rec,
        f1=blink_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
    )

    sample_confusion = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_mask": gt_mask,
        "pred_mask": pred_mask,
    }

    return sample_metrics, sample_confusion
