from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from blink_evaluation.intervals import compute_tiou, expand_interval
from blink_evaluation.types import AnnotationEvent, Match


def _build_cost_matrix(
    gt_events: list[AnnotationEvent],
    pred_events: list[AnnotationEvent],
    pad: float,
    peak_required: bool,
    peak_tolerance: float | None,
) -> np.ndarray:
    n_gt = len(gt_events)
    n_pred = len(pred_events)
    matrix = np.zeros((n_gt, n_pred), dtype=float)

    for i, gt in enumerate(gt_events):
        g_s, g_e = expand_interval(gt.onset, gt.duration, pad)
        for j, pred in enumerate(pred_events):
            p_s, p_e = expand_interval(pred.onset, pred.duration, pad)
            iou = compute_tiou(g_s, g_e, p_s, p_e)

            if peak_required and iou > 0.0:
                gt_peak = gt.peak_time if gt.peak_time is not None else gt.onset + gt.duration / 2
                pred_peak = pred.peak_time if pred.peak_time is not None else pred.onset + pred.duration / 2
                tol = peak_tolerance if peak_tolerance is not None else 0.0
                if abs(gt_peak - pred_peak) > tol:
                    iou = 0.0

            matrix[i, j] = iou

    return matrix


def match_events(
    gt_events: list[AnnotationEvent],
    pred_events: list[AnnotationEvent],
    iou_threshold: float = 0.5,
    pad: float = 0.0,
    peak_required: bool = False,
    peak_tolerance: float | None = None,
) -> tuple[list[Match], list[AnnotationEvent], list[AnnotationEvent]]:
    if not gt_events or not pred_events:
        return [], list(pred_events), list(gt_events)

    cost_expanded = _build_cost_matrix(gt_events, pred_events, pad, peak_required, peak_tolerance)
    cost_raw = _build_cost_matrix(gt_events, pred_events, 0.0, False, None)

    row_ind, col_ind = linear_sum_assignment(cost_expanded, maximize=True)

    matches: list[Match] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for i, j in zip(row_ind, col_ind):
        iou_exp = cost_expanded[i, j]
        iou_raw = cost_raw[i, j]

        if iou_exp < iou_threshold:
            continue

        if peak_required:
            gt = gt_events[i]
            pred = pred_events[j]
            gt_peak = gt.peak_time if gt.peak_time is not None else gt.onset + gt.duration / 2
            pred_peak = pred.peak_time if pred.peak_time is not None else pred.onset + pred.duration / 2
            peak_delta: float | None = abs(gt_peak - pred_peak)
        else:
            peak_delta = None

        matches.append(
            Match(
                gt_index=gt_events[i].index,
                pred_index=pred_events[j].index,
                iou_raw=iou_raw,
                iou_expanded=iou_exp,
                peak_delta=peak_delta,
                accepted=True,
            )
        )
        matched_gt.add(i)
        matched_pred.add(j)

    false_positives = [pred_events[j] for j in range(len(pred_events)) if j not in matched_pred]
    false_negatives = [gt_events[i] for i in range(len(gt_events)) if i not in matched_gt]

    return matches, false_positives, false_negatives
