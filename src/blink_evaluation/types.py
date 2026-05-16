from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnnotationEvent:
    index: int
    onset: float
    duration: float
    description: str
    peak_time: float | None = None


@dataclass
class Match:
    gt_index: int
    pred_index: int
    iou_raw: float
    iou_expanded: float
    peak_delta: float | None
    accepted: bool


@dataclass
class EventMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_iou_raw: float
    mean_iou_expanded: float


@dataclass
class SampleMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


@dataclass
class EvaluationResult:
    event_metrics: EventMetrics
    sample_metrics: SampleMetrics
    matches: list[Match]
    true_positives: list[Match]
    false_positives: list[AnnotationEvent]
    false_negatives: list[AnnotationEvent]
    sample_confusion: dict
