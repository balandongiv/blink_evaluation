from __future__ import annotations

import numpy as np

from blink_evaluation.types import AnnotationEvent


def compute_tiou(g_start: float, g_end: float, p_start: float, p_end: float) -> float:
    intersection = max(0.0, min(g_end, p_end) - max(g_start, p_start))
    union = max(g_end, p_end) - min(g_start, p_start)
    if union <= 0.0:
        return 0.0
    return intersection / union


def expand_interval(onset: float, duration: float, pad: float) -> tuple[float, float]:
    return onset - pad, onset + duration + pad


def annotations_to_binary_array(
    events: list[AnnotationEvent],
    sample_rate: float,
    n_samples: int,
) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for e in events:
        start_idx = int(np.floor(e.onset * sample_rate))
        end_idx = int(np.ceil((e.onset + e.duration) * sample_rate))
        start_idx = max(0, min(start_idx, n_samples))
        end_idx = max(0, min(end_idx, n_samples))
        mask[start_idx:end_idx] = True
    return mask
