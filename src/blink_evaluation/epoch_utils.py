"""Epoch-relative annotation loading and absolute-time enrichment."""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from blink_evaluation.io import dataframe_to_annotations


def load_annotation_as_reference(
    csv_path: Path,
    epoch_duration: float,
) -> pd.DataFrame:
    """Convert a CSV of absolute-onset annotations into epoch-relative blink references.

    The CSV must have ``onset`` and ``duration`` columns (in seconds).
    Returns a DataFrame with columns: epoch_index, blink_onset, blink_duration.
    """
    df = pd.read_csv(csv_path).dropna(subset=["onset", "duration"])
    rows: list[dict] = []
    for _, row in df.iterrows():
        onset_abs = float(row["onset"])
        duration = float(row["duration"])
        epoch_index = int(onset_abs // epoch_duration)
        rows.append(
            {
                "epoch_index": epoch_index,
                "blink_onset": onset_abs - epoch_index * epoch_duration,
                "blink_duration": duration,
            }
        )
    return pd.DataFrame(rows, columns=["epoch_index", "blink_onset", "blink_duration"])


def enrich_absolute_times(frame: pd.DataFrame, epoch_duration: float) -> pd.DataFrame:
    """Add ``absolute_onset_s`` and ``absolute_offset_s`` columns from epoch-relative timings.

    Requires ``epoch_index``, ``blink_onset``, and ``blink_duration`` columns.
    These enriched columns are expected by ``dataframe_to_annotations``.
    """
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["absolute_onset_s"] = (
        out["epoch_index"].astype(float) * epoch_duration
        + out["blink_onset"].astype(float)
    )
    out["absolute_offset_s"] = out["absolute_onset_s"] + out["blink_duration"].astype(float)
    return out


def load_ground_truth_annotations(
    csv_path: Path,
    epoch_duration: float,
) -> mne.Annotations:
    """Load a ground-truth CSV and return absolute-time ``mne.Annotations``.

    Combines :func:`load_annotation_as_reference`, :func:`enrich_absolute_times`,
    and :func:`~blink_evaluation.io.dataframe_to_annotations` into one call.
    """
    df = enrich_absolute_times(load_annotation_as_reference(csv_path, epoch_duration), epoch_duration)
    return dataframe_to_annotations(df)


__all__ = [
    "enrich_absolute_times",
    "load_annotation_as_reference",
    "load_ground_truth_annotations",
]
