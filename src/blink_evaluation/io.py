from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from blink_evaluation.types import AnnotationEvent

_REQUIRED_COLUMNS = {"onset", "duration", "description"}


def dataframe_to_annotations(
    df: pd.DataFrame,
    onset_col: str = "absolute_onset_s",
    duration_col: str = "blink_duration",
    description: str = "blink",
) -> mne.Annotations:
    """Convert a DataFrame with absolute onset/duration columns to mne.Annotations.

    Parameters
    ----------
    df:
        DataFrame produced by ``enrich_absolute_times`` or similar.  Must contain
        *onset_col* and *duration_col* columns.
    onset_col:
        Column holding absolute onset times in seconds.
    duration_col:
        Column holding blink durations in seconds.
    description:
        Annotation label assigned to every row.

    Returns
    -------
    mne.Annotations
    """
    if df.empty:
        return mne.Annotations(onset=[], duration=[], description=[])
    onsets = df[onset_col].astype(float).to_numpy()
    durations = df[duration_col].astype(float).to_numpy()
    descriptions = [description] * len(df)
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


def read_annotation_csv(path: str | Path) -> mne.Annotations:
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file '{path}': {exc}") from exc

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV file '{path}' is missing required columns: {sorted(missing)}"
        )

    try:
        onsets = df["onset"].astype(float).values
        durations = df["duration"].astype(float).values
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"CSV file '{path}' contains non-numeric onset or duration values: {exc}"
        ) from exc

    descriptions = df["description"].astype(str).values
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


def annotations_to_events(
    annotations: mne.Annotations,
    target_label: str,
) -> list[AnnotationEvent]:
    events: list[AnnotationEvent] = []
    for idx, (onset, duration, description) in enumerate(
        zip(annotations.onset, annotations.duration, annotations.description)
    ):
        if description == target_label:
            events.append(
                AnnotationEvent(
                    index=idx,
                    onset=float(onset),
                    duration=float(duration),
                    description=description,
                )
            )
    return events
