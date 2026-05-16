from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from blink_evaluation.types import AnnotationEvent

_REQUIRED_COLUMNS = {"onset", "duration", "description"}


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
