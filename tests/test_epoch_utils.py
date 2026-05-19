"""Unit tests for blink_evaluation.epoch_utils."""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd
import pytest

from blink_evaluation.epoch_utils import (
    enrich_absolute_times,
    load_annotation_as_reference,
    load_ground_truth_annotations,
)


# ---------------------------------------------------------------------------
# enrich_absolute_times
# ---------------------------------------------------------------------------

def _make_epoch_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["epoch_index", "blink_onset", "blink_duration"])


def test_enrich_absolute_times_basic():
    frame = _make_epoch_frame([
        {"epoch_index": 0, "blink_onset": 5.0, "blink_duration": 0.3},
        {"epoch_index": 1, "blink_onset": 2.5, "blink_duration": 0.4},
        {"epoch_index": 2, "blink_onset": 0.0, "blink_duration": 0.2},
    ])
    result = enrich_absolute_times(frame, epoch_duration=60.0)

    assert result["absolute_onset_s"].tolist() == pytest.approx([5.0, 62.5, 120.0])
    assert result["absolute_offset_s"].tolist() == pytest.approx([5.3, 62.9, 120.2])


def test_enrich_absolute_times_does_not_mutate_input():
    frame = _make_epoch_frame([{"epoch_index": 0, "blink_onset": 1.0, "blink_duration": 0.1}])
    original_cols = set(frame.columns)
    enrich_absolute_times(frame, epoch_duration=30.0)
    assert set(frame.columns) == original_cols


def test_enrich_absolute_times_empty_frame():
    empty = pd.DataFrame(columns=["epoch_index", "blink_onset", "blink_duration"])
    result = enrich_absolute_times(empty, epoch_duration=60.0)
    assert result.empty
    assert "absolute_onset_s" not in result.columns


def test_enrich_absolute_times_epoch_boundary():
    # blink at the very start of epoch 3 with epoch_duration=10s
    frame = _make_epoch_frame([{"epoch_index": 3, "blink_onset": 0.0, "blink_duration": 0.5}])
    result = enrich_absolute_times(frame, epoch_duration=10.0)
    assert result["absolute_onset_s"].iloc[0] == pytest.approx(30.0)
    assert result["absolute_offset_s"].iloc[0] == pytest.approx(30.5)


# ---------------------------------------------------------------------------
# load_annotation_as_reference
# ---------------------------------------------------------------------------

def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "annot.csv"
    p.write_text(content)
    return p


def test_load_annotation_as_reference_basic(tmp_path):
    csv = _write_csv(tmp_path, "onset,duration\n65.0,0.3\n125.5,0.2\n")
    result = load_annotation_as_reference(csv, epoch_duration=60.0)

    assert list(result.columns) == ["epoch_index", "blink_onset", "blink_duration"]
    assert result["epoch_index"].tolist() == [1, 2]
    assert result["blink_onset"].tolist() == pytest.approx([5.0, 5.5])
    assert result["blink_duration"].tolist() == pytest.approx([0.3, 0.2])


def test_load_annotation_as_reference_drops_na(tmp_path):
    csv = _write_csv(tmp_path, "onset,duration\n10.0,0.3\n,0.2\n20.0,\n")
    result = load_annotation_as_reference(csv, epoch_duration=60.0)
    assert len(result) == 1
    assert result["blink_onset"].iloc[0] == pytest.approx(10.0)


def test_load_annotation_as_reference_first_epoch(tmp_path):
    csv = _write_csv(tmp_path, "onset,duration\n0.0,0.1\n59.9,0.1\n")
    result = load_annotation_as_reference(csv, epoch_duration=60.0)
    assert result["epoch_index"].tolist() == [0, 0]
    assert result["blink_onset"].tolist() == pytest.approx([0.0, 59.9])


def test_load_annotation_as_reference_exact_epoch_boundary(tmp_path):
    # onset exactly at epoch boundary should map to the next epoch
    csv = _write_csv(tmp_path, "onset,duration\n60.0,0.2\n")
    result = load_annotation_as_reference(csv, epoch_duration=60.0)
    assert result["epoch_index"].iloc[0] == 1
    assert result["blink_onset"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# load_ground_truth_annotations
# ---------------------------------------------------------------------------

def test_load_ground_truth_annotations_returns_mne_annotations(tmp_path):
    csv = _write_csv(tmp_path, "onset,duration\n65.0,0.3\n125.5,0.2\n")
    result = load_ground_truth_annotations(csv, epoch_duration=60.0)
    assert isinstance(result, mne.Annotations)


def test_load_ground_truth_annotations_onset_values(tmp_path):
    csv = _write_csv(tmp_path, "onset,duration\n65.0,0.3\n125.5,0.2\n")
    result = load_ground_truth_annotations(csv, epoch_duration=60.0)
    assert list(result.onset) == pytest.approx([65.0, 125.5])
    assert list(result.duration) == pytest.approx([0.3, 0.2])


def test_load_ground_truth_annotations_descriptions_are_blink(tmp_path):
    csv = _write_csv(tmp_path, "onset,duration\n65.0,0.3\n")
    result = load_ground_truth_annotations(csv, epoch_duration=60.0)
    assert list(result.description) == ["blink"]


def test_load_ground_truth_annotations_empty_csv(tmp_path):
    csv = _write_csv(tmp_path, "onset,duration\n")
    result = load_ground_truth_annotations(csv, epoch_duration=60.0)
    assert len(result) == 0
