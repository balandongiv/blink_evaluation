"""Compare one case CSV against the saved FIF annotations and open it in a plot."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mne


@dataclass
class AnnotationRow:
    onset: float
    duration: float
    description: str


def load_csv_annotations(csv_path: Path) -> List[AnnotationRow]:
    rows: List[AnnotationRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                AnnotationRow(
                    onset=float(row["onset"]),
                    duration=float(row["duration"]),
                    description=row["description"],
                )
            )
    return rows


def load_fif_annotations(fif_path: Path) -> List[AnnotationRow]:
    ann = mne.read_annotations(fif_path)
    return [
        AnnotationRow(onset=float(onset), duration=float(duration), description=desc)
        for onset, duration, desc in zip(ann.onset, ann.duration, ann.description)
    ]


def load_review_raw() -> mne.io.BaseRaw:
    """Load the sample raw recording so annotations can be visualized on it."""
    sample_data_folder = mne.datasets.sample.data_path()
    raw_file = os.path.join(
        sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
    )
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 20.5, fir_design="firwin")
    raw.resample(100)
    desired_channels = [f"EEG 00{idx}" for idx in range(10)]
    available = [ch for ch in desired_channels if ch in raw.ch_names]
    if available:
        raw.pick_channels(available)
    return raw


def resolve_case_path(case_name: str) -> Path:
    case_path = Path(f"{case_name}.csv")
    if not case_path.exists():
        valid = ", ".join(["case_1", "case_2", "case_3"])
        raise FileNotFoundError(f"Missing case file: {case_path}. Choose one of: {valid}")
    return case_path


def compare_annotations(csv_rows: List[AnnotationRow], fif_rows: List[AnnotationRow]) -> None:
    print(f"CSV rows: {len(csv_rows)}")
    print(f"FIF rows: {len(fif_rows)}")
    print()

    max_rows = max(len(csv_rows), len(fif_rows))
    mismatch_count = 0

    for idx in range(max_rows):
        csv_row = csv_rows[idx] if idx < len(csv_rows) else None
        fif_row = fif_rows[idx] if idx < len(fif_rows) else None

        print(f"Row {idx + 1}")
        print(f"  CSV: {csv_row}")
        print(f"  FIF: {fif_row}")

        if csv_row != fif_row:
            mismatch_count += 1
            print("  Status: different")
        else:
            print("  Status: match")
        print()

    if mismatch_count == 0 and len(csv_rows) == len(fif_rows):
        print("All annotations match.")
    else:
        print(f"Found {mismatch_count} differing row(s).")


def main() -> None:
    case_name = sys.argv[1] if len(sys.argv) > 1 else "case_2"
    case_path = resolve_case_path(case_name)
    fif_path = Path("../ground_truth_annotations.fif")

    if not fif_path.exists():
        raise FileNotFoundError(f"Missing FIF file: {fif_path}")

    csv_rows = load_csv_annotations(case_path)
    fif_rows = load_fif_annotations(fif_path)
    print(f"Checking {case_path.name} against {fif_path.name}")
    compare_annotations(csv_rows, fif_rows)

    raw = load_review_raw()
    raw.set_annotations(mne.read_annotations(fif_path))
    print("Opening interactive review plot. Close the window to finish.")
    raw.plot(block=True)


if __name__ == "__main__":
    main()
