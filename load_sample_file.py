"""Load a raw EEG file, annotate it manually, and save the ground-truth labels."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import mne


def prepare_raw(raw_file: str) -> mne.io.BaseRaw:
    """Load and lightly preprocess the raw recording before annotation."""
    print(f"Loading data from: {raw_file}")
    raw = mne.io.read_raw_fif(raw_file, preload=True)

    # Keep only EEG channels, then apply the same preprocessing used for review.
    raw.pick_types(eeg=True)
    raw.filter(0.5, 20.5, fir_design="firwin")
    raw.resample(100)

    # Keep only the first 10 EEG channels if they exist in the file.
    desired_channels = [f"EEG 00{idx}" for idx in range(10)]
    available = [ch for ch in desired_channels if ch in raw.ch_names]
    if available:
        raw.pick_channels(available)

    return raw


def save_annotations(raw: mne.io.BaseRaw, output_stem: Path) -> None:
    """Save annotations in a plain onset/duration/description CSV format."""
    annotations = raw.annotations

    csv_path = output_stem.with_suffix(".csv")

    if len(annotations) == 0:
        print("No annotations were added. Nothing was saved.")
        return

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["onset", "duration", "description"])
        for onset, duration, description in zip(
            annotations.onset, annotations.duration, annotations.description
        ):
            writer.writerow([onset, duration, description])

    print(f"Saved annotations to: {csv_path}")


def main() -> None:
    sample_data_folder = mne.datasets.sample.data_path()
    raw_file = os.path.join(
        sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
    )

    raw = prepare_raw(raw_file)

    print(
        "\nInteractive annotation instructions:\n"
        "- Use the MNE plot window to mark manual annotations.\n"
        "- Set the description to the label you want, for example: eye_blink.\n"
        "- Close the window when you are done annotating.\n"
        "- After closing, the annotations will be saved.\n"
    )

    raw.plot(block=True)

    output_stem = Path("ground_truth_annotations")
    save_annotations(raw, output_stem)


if __name__ == "__main__":
    main()
