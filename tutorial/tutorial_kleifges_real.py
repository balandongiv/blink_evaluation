"""Development tutorial: evaluate_channels on real Kleifges data.

Loads the pickled inputs produced by ``tutorial/10a_kleifges.py`` and the
corresponding FIF file, then calls ``evaluate_channels``.  Use this script
as the live test-bed when developing and refining the ``blink_evaluation``
package — all inputs are real signals from the drowsy-driving dataset.

Run ``tutorial/10a_kleifges.py`` first to regenerate the pickle if the
strategy or preprocessing changes.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import mne
import pandas as pd

# -- path setup ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BLINK_EVAL_SRC = Path(__file__).resolve().parents[1] / "src"
for p in (str(REPO_ROOT), str(BLINK_EVAL_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from blink_evaluation import (
    build_annotations_from_events,
    build_events_masterlist_df,
    evaluate_channels,
    save_scored_annotations_csv,
)
from blink_evaluation.blink_epoch_report import create_blink_epoch_report
from src.io.eeg_channels import load_brain_region_channels, load_raw_with_brain_channels

# -- paths --------------------------------------------------------------------
PICKLE_PATH = Path(
    r"D:\dataset\drowsy_driving_raja_processed\S1\S01_20170519_043933"
    r"\annotation_prediction\kleifges_eval_inputs.pkl"
)
TUTORIAL_DIR = Path(__file__).parent
ANNOTATION_CSV = TUTORIAL_DIR / "ear_eog_predicted_kleifges.csv"
MASTERLIST_CSV = TUTORIAL_DIR / "blink_events_masterlist_kleifges.csv"
REPORT_PATH = TUTORIAL_DIR / "blink_epoch_report_kleifges.html"
BRAIN_REGION_YAML = REPO_ROOT / "brain_region.yaml"


def main() -> None:
    # -- load pickle ----------------------------------------------------------
    print(f"Loading eval inputs from: {PICKLE_PATH}")
    with open(PICKLE_PATH, "rb") as f:
        inputs = pickle.load(f)

    channel_results: list[dict] = inputs["channel_results"]
    gt_annotations: mne.Annotations = inputs["gt_annotations"]
    epoch_duration: float = inputs["epoch_duration"]
    peak_required: bool = inputs["peak_required"]
    peak_tolerance: float | None = inputs["peak_tolerance"]
    fif_path = Path(inputs["fif_path"])

    print(f"  channels      : {[cr['channel'] for cr in channel_results]}")
    print(f"  gt blinks     : {len(gt_annotations)}")
    print(f"  epoch_duration: {epoch_duration}s")
    print(f"  peak_required : {peak_required}, peak_tolerance: {peak_tolerance}")

    # -- load raw for continuous time-series peak detection -------------------
    brain_channels = load_brain_region_channels(BRAIN_REGION_YAML)
    brain_channels = ["E3"]
    raw = load_raw_with_brain_channels(fif_path, brain_channels)
    print(f"\nRaw loaded: {raw.info['nchan']} ch, {raw.times[-1]:.1f}s @ {raw.info['sfreq']}Hz")

    # -- evaluate -------------------------------------------------------------
    scored = evaluate_channels(
        channel_results,
        gt_annotations,
        epoch_duration=epoch_duration,
        peak_required=peak_required,
        peak_tolerance=peak_tolerance,
        raw=raw,
    )

    # -- results --------------------------------------------------------------
    em = scored.best_eval_result.event_metrics
    print(f"\nbest_channel = {scored.best_channel}")
    print(f"tp={em.tp}  fp={em.fp}  fn={em.fn}")
    print(f"precision={em.precision:.4f}  recall={em.recall:.4f}  f1={em.f1:.4f}")
    print(f"\n=== Lane Summary ===")
    print(scored.lane_summary.to_string(index=False))

    # -- build output from tp/fp/fn events ------------------------------------
    result = scored.best_eval_result
    tp_events = result.true_positives
    fp_events = result.false_positives
    fn_events = result.false_negatives

    # masterlist CSV: one row per event with full timing on both sides
    df_masterlist = build_events_masterlist_df(tp_events, fp_events, fn_events)
    # compute onset as average of onset_gt and onset_pred; fall back to whichever is present
    df_masterlist["onset"] = df_masterlist.apply(
        lambda row: (
            (row["onset_gt"] + row["onset_pred"]) / 2.0
            if pd.notna(row["onset_gt"]) and pd.notna(row["onset_pred"])
            else float(row["onset_gt"]) if pd.notna(row["onset_gt"])
            else float(row["onset_pred"]) if pd.notna(row["onset_pred"])
            else 0.0
        ),
        axis=1,
    )
    df_masterlist = df_masterlist.sort_values("onset").reset_index(drop=True)

    df_masterlist.to_csv(MASTERLIST_CSV, index=False)
    print(f"\nMasterlist CSV saved: {MASTERLIST_CSV}")
    print(df_masterlist.to_string(index=False))

    # annotation CSV: tp/fp/fn windows for visual replay on the raw signal
    scored_ann = build_annotations_from_events(tp_events, fp_events, fn_events)
    csv_out = save_scored_annotations_csv(scored_ann, ANNOTATION_CSV)
    print(f"\nScored annotation CSV saved: {csv_out}")

    # -- Per-epoch blink HTML report ------------------------------------------
    saved_reports = create_blink_epoch_report(
        scored,
        df_masterlist,
        epoch_duration=epoch_duration,
        output_path=REPORT_PATH,
        pad_s=0.5,
        sync_offset_s=0.0,
    )
    for p in saved_reports:
        print(f"Blink epoch report saved: {p}")


if __name__ == "__main__":
    main()
