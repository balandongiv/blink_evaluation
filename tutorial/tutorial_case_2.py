"""
Tutorial -- Case 2: Two Spurious Detections (False Positives)
=============================================================
Ground truth : case_1.csv  (14 blink annotations)
Prediction   : case_2.csv  (16 blink annotations -- 14 correct + 2 extra)

Expected outcome
----------------
  TP = 14   FP = 2   FN = 0
  Precision = 0.8750   Recall = 1.0000   F1 = 0.9333

The detector finds every ground-truth blink (recall = 1) but fires
two extra times at onset ~ 10 s and onset ~ 300 s, which have no
matching ground-truth event.  Those two intervals appear red (FP) in
the plot; everything else is green (TP).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mne

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from load_sample_file import prepare_raw

from blink_evaluation import evaluate_annotations, read_annotation_csv
from blink_evaluation.visualization import create_html_report, plot_annotation_comparison

# ---------------------------------------------------------------------------
# 1. Load the MNE sample raw file
# ---------------------------------------------------------------------------
sample_data_folder = mne.datasets.sample.data_path()
raw_file = os.path.join(
    sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
)
raw = prepare_raw(raw_file)

# ---------------------------------------------------------------------------
# 2. Load annotations (GT = case_1, Pred = case_2)
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "tests" / "data"

gt   = read_annotation_csv(DATA_DIR / "case_1.csv")
pred = read_annotation_csv(DATA_DIR / "case_2.csv")

print(f"Ground-truth annotations : {len(gt)}")
print(f"Predicted  annotations   : {len(pred)}  (14 correct + 2 false alarms)")

# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------
result = evaluate_annotations(
    gt,
    pred,
    target_label="blink",
    iou_threshold=0.5,
    pad=0.0,
    sample_rate=raw.info["sfreq"],
    recording_duration=float(raw.times[-1]),
)

# ---------------------------------------------------------------------------
# 4. Print metrics
# ---------------------------------------------------------------------------
em = result.event_metrics
sm = result.sample_metrics

print("\n=== Event-Level Metrics (Case 2 -- 2 False Positives) ===")
print(f"  TP        : {em.tp}   (expected 14)")
print(f"  FP        : {em.fp}   (expected  2)")
print(f"  FN        : {em.fn}   (expected  0)")
print(f"  Precision : {em.precision:.4f}  (expected 0.8750 = 14/16)")
print(f"  Recall    : {em.recall:.4f}  (expected 1.0000)")
print(f"  F1        : {em.f1:.4f}  (expected 0.9333)")
print(f"  Mean IoU  : {em.mean_iou_raw:.4f}")

print("\n=== Sample-Wise Metrics ===")
print(f"  TP (samples) : {sm.tp}")
print(f"  FP (samples) : {sm.fp}   <- from the 2 spurious detections")
print(f"  FN (samples) : {sm.fn}")
print(f"  TN (samples) : {sm.tn}")
print(f"  Accuracy     : {sm.accuracy:.4f}")
print(f"  Precision    : {sm.precision:.4f}")
print(f"  Recall       : {sm.recall:.4f}")
print(f"  F1           : {sm.f1:.4f}")
print(f"  Macro F1     : {sm.macro_f1:.4f}")

print("\nFalse Positives (spurious detections):")
for fp in sorted(result.false_positives, key=lambda e: e.onset):
    print(f"  onset={fp.onset:.4f} s  duration={fp.duration:.4f} s")

print("\nFalse Negatives : (none -- all ground-truth blinks were found)")
for fn in result.false_negatives:
    print(f"  onset={fn.onset:.4f}  duration={fn.duration:.4f}")

# ---------------------------------------------------------------------------
# 5. Time-series plots -- one for each false-positive region
# ---------------------------------------------------------------------------
TUTORIAL_DIR = Path(__file__).parent

# First FP at onset ~ 10 s
fig_fp1 = plot_annotation_comparison(
    raw, gt, pred, result,
    target_label="blink",
    start=8.0, stop=14.0,
    show=False,
)
path_fp1 = TUTORIAL_DIR / "case_2_fp1_t10s.png"
fig_fp1.savefig(path_fp1, dpi=120, bbox_inches="tight")
print(f"\nPlot (FP at t~10 s) : {path_fp1.resolve()}")

# Second FP at onset ~ 300 s
fig_fp2 = plot_annotation_comparison(
    raw, gt, pred, result,
    target_label="blink",
    start=298.0, stop=302.0,
    show=False,
)
path_fp2 = TUTORIAL_DIR / "case_2_fp2_t300s.png"
fig_fp2.savefig(path_fp2, dpi=120, bbox_inches="tight")
print(f"Plot (FP at t~300 s): {path_fp2.resolve()}")

# Overview of the true-positive cluster
fig_tp = plot_annotation_comparison(
    raw, gt, pred, result,
    target_label="blink",
    start=40.0, stop=80.0,
    show=False,
)
path_tp = TUTORIAL_DIR / "case_2_tp_cluster.png"
fig_tp.savefig(path_tp, dpi=120, bbox_inches="tight")
print(f"Plot (TP cluster)   : {path_tp.resolve()}")

# ---------------------------------------------------------------------------
# 6. HTML report
# ---------------------------------------------------------------------------
report_path = TUTORIAL_DIR / "case_2_report.html"
create_html_report(
    raw, gt, pred, result,
    output_path=report_path,
    target_label="blink",
)
print(f"\nHTML report : {report_path.resolve()}")
