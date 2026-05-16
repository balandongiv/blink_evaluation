"""
Tutorial -- Case 3: Missed Blinks and Spurious Detections (FP + FN)
====================================================================
Ground truth : case_1.csv  (14 blink annotations)
Prediction   : case_3.csv  (12 blink annotations -- 10 correct + 2 extra, 4 missed)

Expected outcome
----------------
  TP = 10   FP = 2   FN = 4
  Precision = 0.8333   Recall = 0.7143   F1 = 0.7692

This is a realistic imperfect detector:
  * 10 of the 14 ground-truth blinks are correctly detected  (green in plot)
  * 2 detections have no matching ground-truth event          (red   in plot)
  * 4 ground-truth blinks were never detected                 (blue  in plot)

The four missed blinks form a tight cluster around t = 231-261 s, which
suggests the detector struggled during that period of the recording.
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
# 2. Load annotations (GT = case_1, Pred = case_3)
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "tests" / "data"

gt   = read_annotation_csv(DATA_DIR / "case_1.csv")
pred = read_annotation_csv(DATA_DIR / "case_3.csv")

print(f"Ground-truth annotations : {len(gt)}")
print(f"Predicted  annotations   : {len(pred)}  (10 matched + 2 false alarms; 4 GT missed)")

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

print("\n=== Event-Level Metrics (Case 3 -- Mixed FP + FN) ===")
print(f"  TP        : {em.tp}   (expected 10)")
print(f"  FP        : {em.fp}   (expected  2)")
print(f"  FN        : {em.fn}   (expected  4)")
print(f"  Precision : {em.precision:.4f}  (expected 0.8333 = 10/12)")
print(f"  Recall    : {em.recall:.4f}  (expected 0.7143 = 10/14)")
print(f"  F1        : {em.f1:.4f}  (expected 0.7692)")
print(f"  Mean IoU  : {em.mean_iou_raw:.4f}")

print("\n=== Sample-Wise Metrics ===")
print(f"  TP (samples) : {sm.tp}")
print(f"  FP (samples) : {sm.fp}   <- from the 2 spurious detections")
print(f"  FN (samples) : {sm.fn}   <- from the 4 missed blinks")
print(f"  TN (samples) : {sm.tn}")
print(f"  Accuracy     : {sm.accuracy:.4f}")
print(f"  Precision    : {sm.precision:.4f}")
print(f"  Recall       : {sm.recall:.4f}")
print(f"  F1           : {sm.f1:.4f}")
print(f"  Macro F1     : {sm.macro_f1:.4f}")

print("\nFalse Positives (spurious detections):")
for fp in sorted(result.false_positives, key=lambda e: e.onset):
    print(f"  onset={fp.onset:.4f} s  duration={fp.duration:.4f} s")

print("\nFalse Negatives (missed ground-truth blinks):")
for fn in sorted(result.false_negatives, key=lambda e: e.onset):
    print(f"  onset={fn.onset:.4f} s  duration={fn.duration:.4f} s")

print("\nTrue Positive matches (gt_index -> pred_index, IoU):")
for m in sorted(result.matches, key=lambda m: m.gt_index):
    print(f"  GT[{m.gt_index:2d}] -> Pred[{m.pred_index:2d}]  IoU={m.iou_raw:.4f}")

# ---------------------------------------------------------------------------
# 5. Time-series plots
# ---------------------------------------------------------------------------
TUTORIAL_DIR = Path(__file__).parent

# Overview: early blinks that were correctly detected (t=40-80 s, all TP)
fig_early = plot_annotation_comparison(
    raw, gt, pred, result,
    target_label="blink",
    start=40.0, stop=80.0,
    show=False,
)
path_early = TUTORIAL_DIR / "case_3_early_tps.png"
fig_early.savefig(path_early, dpi=120, bbox_inches="tight")
print(f"\nPlot (early TPs, t=40-80 s)       : {path_early.resolve()}")

# Cluster of missed blinks (t=228-265 s, mix of TP and FN)
fig_fn = plot_annotation_comparison(
    raw, gt, pred, result,
    target_label="blink",
    start=226.0, stop=265.0,
    show=False,
)
path_fn = TUTORIAL_DIR / "case_3_missed_cluster.png"
fig_fn.savefig(path_fn, dpi=120, bbox_inches="tight")
print(f"Plot (missed cluster, t=226-265 s): {path_fn.resolve()}")

# First spurious detection (t~10 s)
fig_fp1 = plot_annotation_comparison(
    raw, gt, pred, result,
    target_label="blink",
    start=8.0, stop=14.0,
    show=False,
)
path_fp1 = TUTORIAL_DIR / "case_3_fp1_t10s.png"
fig_fp1.savefig(path_fp1, dpi=120, bbox_inches="tight")
print(f"Plot (FP at t~10 s)               : {path_fp1.resolve()}")

# ---------------------------------------------------------------------------
# 6. HTML report
# ---------------------------------------------------------------------------
report_path = TUTORIAL_DIR / "case_3_report.html"
create_html_report(
    raw, gt, pred, result,
    output_path=report_path,
    target_label="blink",
)
print(f"\nHTML report : {report_path.resolve()}")
