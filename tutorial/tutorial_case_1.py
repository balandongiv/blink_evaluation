"""
Tutorial — Case 1: Perfect Match
=================================
Ground truth : case_1.csv  (14 blink annotations)
Prediction   : case_1.csv  (identical to ground truth)

Expected outcome
----------------
  TP = 14   FP = 0   FN = 0
  Precision = 1.000   Recall = 1.000   F1 = 1.000

This is the ideal scenario: every ground-truth blink is detected and
no spurious detections are added.  The plot should show all 14 intervals
coloured green (true positives) and nothing in red or blue.
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
# 2. Load annotations (GT = case_1, Pred = case_1)
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "tests" / "data"

gt   = read_annotation_csv(DATA_DIR / "case_1.csv")
pred = read_annotation_csv(DATA_DIR / "case_1.csv")

print(f"Ground-truth annotations : {len(gt)}")
print(f"Predicted  annotations   : {len(pred)}")

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

print("\n=== Event-Level Metrics (Case 1 — Perfect Match) ===")
print(f"  TP        : {em.tp}   (expected 14)")
print(f"  FP        : {em.fp}   (expected  0)")
print(f"  FN        : {em.fn}   (expected  0)")
print(f"  Precision : {em.precision:.4f}  (expected 1.0000)")
print(f"  Recall    : {em.recall:.4f}  (expected 1.0000)")
print(f"  F1        : {em.f1:.4f}  (expected 1.0000)")
print(f"  Mean IoU  : {em.mean_iou_raw:.4f}")

print("\n=== Sample-Wise Metrics ===")
print(f"  TP (samples) : {sm.tp}")
print(f"  FP (samples) : {sm.fp}")
print(f"  FN (samples) : {sm.fn}")
print(f"  TN (samples) : {sm.tn}")
print(f"  Accuracy     : {sm.accuracy:.4f}")
print(f"  Precision    : {sm.precision:.4f}")
print(f"  Recall       : {sm.recall:.4f}")
print(f"  F1           : {sm.f1:.4f}")
print(f"  Macro F1     : {sm.macro_f1:.4f}")

print("\nFalse Positives : (none expected)")
for fp in result.false_positives:
    print(f"  onset={fp.onset:.4f}  duration={fp.duration:.4f}")

print("\nFalse Negatives : (none expected)")
for fn in result.false_negatives:
    print(f"  onset={fn.onset:.4f}  duration={fn.duration:.4f}")

# ---------------------------------------------------------------------------
# 5. Time-series plot — zoom into the cluster of blinks around t=43–75 s
#    where most of the early blinks occur.
# ---------------------------------------------------------------------------
TUTORIAL_DIR = Path(__file__).parent
fig = plot_annotation_comparison(
    raw,
    gt,
    pred,
    result,
    target_label="blink",
    start=40.0,
    stop=80.0,
    show=False,
)
plot_path = TUTORIAL_DIR / "case_1_plot.png"
fig.savefig(plot_path, dpi=120, bbox_inches="tight")
print(f"\nPlot saved to : {plot_path.resolve()}")

# ---------------------------------------------------------------------------
# 6. HTML report
# ---------------------------------------------------------------------------
report_path = TUTORIAL_DIR / "case_1_report.html"
create_html_report(
    raw,
    gt,
    pred,
    result,
    output_path=report_path,
    target_label="blink",
)
print(f"HTML report   : {report_path.resolve()}")
