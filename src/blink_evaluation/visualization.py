from __future__ import annotations

import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne

from blink_evaluation.io import annotations_to_events
from blink_evaluation.types import AnnotationEvent, EvaluationResult, Match

matplotlib.use("Agg")

_TABLE_STYLE = (
    "border-collapse:collapse;font-family:monospace;font-size:13px;"
)
_TH_STYLE = "border:1px solid #aaa;padding:4px 10px;background:#f0f0f0;text-align:left;"
_TD_STYLE = "border:1px solid #aaa;padding:4px 10px;"


def _html_table(headers: list[str], rows: list[list]) -> str:
    ths = "".join(f"<th style='{_TH_STYLE}'>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        tds = "".join(f"<td style='{_TD_STYLE}'>{cell}</td>" for cell in row)
        body += f"<tr>{tds}</tr>"
    return f"<table style='{_TABLE_STYLE}'><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>"


def _pick_channel_index(raw: mne.io.BaseRaw, picks: str | list[str] | None) -> int:
    if picks is None:
        return 0
    if isinstance(picks, list):
        picks = picks[0]
    try:
        return raw.ch_names.index(picks)
    except ValueError:
        return 0


def plot_annotation_comparison(
    raw: mne.io.BaseRaw,
    ground_truth: mne.Annotations,
    prediction: mne.Annotations,
    result: EvaluationResult,
    *,
    target_label: str = "blink",
    start: float | None = None,
    stop: float | None = None,
    picks: str | list[str] | None = None,
    show: bool = False,
) -> plt.Figure:
    ch_idx = _pick_channel_index(raw, picks)
    ch_name = raw.ch_names[ch_idx]

    data_2d, times = raw[ch_idx, :]
    signal = data_2d[0]

    t_start = start if start is not None else float(times[0])
    t_stop = stop if stop is not None else float(times[-1])
    mask = (times >= t_start) & (times <= t_stop)
    times_plot = times[mask]
    signal_plot = signal[mask]

    gt_events = annotations_to_events(ground_truth, target_label)
    pred_events = annotations_to_events(prediction, target_label)

    gt_by_idx = {e.index: e for e in gt_events}
    pred_by_idx = {e.index: e for e in pred_events}

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(times_plot, signal_plot, color="k", linewidth=0.5, alpha=0.6, label=ch_name)

    # TP spans (green)
    tp_labeled = False
    for m in result.true_positives:
        gt = gt_by_idx.get(m.gt_index)
        pred = pred_by_idx.get(m.pred_index)
        if gt is None or pred is None:
            continue
        span_start = min(m.onset_gt, m.onset_pred)
        span_end = max(m.onset_gt + m.duration_gt, m.onset_pred + m.duration_pred)
        label = "TP" if not tp_labeled else None
        ax.axvspan(span_start, span_end, alpha=0.25, color="green", label=label)
        tp_labeled = True

    # FP spans (red)
    fp_labeled = False
    for fp in result.false_positives:
        label = "FP" if not fp_labeled else None
        ax.axvspan(fp.onset, fp.onset + fp.duration_pred, alpha=0.35, color="red", label=label)
        fp_labeled = True

    # FN spans (blue)
    fn_labeled = False
    for fn in result.false_negatives:
        label = "FN" if not fn_labeled else None
        ax.axvspan(fn.onset, fn.onset + fn.duration_gt, alpha=0.35, color="royalblue", label=label)
        fn_labeled = True

    em = result.event_metrics
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Blink Annotation Comparison — TP={em.tp}  FP={em.fp}  FN={em.fn} | "
        f"P={em.precision:.3f}  R={em.recall:.3f}  F1={em.f1:.3f}"
    )
    ax.set_xlim(t_start, t_stop)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def create_html_report(
    raw: mne.io.BaseRaw,
    ground_truth: mne.Annotations,
    prediction: mne.Annotations,
    result: EvaluationResult,
    output_path: str | Path,
    *,
    target_label: str = "blink",
    picks: str | list[str] | None = None,
) -> None:
    report = mne.Report(title="Blink Evaluation Report")

    # --- Event metrics table ---
    em = result.event_metrics
    event_rows = [
        ["TP", em.tp],
        ["FP", em.fp],
        ["FN", em.fn],
        ["Precision", f"{em.precision:.6f}"],
        ["Recall", f"{em.recall:.6f}"],
        ["F1", f"{em.f1:.6f}"],
        ["Mean IoU (raw)", f"{em.mean_iou_raw:.6f}"],
        ["Mean IoU (expanded)", f"{em.mean_iou_expanded:.6f}"],
    ]
    event_html = "<h3>Event-Level Metrics</h3>" + _html_table(["Metric", "Value"], event_rows)
    report.add_html(event_html, title="Event Metrics")

    # --- Sample metrics table ---
    sm = result.sample_metrics
    sample_rows = [
        ["TP (samples)", sm.tp],
        ["FP (samples)", sm.fp],
        ["FN (samples)", sm.fn],
        ["TN (samples)", sm.tn],
        ["Accuracy", f"{sm.accuracy:.6f}"],
        ["Precision", f"{sm.precision:.6f}"],
        ["Recall", f"{sm.recall:.6f}"],
        ["F1", f"{sm.f1:.6f}"],
        ["Micro Precision", f"{sm.micro_precision:.6f}"],
        ["Micro Recall", f"{sm.micro_recall:.6f}"],
        ["Micro F1", f"{sm.micro_f1:.6f}"],
        ["Macro Precision", f"{sm.macro_precision:.6f}"],
        ["Macro Recall", f"{sm.macro_recall:.6f}"],
        ["Macro F1", f"{sm.macro_f1:.6f}"],
    ]
    sample_html = "<h3>Sample-Wise Metrics</h3>" + _html_table(["Metric", "Value"], sample_rows)
    report.add_html(sample_html, title="Sample Metrics")

    # --- Match table ---
    match_rows = [
        [
            m.gt_index,
            m.pred_index,
            f"{m.iou_raw:.4f}",
            f"{m.iou_expanded:.4f}",
            f"{m.peak_delta:.4f}" if m.peak_delta is not None else "N/A",
        ]
        for m in result.matches
    ]
    match_html = "<h3>Matches (True Positives)</h3>" + _html_table(
        ["GT Index", "Pred Index", "IoU Raw", "IoU Expanded", "Peak Delta"],
        match_rows,
    )
    report.add_html(match_html, title="Matches")

    # --- FP table ---
    fp_rows = [
        [fp.index, f"{fp.onset:.6f}", f"{fp.duration_pred:.6f}", fp.description]
        for fp in result.false_positives
    ]
    fp_html = "<h3>False Positives</h3>" + (
        _html_table(["Index", "Onset", "Duration", "Label"], fp_rows) if fp_rows else "<p>None</p>"
    )
    report.add_html(fp_html, title="False Positives")

    # --- FN table ---
    fn_rows = [
        [fn.index, f"{fn.onset:.6f}", f"{fn.duration_gt:.6f}", fn.description]
        for fn in result.false_negatives
    ]
    fn_html = "<h3>False Negatives</h3>" + (
        _html_table(["Index", "Onset", "Duration", "Label"], fn_rows) if fn_rows else "<p>None</p>"
    )
    report.add_html(fn_html, title="False Negatives")

    # --- Time-series comparison plot ---
    fig = plot_annotation_comparison(
        raw,
        ground_truth,
        prediction,
        result,
        target_label=target_label,
        picks=picks,
    )
    report.add_figure(fig, title="Annotation Comparison Plot", image_format="png")
    plt.close(fig)

    report.save(str(output_path), overwrite=True)
