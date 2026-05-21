"""Per-epoch blink visualization in an MNE HTML report.

Creates one figure per blink event (TP, FP, FN) showing ±pad_s of signal
context around the event window.  The report is written as a self-contained
HTML file via :class:`mne.Report`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from blink_evaluation.channel_scoring import ChannelEvaluationResult
from blink_evaluation.types import EvaluationResult

matplotlib.use("Agg")

_TABLE_STYLE = "border-collapse:collapse;font-family:monospace;font-size:13px;"
_TH_STYLE = "border:1px solid #aaa;padding:5px 12px;background:#f0f0f0;text-align:left;"
_TD_STYLE = "border:1px solid #aaa;padding:5px 12px;"
_TD_NUM_STYLE = "border:1px solid #aaa;padding:5px 12px;text-align:right;"


def _html_table(headers: list[str], rows: list[list], *, numeric_cols: set[int] | None = None) -> str:
    numeric_cols = numeric_cols or set()
    ths = "".join(f"<th style='{_TH_STYLE}'>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        tds = "".join(
            f"<td style='{_TD_NUM_STYLE if i in numeric_cols else _TD_STYLE}'>{cell}</td>"
            for i, cell in enumerate(row)
        )
        body += f"<tr>{tds}</tr>"
    return f"<table style='{_TABLE_STYLE}'><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _plot_blink_context(
    signal: np.ndarray,
    sfreq: float,
    epoch_duration: float,
    focus_onset_s: float,
    focus_duration_s: float,
    focus_color: str,
    focus_label: str,
    gt_spans: list[tuple[float, float]],
    all_pred_spans: list[tuple[float, float, str]],
    epoch_index: int,
    channel: str,
    pad_s: float,
) -> plt.Figure:
    """Plot a signal context window centred on one blink event.

    Parameters
    ----------
    signal:
        1-D filtered signal array for the epoch.
    sfreq:
        Sampling frequency in Hz.
    epoch_duration:
        Epoch duration in seconds (used to clip the window).
    focus_onset_s:
        Onset of the focus event in epoch-relative seconds.
    focus_duration_s:
        Duration of the focus event in seconds.
    focus_color:
        Matplotlib colour for the focus span (green=TP, red=FP, orange=FN).
    focus_label:
        Span legend label (e.g. ``"TP"``, ``"FP"``, ``"FN (missed GT)"``).
    gt_spans:
        List of (onset_s, duration_s) for all GT blinks in this epoch.
    all_pred_spans:
        List of (onset_s, duration_s, label) for every predicted blink in
        this epoch so neighbouring detections are visible in context.
    epoch_index:
        Epoch number shown in the title.
    channel:
        Channel name shown in the title and legend.
    pad_s:
        Seconds of context before and after the focus event.

    Returns
    -------
    plt.Figure
    """
    t_arr = np.arange(len(signal)) / sfreq
    t_win_start = max(0.0, focus_onset_s - pad_s)
    t_win_end = min(epoch_duration, focus_onset_s + focus_duration_s + pad_s)

    mask = (t_arr >= t_win_start) & (t_arr <= t_win_end)
    t_plot = t_arr[mask]
    s_plot = signal[mask]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_plot, s_plot, color="k", linewidth=0.8, label=channel)

    # Overlay all predicted blinks; use equal alpha so multiple TP/FP spans in
    # the same window are all clearly visible.  Focus event is slightly stronger.
    _PRED_COLORS = {"TP": "green", "FP": "red", "FN (missed GT)": "orange"}
    label_shown: set[str] = set()
    for p_onset, p_dur, p_lbl in all_pred_spans:
        if p_onset + p_dur < t_win_start or p_onset > t_win_end:
            continue
        is_focus = (
            abs(p_onset - focus_onset_s) < 1e-6 and abs(p_dur - focus_duration_s) < 1e-6
        )
        alpha = 0.42 if is_focus else 0.30
        color = _PRED_COLORS.get(p_lbl, "orange")
        lbl = p_lbl if p_lbl not in label_shown else None
        if lbl is not None:
            label_shown.add(lbl)
        ax.axvspan(p_onset, p_onset + p_dur, alpha=alpha, color=color, label=lbl)

    gt_labeled = False
    for gt_onset, gt_dur in gt_spans:
        if gt_onset + gt_dur < t_win_start or gt_onset > t_win_end:
            continue
        lbl = "GT" if not gt_labeled else None
        ax.axvspan(gt_onset, gt_onset + gt_dur, alpha=0.25, color="royalblue", label=lbl)
        gt_labeled = True

    ax.set_xlim(t_win_start, t_win_end)
    ax.set_xlabel("Time in epoch (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Epoch {epoch_index} | {focus_label} | Ch: {channel} | "
        f"onset={focus_onset_s:.3f}s  dur={focus_duration_s:.3f}s"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_blink_epoch_report(
    scored: ChannelEvaluationResult,
    df_masterlist: pd.DataFrame,
    epoch_duration: float,
    output_path: Path | str,
    *,
    pad_s: float = 0.5,
    target_label: str = "blink",
    max_figures_per_file: int = 300,
    csv_path: Path | str | None = None,
    sync_offset_s: float = 0.0,
) -> list[Path]:
    """Create MNE HTML report(s) with one figure per blink event.

    Each figure shows ±``pad_s`` seconds of signal context around one blink.
    All TP/FP/FN spans visible within the same window are rendered at equal
    opacity so multiple annotations are always clearly visible.  Colour coding:

    - **green** — true positive (TP)
    - **red** — false positive (FP)
    - **orange** — missed GT blink (FN)
    - **royalblue** (faint) — ground-truth span reference

    When the total number of figures exceeds *max_figures_per_file*, the output
    is split into multiple files named ``{stem}_part01.html``,
    ``{stem}_part02.html``, etc.  Every file contains the full statistics
    summary so each part is self-contained.

    Parameters
    ----------
    scored:
        Output of :func:`~blink_evaluation.channel_scoring.evaluate_channels`.
    df_masterlist:
        Event masterlist DataFrame (one row per TP/FP/FN event) as returned by
        :func:`~blink_evaluation.prediction_annotation.build_events_masterlist_df`.
        Required columns: ``status``, ``onset_gt``, ``onset_pred``,
        ``duration_gt``, ``duration_pred``.  Row order is preserved in the
        report — pre-sort by a combined onset column for chronological figures.
    epoch_duration:
        Duration of each epoch in seconds.
    output_path:
        Base path for the saved HTML file(s).
    pad_s:
        Seconds of context before and after each blink.  Default 0.5.
    target_label:
        Annotation label used during evaluation.
    max_figures_per_file:
        Maximum number of blink figures per HTML file.  Default 300.
    csv_path:
        Path of the ground-truth CSV file picked for this evaluation.
        Displayed in the report summary alongside ``sync_offset_s``.
    sync_offset_s:
        Time shift (in seconds) applied to the CSV annotations before
        evaluation (positive = annotations shifted later).  Displayed in
        the report summary alongside ``csv_path``.

    Returns
    -------
    list[Path]
        Paths of all HTML files written (one or more).

    Raises
    ------
    ValueError
        When *scored* has no best-channel result or ``signal_by_epoch`` is empty.
    """
    if (
        scored.best_channel_result is None
        or scored.best_eval_result is None
        or scored.best_predicted is None
    ):
        raise ValueError("scored has no best channel result")

    cr = scored.best_channel_result
    channel = scored.best_channel or ""
    signal_by_epoch: dict[int, np.ndarray] = cr["signal_by_epoch"]
    result: EvaluationResult = scored.best_eval_result

    if not signal_by_epoch:
        raise ValueError("signal_by_epoch is empty; cannot derive sfreq")
    first_sig = next(iter(signal_by_epoch.values()))
    sfreq: float = first_sig.size / epoch_duration

    # -- Build GT spans per epoch from df_masterlist (TP and FN rows have onset_gt)
    gt_by_epoch: dict[int, list[tuple[float, float]]] = {}
    for _, row in df_masterlist.iterrows():
        if pd.notna(row["onset_gt"]) and pd.notna(row["duration_gt"]):
            ep = int(float(row["onset_gt"]) // epoch_duration)
            rel = float(row["onset_gt"]) - ep * epoch_duration
            gt_by_epoch.setdefault(ep, []).append((rel, float(row["duration_gt"])))

    # -- Build predicted spans per epoch from df_masterlist
    pred_spans_by_epoch: dict[int, list[tuple[float, float, str]]] = {}
    for _, row in df_masterlist.iterrows():
        status = str(row["status"])
        if status in ("tp", "fp") and pd.notna(row["onset_pred"]) and pd.notna(row["duration_pred"]):
            ep = int(float(row["onset_pred"]) // epoch_duration)
            rel = float(row["onset_pred"]) - ep * epoch_duration
            lbl = "TP" if status == "tp" else "FP"
            pred_spans_by_epoch.setdefault(ep, []).append((rel, float(row["duration_pred"]), lbl))
        elif status == "fn" and pd.notna(row["onset_gt"]) and pd.notna(row["duration_gt"]):
            ep = int(float(row["onset_gt"]) // epoch_duration)
            rel = float(row["onset_gt"]) - ep * epoch_duration
            pred_spans_by_epoch.setdefault(ep, []).append((rel, float(row["duration_gt"]), "FN (missed GT)"))

    # -- Build flat list of figure specs from df_masterlist (row order = figure order)
    specs: list[dict] = []
    for _, row in df_masterlist.iterrows():
        status = str(row["status"])
        if status == "fn":
            if pd.isna(row["onset_gt"]):
                continue
            ep = int(float(row["onset_gt"]) // epoch_duration)
            if signal_by_epoch.get(ep) is None:
                continue
            rel_onset = float(row["onset_gt"]) - ep * epoch_duration
            specs.append({
                "ep_idx": ep,
                "focus_onset": rel_onset,
                "focus_dur": float(row["duration_gt"]),
                "color": "orange",
                "label": "FN (missed GT)",
                "gt_spans": gt_by_epoch.get(ep, []),
                "all_pred_spans": pred_spans_by_epoch.get(ep, []),
                "fig_title": f"Ep {ep:04d} | FN | {rel_onset:.3f}s",
                "tag": "FN",
            })
        else:
            if pd.isna(row["onset_pred"]):
                continue
            ep = int(float(row["onset_pred"]) // epoch_duration)
            if signal_by_epoch.get(ep) is None:
                continue
            rel_onset = float(row["onset_pred"]) - ep * epoch_duration
            color = "green" if status == "tp" else "red"
            label = "TP" if status == "tp" else "FP"
            specs.append({
                "ep_idx": ep,
                "focus_onset": rel_onset,
                "focus_dur": float(row["duration_pred"]),
                "color": color,
                "label": label,
                "gt_spans": gt_by_epoch.get(ep, []),
                "all_pred_spans": pred_spans_by_epoch.get(ep, []),
                "fig_title": f"Ep {ep:04d} | {label} | {rel_onset:.3f}s",
                "tag": label,
            })

    # -- Build summary HTML (shared across all parts) --------------------------
    em = result.event_metrics
    sm = result.sample_metrics

    recording_info_table = _html_table(
        ["Field", "Value"],
        [
            ["Ground-truth CSV", str(csv_path) if csv_path is not None else "—"],
            ["Sync offset (s)", f"{sync_offset_s:+.3f}"],
        ],
    )

    event_table = _html_table(
        ["Metric", "Value"],
        [
            ["Best channel", channel],
            ["GT blinks (total)", em.tp + em.fn],
            ["Detected blinks", int(df_masterlist["status"].isin(["tp", "fp"]).sum())],
            ["TP (events)", em.tp],
            ["FP (events)", em.fp],
            ["FN (events)", em.fn],
            ["TN (samples)", sm.tn],
            ["Precision", f"{em.precision:.4f}"],
            ["Recall", f"{em.recall:.4f}"],
            ["F1", f"{em.f1:.4f}"],
            ["Mean IoU (raw)", f"{em.mean_iou_raw:.4f}"],
            ["Mean IoU (expanded)", f"{em.mean_iou_expanded:.4f}"],
        ],
        numeric_cols={1},
    )
    sample_table = _html_table(
        ["Metric", "Value"],
        [
            ["TP (samples)", sm.tp],
            ["FP (samples)", sm.fp],
            ["FN (samples)", sm.fn],
            ["TN (samples)", sm.tn],
            ["Accuracy", f"{sm.accuracy:.4f}"],
            ["Precision", f"{sm.precision:.4f}"],
            ["Recall", f"{sm.recall:.4f}"],
            ["F1", f"{sm.f1:.4f}"],
            ["Macro Precision", f"{sm.macro_precision:.4f}"],
            ["Macro Recall", f"{sm.macro_recall:.4f}"],
            ["Macro F1", f"{sm.macro_f1:.4f}"],
            ["Micro Precision", f"{sm.micro_precision:.4f}"],
            ["Micro Recall", f"{sm.micro_recall:.4f}"],
            ["Micro F1", f"{sm.micro_f1:.4f}"],
        ],
        numeric_cols={1},
    )
    lane_rows = [
        [
            row["channel"],
            int(row["tp"]),
            int(row["fp"]),
            int(row["fn"]),
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1']:.4f}",
            int(row["mapped_candidate_count"]),
        ]
        for _, row in scored.lane_summary.iterrows()
    ]
    lane_table = _html_table(
        ["Channel", "TP", "FP", "FN", "Precision", "Recall", "F1", "Detections"],
        lane_rows,
        numeric_cols={1, 2, 3, 4, 5, 6, 7},
    )
    summary_html = (
        "<h3>Recording Info</h3>" + recording_info_table
        + "<h3 style='margin-top:18px'>Event-Level Metrics</h3>" + event_table
        + "<h3 style='margin-top:18px'>Sample-Level Metrics</h3>" + sample_table
        + "<h3 style='margin-top:18px'>Per-Channel Lane Summary</h3>" + lane_table
    )

    # -- Split specs into chunks and write one report per chunk ----------------
    chunks = [
        specs[i: i + max_figures_per_file]
        for i in range(0, max(len(specs), 1), max_figures_per_file)
    ]
    n_parts = len(chunks)
    base = Path(output_path)
    saved_paths: list[Path] = []

    for part_idx, chunk in enumerate(chunks):
        if n_parts == 1:
            part_path = base
            part_label = ""
        else:
            part_path = base.with_stem(f"{base.stem}_part{part_idx + 1:02d}")
            part_label = f" (part {part_idx + 1}/{n_parts})"

        report = mne.Report(title=f"Blink Epoch Report — {channel}{part_label}")

        # Summary in every part so each file is self-contained
        part_summary = summary_html
        if n_parts > 1:
            nav_parts = []
            for p in range(n_parts):
                part_name = base.with_stem(f"{base.stem}_part{p + 1:02d}").name
                nav_parts.append(f"<a href='{part_name}'>Part {p + 1}</a>")
            nav_links = "&nbsp;&nbsp;".join(nav_parts)
            fig_start = part_idx * max_figures_per_file + 1
            fig_end = min((part_idx + 1) * max_figures_per_file, len(specs))
            part_summary = (
                f"<p><strong>Part {part_idx + 1} of {n_parts}</strong> — "
                f"figures {fig_start}–{fig_end} of {len(specs)} total"
                f"&nbsp;|&nbsp; {nav_links}</p>"
                + summary_html
            )

        report.add_html(part_summary, title="Summary", tags=("summary",))

        for spec in chunk:
            signal = signal_by_epoch[spec["ep_idx"]]
            fig = _plot_blink_context(
                signal,
                sfreq,
                epoch_duration,
                spec["focus_onset"],
                spec["focus_dur"],
                spec["color"],
                spec["label"],
                spec["gt_spans"],
                spec["all_pred_spans"],
                spec["ep_idx"],
                channel,
                pad_s,
            )
            report.add_figure(
                fig,
                title=spec["fig_title"],
                image_format="png",
                tags=(spec["tag"],),
            )
            plt.close(fig)

        report.save(str(part_path), overwrite=True)
        saved_paths.append(part_path)

    return saved_paths


__all__ = ["create_blink_epoch_report"]
