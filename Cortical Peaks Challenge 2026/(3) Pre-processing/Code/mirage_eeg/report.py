"""Build per-run and combined HTML QC/ERDS reports (mne.Report)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from .epochs import CLASS_NAMES
from .erds import DEFAULT_CHANNELS, DEFAULT_FREQS, compute_erds_tfr, plot_erds_map


def _drop_log_summary(epochs: mne.Epochs) -> tuple[int, int, dict[str, int]]:
    n_total = len(epochs.drop_log)
    n_kept = len(epochs)
    n_dropped = n_total - n_kept
    reasons: dict[str, int] = {}
    for entry in epochs.drop_log:
        for reason in entry:
            reasons[reason] = reasons.get(reason, 0) + 1
    return n_total, n_dropped, reasons


def _class_counts(epochs: mne.Epochs) -> dict[str, int]:
    return {CLASS_NAMES[eid]: int(np.sum(epochs.events[:, 2] == eid)) for eid in CLASS_NAMES}


def _trial_counts_html(epochs: mne.Epochs) -> str:
    df = pd.DataFrame([_class_counts(epochs)])
    n_total, n_dropped, reasons = _drop_log_summary(epochs)
    html = f"<p>{n_dropped} of {n_total} epochs dropped"
    if reasons:
        html += " (" + ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items())) + ")"
    html += f". {len(epochs)} kept, per class:</p>"
    html += df.to_html(index=False)
    return html


def _ptp_qc_html(epochs: mne.Epochs) -> str:
    """Per-epoch peak-to-peak amplitude, as an informational artifact indicator
    (no rejection applied - see mirage_eeg.epochs.DEFAULT_REJECT_UV)."""
    data = epochs.get_data(picks="data", verbose=False)
    ptp_uv = (data.max(axis=2) - data.min(axis=2)) * 1e6
    per_epoch_max = ptp_uv.max(axis=1)
    pct = np.percentile(per_epoch_max, [0, 25, 50, 75, 90, 100])
    n_over_500 = int((per_epoch_max > 500).sum())
    return (
        "<p>Per-epoch max peak-to-peak amplitude (any channel), informational only - "
        "no epochs were dropped for this (see Overview). High values suggest movement/"
        "muscle/blink artifact worth ICA cleanup (WP3, still open).</p>"
        f"<p>Percentiles (0/25/50/75/90/100%), µV: "
        f"{', '.join(f'{v:.0f}' for v in pct)}. "
        f"{n_over_500} of {len(epochs)} epochs exceed 500µV.</p>"
    )


def _erds_section(
    report: mne.Report,
    epochs: mne.Epochs,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    freqs: np.ndarray = DEFAULT_FREQS,
    tags: tuple[str, ...] = ("erds",),
) -> None:
    for eid, class_name in CLASS_NAMES.items():
        class_epochs = epochs[class_name]
        if len(class_epochs) == 0:
            continue
        tfr = compute_erds_tfr(class_epochs, channels=channels, freqs=freqs)
        fig = plot_erds_map(tfr, title=f"ERDS: {class_name} (n={len(class_epochs)})")
        report.add_figure(
            fig,
            title=f"ERDS: {class_name}",
            caption=(
                f"Percent-baseline ERD/S ({', '.join(tfr.ch_names)}), "
                f"baseline={epochs.baseline}. Red = desynchronization (ERD, power "
                "decrease), blue = synchronization (ERS, power increase). Dotted "
                "line marks cue onset."
            ),
            tags=tags + (class_name,),
        )
        plt.close(fig)


def _overview_html(raw: mne.io.BaseRaw, epochs: mne.Epochs, n_components: int, n_excluded: int, reference: str) -> str:
    reject = epochs.reject.get("eeg") if epochs.reject else None
    reject_txt = f"{reject * 1e6:.0f}µV peak-to-peak epoch reject" if reject else "no epoch rejection"
    ref_txt = "common-average (CAR)" if reference == "car" else "surface Laplacian (CSD)"
    return (
        f"<p>{raw.n_times / raw.info['sfreq']:.1f}s recording at {raw.info['sfreq']:.0f} Hz, "
        f"{len(raw.ch_names)} EEG channels. Filtering: notch 50 Hz, band-pass, ICA "
        f"({n_excluded}/{n_components} components excluded via ICLabel - see ICA section below), "
        f"final reference: {ref_txt}. Beyond ICA, only a {reject_txt}.</p>"
    )


def _ica_section_html(labels_df: pd.DataFrame) -> str:
    excluded = labels_df[labels_df["excluded"]]
    html = (
        "<p>Components rejected if ICLabel's predicted (argmax) label is a non-brain "
        "artifact class - no per-class probability threshold (see mirage_eeg.preprocess "
        "docstring for why: the reference project's manually-tuned thresholds were fit to a "
        "different subject/montage and don't transfer here). 'other' is kept deliberately - "
        "not confident enough to call it an artifact.</p>"
    )
    html += f"<p>{len(excluded)} of {len(labels_df)} components excluded:</p>"
    html += labels_df.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    return html


def build_run_report(
    run_name: str,
    raw: mne.io.BaseRaw,
    epochs: mne.Epochs,
    events: np.ndarray,
    ica: mne.preprocessing.ICA,
    ica_labels_df: pd.DataFrame,
    reference: str,
    out_dir: Path,
    erds_channels: tuple[str, ...] = DEFAULT_CHANNELS,
    erds_freqs: np.ndarray = DEFAULT_FREQS,
) -> Path:
    report = mne.Report(title=f"EEG QC + ERDS - {run_name}")

    report.add_html(
        _overview_html(raw, epochs, ica.n_components_, len(ica.exclude), reference),
        title="Overview",
        tags=("overview",),
    )
    report.add_events(events, title="Class-cue markers", sfreq=raw.info["sfreq"], tags=("overview",))
    report.add_html(_trial_counts_html(epochs), title="Trial counts", tags=("overview",))
    report.add_html(_ptp_qc_html(epochs), title="Artifact amplitude (QC)", tags=("overview",))

    report.add_html(_ica_section_html(ica_labels_df), title="ICA components", tags=("ica",))
    if ica.exclude:
        figs = ica.plot_components(picks=ica.exclude, show=False)
        figs = figs if isinstance(figs, list) else [figs]
        for i, fig in enumerate(figs):
            report.add_figure(
                fig,
                title=f"Excluded component topographies{'' if len(figs) == 1 else f' ({i + 1}/{len(figs)})'}",
                caption="Topomaps of the excluded ICA components.",
                tags=("ica",),
            )
            plt.close(fig)

    report.add_html(
        "<p>Power spectral density across all channels, after ICA cleanup and "
        "re-referencing. The notch at 50 Hz confirms the mains filter worked.</p>",
        title="Cleaned data: what this shows",
        tags=("raw",),
    )
    report.add_raw(raw, title="Cleaned data overview", psd=True, butterfly=False, tags=("raw",))

    _erds_section(report, epochs, channels=erds_channels, freqs=erds_freqs)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_report.html"
    report.save(out_path, overwrite=True, open_browser=False, verbose=False)
    return out_path


def build_combined_report(
    runs: list[tuple[str, mne.Epochs, mne.preprocessing.ICA]],
    out_dir: Path,
    reference: str,
    erds_channels: tuple[str, ...] = DEFAULT_CHANNELS,
    erds_freqs: np.ndarray = DEFAULT_FREQS,
) -> Path:
    report = mne.Report(title=f"Combined EEG QC + ERDS ({len(runs)} runs)")

    ref_txt = "common-average (CAR)" if reference == "car" else "surface Laplacian (CSD)"
    rows = []
    for name, epochs, ica in runs:
        _, n_dropped, _ = _drop_log_summary(epochs)
        row = {
            "run": name,
            "kept": len(epochs),
            "dropped": n_dropped,
            "ica_excluded": f"{len(ica.exclude)}/{ica.n_components_}",
        }
        row.update(_class_counts(epochs))
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    report.add_html(
        f"<p>{len(runs)} run(s) combined: {', '.join(name for name, _, _ in runs)}. "
        f"ICA + {ref_txt} reference on each run (see per-run reports for excluded "
        f"component detail).</p>" + summary_df.to_html(index=False),
        title="Per-run summary",
        tags=("overview",),
    )

    combined_epochs = mne.concatenate_epochs([epochs for _, epochs, _ in runs])
    report.add_html(
        _trial_counts_html(combined_epochs), title="Combined trial counts", tags=("overview",)
    )
    report.add_html(_ptp_qc_html(combined_epochs), title="Artifact amplitude (QC)", tags=("overview",))
    _erds_section(report, combined_epochs, channels=erds_channels, freqs=erds_freqs, tags=("erds", "combined"))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_report.html"
    report.save(out_path, overwrite=True, open_browser=False, verbose=False)
    return out_path
