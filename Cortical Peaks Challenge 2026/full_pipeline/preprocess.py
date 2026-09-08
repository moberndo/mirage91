"""
Pipeline order:
1. notch + band-pass
2. common-average reference (CAR)
3. (optional, off by default) ICA fit + ICLabel rejection + apply
4. final re-reference: keep CAR, or switch to a surface Laplacian (CSD)
Artifact-epoch rejection (reject_artifact_epochs) happens downstream, after
epoching — see run_pipeline.py.
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

MAINS_FREQ_HZ = 50.0  # Hz Graz, Austria

#but not all that important since we dont use ICA and can't classify that specific without it 
NON_BRAIN_LABELS = {
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
}


def filter_raw(
    raw: mne.io.BaseRaw,
    notch_freq: float = MAINS_FREQ_HZ,
    highpass_freq: float = 1.0,
    lowpass_freq: float | None = 60.0,
    order: int = 4,
) -> mne.io.BaseRaw:
    """Notch + band-pass filter, CAUSAL (via scipy lfilter, not filtfilt).

    Changed from zero-phase to causal on purpose: this makes the offline
    filter response identical to what you'll get online. Zero-
    phase filtering (the previous version, and what MNE's raw.filter/
    notch_filter do by default) uses future samples via filtfilt — great for
    offline-only analysis, but it's literally impossible to replicate live,
    since a live sample stream has no "future" to filter with. This removes
    that offline/online mismatch entirely, at the cost of a small phase
    delay (inherent to causal filtering, unavoidable, and also present in
    your eventual online system regardless).
    """
    import numpy as np
    import scipy.signal as sps

    fs = raw.info["sfreq"]
    data = raw.get_data()

    b_notch, a_notch = sps.iirnotch(w0=notch_freq, Q=30, fs=fs)
    data = sps.lfilter(b_notch, a_notch, data, axis=-1)

    nyquist = fs / 2
    low = highpass_freq / nyquist
    if lowpass_freq is not None:
        high = lowpass_freq / nyquist
        b_bp, a_bp = sps.butter(order, [low, high], btype="band")
    else:
        b_bp, a_bp = sps.butter(order, low, btype="high")
    data = sps.lfilter(b_bp, a_bp, data, axis=-1)

    raw_filtered = mne.io.RawArray(data, raw.info.copy(), verbose=False)
    raw_filtered.set_annotations(raw.annotations)
    return raw_filtered

#not that important since we shouldn't use it
def fit_ica(raw: mne.io.BaseRaw, random_state: int = 67) -> mne.preprocessing.ICA:
    ica = mne.preprocessing.ICA(
        n_components=None,
        method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw, verbose=False)
    return ica


def label_and_reject_components(
    raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA
) -> pd.DataFrame:
    from mne_icalabel import label_components

    ic_labels = label_components(raw, ica, method="iclabel")
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]

    exclude = [idx for idx, label in enumerate(labels) if label in NON_BRAIN_LABELS]
    ica.exclude = exclude

    return pd.DataFrame(
        {
            "component": range(len(labels)),
            "predicted_label": labels,
            "probability": probs,
            "excluded": [idx in exclude for idx in range(len(labels))],
        }
    )


def apply_reference(raw: mne.io.BaseRaw, method: str = "car") -> mne.io.BaseRaw:
    if method == "car":
        return raw
    if method == "laplacian":
        return mne.preprocessing.compute_current_source_density(raw, verbose=False)
    raise ValueError(f"unknown reference method: {method!r} (expected 'car' or 'laplacian')")


def preprocess_run(
    raw: mne.io.BaseRaw,
    reference: str = "car",
    notch_freq: float = MAINS_FREQ_HZ,
    highpass_freq: float = 1.0,
    lowpass_freq: float | None = 40.0,
    random_state: int = 67,
    use_ica: bool = False,
) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA | None, pd.DataFrame | None]:
    """Full chain: filter -> CAR -> (optional ICA fit/reject/apply) -> final reference.

    use_ica defaults to False (see module docstring). Set True only to A/B
    against threshold-based rejection. Returns (clean_raw, ica_or_None,
    component_labels_df_or_None).
    """
    raw = filter_raw(raw, notch_freq, highpass_freq, lowpass_freq)
    raw.set_eeg_reference("average", verbose=False)

    if not use_ica:
        raw_clean = apply_reference(raw, method=reference)
        return raw_clean, None, None

    ica = fit_ica(raw, random_state=random_state)
    labels_df = label_and_reject_components(raw, ica)

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    raw_clean = apply_reference(raw_clean, method=reference)
    return raw_clean, ica, labels_df


# ---------------------------------------------------------------------------
# Fit-once / apply-only helpers for the eventual online pipeline.
# Not used by the offline benchmark below
# ---------------------------------------------------------------------------

def fit_and_save_ica(
    raw: mne.io.BaseRaw, save_path: str | Path, random_state: int = 67
) -> tuple[mne.preprocessing.ICA, pd.DataFrame]:
    """Fit ICA once on a calibration recording and persist it (unmixing
    matrix + rejected-component list) so it can be reused without refitting."""
    ica = fit_ica(raw, random_state=random_state)
    labels_df = label_and_reject_components(raw, ica)
    ica.save(save_path, overwrite=True, verbose=False)
    return ica, labels_df


def load_and_apply_ica(raw: mne.io.BaseRaw, load_path: str | Path) -> mne.io.BaseRaw:
    """Apply a previously-fit, previously-reviewed ICA solution to new data.
    No refitting, no component review — this is the online-safe operation."""
    ica = mne.preprocessing.read_ica(load_path, verbose=False)
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean


# ---------------------------------------------------------------------------
# ICA replacement: causal, online-safe artifact rejection at the epoch level.
# Satisfies TE-3 without the offline/online drift problem ICA has.
# ---------------------------------------------------------------------------

def reject_artifact_epochs(
    X: "object",  # np.ndarray, (n_epochs, n_channels, n_times)
    #very conservative thresholds, we dont want to throw away too much data since we have very little data
    ptp_threshold: float = 400e-6,
    grad_threshold: float = 25e-6,
    max_bad_channel_frac: float = 0.5,
) -> "object":
    """Flag epochs likely contaminated by blinks/muscle/movement artifacts,
    using only information inside that epoch's own window (so it behaves
    identically offline and online).

    this checks per-channel, then rejects the epoch only if more
    than max_bad_channel_frac of channels are bad. 

    Rejection is kept (not disabled) to satisfy competition rule
    TE-3, which requires some form of artifact handling.

    - ptp_threshold: max allowed peak-to-peak amplitude per channel (volts).
    - grad_threshold: max allowed sample-to-sample jump per channel (volts).
    - max_bad_channel_frac: epoch is rejected only if more than this
      fraction of channels exceed either threshold.

    Returns a boolean mask, shape (n_epochs,), True = keep.
    """
    import numpy as np

    X = np.asarray(X)
    ptp = X.max(axis=-1) - X.min(axis=-1)            # (n_epochs, n_channels)
    grad = np.abs(np.diff(X, axis=-1)).max(axis=-1)  # (n_epochs, n_channels)

    channel_bad = (ptp > ptp_threshold) | (grad > grad_threshold)
    bad_frac = channel_bad.mean(axis=-1)
    return bad_frac <= max_bad_channel_frac