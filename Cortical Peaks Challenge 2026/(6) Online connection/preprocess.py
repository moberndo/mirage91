"""Filtering, ICA artifact rejection, and re-referencing (CAR or Laplacian/CSD).

Pipeline order matters here:
1. notch + band-pass
2. common-average reference (CAR) - required before ICA: ICLabel was trained
   on extended-infomax ICA of average-referenced, 1-100 Hz data, and its
   accuracy is "negatively impacted" (per its own docstring) if you skip this.
3. fit ICA (picard, extended - a fast equivalent of extended infomax) and
   reject components via ICLabel.
4. apply ICA to remove the rejected components.
5. final re-reference: keep CAR, or switch to a surface Laplacian (CSD).
   CSD is reference-invariant (it's built from potential *differences*), so
   it's safe to apply after CAR + ICA without "undoing" step 2 first.

On ICA thresholds: reference/eeg_drift/preprocess.py's ICLABEL_THRESHOLDS were
manually tuned by that project's supervisor against one specific subject's
64-channel PhysioNet-style recordings (see reference/scripts/run_ica_review.py
+ derive_thresholds.py) - they have no basis for this 32-channel mirage91
montage/subject and would just be guessing dressed up as precision. Instead
this uses ICLabel's own documented default: reject a component if its
*predicted* (argmax) label is a non-brain artifact class, no per-class
probability threshold. "other" is kept deliberately - it means ICLabel isn't
confident enough to call it an artifact, so treating it as one would be
guessing in the opposite direction. If a pilot/supervisor later reviews
excluded components (see the "ICA components" section in each report) and
disagrees case-by-case, that's the point at which per-class thresholds
tuned to this montage would actually mean something - not before.
"""

from __future__ import annotations

import mne
import pandas as pd

MAINS_FREQ_HZ = 50.0  # Graz, Austria

# ICLabel's 7 classes; components predicted as one of these are excluded.
# "brain" and "other" are kept - see module docstring.
NON_BRAIN_LABELS = {"muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"}


def filter_raw(
    raw: mne.io.BaseRaw,
    notch_freq: float = MAINS_FREQ_HZ,
    highpass_freq: float = 1.0,
    lowpass_freq: float | None = 100.0,
) -> mne.io.BaseRaw:
    """Notch + band-pass filter. Zero-phase IIR (Butterworth), matching the
    reference eeg_drift.preprocess convention (reference/eeg_drift/preprocess.py)."""
    raw = raw.copy()

    raw.notch_filter(
        freqs=[notch_freq],
        notch_widths=2.0,
        method="iir",
        iir_params=dict(order=2, ftype="butter"),
        verbose=False,
    )
    raw.filter(
        l_freq=highpass_freq,
        h_freq=lowpass_freq,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        verbose=False,
    )
    return raw


def fit_ica(raw: mne.io.BaseRaw, random_state: int = 42) -> mne.preprocessing.ICA:
    """Fit ICA using Picard with extended-Infomax-equivalent settings."""
    ica = mne.preprocessing.ICA(
        n_components=None,
        method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw, verbose=False)
    return ica


def label_and_reject_components(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA) -> pd.DataFrame:
    """Run ICLabel and mark non-brain/other components as bad (ica.exclude).

    Returns a per-component table (label, probability, excluded) for the
    report - the ICA section is meant to be human-checkable, not a black box.
    """
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
    """Final re-reference: 'car' (already applied pre-ICA, kept as-is) or
    'laplacian' (surface Laplacian / current source density - sharpens
    focal, lateralized activity like C3/C4 motor ERD that CAR tends to blur
    with sparse coverage; see reference/Plan.pdf's WP3 re-referencing note)."""
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
    lowpass_freq: float | None = 100.0,
    random_state: int = 42,
) -> tuple[mne.io.BaseRaw, mne.preprocessing.ICA, pd.DataFrame]:
    """Full chain: filter -> CAR -> ICA fit/reject/apply -> final reference.

    Returns (clean_raw, ica, component_labels_df).
    """
    raw = filter_raw(raw, notch_freq, highpass_freq, lowpass_freq)
    raw.set_eeg_reference("average", verbose=False)

    ica = fit_ica(raw, random_state=random_state)
    labels_df = label_and_reject_components(raw, ica)

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    raw_clean = apply_reference(raw_clean, method=reference)
    return raw_clean, ica, labels_df
