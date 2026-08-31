"""ERDS map computation (event-related desynchronization/synchronization).

Mirrors reference/assignment5/skeleton_assignmnet_5_1.py's approach:
per-epoch multitaper TFR, percent-baseline normalized, plotted per channel.
https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.colors import TwoSlopeNorm

DEFAULT_FREQS = np.arange(4, 36)
DEFAULT_CHANNELS = ("C3", "Cz", "C4")
ERDS_VMIN, ERDS_VMAX = -1, 1.5


def compute_erds_tfr(
    epochs: mne.Epochs,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    freqs: np.ndarray = DEFAULT_FREQS,
    baseline: tuple[float, float] | None = None,
    decim: int = 4,
) -> mne.time_frequency.EpochsTFR:
    """Per-epoch multitaper TFR on `channels`, percent-baseline normalized."""
    picks = [ch for ch in channels if ch in epochs.ch_names]
    if not picks:
        raise ValueError(f"none of {channels} present in epochs ({epochs.ch_names})")

    if baseline is None:
        baseline = (epochs.tmin, 0.0)

    tfr = epochs.compute_tfr(
        method="multitaper",
        freqs=freqs,
        n_cycles=freqs,
        use_fft=True,
        return_itc=False,
        average=False,
        decim=decim,
        picks=picks,
        verbose=False,
    )
    tfr.apply_baseline(baseline, mode="percent", verbose=False)
    return tfr


def plot_erds_map(tfr: mne.time_frequency.BaseTFR, title: str) -> plt.Figure:
    """One row of per-channel ERDS maps, averaged across the TFR's epochs."""
    channels = tfr.ch_names
    cnorm = TwoSlopeNorm(vmin=ERDS_VMIN, vcenter=0, vmax=ERDS_VMAX)

    avg = tfr.average() if hasattr(tfr, "average") else tfr
    fig, axes = plt.subplots(
        1,
        len(channels) + 1,
        figsize=(3.2 * len(channels) + 1, 4),
        gridspec_kw={"width_ratios": [10] * len(channels) + [1]},
    )
    for ch, ax in enumerate(axes[:-1]):
        avg.plot([ch], cmap="RdBu", cnorm=cnorm, axes=ax, colorbar=False, show=False)
        ax.set_title(channels[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle(title)
    fig.tight_layout()
    return fig
