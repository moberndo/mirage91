"""Epoching around class-cue onsets.

Trial timing (from paradigm/bci_rotation_cross.psyexp and cross-checked
against the actual marker spacing in the recordings): a class cue is shown
for ~2.5s, followed by a ~4.15s pause before the next cue (~6.65s/trial
total). Defaults below keep tmin inside the preceding pause (safe rest
baseline) and tmax well short of the next cue.
"""

from __future__ import annotations

import mne
import numpy as np

from .io import EVENT_ID

CLASS_NAMES = {v: k for k, v in EVENT_ID.items()}

DEFAULT_TMIN = -2.0
DEFAULT_TMAX = 4.0
DEFAULT_BASELINE = (-2.0, 0.0)

# No default amplitude-based rejection: this data has no ICA/artifact removal
# yet (WP3 still open per reference/Plan.pdf), and un-cleaned motor-imagery
# EEG routinely has per-epoch peak-to-peak swings of 300-700uV (real muscle/
# movement artifact, not just noise) - a naive ~200uV threshold drops ~90% of
# trials, which defeats a report whose whole point is to show data quality.
# Peak-to-peak stats are surfaced in the report instead (see report.py).
DEFAULT_REJECT_UV: float | None = None


def make_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    baseline: tuple[float, float] = DEFAULT_BASELINE,
    reject_uv: float | None = DEFAULT_REJECT_UV,
) -> mne.Epochs:
    reject = dict(eeg=reject_uv * 1e-6) if reject_uv else None
    return mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        verbose=False,
    )
