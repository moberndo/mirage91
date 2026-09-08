"""Shared epoching. One design decision this file encodes: we epoch ONCE per
session with a WIDE window, and let each feature extractor crop out the
sub-window it actually needs (e.g. bandpower/CSP want 1-3s post-cue, ERDS
additionally wants a pre-cue baseline). That's what lets bandpower, CSP,
FBCSP, Riemannian and ERDS features all share one epoching pass instead of
each re-epoching the raw data with their own tmin/tmax, which is what the two
original scripts did independently (and part of why they couldn't be merged
without modification).

ASSUMPTION (flagged for you to check): DEFAULT_TMIN/TMAX/BASELINE below are
my best guess at values that satisfy every feature extractor:
- bandpower/CSP (feature_extraction.py) want samples from 1.0-3.0s post-cue
- ERDS (train_classifier_rest_vs_movement.py) wants a baseline window,
  defaulting to (-2.0, 0.0), plus a post-cue window up to 3.5s
So the epoch needs to span at least -2.0 to 3.5s. Adjust if your paradigm's
cue timing / trial length differs.
"""

from __future__ import annotations

import mne

DEFAULT_TMIN = -2.0
DEFAULT_TMAX = 4.0
DEFAULT_BASELINE = None  # no MNE-level baseline correction; ERDSFeatures does
                          # its own baseline-relative normalization internally


def make_epochs(
    raw: mne.io.BaseRaw,
    events,
    event_id: dict,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    baseline=DEFAULT_BASELINE,
) -> mne.Epochs:
    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )
