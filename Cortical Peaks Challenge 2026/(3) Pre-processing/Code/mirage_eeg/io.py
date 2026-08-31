"""Load raw EEG + paradigm markers from a LabRecorder .xdf recording.

Written against the "Cortical Peaks Challenge 2026" recordings: a
'BrainVision RDA' EEG stream (32 EEG channels + x_dir/y_dir/z_dir
accelerometer channels, 500 Hz) and a 'paradigm' Markers stream pushing
one of left_hand/right_hand/feet/rest at cue onset and 'pause' at the
inter-trial interval onset (see paradigm/bci_rotation_cross.psyexp in the
mirage91 repo). Only the class-cue labels are trial onsets; 'pause',
'post_block', etc. are not epoched.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pyxdf

EEG_STREAM_NAME = "BrainVision RDA"
MARKER_STREAM_NAME = "paradigm"
ACCEL_CHANNELS = ("x_dir", "y_dir", "z_dir")
EVENT_ID = {"left_hand": 1, "right_hand": 2, "feet": 3, "rest": 4}


def load_xdf_run(path: Path) -> tuple[mne.io.RawArray, np.ndarray]:
    """Load one .xdf recording into an MNE Raw (EEG only) + MNE events array."""
    streams, _ = pyxdf.load_xdf(str(path))
    eeg_stream = next(s for s in streams if s["info"]["name"][0] == EEG_STREAM_NAME)
    marker_stream = next(s for s in streams if s["info"]["name"][0] == MARKER_STREAM_NAME)

    ch_entries = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ch_names = [c["label"][0] for c in ch_entries]
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])

    # BrainVision RDA streams EEG channels in microvolts (per-channel 'unit'
    # in the stream desc); MNE expects volts internally.
    data = np.asarray(eeg_stream["time_series"], dtype=float).T  # (n_channels, n_samples)
    eeg_mask = np.array([name not in ACCEL_CHANNELS for name in ch_names])
    data[eeg_mask] *= 1e-6

    ch_types = ["misc" if name in ACCEL_CHANNELS else "eeg" for name in ch_names]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.drop_channels(list(ACCEL_CHANNELS))

    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False, on_missing="warn")

    labels = [str(sample[0]).strip() for sample in marker_stream["time_series"]]
    marker_ts = np.asarray(marker_stream["time_stamps"])
    eeg_t0 = np.asarray(eeg_stream["time_stamps"])[0]

    events = []
    for label, ts in zip(labels, marker_ts):
        if label not in EVENT_ID:
            continue
        sample = int(round((ts - eeg_t0) * sfreq))
        if 0 <= sample < raw.n_times:
            events.append((sample, 0, EVENT_ID[label]))

    if not events:
        raise ValueError(f"{path}: no class-cue markers found in '{MARKER_STREAM_NAME}' stream")

    events_arr = np.array(sorted(events), dtype=int)
    return raw, events_arr


def find_xdf_runs(recordings_dir: Path) -> list[Path]:
    """All .xdf files in a recordings directory, sorted by name."""
    return sorted(recordings_dir.glob("*.xdf"))
