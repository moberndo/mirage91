"""Main where I run the programm in the necessary steps.
Done by Michael with Python 3.14.7. For packages refer to the imports"""

import matplotlib.pyplot as plt
import mne
import numpy as np
import pyxdf
from feature_extraction import (
    extract_bandpower_features,
    extract_csp_features,
)
from preprocess import preprocess_run
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================
# CONFIG — Just set True or False to enable/disable
# ============================================================

RUN_CSP = True
RUN_BANDPOWER = False
RUN_LDA = True
RUN_SVM = True

SESSION_FILES = [
    "recordings/session1_run1.xdf",
    "recordings/session2_run2.xdf",
    "recordings/untitled.xdf",
]

BANDPOWER_CLASS_CONFIGS = {
    "left_hand":  ("C3", 5, 10),
    "right_hand": ("C3", 5, 11),
    "feet":       ("C3", 5, 13),
    "rest":       ("C3", 5, 11),
}


# ============================================================
# loading the data
# ============================================================

def load_xdf_to_raw(filepath):
    streams, header = pyxdf.load_xdf(filepath)

    eeg_stream = None
    marker_stream = None
    for s in streams:
        stype = s['info']['type'][0]
        if stype == 'EEG':
            eeg_stream = s
        elif stype == 'Markers' and s['info']['name'][0] == 'paradigm':
            marker_stream = s

    if eeg_stream is None:
        raise ValueError("No EEG stream found")

    ch_names = [c['label'][0] for c in
                eeg_stream['info']['desc'][0]['channels'][0]['channel']]
    sfreq = float(eeg_stream['info']['nominal_srate'][0])

    ch_types = ['misc' if ch in ('x_dir', 'y_dir', 'z_dir') else 'eeg'
                for ch in ch_names]

    data = eeg_stream['time_series'].T
    data = data * 1e-6

    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.rename_channels({'FP2': 'Fp2'})

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')

    if marker_stream is not None:
        eeg_t0 = eeg_stream['time_stamps'][0]
        onsets = marker_stream['time_stamps'] - eeg_t0
        descriptions = [label[0].strip() for label in marker_stream['time_series']]
        annotations = mne.Annotations(onset=onsets, duration=0.0, description=descriptions)
        raw.set_annotations(annotations)

    return raw


def load_all_sessions():
    sessions = []
    for i, path in enumerate(SESSION_FILES, start=1):
        raw = load_xdf_to_raw(path)
        print(f"session{i} ---------------------------- loaded")
        print(raw)
        print(raw.annotations)
        sessions.append(raw)
    return sessions


def preprocess_all_sessions(sessions):
    preprocessed = []
    for i, raw in enumerate(sessions, start=1):
        raw_clean, ica, labels_df = preprocess_run(raw)
        preprocessed.append(raw_clean)
    return preprocessed


# ============================================================
# CSP feature extraction
# ============================================================

def run_csp(sessions_preprocessed):
    csp_results = []
    for i, raw in enumerate(sessions_preprocessed):
        features, y = extract_csp_features(raw)
        csp_results.append((features, y))
        print(f"session{i+1} CSP features: {features.shape}, labels: {y.shape}")

    all_features = np.concatenate([r[0] for r in csp_results])
    all_labels = np.concatenate([r[1] for r in csp_results])

    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, all_features, all_labels, cv=5)
    print(f"\nPooled CSP + LDA CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    return all_features, all_labels


# ============================================================
# BANDPOWER feature extraction
# ============================================================

def run_bandpower(sessions_preprocessed):
    bp_results = []
    for i, raw in enumerate(sessions_preprocessed):
        features, y = extract_bandpower_features(raw, BANDPOWER_CLASS_CONFIGS)
        bp_results.append((features, y))
        print(f"session{i+1} bandpower features: {features.shape}, labels: {y.shape}")

    all_features = np.concatenate([r[0] for r in bp_results])
    all_labels = np.concatenate([r[1] for r in bp_results])

    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, all_features, all_labels, cv=5)
    print(f"\nPooled Bandpower + LDA CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    return all_features, all_labels


# ============================================================
# LDA (to test)
# ============================================================

def run_lda(all_features, all_labels, label=""):
    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, all_features, all_labels, cv=5)
    print(f"\n{label} LDA CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")


# ============================================================
# SVM
# ============================================================

def run_svm(all_features, all_labels, label=""):
    svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))
    scores = cross_val_score(svm_pipeline, all_features, all_labels, cv=5)
    print(f"\n{label} SVM CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")


# ============================================================
# MAIN
# ============================================================

def main():
    sessions = load_all_sessions()
    sessions_preprocessed = preprocess_all_sessions(sessions)

    csp_features = csp_labels = None
    if RUN_CSP:
        csp_features, csp_labels = run_csp(sessions_preprocessed)

    bp_features = bp_labels = None
    if RUN_BANDPOWER:
        bp_features, bp_labels = run_bandpower(sessions_preprocessed)

    if RUN_LDA:
        if csp_features is not None:
            run_lda(csp_features, csp_labels, label="CSP")
        if bp_features is not None:
            run_lda(bp_features, bp_labels, label="Bandpower")

    if RUN_SVM:
        if csp_features is not None:
            run_svm(csp_features, csp_labels, label="CSP")
        if bp_features is not None:
            run_svm(bp_features, bp_labels, label="Bandpower")


if __name__ == "__main__":
    main()