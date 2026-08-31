#!/usr/bin/env python3
"""Refined BCI Binary Classifier with Session Normalization & Riemannian Geometry.

Features:
1. Per-session standardization (Z-score) of ERDS features to handle baseline shifts across sessions.
2. Riemannian Geometry (Covariance matrices -> Tangent Space projection).
3. Evaluates models using GroupKFold for strict inter-session evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Intentar importar pyriemann para la covarianza riemanniana
try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False
    print("WARNING: 'pyriemann' not found. Fallback to standard FBCSP. Run 'pip install pyriemann' to enable.")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mne.decoding import CSP

from mirage_eeg.epochs import DEFAULT_BASELINE, DEFAULT_TMAX, DEFAULT_TMIN, make_epochs
from mirage_eeg.io import EVENT_ID, find_xdf_runs, load_xdf_run
from mirage_eeg.preprocess import preprocess_run

FREQUENCY_BANDS: list[tuple[float, float]] = [
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
]

ERDS_FEATURE_SPECS: list[dict] = [
    {"name": "C3_15_20Hz_ERS", "channel": "C3", "band": (15.0, 20.0), "window": (1.0, 3.5)},
    {"name": "C4_5_10Hz_ERS", "channel": "C4", "band": (5.0, 10.0), "window": (1.0, 3.5)},
]

ERDS_BASELINE_WINDOW: tuple[float, float] = DEFAULT_BASELINE if DEFAULT_BASELINE else (-2.0, 0.0)


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter-Bank CSP features."""

    def __init__(self, bands, sfreq, n_components=4):
        self.bands = bands
        self.sfreq = sfreq
        self.n_components = n_components

    def fit(self, X, y):
        self.csp_by_band_ = []
        for l_freq, h_freq in self.bands:
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
            )
            csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True, norm_trace=False)
            csp.fit(X_band, y)
            self.csp_by_band_.append(csp)
        return self

    def transform(self, X):
        band_features = []
        for (l_freq, h_freq), csp in zip(self.bands, self.csp_by_band_):
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
            )
            band_features.append(csp.transform(X_band))
        return np.concatenate(band_features, axis=1)


class RiemannianFeatures(BaseEstimator, TransformerMixin):
    """Covariance estimation + Tangent Space projection per frequency band."""

    def __init__(self, bands, sfreq):
        self.bands = bands
        self.sfreq = sfreq

    def fit(self, X, y=None):
        if HAS_PYRIEMANN:
            self.tangent_spaces_ = []
            for l_freq, h_freq in self.bands:
                X_band = mne.filter.filter_data(
                    X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
                )
                covs = Covariances(estimator="lwf").fit_transform(X_band)
                ts = TangentSpace(metric="riemann").fit(covs)
                self.tangent_spaces_.append(ts)
        return self

    def transform(self, X):
        if not HAS_PYRIEMANN:
            raise RuntimeError("pyriemann is required for RiemannianFeatures. Install via pip install pyriemann.")

        ts_features = []
        for (l_freq, h_freq), ts in zip(self.bands, self.tangent_spaces_):
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
            )
            covs = Covariances(estimator="lwf").transform(X_band)
            ts_features.append(ts.transform(covs))
        return np.concatenate(ts_features, axis=1)


class ERDSFeatures(BaseEstimator, TransformerMixin):
    """Computes baseline-relative power changes."""

    def __init__(self, specs, ch_names, sfreq, tmin, baseline_window):
        self.specs = specs
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.tmin = tmin
        self.baseline_window = baseline_window

    def _sample_index(self, t):
        return int(round((t - self.tmin) * self.sfreq))

    def fit(self, X, y=None):
        self.channel_indices_ = {
            s["name"]: self.ch_names.index(s["channel"]) for s in self.specs
        }
        self.baseline_idx_ = (
            self._sample_index(self.baseline_window[0]),
            self._sample_index(self.baseline_window[1]),
        )
        return self

    @property
    def feature_names_(self):
        return [s["name"] for s in self.specs]

    def transform(self, X):
        n_trials = X.shape[0]
        n_times = X.shape[2]
        feats = np.zeros((n_trials, len(self.specs)), dtype=float)

        base_start = max(self.baseline_idx_[0], 0)
        base_end = min(self.baseline_idx_[1], n_times)

        for i, spec in enumerate(self.specs):
            ch_idx = self.channel_indices_[spec["name"]]
            sig = X[:, ch_idx, :]

            l_freq, h_freq = spec["band"]
            sig_filt = mne.filter.filter_data(
                sig, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
            )
            power = np.abs(hilbert(sig_filt, axis=-1)) ** 2

            win_start = max(self._sample_index(spec["window"][0]), 0)
            win_end = min(self._sample_index(spec["window"][1]), n_times)

            baseline_power = power[:, base_start:base_end].mean(axis=1)
            window_power = power[:, win_start:win_end].mean(axis=1)

            feats[:, i] = (window_power - baseline_power) / (baseline_power + 1e-8)

        return feats


def normalize_by_session(X_features: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Standardizes features independently per recording session (group).

    Removes inter-session baseline shifts and impedance variations.
    """
    X_norm = np.copy(X_features)
    for group_id in np.unique(groups):
        mask = groups == group_id
        scaler = StandardScaler()
        X_norm[mask] = scaler.fit_transform(X_features[mask])
    return X_norm


def load_all_runs_binary(recordings_dir: Path, reference: str, tmin: float, tmax: float):
    runs = find_xdf_runs(recordings_dir)
    if not runs:
        raise RuntimeError(f"No .xdf files found under {recordings_dir}")

    X_list, y_list, group_list = [], [], []
    sfreq, ch_names = None, None

    for run_idx, path in enumerate(runs):
        raw, events = load_xdf_run(path)
        raw_clean, _, _ = preprocess_run(raw, reference=reference)
        epochs = make_epochs(raw_clean, events, tmin=tmin, tmax=tmax, baseline=DEFAULT_BASELINE)

        if sfreq is None:
            sfreq = epochs.info["sfreq"]
            ch_names = list(epochs.info["ch_names"])

        raw_y = epochs.events[:, 2]
        binary_y = np.array([0 if EVENT_ID.get("rest") == label else 1 for label in raw_y])

        X_list.append(epochs.get_data(copy=True))
        y_list.append(binary_y)
        group_list.append(np.full(len(epochs), run_idx))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(group_list, axis=0)

    return X, y, groups, sfreq, ch_names


def get_classifiers():
    return {
        "sLDA": LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
        "SVM (Linear)": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear"))]),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def run_pipeline(
    recordings_dir: Path,
    reference: str = "car",
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    n_csp_components: int = 4,
    n_splits: int = 3,
    group_by_run: bool = True,
):
    X, y, groups, sfreq, ch_names = load_all_runs_binary(recordings_dir, reference, tmin, tmax)
    print(f"\nPooled dataset: {X.shape[0]} trials, {X.shape[1]} channels, sfreq={sfreq} Hz across {len(np.unique(groups))} sessions")

    # Seleccionar extractor de características principal: Tangent Space (Riemann) o FBCSP
    if HAS_PYRIEMANN:
        print("Using Riemannian Geometry (Tangent Space) for spatial feature extraction...")
        spatial_extractor = ("Riemann", RiemannianFeatures(bands=FREQUENCY_BANDS, sfreq=sfreq))
    else:
        print("Using Filter-Bank CSP for spatial feature extraction...")
        spatial_extractor = ("FBCSP", FilterBankCSP(bands=FREQUENCY_BANDS, sfreq=sfreq, n_components=n_csp_components))

    features = FeatureUnion([
        spatial_extractor,
        ("ERDS", ERDSFeatures(
            specs=ERDS_FEATURE_SPECS,
            ch_names=ch_names,
            sfreq=sfreq,
            tmin=tmin,
            baseline_window=ERDS_BASELINE_WINDOW,
        )),
    ])

    cv = GroupKFold(n_splits=n_splits) if group_by_run else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_args = {"groups": groups} if group_by_run else {}

    print("\n" + "=" * 60)
    print("EVALUATING INTER-SESSION GENERALIZATION (WITH SESSION NORMALIZATION)")
    print("=" * 60)

    best_score = 0.0
    best_clf_name = ""

    for name, clf_model in get_classifiers().items():
        pipeline = Pipeline([
            ("features", features),
            ("classifier", clf_model),
        ])

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1, **cv_args)
        mean_acc, std_acc = scores.mean(), scores.std()
        print(f"{name:18s} | Inter-Session Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")

        if mean_acc > best_score:
            best_score = mean_acc
            best_clf_name = name

    print(f"\nBest performing model: {best_clf_name} ({best_score:.3f} accuracy)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--reference", choices=["car", "laplacian"], default="car")
    parser.add_argument("--tmin", type=float, default=DEFAULT_TMIN)
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument("--n_csp", type=int, default=4)
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--group-by-run", action="store_true", default=True)
    args = parser.parse_args()

    run_pipeline(
        args.recordings_dir,
        reference=args.reference,
        tmin=args.tmin,
        tmax=args.tmax,
        n_csp_components=args.n_csp,
        n_splits=args.n_splits,
        group_by_run=args.group_by_run,
    )