"""Feature extractors, unified behind the sklearn Transformer interface
(fit(X, y), transform(X) -> feature matrix), where X is always an epoched
array of shape (n_trials, n_channels, n_times) as returned by
epochs.make_epochs(...).get_data().
"""

from __future__ import annotations

import mne
import numpy as np
from scipy.linalg import eigh
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False


## 1. Utility functions
#here we sample the index of a time point in seconds relative to the epoch's tmin, given the sampling frequency
def _sample_index(t: float, tmin: float, sfreq: float) -> int:
    return int(round((t - tmin) * sfreq))

#here we crop the time axis of an (n_trials, n_channels, n_times) array to a window given in seconds relative to the epoch's tmin
def _crop(X: np.ndarray, tmin: float, sfreq: float, window: tuple[float, float]) -> np.ndarray:
    n_times = X.shape[-1]
    start = max(_sample_index(window[0], tmin, sfreq), 0)
    end = min(_sample_index(window[1], tmin, sfreq), n_times)
    return X[..., start:end]


# ============================================================
# 1. Bandpower features 
# ============================================================

class BandpowerFeatures(BaseEstimator, TransformerMixin):
    #One log-bandpower feature per (channel, band) config
    
    def __init__(self, class_configs, ch_names, sfreq, tmin, window=(0.5, 3.5)):
        self.class_configs = class_configs
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.tmin = tmin
        self.window = window


    #fit() just precomputes the channel indices for each class config, so we don't have to look them up every time in transform()
    def fit(self, X, y=None):
        self.channel_indices_ = {
            name: self.ch_names.index(ch)
            for name, (ch, _, _) in self.class_configs.items()
        }
        return self

    # transform() computes the log-bandpower features for each trial, using the precomputed channel indices and the specified frequency bands and time window
    def transform(self, X):
        n_trials = X.shape[0]
        feats = np.zeros((n_trials, len(self.class_configs)), dtype=float)
        for i, (name, (ch, l_freq, h_freq)) in enumerate(self.class_configs.items()):
            ch_idx = self.channel_indices_[name]
            sig = X[:, ch_idx, :]
            sig_filt = mne.filter.filter_data(
                sig, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq,
                method="iir", verbose=False
            )
            sig_win = _crop(sig_filt[:, None, :], self.tmin, self.sfreq, self.window)[:, 0, :]
            feats[:, i] = np.log(np.sum(sig_win ** 2, axis=-1) + 1e-12)
        return feats


# ============================================================
# 2. CSP features — one-vs-rest, per class
# ============================================================

def _compute_cov(trial: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    cov = np.cov(trial)
    cov /= np.trace(cov)
    cov += reg * np.eye(cov.shape[0]) #is added so that the covariance matrix is positive definite and invertible
    return cov


def _csp_filters_binary(cov_a: np.ndarray, cov_b: np.ndarray, n_components: int) -> np.ndarray:
    eigvals, eigvecs = eigh(cov_a, cov_a + cov_b)
    idx = np.argsort(eigvals)
    sorted_vecs = eigvecs[:, idx]
    n = n_components // 2
    filters = np.concatenate([sorted_vecs[:, :n], sorted_vecs[:, -n:]], axis=1)
    return filters.T


class CSPFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=6, l_freq=5.0, h_freq=25.0, sfreq=None, tmin=0.0, window=(1.0, 3.0)):
        self.n_components = n_components
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.tmin = tmin
        self.window = window

    # fit() computes the class-wise covariance matrices and derives the CSP filters for each class, storing them for later use in transform().
    def fit(self, X, y):
        X_filt = mne.filter.filter_data(
            X, sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq,
            method="iir", verbose=False
        )
        X_win = _crop(X_filt, self.tmin, self.sfreq, self.window)

        classes = np.unique(y)
        self.classes_ = classes
        class_covs = {}
        for c in classes:
            trials = X_win[y == c]
            covs = [_compute_cov(trial) for trial in trials]
            class_covs[c] = np.mean(covs, axis=0)

        self.filters_ = {}
        for c in classes:
            cov_c = class_covs[c]
            cov_rest = np.mean([class_covs[other] for other in classes if other != c], axis=0)
            self.filters_[c] = _csp_filters_binary(cov_c, cov_rest, self.n_components)
        return self

    # transform() applies the precomputed CSP filters to each trial, computing the log-variance of the filtered signals as features.
    def transform(self, X):
        X_filt = mne.filter.filter_data(
            X, sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq,
            method="iir", verbose=False
        )
        X_win = _crop(X_filt, self.tmin, self.sfreq, self.window)

        feature_list = []
        for trial in X_win:
            trial_feats = []
            for c in self.classes_:
                W = self.filters_[c]
                filtered = W @ trial
                log_var = np.log(np.var(filtered, axis=1) + 1e-12) #with the var we square and center 
                trial_feats.append(log_var)
            feature_list.append(np.concatenate(trial_feats))
        return np.array(feature_list)


# ============================================================
# 3. Filter-Bank CSP (from train_classifier_rest_vs_movement.py)
# ============================================================

class FilterBankCSP(BaseEstimator, TransformerMixin):
    def __init__(self, bands, sfreq, n_components=4, tmin=0.0, window=(1.0, 3.0)):
        self.bands = bands
        self.sfreq = sfreq
        self.n_components = n_components
        self.tmin = tmin
        self.window = window

    def fit(self, X, y):
        from mne.decoding import CSP

        self.csp_by_band_ = []
        for l_freq, h_freq in self.bands:
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
            )
            X_win = _crop(X_band, self.tmin, self.sfreq, self.window)
            csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True, norm_trace=False)
            csp.fit(X_win, y)
            self.csp_by_band_.append(csp)
        return self

    def transform(self, X):
        band_features = []
        for (l_freq, h_freq), csp in zip(self.bands, self.csp_by_band_):
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
            )
            X_win = _crop(X_band, self.tmin, self.sfreq, self.window)
            band_features.append(csp.transform(X_win))
        return np.concatenate(band_features, axis=1)


# ============================================================
# 4. Riemannian tangent-space features 
# ============================================================

class RiemannianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, bands, sfreq, tmin=0.0, window=(1.0, 3.0)):
        self.bands = bands
        self.sfreq = sfreq
        self.tmin = tmin
        self.window = window

    def fit(self, X, y=None):
        self.tangent_spaces_ = []
        for l_freq, h_freq in self.bands:
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
            )
            X_win = _crop(X_band, self.tmin, self.sfreq, self.window)
            covs = Covariances(estimator="lwf").fit_transform(X_win)
            ts = TangentSpace(metric="riemann").fit(covs)
            self.tangent_spaces_.append(ts)
        return self

    def transform(self, X):
        ts_features = []
        for (l_freq, h_freq), ts in zip(self.bands, self.tangent_spaces_):
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
            )
            X_win = _crop(X_band, self.tmin, self.sfreq, self.window)
            covs = Covariances(estimator="lwf").transform(X_win)
            ts_features.append(ts.transform(covs))
        return np.concatenate(ts_features, axis=1)


# ============================================================
# 5. ERDS (baseline-relative power) features 
# ============================================================

class ERDSFeatures(BaseEstimator, TransformerMixin):
    """Baseline-relative power change per (channel, band, window) spec. Needs
    the epoch to include a pre-cue baseline window."""

    def __init__(self, specs, ch_names, sfreq, tmin, baseline_window):
        self.specs = specs
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.tmin = tmin
        self.baseline_window = baseline_window

    def fit(self, X, y=None):
        self.channel_indices_ = {s["name"]: self.ch_names.index(s["channel"]) for s in self.specs}
        self.baseline_idx_ = (
            _sample_index(self.baseline_window[0], self.tmin, self.sfreq),
            _sample_index(self.baseline_window[1], self.tmin, self.sfreq),
        )
        return self

    def transform(self, X):
        n_trials, n_times = X.shape[0], X.shape[2]
        feats = np.zeros((n_trials, len(self.specs)), dtype=float)

        base_start = max(self.baseline_idx_[0], 0)
        base_end = min(self.baseline_idx_[1], n_times)

        for i, spec in enumerate(self.specs):
            ch_idx = self.channel_indices_[spec["name"]]
            sig = X[:, ch_idx, :]
            l_freq, h_freq = spec["band"]
            sig_filt = mne.filter.filter_data(sig, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False)
            power = np.abs(hilbert(sig_filt, axis=-1)) ** 2

            win_start = max(_sample_index(spec["window"][0], self.tmin, self.sfreq), 0)
            win_end = min(_sample_index(spec["window"][1], self.tmin, self.sfreq), n_times)

            baseline_power = power[:, base_start:base_end].mean(axis=1)
            window_power = power[:, win_start:win_end].mean(axis=1)
            feats[:, i] = (window_power - baseline_power) / (baseline_power + 1e-8)

        return feats
