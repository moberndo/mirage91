import mne
import numpy as np
from scipy.linalg import eigh

# we have 4 classes: left hand, right hand, feet and rest 
#============================================================
#1. Bandpower features
#============================================================
"""
    steps according to the NBCI assignment 5
    1. load preprocessed data -> we can also do that in the main 
    2. epoch the data -> can also do that in the main 
    3. compute and plot ERDS maps for each class -> schon von paul gemacht kann funktion verwenden, anschauen und klassen bestimmen 
    4. Choose features from the ERDS maps 
    5. select the channel that is used as feature
    6. apply bandpass filter to the selected frequency band
    7. for each trial, extract a 2sec segment of data (1 to 3s after cue) square each sample and sum up the squared values in the segment (idk exactly why we do that we did that in the assignment)
    8. calculate the log of the sum 
    9. Done after this you can do LDA (or SVM)

"""

def get_events(raw):
    """Extract events from the annotations already attached to raw,
    keeping only the 4 real classes (drop 'pause' and 'post_block')."""
    event_mapping = {"left_hand": 1, "right_hand": 2, "feet": 3, "rest": 4}
    events, found_ids = mne.events_from_annotations(
        raw, event_id=event_mapping, verbose=False
    )
    return events, event_mapping

def extract_bandpower_features(raw, class_configs, tmin=1.0, tmax=3.0): #tried different tmin and tmax these worked best for me but feel free to change
    event_mapping = {"left_hand": 1, "right_hand": 2, "feet": 3, "rest": 4}
    events, _ = mne.events_from_annotations(raw, event_id=event_mapping, verbose=False)

    feature_columns = []
    labels = None

    for class_name, (channel, l_freq, h_freq) in class_configs.items():
        raw_ch = raw.copy().pick([channel])
        raw_ch.filter(l_freq, h_freq, method="iir", verbose=False)

        epochs = mne.Epochs(
            raw_ch, events, event_id=event_mapping,
            tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False,
        )

        data = epochs.get_data(copy=False)          # (n_epochs, 1, n_times)
        log_power = np.log(np.sum(data ** 2, axis=-1)).squeeze()  # (n_epochs,)
        feature_columns.append(log_power)
        labels = epochs.events[:, -1]

    features = np.column_stack(feature_columns)  # (n_epochs, n_classes)
    return features, labels

#============================================================
#2. Spatial filtering: Common Spatial Patterns (CSP)
#============================================================
"""
    steps also according to the NBCI assignment 5
    1. bandpass filter the preprocessed data 
    2. Epoch 
    3. construct epoch matrix 
    4. compute normalized covariance matrix for each epoch
    5. Average the covariance matrices seperately for the 4 classes (left hand, right hand, feet, rest)
    6. solve the generalized eigenvalue problem  (mit scipy.linalg.eigh) to get the CSP filters
    7. select the first and last n_components eigenvectors (n_components = 4
    8. extract features by applying the CSP filters to the epochs and calculating the log-variance of the filtered signals
    9. compute variance along the time axis and apply log to the variance to get the features 
    10. also create an y_train array!
    11. Done after this you can do LDA (or SVM)
"""

#by doing the ICA and removing components the matrix is not of full rank anymore, which means we cant compute 
#the eigenvalues that we need for the CSP. Therefore we use regularization to make the covariance matrix positive definite and full rank again.
def compute_cov(trial, reg=1e-6):
    cov = np.cov(trial)
    cov /= np.trace(cov)
    cov += reg * np.eye(cov.shape[0])
    return cov
    
def extract_csp_features(raw, n_components=6, l_freq=5.0, h_freq=25.0, tmin=1.0, tmax=3.0):
    raw_filtered = raw.copy().pick(["eeg", "csd"])
    raw_filtered.filter(l_freq, h_freq, method="iir", verbose=False)

    events, event_mapping = get_events(raw_filtered)
    epochs = mne.Epochs(
        raw_filtered, events, event_id=event_mapping,
        tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False,
    )
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]

    classes = np.unique(y)
    class_covs = {}
    for c in classes:
        trials = X[y == c]
        covs = [compute_cov(trial) for trial in trials]
        class_covs[c] = np.mean(covs, axis=0)

    #csp is inherently binary; here is where the actual csp part happens where we look at one class and the other 3 are grouped togehter we do that with each     
    def csp_filters_binary(cov_a, cov_b, n_components):
        # 6. generalized eigenvalue problem
        eigvals, eigvecs = eigh(cov_a, cov_a + cov_b)
        # 7. take first and last n_components/2 eigenvectors
        idx = np.argsort(eigvals)
        sorted_vecs = eigvecs[:, idx]
        n = n_components // 2
        filters = np.concatenate(
            [sorted_vecs[:, :n], sorted_vecs[:, -n:]], axis=1
        )
        return filters.T  # (n_components, n_channels)

    # One vs the rest CSP filters per class
    all_filters = {}
    for c in classes:
        cov_c = class_covs[c]
        cov_rest = np.mean(
            [class_covs[other] for other in classes if other != c], axis=0
        )
        all_filters[c] = csp_filters_binary(cov_c, cov_rest, n_components)

    feature_list = []
    for trial in X:
        trial_feats = []
        for c in classes:
            W = all_filters[c]
            filtered = W @ trial          # (n_components, n_times)
            var = np.var(filtered, axis=1)
            log_var = np.log(var)
            trial_feats.append(log_var)
        feature_list.append(np.concatenate(trial_feats))

    features = np.array(feature_list)  # (n_epochs, n_classes * n_components)

    y_train = y

    return features, y_train
