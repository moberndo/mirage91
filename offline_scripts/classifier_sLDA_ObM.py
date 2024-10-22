"""
Title: Main offline processing pipeline
Authors: Mirage 91
"""
''' IMPORTS PYTHON'''
import numpy as np
from numpy import load, arange
import mne
from itertools import combinations
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import pickle
from scipy.signal import butter, filtfilt

''' FUNCTIONS '''
# Define the bandpass filter along the time axis
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=2)
    return filtered_data


''' MAIN '''    

# data = load('./features/csp_features.npy', allow_pickle=True)
features = load('./features/normalized_cleaned_eeg_data.npy', allow_pickle=True)
# Load the labels
labels = load('./features/normalized_cleaned_eeg_labels.npy', allow_pickle=True)
# Sample frequency
fs = 200
# Calculate CSP features
n_classes = 4 # Assuming 4 classes
n_components = 6  # Number of components for CSP

# Apply filterbank to the data to find the best possible filterband combination
filter_start = 1 # Hz
filter_stop = 45 # Hz
filter_step = 4 # Hz
filterbank_freqs = arange(filter_start, filter_stop, filter_step)
filterbank_freqs = [(filterbank_freqs[idx], filterbank_freqs[idx+1]) for idx in range(len(filterbank_freqs)-1)]

csp_features = []
csp_models = []
for idx, filter_freqs in enumerate(filterbank_freqs):
    low_freq = filter_freqs[0]
    high_freq = filter_freqs[1]
    filtered_data = bandpass_filter(features, lowcut=low_freq, highcut=high_freq, fs=200, order=4)
    # cut out 0.5 after cue onset until 4.0 seconds after cue onset
    start_cut = int(1.5 * fs) # 2.5 seconds into the signal
    end_cut = int(5.5 * fs) # 6.0 seconds into the signal
    filtered_data = filtered_data[:, :, start_cut:end_cut]

    csp = CSP(n_components=n_components, reg='shrinkage', log=True, cov_est='epoch')
    # Fit CSP on the data and extract features
    csp.fit(filtered_data, labels)

    csp_models.append(csp)
    csp_features.append(csp.transform(filtered_data))

# Make all CSP combinations
num_fbands_idx = list(range(len(csp_features)))
fband_combinations = list(combinations(num_fbands_idx, 4))

csp_combinations = []
for fband_comb in fband_combinations:
    features = []
    for band_idx in fband_comb:
        features.append(csp_features[band_idx])
    # check shape and reshape ...
    ...
    csp_combinations.append(features)
# check csp_combinations shape, should be (feautres, n_comps, n_timepoints)
...

# Define sLDA
best_bands = (0, []) # stores 10 fold cross val acc and the band combinations
for idx, csp_bands in enumerate(csp_combinations):
    # Initialize sLDA
    slda = LDA(shrinkage='auto', solver='eigen')
    cv = StratifiedKFold(n_splits=10)
    # Calcualte cross-val
    cross_val_scores = cross_val_score(estimator=slda, X=csp_bands, y=labels, cv=cv, scoring='accuracy')
    print(f'sLDA 10-fold cross-validation accuracy: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}')
    if cross_val_score.mean() > best_bands[0]:
        best_bands[0] = cross_val_score.mean()
        best_bands[1] = fband_combinations[idx] # save the indices of the best feature freq bands

# Print results:
print(f'Best accuracy is acchieved for {best_bands[1]} with an accuracy of {best_bands[0]*100}%.')


# Train the best sLDA model
slda = LDA(shrinkage='auto', solver='eigen')
slda.fit(features, labels)
# Save the sLDA model
with open('classifier_results/slda.pkl', 'wb') as f:
    pickle.dump(slda, f)

# Save the CSP model
with open('classifier_results/csp.pkl', 'wb') as f:
   pickle.dump(csp, f)