"""
Title: Main offline processing pipeline
Authors: Mirage 91
"""
''' IMPORTS PYTHON'''
import numpy as np
from numpy import load, arange, array, reshape
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
n_classes = 2 # Assuming 4 classes
n_components = 6  # Number of components for CSP

# Apply filterbank to the data to find the best possible filterband combination
filter_start = 5 # Hz
filter_stop = 40 # Hz
filter_step = 4 # Hz
filterbank_freqs = arange(filter_start, filter_stop, filter_step)
filterbank_freqs = [(filterbank_freqs[idx], filterbank_freqs[idx+1]) for idx in range(len(filterbank_freqs)-1)]

# Iterate over all combinations of 2 classes


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
num_combs = 6
num_fbands_idx = list(range(len(csp_features)))
fband_combinations = list(combinations(num_fbands_idx, num_combs))

csp_combinations = []
for fband_comb in fband_combinations:
    feature = []
    for band_idx in fband_comb:
        feature.append(csp_features[band_idx])
    # check shape and reshape ...
    feature = array(feature)
    csp_combinations.append(feature)
# check csp_combinations shape, should be (feautres, n_comps, n_timepoints)
csp_combinations = reshape(array(csp_combinations), (len(fband_combinations), -1, num_combs, 6))

# Define sLDA
best_bands = [0, []] # stores 10 fold cross val acc and the band combinations
for idx in range(csp_combinations.shape[0]):
    # Initialize sLDA
    slda = LDA(shrinkage='auto', solver='eigen')
    cv = StratifiedKFold(n_splits=10)
    # Calcualte cross-val
    X_ = csp_combinations[idx, :, :, :]
    X_ = reshape(X_, (-1, int(X_.shape[1] * X_.shape[2])))
    cross_val_scores = cross_val_score(estimator=slda, X=X_, y=labels, cv=cv, scoring='accuracy')
    print(f'sLDA 10-fold cross-validation accuracy: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}')
    if cross_val_scores.mean() > best_bands[0]:
        best_bands[0] = cross_val_scores.mean()
        best_bands[1] = fband_combinations[idx] # save the indices of the best feature freq bands

# Print results:
print(f'\n Best accuracy is acchieved for {best_bands[1]} with an accuracy of {best_bands[0]*100:.2f}%.')
print(f'The filterbank bands are:')
for idx in best_bands[1]:
    print(f'{filterbank_freqs[idx]} Hz')
    
features = [csp_features[idx] for idx in best_bands[1]]
# change shape of features
...
csps = [csp_models[idx] for idx in best_bands[1]]

# Train the best sLDA model
slda = LDA(shrinkage='auto', solver='eigen')
slda.fit(features, labels)
# Save the sLDA model
with open('classifier_results/slda.pkl', 'wb') as f:
    pickle.dump(slda, f)

# Save the CSP model
with open('classifier_results/csp.pkl', 'wb') as f:
   pickle.dump(csps, f)