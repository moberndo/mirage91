"""
Title: Main offline processing pipeline
Authors: Mirage 91
"""
''' IMPORTS PYTHON'''
import numpy as np
from numpy import load
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import pickle


''' SETTINGS '''
...

''' MAIN '''    

# data = load('./features/csp_features.npy', allow_pickle=True)
features = load('./features/csp_features.npy', allow_pickle=True)
# take the 0th frequency band
features = features[0, :, :]
# Load the labels
labels = load('./features/csp_labels.npy', allow_pickle=True)

# Calculate CSP features
# n_classes = 4 # Assuming 4 classes
# n_components = 10  # Number of components for CSP
# csp = CSP(n_components=n_components, log=True, cov_est='epoch')

# Define sLDA
slda = LDA(shrinkage='auto', solver='eigen')
name = 'sLDA'

testing = True
if testing:
    cv = StratifiedKFold(n_splits=10)
    # Calcualte cross-val
    cross_val_scores = cross_val_score(estimator=slda, X=features, y=labels, cv=cv, scoring='accuracy')
    print(f'{name} 10-fold cross-validation accuracy: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}')

# X_train, X_test, y_train, y_test = train_test_split(features[0, :, :], labels, test_size=0.33, random_state=42)
slda.fit(features, labels)
# slda.predict(X_test)

# Save the sLDA model weights
with open('classifier_results/slda_weights.pkl', 'wb') as f:
    pickle.dump(slda, f)


# Save CSP params
#csp = CSP(n_components=6, log=True, cov_est='epoch')
#csp.fit(features, labels)
#csp_params = csp.get_params()

# Save the CSP model weights
#with open('classifier_results/csp_params.pkl', 'wb') as f:
#    pickle.dump(csp_params, f)