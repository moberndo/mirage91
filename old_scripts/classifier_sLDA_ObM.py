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

data = load('./features/csp_features.npy', allow_pickle=True)
features = np.vstack(data[:, 1])
labels = [str(elem) for elem in data[:, 0]]

# Calculate CSP features
n_classes = 4 # Assuming 4 classes
n_components = 10  # Number of components for CSP
csp = CSP(n_components=n_components, log=True, cov_est='epoch')

# Define sLDA
slda = LDA(shrinkage='auto', solver='eigen')
name = 'sLDA'

cv = StratifiedKFold(n_splits=10)

# Calcualte cross-val
cross_val_scores = cross_val_score(estimator=slda, X=features, y=labels, cv=cv, scoring='accuracy')
print(f'{name} 10-fold cross-validation accuracy: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}')

# Split data into training and testing
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train sLDA classifier

# slda.fit(X_train, y_train)

# Make predictions
# y_predict = slda.predict(X=X_test)


# Save the CSP and sLDA model weights
with open('csp_slda_weights.pkl', 'wb') as f:
    pickle.dump({'csp': csp, 'slda': slda}, f)
