"""
Title: sLDA Classifier
Authors: Mirage91
"""
''' IMPORTS PYTHON'''
from numpy import load, mean, array
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

''' IMPORTS CUSTOM '''
from classifier_functions import sLDA

''' SETTINGS '''
path = Path(__file__).parent / 'features' / 'csp_features.npy'

''' MAIN '''
# Read Features
features = load(path, allow_pickle=True)
labels = array([entry for entry in features[:,0]])
data = array([entry for entry in features[:,1]])

# Initialize the sLDA classifier (with shrinkage enabled)
slda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

# Create 10-fold cross-validation strategy
cv = StratifiedKFold(n_splits=10)

# Perform 10-fold cross-validation on CSP features
cross_val_scores = cross_val_score(estimator=slda, X=data, y=labels, cv=cv, scoring='accuracy')

print(f'Cross-validation accuracy scores: {cross_val_scores}')
print(f'Mean accuracy: {mean(cross_val_scores):.2f}')
