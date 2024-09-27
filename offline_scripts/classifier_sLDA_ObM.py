"""
Title: Main offline processing pipeline
Authors: Mirage 91
"""
''' IMPORTS PYTHON'''
import numpy as np
import matplotlib.pyplot as plt

''' IMPORTS CUSTOM '''
from custom_functions import Pilot
from classifier_functions import sLDA
from classifier_functions import AdvancedPipelineEvaluator

''' SETTINGS '''
pilot_name = 'Antonio'

# ALL Sessions 4 classes 18th + 26th July 2024
# path = [
#         r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_18\block_1.xdf',
#         r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_18\block_2.xdf',
#         r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_18\block_3.xdf',
#         r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_26\block_1.xdf',
#         r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_26\block_2.xdf',
#         r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_26\block_3.xdf'
#         ]
# #Session 26th July 2024
# path = [
#         r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_26\block_1.xdf',
#         r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_26\block_2.xdf',
#         r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_26\block_3.xdf'
#         ]

#Session 18th July 2024
path = [
        r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_18\block_1.xdf',
        r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_18\block_2.xdf',
        r'C:\Users\Markus\Mirage91\Program\venv\mirage91\recordings\2024_07_18\block_3.xdf'
        ]
paradigm = {'pre_cue':2.0, 'left_hand':4.5, 'left_foot':4.5, 'right_foot':4.5, 'mental_singing':4.5, 'pause':2.5, 'post_run':0.0}
tasks = ['left_hand', 'left_foot', 'right_foot', 'mental_singing']


# Session 10th June 2024
# path = [r'C:\Users\Markus\Mirage91\Program\mirage91\recordings\2024_06_10\block_1.xdf']
# paradigm = {'pre_cue':2.0, 'hand':4.5, 'foot':4.5, 'pause':2.5}
# tasks = ['hand', 'foot']

''' MAIN '''
# define pilot
pilot = Pilot(pilot_name)

# load eeg and process it
pilot.set_eeg(path, paradigm, tasks)
pilot.eeg.processing_pipeline()
#pilot.eeg.show_erds()
# pilot.eeg.show_erds_bp(tasks)

# get train and test set
X_train, X_test, Y_train, Y_test = pilot.eeg.split_train_test()

# extract features bandpower
print('\n BANDPOWER + LDA: \n ')
modality = 'bp'
pilot.eeg.extract_features(modality=modality)
X_train_feat, X_test_feat, Y_train_feat, Y_test_feat = pilot.eeg.get_features()
# choose classifier
classifier = sLDA(X_train_feat, X_test_feat, Y_train_feat, Y_test_feat)
classifier.train_and_test(modality=modality)

# extract features csp
print('\n CSP + LDA: \n')
modality = 'csp'
pilot.eeg.extract_features(modality=modality)
X_train_feat, X_test_feat, Y_train_feat, Y_test_feat = pilot.eeg.get_features()
# choose classifier
classifier = sLDA(X_train_feat, X_test_feat, Y_train_feat, Y_test_feat)
classifier.train_and_test(modality=modality)

# print('\n PIPELINE TESTING ... \n')
# ## ONLY FOR EVALUATING NEW THINGS
# # find best pipeline and parameters automatically (see SLDA.PY)
# evaluator = AdvancedPipelineEvaluator(X_train, X_test, Y_train, Y_test)
# evaluator.evaluate_all_pipelines()

# show results
plt.show()