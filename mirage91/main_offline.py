"""
Title: Main offline processing pipeline
Authors: Mirage 91
"""
''' IMPORTS PYTHON'''
import numpy as np
import matplotlib.pyplot as plt

''' IMPORTS CUSTOM '''
from mirage91_pipeline import Pilot
from classifier import sLDA

''' SETTINGS '''
pilot_name = 'Patrick'
# Session 18th July 2024
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
pilot.eeg.show_erds()
# pilot.eeg.show_erds_bp(tasks)

# extract features
pilot.eeg.extract_features()
X, Y = pilot.eeg.get_features()

# choose classifier
classifier = sLDA(X, Y, train_size=0.7)
classifier.train_and_test()

# show results
plt.show()