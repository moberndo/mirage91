"""
Title: Main offline processing pipeline
Authors: Mirage 91
"""
''' IMPORTS PYTHON'''
import numpy as np
import matplotlib.pyplot as plt

''' IMPORTS CUSTOM '''
from custom_functions import Pilot

''' SETTINGS '''
pilot_name = 'Antonio'

#Session data
path = r'./raw_data'
paradigm = {'pre_cue':2.0, 'left_hand':4.5, 'right_hand':4.5, 'feet':4.5, 'mental_singing':4.5, 'pause':2.5, 'post_block':0.0}
tasks = ['left_hand', 'right_hand', 'feet', 'mental_singing']


''' MAIN '''
# define pilot
pilot = Pilot(pilot_name)

# Load the EEG data + paradigm stream
pilot.set_eeg(path, paradigm, tasks)
# Preprocess the data
pilot.eeg.preprocessing()
# Epoch the data and save the epoched data as a .npy file in the "featues" folder
pilot.eeg.epoching_and_rejecting()

# Apply ICA and then feature extraction (CSP)
pilot.eeg.processing_pipeline() # change this




# extract features bandpower
print('\n BANDPOWER + LDA: \n ')
modality = 'bp'
pilot.eeg.extract_features(modality=modality)
# save features in "feature" folder
...

# extract features csp
print('\n CSP + LDA: \n')
modality = 'csp'
pilot.eeg.extract_features(modality=modality)
# save features in "feature" folder
...



# show results
plt.show()