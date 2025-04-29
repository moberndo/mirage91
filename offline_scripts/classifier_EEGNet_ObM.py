''' ######################################################################
        Author: Markus E. Oberndorfer
        Title: Pre-Trained EEGNetv4 classifier
###################################################################### '''

''' ### IMPORTS ### '''
# basic imports
from pathlib import Path
from tempfile import mkdtemp
from copy import deepcopy
from numpy import load
# advanced imports
import torch
from torch import nn
from braindecode import EEGClassifier
from braindecode.models import EEGNetv4
from huggingface_hub import hf_hub_download
# custom imports
...

''' ### CONSTANTS ### '''
path_data = Path(__file__).parent / 'features'/ 'cleaned_epoched_eeg.npy'

''' ### FUNCTIONS ### '''
def normalize_data(data):
    ...

def load_data(path_data, normalize=True):
    data_ = load(str(path_data), allow_pickle=True)
    # Unzip the data structure

    # Normalize the data (dependend on session number)
    if normalize:
        data = normalize_data(data)

    data = ...
    labels = ...
    return data, labels

''' ### MAIN ### '''
# Loading the data
data, labels = load_data(path_data=path_data)

# Load the classifier architecture
...
# Load the  pre-trained parameters
...
# Freeze the layers except the input and output layer
...