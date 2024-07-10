
"""
Title: class file: Pilot
Authors: Mirage 91
"""
# imports python
from .eeg import EEG

# main
class Pilot:
    def __init__(self, name):
        '''
        Initialize the Pilot class with the name of the pilot.
        Input:
            - name: Name of the pilot [str]
        '''
        self.name = name
        self.eeg = None
        self.classifier = None
    
    def set_eeg(self, path, paradigm):
        '''
        Set the EEG data for the pilot.
        Input:
            - path: Path to the data [str]
            - paradigm: Paradigm that was used to record the data [dict]
        '''
        self.eeg = EEG(path, paradigm)

    def eeg(self):
        '''
        This function is called to obtain the EEG.
        Input: -
        '''
        if self.eeg is not None:
            return self.eeg
        else:
            raise ValueError('EEG data has not been defined yet. Use "set_eeg()" method.')
        
    