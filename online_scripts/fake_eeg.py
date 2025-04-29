"""
Title: Main for online processing
Authors: Mirage 91
"""

''' IMPORTS '''
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

''' SETTINGS '''
# Define the number of channels and sampling rate
n_channels = 9
sampling_rate = 100  # Hz

''' MAIN '''
# Define the StreamInfo object (metadata about the stream)
# Name: 'FakeEEG', Type: 'EEG', 8 channels, sampling rate of 100 Hz, and float32 format
info = StreamInfo(name='FakeEEG', type='EEG', channel_count=n_channels, nominal_srate=sampling_rate, channel_format='float32', source_id='eeg91')

# Create an outlet to send the data
outlet = StreamOutlet(info)

print("Now sending data...")

# Generate a fake EEG signal (random noise in this case)
# This could be replaced with a sine wave, real data, etc.
try:
    while True:
        # Generate a random sample with 32 channels
        sample = np.random.rand(n_channels).tolist()
        
        # Push the sample to the LSL stream
        outlet.push_sample(sample)
        
        # Sleep to simulate the actual sampling rate (100 Hz = 0.01s interval)
        time.sleep(1.0 / sampling_rate)

except KeyboardInterrupt:
    print("Stream stopped.")