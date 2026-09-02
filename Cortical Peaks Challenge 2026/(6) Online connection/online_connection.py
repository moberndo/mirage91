"""
Title: Main for online processing
Authors: Mirage 91
"""

''' PYTHON IMPORTS '''
from pylsl import StreamInfo, StreamOutlet
from OnlineProcessingPipeline import OnlineProcessingPipeline as pipe
import time
from pathlib import Path
import pickle
import numpy as np
from numpy import ones, append, array, copy, reshape, expand_dims, zeros_like, zeros
import torch
from torch import load, from_numpy
import scipy.signal as signal
# import classifier ...
...
''' CUSTOM IMPORTS'''
from classifier_functions import EEGNet


''' SETTINGS '''
classifier_params = Path(__file__).parent / 'classifier_params' /  '4classes_eegnet_model.pth'# 'lmda_params.pt'
eeg_fs = 100 # 512 # Hz
cutoff_freq1 = [0.5, 30] # Hz
filterorder = 4
fs_downsampled = 2
ftype = 'butter' # Butterworth
btype = 'band'  # or 'lowpass', 'highpass'
channel_count = 4  # for the classifier stream according to the game and predictor
length_of_window = 0.5  # second window moving average
t_timeout = 5

''' ################################################################## '''
''' #      READ EEG STREAM                                             '''
''' ################################################################## '''
stream_eeg = pipe.ResolveCreateStream()

''' ################################################################## '''
''' #     INITIALIZE ONLINE FILTERING                                  '''
''' ################################################################## '''

def butter_bandpass(lowpass, highpass, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    low = lowpass / nyquist
    high = highpass / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def notch_filter(w0, Q, fs):
    b, a = signal.iirnotch(w0=w0, Q=Q, fs=fs)
    return b, a

b_bp, a_bp = butter_bandpass(cutoff_freq1[0], cutoff_freq1[1], fs=500, order=4)
bp_state = signal.lfilter_zi(b_bp, a_bp)
bp_state = np.tile(bp_state, (32, 1))

b_notch, a_notch = notch_filter(w0=50, Q=30, fs=500)
notch_state = signal.lfilter_zi(b_notch, a_notch)
notch_state = np.tile(notch_state, (32, 1))

filters = [(b_notch, a_notch), (b_bp, a_bp)]
filter_state = [notch_state, bp_state]

#bandpass_filter = pipe.OnlineFilter(filterorder, cutoff_freq1, stream_eeg.fs,
#                                    btype, ftype, stream_eeg.n_channels - 3)  # drop the x, y, z channels##
#
#filters = bandpass_filter#
#
# csp_filters = pipe.OnlineFilter(filterorder, )
#notch_filter = pipe.NotchFilter(stream_eeg.n_channels - 3) # drop the x, y, z channels
#filter_mov_avg = pipe.MovingAverageFilter(length_of_window, stream_eeg.fs,
#                                          stream_eeg.n_channels)
#downsampling_ratio = stream_eeg.fs / fs_downsampled  #downsampling ratio must be an integer
#dec_filter = pipe.DecimationFilter(downsampling_ratio)



''' ################################################################## '''
''' #     CREATE OUTLET STREAM                                         '''
''' ################################################################## '''
print('Creating the classifier stream info...')
info = StreamInfo(name='ClassifierOutput', type='ClassProb', nominal_srate=10,
                  channel_count=3, channel_format='float32', source_id='classifier91')
print('Opening the classifier outlet...')
outlet_classifier = StreamOutlet(info)

''' ################################################################## '''
''' #     INITIALIZE CLASSIFIER                                        '''
''' ################################################################## '''
# Define architecture
# model = LMDA(num_classes=4, chans=32, samples=267, channel_depth1=24, channel_depth2=7)
n_classes, dropoutRate, kernelLength, kernelLength2, F1, D = 3, 0.5, 64, 16, 8, 2
F2 = F1 * D
chans = 32
samples = 500
model = EEGNet(n_classes, chans, samples, dropoutRate, kernelLength, kernelLength2, F1, D, F2)
# model = EEGNet(channels=32, n_classes=2, samples=500)
# Load the saved weights into the model
# model.load_state_dict(load(classifier_params, weights_only=True))
checkpoint = torch.load('./classifier_params/model_and_optimizer_3class_EEGNet_comp_all.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# model = torch.load('./classifier_params/entire_model.pth')
# Initialize
model.eval()

''' ################################################################## '''
''' #     START ONLINE PROCESSING                                      '''
''' ################################################################## '''
decoding = True
buffer_size = 2000
buffer = ones(shape=(32,1))
t_start_timeout = time.time()

# Pull the first chunk 
chunk, timestamps = stream_eeg.inlet.pull_chunk()
# Wait for 2 seconds
time.sleep(2)

while decoding:
    # Get a new chunk, load buffer and apply preprocessing
    while 1:
        time.sleep(0.1)
        chunk, timestamps = stream_eeg.inlet.pull_chunk()

        chunk = array(chunk).T
        chunk = chunk[0:32,:]
        # print(chunk.shape)

        # if chunk:
        buffer = append(buffer, chunk, axis=1)
        #cut buffer here
        if buffer.shape[1] >= buffer_size:
            buffer = buffer[:, -buffer_size:]
            processed_chunk, filter_state = pipe.apply_pipeline(buffer, filters=filters, filter_states=filter_state)
            break

    # Create a copy to remove negative strides
    processed_chunk = copy(processed_chunk)
    processed_chunk = expand_dims(processed_chunk, axis=[0,1])
    # processed_chunk = reshape(processed_chunk, (1, processed_chunk.shape[0], processed_chunk.shape[1]))

    # Convert the NumPy array to a Tensor
    processed_chunk_tensor = from_numpy(processed_chunk).float()
    # Make a prediction
    with torch.no_grad():
        predicted_class = model(processed_chunk_tensor)


    # print(predicted_class.numpy())
    outlet_classifier.push_chunk(predicted_class.numpy())
    print(predicted_class.numpy())
    time.sleep(0.3)

        # stop the decoder when no EEG samples received
    t_start_timeout = time.time()
    if time.time() - t_start_timeout > t_timeout:
        print("No EEG samples received for %s seconds... decoding is stopped" % t_timeout)
        decoding = False
