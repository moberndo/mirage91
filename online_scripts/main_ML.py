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
from numpy import ones, append, array, copy, mean, newaxis, shape, round
#from torch import load, from_numpy
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# import classifier ...
...
''' CUSTOM IMPORTS'''
# from classifier_functions import EEGNetModel


''' SETTINGS '''
print('Starting Online Processing File...')

classifier_params = Path(__file__).resolve().parent.parent / 'offline_scripts' / 'classifier_results'
# Path(__file__).parent / 'classifier_params' /  '4classes_eegnet_model.pth'# 'lmda_params.pt'
eeg_fs = 100 # 512 # Hz
cutoff_freq1 = [1, 45] # Hz
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
bandpass_filter = pipe.OnlineFilter(filterorder, cutoff_freq1, stream_eeg.fs,
                                    btype, ftype, stream_eeg.n_channels - 3)  # drop the x, y, z channels
filters = bandpass_filter
notch_filter = pipe.NotchFilter(stream_eeg.n_channels - 3) # drop the x, y, z channels
#filter_mov_avg = pipe.MovingAverageFilter(length_of_window, stream_eeg.fs,
#                                          stream_eeg.n_channels)
downsampling_ratio = stream_eeg.fs / fs_downsampled  #downsampling ratio must be an integer
dec_filter = pipe.DecimationFilter(downsampling_ratio)

''' ################################################################## '''
''' #     CREATE OUTLET STREAM                                         '''
''' ################################################################## '''
print('Creating the classifier stream info...')
info2 = StreamInfo(name='ClassifierOutput', type='ClassProb', nominal_srate=10,
                  channel_count=4, channel_format='float32', source_id='classifier91')
print('Opening the classifier outlet...')
outlet_classifier = StreamOutlet(info2)

''' ################################################################## '''
''' #     INITIALIZE CLASSIFIER                                        '''
''' ################################################################## '''
# Initialize CSP
# Load the entire CSP model
with open(classifier_params / 'csp_model.pkl', 'rb') as file:
    csp = pickle.load(file)

# Initilaize sLDA
with open(classifier_params / 'slda_weights.pkl', 'rb') as file:
    slda = pickle.load(file)

''' ################################################################## '''
''' #     START ONLINE PROCESSING                                      '''
''' ################################################################## '''
decoding = True
buffer_size = 300
buffer = ones(shape=(32,1))
t_start_timeout = time.time()

# Pull the first chunk 
chunk, timestamps = stream_eeg.inlet.pull_chunk()
# Wait for 2 seconds
time.sleep(2)

while decoding:
    # Get a new chunk, load buffer and apply preprocessing
    while 1:
        time.sleep(0.5)
        chunk, timestamps = stream_eeg.inlet.pull_chunk()

        chunk = array(chunk).T
        chunk = chunk[0:32,:]
        # print(chunk.shape)

        # if chunk:
        buffer = append(buffer, chunk, axis=1)
        if buffer.shape[1] >= buffer_size:
            processed_chunk, notch_filter, filters = pipe.apply_pipeline(buffer, filters, notch_filter, CSP_filter=[(1, 10)])
            break

    # Create a copy to remove negative strides
    processed_chunk = copy(processed_chunk)  

    csp_features = csp.transform(processed_chunk)
    #csp_features = csp_features[:,newaxis]
    predicted_class = slda.predict_proba(csp_features)
    predicted_class = list(predicted_class[0])  #round(predicted_class,1)
    #predicted_class = predicted_class.T
    print(predicted_class)
    
    # print(predicted_class.numpy())
    #outlet_classifier.push_chunk(predicted_class.numpy())
    #outlet_classifier.push_chunk(predicted_class)
    outlet_classifier.push_chunk(predicted_class)
    
        # stop the decoder when no EEG samples received
    t_start_timeout = time.time()
    if time.time() - t_start_timeout > t_timeout:
        print("No EEG samples received for %s seconds... decoding is stopped" % t_timeout)
        decoding = False
