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
from numpy import ones, append, array, copy
import torch
from torch import load, from_numpy
# import classifier ...
...
''' CUSTOM IMPORTS'''
from classifier_functions.LMDA_modified import LMDA


''' SETTINGS '''
classifier_params = Path(__file__).parent / 'classifier_params' / 'lmda_params.pt'
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
                                    btype, ftype, stream_eeg.n_channels)
filters = [bandpass_filter]
#filter_mov_avg = pipe.MovingAverageFilter(length_of_window, stream_eeg.fs,
#                                          stream_eeg.n_channels)
downsampling_ratio = stream_eeg.fs / fs_downsampled  #downsampling ratio must be an integer
dec_filter = pipe.DecimationFilter(downsampling_ratio)

''' ################################################################## '''
''' #     CREATE OUTLET STREAM                                         '''
''' ################################################################## '''
print('Creating the classifier stream info...')
info = StreamInfo(name='ClassifierOutput', type='ClassProb', nominal_srate=10,
                  channel_count=4, channel_format='float32', source_id='classifier91')
print('Opening the classifier outlet...')
outlet_classifier = StreamOutlet(info)

''' ################################################################## '''
''' #     INITIALIZE CLASSIFIER                                        '''
''' ################################################################## '''
# Define architecture
model = LMDA(num_classes=4, chans=32, samples=267, channel_depth1=24, channel_depth2=7)
# Load the saved weights into the model
model.load_state_dict(load(classifier_params, weights_only=True))

# Initialize
model.eval()

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
            processed_chunk = pipe.apply_pipeline(buffer, filters, dec_filter)
            break

    # Create a copy to remove negative strides
    processed_chunk = copy(processed_chunk)  

    # Convert the NumPy array to a Tensor
    processed_chunk_tensor = from_numpy(processed_chunk).float()
    # Make a prediction
    with torch.no_grad():
        predicted_class = model(processed_chunk_tensor)


    # print(predicted_class.numpy())
    outlet_classifier.push_chunk(predicted_class.numpy())

        # stop the decoder when no EEG samples received
    t_start_timeout = time.time()
    if time.time() - t_start_timeout > t_timeout:
        print("No EEG samples received for %s seconds... decoding is stopped" % t_timeout)
        decoding = False
