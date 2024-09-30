"""
Title: Main for online processing
Authors: Mirage 91
"""

from pylsl import StreamInfo, StreamOutlet
from OnlineProcessingPipeline import OnlineProcessingPipeline as pipe
import time
import numpy as np
import pickle
# import sLDA #import the classifier


# initialize variables
# TODO: Edit initializing variables, once the preprocessing pipeline is fixed
cutoff_freq1 = [3, 5]
cutoff_freq2 = [10, 15]
filterorder = 4
fs_downsampled = 16
ftype = 'butter'
btype = 'band'  # or 'lowpass', 'highpass'
channel_count = 4  # for the classifier stream according to the game and predictor
length_of_window = 1  #second window moving average
t_timeout = 5

# looking for an EEG-stream
stream_eeg = pipe.ResolveCreateStream()

# initialize filters
filter_alpha = pipe.OnlineFilter(filterorder, cutoff_freq1, stream_eeg.fs, btype, ftype, stream_eeg.n_channels)
filter_beta = pipe.OnlineFilter(filterorder, cutoff_freq2, stream_eeg.fs, btype, ftype, stream_eeg.n_channels)
filters = [filter_alpha, filter_beta]
filter_mov_avg = pipe.MovingAverageFilter(length_of_window, stream_eeg.fs, stream_eeg.n_channels)
downsampling_ratio = int(np.round(stream_eeg.fs / fs_downsampled,0))  #downsampling ratio must be an integer
dec_filter = pipe.DecimationFilter(downsampling_ratio)

# initialize Classifier
with open('lda_model.pk', 'rb') as file:
    lda_model_ObM = pickle.load(file)
classifier = lda_model_ObM.classifier()


print('Creating the classifier stream info...')
info_classifier = StreamInfo('classifier', 'classifier', channel_count, 0, 'float32')
print('Opening the classifier outlet...')
outlet_classifier = StreamOutlet(info_classifier)

# Online processing
decoding = True
t_start_timeout = time.time()
while decoding:
    # get a new chunk
    chunk, timestamps = stream_eeg.inlet.pull_chunk()
    if chunk:
        # modify pipeline
        processed_chunk = pipe.run_pipeline_sLDA_bp_ObM(chunk, filters, filter_mov_avg, dec_filter)
        if processed_chunk.size > 0:
            ############################################################################################################
            ###                                     Begin Classifier                                                 ###
            ###                         TODO: Edit Classifier once it is fixed
            ############################################################################################################
            # predictor should look something like this
            # [predicted_class, linear_scores, class_probabilities] = sLDA.classifier.predict(processed_chunk[:,-1])
            # #just dummy variable, edit accordingly to the classifier
            predicted_probabilities = classifier.predict_proba(np.transpose(processed_chunk))

            ############################################################################################################
            ###                                     End Classifier                                                   ###
            ############################################################################################################

            # prepare the predicted chunk to be streamed
            if predicted_probabilities:
                # TODO: edit the out_chunk such that it includes the predicted classes and is suitable for the game
                out_chunk = [np.transpose(predicted_probabilities)]
                outlet_classifier.push_chunk(out_chunk)


        # stop the decoder when no EEG samples received
        t_start_timeout = time.time()
    if time.time() - t_start_timeout > t_timeout:
        print("No EEG samples received for %s seconds... decoding is stopped" % t_timeout)
        decoding = False
