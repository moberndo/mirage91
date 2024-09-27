"""
Title: Class Online processing pipeline
Authors: Mirage 91
"""

import numpy as np
import scipy.signal as signal


class OnlineProcessingPipeline:
    class OnlineFilter:
        # Create an online filter
        def __init__(self, filterorder, cutoff_freq, fs, btype, ftype, n_channels):
            b, a = signal.iirfilter(filterorder, Wn=cutoff_freq, fs=fs, btype=btype, ftype=ftype)
            z = signal.lfilter_zi(b, a)
            z = np.tile(z, (n_channels, 1))
            self.b = b
            self.a = a
            self.z = z

    class ResolveCreateStream:
        # Looking for an EEG-stream
        def __init__(self) -> None:
            from pylsl import StreamInlet, resolve_byprop
            print("looking for an EEG stream...")
            self.streams = resolve_byprop('type', 'EEG', timeout=10)
            if len(self.streams) < 1:
                raise TimeoutError('No EEG-Stream could be found, please attach EEG')
            self.fs = self.streams[0].nominal_srate()
            self.n_channels = self.streams[0].channel_count()

            # create a new inlet to read from the stream
            self.inlet = StreamInlet(self.streams[0])
            print("An inlet was successfully created")

    class MovingAverageFilter:
        def __init__(self, length_of_window, fs, n_channels):
            # lenght_of_window is how long in seconds should it average
            windowsize = length_of_window * fs
            self.b = (1 / windowsize) * np.ones((int(windowsize), 1))
            self.b = self.b.flatten()
            self.z = signal.lfilter_zi(self.b, 1)
            self.z = np.tile(self.z, (n_channels * 2, 1))
            self.a = 1

    class DecimationFilter:
        # used for downsampling, ringbuffer is necessary because chunks have different shapes
        def __init__(self, downsampling_factor):
            self.downsampling_factor = int(np.floor(downsampling_factor))
            self.ringbuffer = range(self.downsampling_factor)

        def decimate_by_dwnf(self, chunk):
            length_of_chunk = np.shape(chunk)[1]
            decimated_chunk = chunk[:, self.ringbuffer[0]::self.downsampling_factor]
            self.ringbuffer = np.roll(self.ringbuffer, length_of_chunk)
            return decimated_chunk

    def run_pipeline_dummy(chunk, filters, mov_avg_filter, dec_filter):
        # Transpose the chunks into dimensions of (channels x samples)
        chunk = np.transpose(chunk)
        ################################################################################################################
        ###                             Begin processing pipeline                                                    ###
        ###               TODO: adjust this part once the preprocessing pipeline is fixed                            ###
        ###                           This is just a dummy pipeline                                                  ###
        ################################################################################################################

        # online filtering with butterworth filter
        # filtered_chunk has the dimension of (number_of_filters, channels, samples)
        filtered_chunk = np.tile(chunk, (len(filters), 1, 1))
        for i, filter_i in enumerate(filters, start=0):
            filtered_chunk[i, :, :], filter_i.z = signal.lfilter(filter_i.b, filter_i.a, chunk, axis=1, zi=filter_i.z)

        # CAR - Rereferencing
        filtered_chunk = filtered_chunk - np.nanmean(filtered_chunk, axis=1, keepdims=True)

        # concatenate features such that (number_of_filters, channels, samples) -> (number_of_filters*channels, samples)
        concat_features = filtered_chunk[0, :, :]
        for mat in range(1, filtered_chunk.shape[0]):
            concat_features = np.concatenate((concat_features, filtered_chunk[mat, :, :]), axis=0)

        # instantaneous power by squaring
        concat_features = concat_features ** 2

        # moving average filter
        mov_avg_chunk, mov_avg_filter.z = signal.lfilter(mov_avg_filter.b, mov_avg_filter.a, concat_features, axis=1,
                                                         zi=mov_avg_filter.z)
        # logarithm of the powers
        log_chunk = np.log10(mov_avg_chunk)

        # #downsampling
        downsampled_eeg_chunk = dec_filter.decimate_by_dwnf(log_chunk)
        ################################################################################################################
        ###                                  End Processing pipeline                                                 ###
        ################################################################################################################

        # downsampled_eeg_chunk should look like (channels, downsampled_samples)
        return downsampled_eeg_chunk
