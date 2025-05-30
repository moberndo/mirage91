"""
Title: Class Online processing pipeline
Authors: Mirage 91
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, lfilter, resample


class OnlineProcessingPipeline:
    class OnlineFilter:
        # Create an online filter
        def __init__(self, filterorder, cutoff_freq, fs, btype, ftype, n_channels):
            b, a = signal.iirfilter(filterorder, Wn=cutoff_freq, fs=fs, btype=btype, ftype=ftype)
            z = signal.lfilter_zi(b, a)
            z = np.tile(z, (n_channels, 1))
            print('here')
            self.b = b
            self.a = a
            self.z = z

    class NotchFilter:
        def __init__(self, n_channels):
            b, a = signal.iirnotch(50, 30, fs=500)
            z = signal.lfilter_zi(b, a)
            z = np.tile(z, (n_channels, 1))
            self.b = b
            self.a = a
            self.z = z

    class ResolveCreateStream:
        # Looking for an EEG-stream
        def __init__(self) -> None:
            from pylsl import StreamInlet, resolve_byprop
            print("Looking for an EEG stream...")
            self.streams = resolve_byprop(prop='type', value='EEG', timeout=10)
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
        
    # def apply_pipeline(chunk, filters, dec_filter, buffer_size=350, output_size=267,
    #                    lowpass=1, highpass=45):
    #     def butter_bandpass(lowpass, highpass, fs, order=4):
    #         nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    #         low = lowpass / nyquist
    #         high = highpass / nyquist
    #         b, a = butter(order, [low, high], btype='band')
    #         return b, a
    #
    #     def apply_bandpass_filter(data, lowpass, highpass, fs, order=4):
    #         # Get the filter coefficients
    #         b, a = butter_bandpass(lowpass, highpass, fs, order=order)
    #         # Apply the filter along the second axis (axis=1) for each channel
    #         filtered_data = lfilter(b, a, data, axis=1)
    #         return filtered_data
    #
    #     # Cut buffer to buffer size
    #     buffer = chunk[:,-buffer_size:]
    #     filtered_buffer = apply_bandpass_filter(buffer, lowpass=lowpass, highpass=highpass, fs=512)[:,:output_size]
    #     filtered_buffer = np.reshape(filtered_buffer, newshape=(1, 1, 32, output_size)).astype('double')

    # Normalize buffer
    #buffer_mean = np.mean(filtered_buffer, axis=1, keepdims=True) # [:, :, self._fs*0.5 : self._fs*2]
    #buffer_std = np.std(filtered_buffer, axis=1, keepdims=True) # [:, :, self._fs*0.5 : self._fs*2]
    #filtered_buffer = (filtered_buffer - buffer_mean) / buffer_std
    #filtered_buffer = np.reshape(filtered_buffer, newshape=(1, 32, buffer_size))

    def apply_pipeline(chunk, filters, filter_states, buffer_size=2000):
        ###########################
        # Filtering
        # Define filter params
        b_notch, a_notch = filters[0]
        notch_state = filter_states[0]
        b_bp, a_bp = filters[1]
        bp_state = filter_states[1]

        # Apply notch
        if notch_state is not None:
            chunk, notch_state = signal.lfilter(b_notch, a_notch, chunk, zi=notch_state)

        # Apply bandpass
        if bp_state is not None:
            filtered_data, bp_state = signal.lfilter(b_bp, a_bp, chunk, zi=bp_state)
        
        # Resample to 200 Hz
        num_new_samples = int(buffer_size * 125 / 500)
        filtered_data = resample(filtered_data, num=num_new_samples, axis=1)

        return filtered_data, [notch_state, bp_state]

    def apply_pipeline_ML(chunk, filters, notch_filter, buffer_size=300, CSP_filter=[], CSP_models=[]):
        def butter_bandpass(lowpass, highpass, fs, order=4):
            nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
            low = lowpass / nyquist
            high = highpass / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a
        
        def apply_bandpass_filter(data, lowpass, highpass, fs, order=4):
            # Get the filter coefficients
            b, a = butter_bandpass(lowpass, highpass, fs, order=order)
        
            # Apply the filter along the second axis (axis=1) for each channel
            filtered_data = lfilter(b, a, data, axis=2)
        
            return filtered_data
        # 
        # Cut buffer to buffer size
        # # buffer = chunk[:,-buffer_size:]
        #
        # filtered_buffer = apply_bandpass_filter(chunk, lowpass=1, highpass=45, fs=512)[:,:267]
        # filtered_buffer = np.reshape(filtered_buffer, newshape=(1, 1, 32, 267)).astype('double')

        ####edited
        #Apply the filter along the second axis (axis=1) for each channel
        # print('here')
        chunk, notch_filter.z = signal.lfilter(notch_filter.b, notch_filter.a, chunk, axis=1, zi=notch_filter.z)
        filtered_buffer, filters.z = signal.lfilter(filters.b, filters.a, chunk, axis=1, zi=filters.z)
        filtered_buffer = filtered_buffer[:, -buffer_size:]
        # Normalize buffer
        buffer_mean = np.mean(filtered_buffer, axis=1, keepdims=True) # [:, :, self._fs*0.5 : self._fs*2]
        buffer_std = np.std(filtered_buffer, axis=1, keepdims=True) # [:, :, self._fs*0.5 : self._fs*2]
        filtered_buffer = (filtered_buffer - buffer_mean) / buffer_std
        filtered_buffer = np.reshape(filtered_buffer, newshape=(1, 32, buffer_size))
        # Resample to 200 Hz
        num_new_samples = int(buffer_size * 200 / 500)
        buffer = resample(filtered_buffer, num=num_new_samples, axis=2)
        # Filter for CSP
        csp_buffer = []
        for filter in CSP_filter:
            csp_buffer.append(apply_bandpass_filter(data=buffer, lowpass=filter[0], highpass=filter[1], fs=200))

        csp_features = []
        for idx, buff in enumerate(csp_buffer):
            csp_features.append(CSP_models[idx](buff))

        # reshape csp_features
        buffer = ...
        # Option 1:
        # Return filter state
        ...

        return buffer, notch_filter, filters  # shape of filtered_buffer is [1, 1, 32, 267]

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
