"""
Title: class file: EEG
Authors: Mirage 91
"""

# imports python
import mne
from mne import read_epochs
from mne.time_frequency import tfr_multitaper, psd_array_welch
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne.decoding import CSP
from pyxdf import load_xdf
from autoreject import AutoReject
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from numpy import isin, max, arange, mean, array, reshape, linspace, save, std
import numpy as np
from collections import defaultdict
from os.path import isfile
import os
from scipy.signal import butter, lfilter, iirnotch
from sklearn.model_selection import train_test_split
from typing import List, Dict
import pickle
    
# main
class EEG:
    def __init__(self, path:str, paradigm:Dict[str, float], tasks:List[str]):
        '''
        Initialization of the EEG class.
        Input: 
            - path: Path where the eeg data is stored in form of an .xdf file. [str]
            - paradigm: Paradigm that was used to record the data [dict]
        '''
        self._path = path
        # self._path_epochs = '/'.join(self._path[0].split('\\')[:-1]) + '/preprocessed_epo.fif'
        self._paradigm_cues = paradigm
        self._tasks = tasks
        self._eeg_stream, self._paradigm_stream, self._session_number = self._load_xdf_file()
        self._raw_mne = self._create_mne_objects()
        self._fs = 500


    def _load_xdf_file(self):
        '''
        Loading the .xdf file of the EEG data.
        Input: -
        Output:
            - eeg_stream: ...
            - paradigm_stream: ...
        '''
        eeg_streams = []
        paradigm_streams = []
        session_numbers = []

        # Find all the different session days
        folders = [os.path.join(self._path, f) for f in os.listdir(self._path) if os.path.isdir(os.path.join(self._path, f))]
        
        for idx, folder in enumerate(folders):
            xdf_paths = [os.path.join(folders[idx], xdf) for xdf in os.listdir(folders[idx]) if xdf.endswith('.xdf')]
            for xdf_path in xdf_paths:
                new_eeg_stream, new_paradigm_stream = self._get_correct_streams(xdf_path)
                eeg_streams.append(new_eeg_stream)
                # If the session date is the 30.08., then change the markers from right_hand to left_hand
                if folder == './raw_data/2024_08_30':
                    # change paradigm stream
                    for idx_, elem in enumerate(new_paradigm_stream['time_series']):
                        
                        if elem == ['right_hand']:
                            new_paradigm_stream['time_series'][idx_] = ['left_hand']
                paradigm_streams.append(new_paradigm_stream)
                session_numbers.append(folder[-10:])

        return eeg_streams, paradigm_streams, session_numbers
    
    def _get_correct_streams(self, path:str):
        '''
        Has the input path for on .xdf block an returns the eeg and paradigm stream

        Input:
            - path: Path to the EEG recording stored sa .xdf file. [str]
        Output: 
            - eeg_stream: Dictionary object that contains the EEG stream. [dict]
            - paradigm_stream: Dictionary object that contains the EEG stream. [dict]
        '''
        stream_data, _ = load_xdf(path)
        for idx, stream in enumerate(stream_data):
            if stream['info']['name'][0] == 'BrainVision RDA':  
                eeg_stream = stream_data[idx]
            elif stream['info']['name'][0] == 'paradigm':
                paradigm_stream = stream_data[idx]
        
        return eeg_stream, paradigm_stream

    def _create_mne_objects(self):

        def butter_bandpass(lowpass, highpass, fs, order=4):
            nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
            low = lowpass / nyquist
            high = highpass / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a
        
        def notch_filter(w0, Q, fs):
            b, a = iirnotch(w0=w0, Q=Q, fs=fs)
            return b, a
        
        def apply_notch_filter(data, w0, Q=30, fs=500, axis=2):
            b, a = notch_filter(w0=w0, Q=Q, fs=fs)
            filtered_data = lfilter(b, a, data, axis=axis)
            return filtered_data
        
        def apply_bandpass_filter(data, lowpass, highpass, fs, order=4, axis=2):
            # Get the filter coefficients
            b, a = butter_bandpass(lowpass, highpass, fs, order=order)
        
            # Apply the filter along the second axis (axis=1) for each channel
            filtered_data = lfilter(b, a, data, axis=axis)
        
            return filtered_data
        
        '''
        Create an MNE object for the EEG data for easier use of the built in methods.
        Input: -
        Output: 
            - raw: MNE object that contains the [list of mne.io.RawArray]
        '''
        raw_mnes = []
        for eeg_stream, paradigm_stream, session_num in zip(self._eeg_stream, self._paradigm_stream, self._session_number):
            eeg_data = eeg_stream['time_series'].T
            eeg_time = eeg_stream['time_stamps']
            paradigm_data = paradigm_stream['time_series']
            paradigm_time = paradigm_stream['time_stamps']
            if paradigm_data[-1] == ['post_run']:
                paradigm_data = paradigm_data[:-1]
                paradigm_time = paradigm_time[:-1]
            fs = float(eeg_stream['info']['nominal_srate'][0])
            self._fs = fs
            # find the first timepoint for both streams
            eeg_first_timestep = float(eeg_time[0])
            cue_first_timestep = float(paradigm_time[0])

            # determine if eeg stream started first or paradigm stream
            if eeg_first_timestep < cue_first_timestep:
                absolut_startpoint = eeg_first_timestep
            else:
                absolut_startpoint = cue_first_timestep

            eeg_time -= absolut_startpoint
            paradigm_time -= absolut_startpoint
            
            # Filter eeg data here
            eeg_data = apply_bandpass_filter(data=eeg_data, lowpass=0.5, highpass=35,
                                             fs=500, order=4, axis=1)
            
            eeg_data = apply_notch_filter(data=eeg_data, w0=50, Q=30, fs=500, axis=1)

            stream_dict = {'eeg_data':eeg_data, 'eeg_time':eeg_time,
                           'paradigm_data':paradigm_data, 'paradigm_time':paradigm_time}
            # define channel names
            channel_names = [element['label'][0] for element in eeg_stream['info']['desc'][0]['channels'][0]['channel']]

            raw = mne.io.RawArray(eeg_data,
                                mne.create_info(ch_names=channel_names,
                                                sfreq=fs,
                                                ch_types='eeg'),
                                verbose=False)
            raw_mnes.append((raw, stream_dict, session_num))
        return raw_mnes
    
    @staticmethod
    def _save_epochs_as_npy(self, filename, session_number=None):
        features = self.norm_session_epoch
        path = './features/' + filename + '_data.npy'
        save(path, features)
        labels = self.norm_session_event
        path = './features/' + filename + '_labels.npy'
        save(path, labels)
        # for epoch_data, epoch_event, session_num in zip(epochs.get_data(copy=True), epochs.events[:, -1], session_number):
        #     # epoch_data: the actual EEG data for the epoch
        #     # epoch_event: the event_id for the corresponding event
        #     epochs_data_list.append((epoch_event, epoch_data, session_num))
        #     # print(f"{epoch_event.size} {epoch_data.shape} {session_num}")

        # filepath = './features/' + filename + '.npy'
        # epochs_data_list = np.array(epochs_data_list, dtype=object)
        # save(filepath, epochs_data_list)
    
    def preprocessing(self, plot=True):
        '''
        Applying filters, dropping channels and resampling of the EEG data.
        Input: 
            - plot: Boolean variable to decide wether the MNE sensor data should be plotted.
        Output: -
        '''
        for idx, _ in enumerate(self._raw_mne):
            # drop the spatial sensor channels and define eeg setup
            self._raw_mne[idx][0].drop_channels(['x_dir', 'y_dir', 'z_dir'])
            self._raw_mne[idx][0].set_montage("standard_1020")

            # find bad channels and interpolate them for replacement
            #bad_channels = mne.preprocessing.find_bad_channels_lof(self._raw_mne[idx][0])
            #self._raw_mne[idx][0].info['bads'].extend(bad_channels)
            #self._raw_mne[idx][0].interpolate_bads()

            # rereference with common average reference (CAR)
            # self._raw_mne[idx][0].set_eeg_reference("average",projection=False)

            # apply standard high, low and notch filters
            #self._raw_mne[idx][0].filter(1., None, method='iir')  # verbose=False)  # Highpass filter at 1 Hz
            #self._raw_mne[idx][0].notch_filter(50., method='iir')  # verbose=False)  # Notch filter at 50 Hz
            #self._raw_mne[idx][0].filter(None, 45., method='iir')  # , verbose=False)  # Anti-aliasing filter at 45 Hz

            # after applying the highpass, we can resample to 200Hz
            self._raw_mne[idx][0].resample(125)

            #if plot:
            #    scale=dict(eeg=100e-6, eog=150e-6)
            #    self._raw_mne.plot(n_channels=32,scalings=scale,start=36, duration=16, show_scrollbars=False)
            #    self._raw_mne.plot_sensors(show_names=True)


    def epoching_and_rejecting(self, save_eeglab=False, save_npy=False):
        '''
        Reshaping the preprocessed EEG into Epochs and applying automated epoch rejection.
        Input: -
        Output: -
        '''
        epochs_list = []
        session_numbers = []
        for _, (raw_mne, streams, session_num) in enumerate(self._raw_mne):
            cue_names = [element[0] for element in streams['paradigm_data']]
            cue_durations = [self._paradigm_cues[cue] for cue in cue_names]
            annotations = mne.Annotations(onset=streams['paradigm_time'],
                                        duration=cue_durations,
                                        description=cue_names,
                                        )
            raw_mne.set_annotations(annotations)

            # extract events from annotations
            event_id = {'left_hand': 1, 'right_hand': 2, 'feet':3, 'mental_singing':4}
            events, event_id = mne.events_from_annotations(raw_mne, event_id=event_id)

            # create mne Epochs object
            new_epoch = mne.Epochs(
                raw=raw_mne,
                events=events,
                event_id=event_id,
                tmin=-2,
                tmax=5,
                detrend=1,
                picks=None,
                baseline=(-1.0, -0.5),
                preload=True
            )
            # Drop bad epochs
            # reject_criteria = dict(eeg=150e-6)  # 100 µV # max(self._eeg_data*0.8)
            # new_epoch.drop_bad(reject=reject_criteria)

            epochs_list.append((new_epoch, session_num))

            # append session numbers
            # new_session_nums = [session_num]*len(new_epoch)
            # session_numbers.extend(new_session_nums)

        # Concatenate epochs for further processing in MNE, if needed
        # self.epochs = mne.concatenate_epochs([new_epoch[0] for new_epoch in epochs_list])
        self.epochs_list = epochs_list
        # self._session_numbers = session_numbers
        # EEG._save_epochs_as_npy(epochs=self.epochs, filename='epoched_eeg', session_number=session_numbers)
        self.session_epochs = []
        session_dict = defaultdict(list)
        for epoch, session_num in self.epochs_list:
            session_dict[session_num].append(epoch)
        
        for key in session_dict.keys():
            # Concatenate the epochs from every session
            session_epoch = mne.concatenate_epochs([new_epoch for new_epoch in session_dict[key]])
            self.session_epochs.append((session_epoch, key))  
        print('here')  

        # Save epochs

        

    def _apply_ica(self):
        '''
        Applying ICA to the preprocessed EEG data.
        Input: -
        Output: -
        '''
        self.epochs = []
        for epoch, session_num in self.session_epochs:
            # After concatenating the epochs Autoreject the bad trials
            ica = ICA(
                n_components=None,
                method='fastica',
                fit_params=None,
                max_iter='auto',
                random_state=91
            )
            
            # Fit ICA to epochs
            ica.fit(epoch)
            # Get labels of ICA and print them
            ic_labels = label_components(epoch, ica, method="iclabel")
            print(ic_labels["labels"])
            # Plot ICA components, sources and properties
            # ica.plot_components()
            # ica.plot_sources(self.epochs, show=True)
            # ica.plot_properties(self.epochs, show=True)

            # Define the ICA components that should be excluded
            lst_exclude = []
            for num, label in zip(list(range(32)), ic_labels["labels"]):
                if label != 'brain' and label != 'other':
                    lst_exclude.append(num)
                    
            ica.exclude = lst_exclude

            # Apply ICA to epochs
            print('Apply ICA now')
            epoch = ica.apply(epoch)
            epoch.apply_baseline()
            print('ICA finished!')

            # ar = AutoReject(verbose=True)
            # epoch = ar.fit_transform(epoch)

            self.epochs.append((epoch, session_num))

    def _normalize(self):
        
        
        # self.norm_dict = defaultdict(list)
        self.norm_epochs = []
        norm_session_epoch = []
        norm_session_event = []
        for session, session_num in self.epochs:
            # Normalize all epochs
            session_mean = mean(session.get_data(), axis=2, keepdims=True) # [:, :, self._fs*0.5 : self._fs*2]
            session_std = std(session.get_data(), axis=2, keepdims=True) # [:, :, self._fs*0.5 : self._fs*2]

            norm_session_epoch.append((session.get_data() - session_mean) / session_std)
            norm_session_event.append(session.events[:, -1])
            # self.norm_dict[session_num] = (norm_session_epoch, session.events[:, -1])
            print('here')
        self.norm_session_epoch = np.vstack(norm_session_epoch)
        self.norm_session_event = np.hstack(norm_session_event)
    
    def processing_pipeline(self):
        '''
        Pipeline function that is called to apply the preprocessing, epoching, epoch-rejecting and ICA.
        Input: -
        Output: -
        '''

        #print(f'EVENT IDs: {self.epochs_list[0].event_id}')
        #if np.isnan(self.epochs_list[0].get_data(copy=False).any()):
        #    raise ValueError('Epochs contain NaNs, please check the processing pipeline.')
        
        if 0:
            self._apply_ica()
        # Save cleaned epochs as .npy file
        # EEG._save_epochs_as_npy(self.epochs, 'cleaned_epoched_eeg', session_number=self._session_numbers)

        # Normalize data
        # self._normalize()
        # EEG._save_epochs_as_npy(self, 'normalized_cleaned_eeg')
        # Create CSP data and save it
        # self._extract_csp_and_save()


    def _extract_csp_and_save(self, n_components=6, file_name='csp_features.npy'):
        '''
        Extract CSP features from MNE epochs object and save them as .npy file
        Input: 
            - epochs:
            - n_components: 
            - file_name: 
        Output:
            - 
        '''
        from scipy.signal import butter, lfilter

        # Define the bandpass filter along the time axis
        def bandpass_filter(data, lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs  # Nyquist frequency
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            filtered_data = lfilter(b, a, data, axis=2)
            return filtered_data

        # Get the numpy file from the epchoched MNE data
        x = self.norm_session_epoch # epochs.get_data()
        y = self.norm_session_event
        # Apply filterbank to the data to find the best possible filterband combination
        filter_start = 1 # Hz
        filter_stop = 45 # Hz
        filter_step = 4 # Hz
        filterbank_freqs = arange(filter_start, filter_stop, filter_step)
        filterbank_freqs = [(filterbank_freqs[idx], filterbank_freqs[idx+1]) for idx in range(len(filterbank_freqs)-1)]
        # filtered_session_data = []
        # for key in epochs.keys():
        #     session_data = epochs[key]
        #     filtered_per_session = []
        csp_features = []
        csp_models = []
        for idx, filter_freqs in enumerate(filterbank_freqs):
            low_freq = filter_freqs[0]
            high_freq = filter_freqs[1]
            filtered_data = bandpass_filter(x, lowcut=low_freq, highcut=high_freq, fs=200, order=4)
            # cut out -0.5 before cue onset until 3.5 seconds after cue onset
            start_cut = int(1.5 * self._fs) # 1.5 seconds into the signal
            end_cut = int(5.5 * self._fs) # 5.5 seconds into the signal
            filtered_data = filtered_data[:, :, start_cut:end_cut]
            # append bandpass filtered data

            print('All filterbank filters applied.')
            # num_epochs = filtered_data.shape[0]
            # Calculate CSP for every filterband
            # for idx in range(num_epochs):
            # print(f'Calculating {idx+1}/{len(filtered_per_session)} filterbank CSP now... ')
            # Initialize CSP object
            csp = CSP(n_components=n_components, reg='shrinkage', log=True, cov_est='epoch')
            # Fit CSP on the data and extract features
            csp.fit(filtered_data, y)
            # csp.plot_patterns(self._raw_mne[0][0].info, ch_type="eeg", units="Patterns (AU)")

            csp_models.append(csp)
            # if idx == 0:
                # Save the CSP model weights
                # Save the entire CSP object
                # with open('classifier_results/csp_model.pkl', 'wb') as f:
                #     pickle.dump(csp, f)
            csp_features.append(csp.transform(filtered_data))
        
        with open('classifier_results/csp_model.pkl', 'wb') as f:
            pickle.dump(csp, f)
        # Create a list of tuples (label, feature) for each epoch
        # features_with_labels = [(feature, y) for feature in csp_features]
        # Add this filterband CSP to the list
        csp_features = array(csp_features)
        print('Finished')
        # Save the CSP features as .npy file
        file_path = './features/'
        filename_features = file_path + 'csp_features.npy'
        filename_labels =  file_path + 'csp_labels.npy'
        # features_with_labels = np.array(features_with_labels, dtype=object)
        save(filename_features, csp_features)
        save(filename_labels, y)
        print(f'CSP features saved to {file_path}')
        
    
    def show_erds(self):
        ''' 
        This function is called to plot the ERDS maps. They are calculated with the built-in MNE function "epoch.comute_tfr()".
        '''
        freqs = arange(2, 36)
        baseline = -1., -0.5
        vmin, vmax = -1, 1.5
        cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        channel_names = self.epochs.ch_names
        tmin=-2
        tmax=5
        #tasks = ['mental_singing', 'left_foot', 'right_foot', 'left_hand']
        
        # erds_hand = self.epochs['left_foot'].compute_tfr('multitaper', return_itc=False, freqs=freqs, average=True)
        #erds_foot.crop(-1.5,4.5).apply_baseline(baseline,mode="percent")
        # for channel in chan_of_interest:
        for task in self._tasks:
            erds_ = self.epochs[task].compute_tfr('multitaper', return_itc=False, freqs=freqs, average=True)
            erds_.apply_baseline(baseline=baseline, mode='percent')
            # erds_img = []
            # for channel in channel_names:
            #     erds_img.append(erds_.plot(picks=channel, tmin=tmin, tmax=tmax, fmin=freqs[0], fmax=freqs[-1],
            #                                         baseline=baseline, mode='percent', vlim=(-1, 1.5), cnorm=cnorm,
            #                                         cmap='RdBu', colorbar=True,title=task + channel, show=False, verbose=False)[0])
            
            # for idx, img in enumerate(erds_img):
            #     img_name = task + '_' + chan_of_interest[idx] + '.png'
            #     img.savefig(img_name)
            # Define the arrangement of the channels on the EEG cap
            channel_positions = {'AFz': (0, 4),
                                 'F5': (1, 1),
                                'F3': (1, 2),
                                'F1': (1, 3),
                                'Fz': (1, 4),
                                'F2': (1, 5),
                                'F4': (1, 6),
                                'F6': (1, 7),
                                'FC5': (2, 1),
                                'FC3': (2, 2),
                                'FC1': (2, 3),
                                'FCz': (2, 4),
                                'FC2': (2, 5),
                                'FC4': (2, 6),
                                'FC6': (2, 7),
                                'C5': (3, 1),
                                'C3': (3, 2),
                                'C1': (3, 3),
                                'Cz': (3, 4),
                                'C2': (3, 5),
                                'C4': (3, 6),
                                'C6': (3, 7),
                                'CP5': (4, 1),
                                'CP3': (4, 2),
                                'CP1': (4, 3),
                                'CPz': (4, 4),
                                'CP2': (4, 5),
                                'CP4': (4, 6),
                                'CP6': (4, 7),
                                'P1': (5, 3),
                                'Pz': (5, 4),
                                'P2': (5, 5),
                                }

            # Create a figure to plot the ERD maps
            fig, ax = plt.subplots(6, 8, figsize=(16, 10))  # Adjust the figure size as needed
            fig.suptitle(f'ERDS maps for {task}')

            # Hide all axes
            for a in ax.flatten():
                a.axis('off')

            # Assuming erds_img is a list of data arrays for each channel
            for idx, channel in enumerate(channel_names):
                pos = channel_positions[channel]
                data = erds_.data[erds_.ch_names.index(channel)]  # Extract the data for the channel
                tmin_idx = int((tmin - erds_.times[0]) / (erds_.times[1] - erds_.times[0]))
                tmax_idx = int((tmax - erds_.times[0]) / (erds_.times[1] - erds_.times[0]))
                fmin_idx = int((freqs[0] - erds_.freqs[0]) / (erds_.freqs[1] - erds_.freqs[0]))
                fmax_idx = int((freqs[-1] - erds_.freqs[0]) / (erds_.freqs[1] - erds_.freqs[0]))
                
                # Plot the data
                im = ax[pos].imshow(data[fmin_idx:fmax_idx, tmin_idx:tmax_idx], aspect='auto', origin='lower', 
                                    extent=[tmin, tmax, freqs[0], freqs[-1]], cmap='RdBu')
                ax[pos].set_title(channel)
                ax[pos].set_xticks([-2, 0, 4.5])
                ax[pos].set_xticklabels(['-2', '0', '4.5'])
                ax[pos].axis('on')

            # Add a single colorbar for the entire figure
            # fig.colorbar(im, ax=ax.ravel().tolist(), location='right', orientation='vertical', fraction=0.05, pad=1)
            
            # # Plot the ERD maps in the corresponding positions
            # for idx, channel in enumerate(channel_names):
            #     pos = channel_positions[channel]
            #     fig_canvas = erds_img[idx].canvas
            #     fig_canvas.draw()
            #     img_data = np.frombuffer(fig_canvas.tostring_rgb(), dtype=np.uint8)
            #     img_data = img_data.reshape(fig_canvas.get_width_height()[::-1] + (3,))
                
            #     ax[pos].imshow(img_data, aspect='auto')
            #     ax[pos].set_title(channel)
            #     ax[pos].axis('on')

            # Adjust layout
            # plt.tight_layout()

        # erds_foot = self.epochs['right_foot'].compute_tfr('multitaper', return_itc=False, freqs=freqs, average=True)
        # #erds_foot.crop(-1.5,4.5).apply_baseline(baseline,mode="percent")
        # # for channel in chan_of_interest:
        # erds_foot_img = []
        # for channel in chan_of_interest:
        #     erds_foot_img.append(erds_foot.plot(picks=channel, tmin=tmin, tmax=tmax, fmin=freqs[0], fmax=freqs[-1],
        #                                         baseline=baseline, mode='percent', vlim=(-1, 1.5), cnorm=cnorm,
        #                                         cmap='RdBu', colorbar=True,title='right_foot ' + channel, show=False, verbose=False)[0])
        
        # for idx, img in enumerate(erds_foot_img):
        #     img_name = 'right_foot_' + chan_of_interest[idx] + '.png'
        #     img.savefig(img_name)




if __name__ == '__main__':
    ...
