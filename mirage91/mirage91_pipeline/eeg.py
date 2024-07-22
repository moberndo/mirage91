"""
Title: class file: EEG
Authors: Mirage 91
"""
# imports python
import mne
from mne import read_epochs
from mne.time_frequency import tfr_multitaper, psd_array_welch
from mne.preprocessing import ICA
from pyxdf import load_xdf
from autoreject import AutoReject
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from numpy import isin, max, arange, mean, array, reshape, linspace
from os.path import isfile

# main
class EEG:
    def __init__(self, path:str, paradigm:dict):
        '''
        Initialization of the EEG class.
        Input: 
            - path: Path where the eeg data is stored in form of an .xdf file. [str]
            - paradigm: Paradigm that was used to record the data [dict]
        '''
        self._path = path
        # self._path_epochs = '/'.join(self._path.split('\\')[:-1]) + '/epochs.fif'
        self._paradigm_cues = paradigm
        self._eeg_stream, self._paradigm_stream = self._load_xdf_file()
        # self._eeg_data = self._eeg_stream['time_series'].T
        # self._eeg_time = self._eeg_stream['time_stamps']
        # self._paradigm_data = self._paradigm_stream['time_series']
        # self._paradigm_time = self._paradigm_stream['time_stamps']
        # self._fs = float(self._eeg_stream['info']['nominal_srate'][0])
        self._raw_mne = self._create_mne_objects()
        self.featureX = None
        self.feautreY = None

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

        for path in self._path:
            new_eeg_stream, new_paradigm_stream = self._get_correct_streams(path)
            eeg_streams.append(new_eeg_stream)
            paradigm_streams.append(new_paradigm_stream)

        return eeg_streams, paradigm_streams
    
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
        '''
        Create an MNE object for the EEG data for easier use of the built in methods.
        Input: -
        Output: 
            - raw: MNE object that contains the [list of mne.io.RawArray]
        '''
        raw_mnes = []
        for eeg_stream, paradigm_stream in zip(self._eeg_stream, self._paradigm_stream):
            eeg_data = eeg_stream['time_series'].T
            eeg_time = eeg_stream['time_stamps']
            paradigm_data = paradigm_stream['time_series']
            paradigm_time = paradigm_stream['time_stamps']
            fs = float(eeg_stream['info']['nominal_srate'][0])
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

            # define channel names
            channel_names = [element['label'][0] for element in eeg_stream['info']['desc'][0]['channels'][0]['channel']]

            raw = mne.io.RawArray(eeg_data,
                                mne.create_info(ch_names=channel_names,
                                                sfreq=fs,
                                                ch_types='eeg'),
                                verbose=False)
            raw_mnes.append(raw)
        return raw_mnes
    
    def _preprocessing(self, plot=False):
        '''
        Applying filters, dropping channels and resampling of the EEG data.
        Input: 
            - plot: Boolean variable to decide wether the MNE sensor data should be plotted.
        Output: -
        '''
        for idx, _ in enumerate(self._raw_mne):
            # drop the spatial sensor channels and define eeg setup
            self._raw_mne[idx].drop_channels(['x_dir', 'y_dir', 'z_dir'])
            self._raw_mne[idx].set_montage("standard_1020")

            # find bad channels and interpolate them for replacement
            bad_channels = mne.preprocessing.find_bad_channels_lof(self._raw_mne[idx])
            self._raw_mne[idx].info['bads'].extend(bad_channels)
            self._raw_mne[idx].interpolate_bads()

            # rereference with common average reference (CAR)
            self._raw_mne[idx].set_eeg_reference("average",projection=False)

            # apply standard high, low and notch filters
            self._raw_mne[idx].filter(1., None, method='iir', )  # verbose=False)  # Highpass filter at 1 Hz
            self._raw_mne[idx].notch_filter(50., method='iir')  # verbose=False)  # Notch filter at 50 Hz
            self._raw_mne[idx].filter(None, 80., method='iir')  # , verbose=False)  # Anti-aliasing filter at 80 Hz

            # after applying the highpass, we can resample to 250Hz
            self._raw_mne[idx].resample(250)
        
        # if plot:
        #     scale=dict(eeg=100e-6, eog=150e-6)
        #     self._raw_mne.plot(n_channels=32,scalings=scale,start=36, duration=16, show_scrollbars=False)
        #     self._raw_mne.plot_sensors(show_names=True)

    def _epoching_and_rejecting(self):
        '''
        Reshaping the preprocessed EEG into Epochs and applying automated epoch rejection.
        Input: -
        Output: -
        '''
        cue_names = [element[0] for element in self._paradigm_data]
        cue_durations = [self._paradigm_cues[cue] for cue in cue_names]
        annotations = mne.Annotations(onset=self._paradigm_time,
                                    duration=cue_durations,
                                    description=cue_names,
                                    )
        self._raw_mne.set_annotations(annotations)

        # extract events from annotations
        events, event_id = mne.events_from_annotations(self._raw_mne)
        events = events[isin(events[:, 2], (1,2)), :]   # remove pre_cue and pause periods

        # create mne Epochs object
        self.epochs = mne.Epochs(
            raw=self._raw_mne,
            events=events,
            event_id=dict(foot=event_id['foot'], hand=event_id['hand']),
            tmin=-2,
            tmax=5,
            detrend=1,
            picks=None,
            baseline=(-1.5, -0.5),
            preload=True
        )
        #reject_criteria = dict(eeg=max(self._eeg_data*0.8))  # 100 µV
        #epochs.drop_bad(reject=reject_criteria)
        ar = AutoReject()
        self.epochs = ar.fit_transform(self.epochs)

        # save processed epochs
        self.epochs.save(self._path_epochs)

    def _apply_ica(self):
        '''
        Applying ICA to the preprocessed EEG data.
        Input: -
        Output: -
        '''
        ica = ICA(
            n_components=len(self.epochs.ch_names),
            method='fastica',
            fit_params=None,
            max_iter='auto',
            random_state=42
        )
        
        ica.fit(self.epochs)
        ica.plot_components(show=False)
        ica.plot_sources(self.epochs, show=False)
        ica.plot_properties(self.epochs, show=False)
        exclude = [0, 1, 2, 3, 4, 9, 21]
        self.epochs = ica.apply(self.epochs, exclude=exclude)

    def processing_pipeline(self):
        '''
        Pipeline function that is called to apply the preprocessing, epoching, epoch-rejecting and ICA.
        Input: -
        Output: -
        '''
        if isfile(self._path_epochs):
            self.epochs = read_epochs(self._path_epochs)
        else:
            self._preprocessing()
            self._epoching_and_rejecting()
        self._apply_ica()

    def extract_features(self):
        '''
        Extracting features from the processed and epoched EEG data.
        Input: -
        Output: -
        '''
        ### Feature 1: Bandpower ###
        # find optimal frequency bands
        freq_bands = [(3, 5),  # lower and upper freq-band limits
                      (10, 15),]
        # cut epochs dependend on frequency bands
        freq_epochs = [self.epochs.filter(l_freq=band[0], h_freq=band[1], method='iir') for band in freq_bands]
        # square the time-data and calculate the average
        # epoch.get_data() returns a numpy arrray of shape (num_epochs, num_channels, num_timepoints)
        bp_features = array([mean(epoch.get_data(copy=False)**2, axis=2) for epoch in freq_epochs])
        bp_features = reshape(bp_features, newshape=(bp_features.shape[1], -1))
        ground_truth = self.epochs.events[:, 2]

        self.featureX = bp_features
        self.featureY = ground_truth

    def get_features(self):
        '''
        This function is called to obtain the features from the EEG class that are needed to train a classifier.
        Input: -
        Output: 
            - featureX: Feature vector of the EEG data
            - featureY: Target vector according to featureX.
        '''
        if self.featureX is not None and self.featureY is not None:
            return self.featureX, self.featureY
        else:
            raise ValueError('Features have not been extracted yet. Use the "extract_features()" method.')
    
    def show_erds(self):
        ''' 
        This function is called to plot the ERDS maps. They are calculated with the built-in MNE function "epoch.comute_tfr()".
        '''
        freqs = arange(2, 36)
        baseline = -1.5, -0.5
        vmin, vmax = -1, 1.5
        cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        #epochs.drop_bad(reject=reject_criteria)
        ## ERDS maps foot C3,Cz,C4
        chan_of_interest = ['C3', 'C1', 'Cz', 'C2', 'C4']
        tmin=-2
        tmax=5
        erds_hand = self.epochs['hand'].compute_tfr('multitaper', return_itc=False, freqs=freqs, average=True)
        #erds_foot.crop(-1.5,4.5).apply_baseline(baseline,mode="percent")
        # for channel in chan_of_interest:
        erds_hand_img = []
        for channel in chan_of_interest:
            erds_hand_img.append(erds_hand.plot(picks=channel, tmin=tmin, tmax=tmax, fmin=freqs[0], fmax=freqs[-1],
                                                baseline=baseline, mode='percent', vlim=(-1, 1.5), cnorm=cnorm,
                                                cmap='RdBu', colorbar=True,title='hand' + channel, show=False, verbose=False)[0])
        
        for idx, img in enumerate(erds_hand_img):
            img_name = 'hand_' + chan_of_interest[idx] + '.png'
            img.savefig(img_name)

        erds_foot = self.epochs['foot'].compute_tfr('multitaper', return_itc=False, freqs=freqs, average=True)
        #erds_foot.crop(-1.5,4.5).apply_baseline(baseline,mode="percent")
        # for channel in chan_of_interest:
        erds_foot_img = []
        for channel in chan_of_interest:
            erds_foot_img.append(erds_foot.plot(picks=channel, tmin=tmin, tmax=tmax, fmin=freqs[0], fmax=freqs[-1],
                                                baseline=baseline, mode='percent', vlim=(-1, 1.5), cnorm=cnorm,
                                                cmap='RdBu', colorbar=True,title='foot ' + channel, show=False, verbose=False)[0])
        
        for idx, img in enumerate(erds_foot_img):
            img_name = 'foot_' + chan_of_interest[idx] + '.png'
            img.savefig(img_name)







if __name__ == '__main__':
    path = r'C:\Users\Markus\Mirage91\Program\venv\mirage91\_storage\block_patrick.xdf'
    paradigm = {'pre_cue':2.0, 'hand':4.5, 'foot':4.5, 'pause':2.5}
    eeg = EEG(path=path, paradigm=paradigm)
    eeg.processing_pipeline()
    eeg.show_erds()
    plt.show()