import os
import numpy as np
import array
import time
import scipy.io as scio

from struct import unpack
from pybmi.data.brPY.brpylib import NsxFile, NevFile
from pybmi.utils.utils import npc_remove, check_params, check_file
from pybmi.signals.spectrogram import pmtm, tfrscalo
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial


class BMIReader():
    """
    Reading the neural data and behavioral data from the BMI system.
    The neural data support mat files or NSx files from Blackrock,
    when reading NSx files, use brPY toolkit provided by Blackrock
    Microsystems. Be note that the value the brPY get is in unit of uV.
    It is 1/4 of the raw values; the behavioral data support mat
    files and BMI format file (BMI file contain a header and the
    behavioral data), get the details from the bmisoft system.

    Parameters
    ----------
    neu_binsize : float
        neu_binsize specified the length of the neural data used to
        compute the LFP. A larger bin size can get more accurate LFP,
        but less online compitiable. Unit: seconds.
    beh_binsize : float
        beh_binsize specified the time resolution of the processed BMI data.
        Normally it equal to the bin size of corresponding neural data.
        Unit: seconds.
    nbands : int
        Number of frequency bands will be generated.
    method : str, optional
        Method used to compute the spectrogram. It can be either
        'pmtm', 'wavelet', or 'count'. Default: 'pmtm'.
        When 'pmtm' is used, the additional parameters needed:
            nw : float, optional
                The "time-bandwidth product" for the DPSS used as data windows.
                Typical choices are 2, 5/2, 3, 7/2, or 4. Default: 2.5.
            nfft : int, optional
                Specifies the FFT length used to calculate the PSD estimates.
                Default: 1024.
            bands : {list, tuple, ndarray}, optional
                The frequency bands desired. Default: [[70, 200]]
        When 'wavelet' is used, the addtional parameters needed:
            edge_size : float, optional
                The time length used to reduce the edge effects in the
                scalogram calculation. The data length used to compute
                scalogram equals to [binsize + 2 * edge_size]. Unit: second.
                Default: 0.0.
            wave : int, optional
                Half length of the Morlet analyzing wavelet at coarsest
                scale. If WAVE = 0, the Mexican hat is used. WAVE can also be
                a vector containing the time samples of any bandpass
                function, at any scale. Default: 7.
            fmin, fmax : float, optional
                Respectively lower and upper frequency bounds of
                the analyzed signal. These parameters fix the equivalent
                frequency bandwidth (expressed in Hz).
                FMIN and FMAX must be > 0 and <= fs / 2.
            ncontext : int, optional
                The number of context for the current time step. Note that only
                history are considered to be the context. Default: 0.
    label_success : int, optional
        label_success is the label used to point out the success trial in the
        behavioral data.
    label_failure : int, optional
        label_failure is the label used to point out the failure trial in the
        behavioral data.
    njobs : int, optional
        NJOBS define how many workers processes parallelly to calculate
        the LFP.
    verbose : bool, optional
        To print a log on the terminal which contain the basic
        information of the readed files, e.g., the header, file size,
        file version etc. Default: False.

    Attributes
    ----------
    h : dict
        Readed header information from specified BMI files. It contain
        the information of experiment like subject, paradigm, date,
        comment and so on, also, the information of data structure is
        included, such as the dimension of the label, the dimension of
        spike data, the dimension of kinematic data and sampling rate.
    raw : ndarray
        The data matrix readed from the BMI file, which didn't do any
        processing.
    data : ndarray
        Processed data matrix. e.g., binned, masked out some labels.
    stat : dict
        For the BCI paradigm, some statistics data can analyzed from
        the raw data, e.g., success rate, number of trials count. The
        stat dict contain these statistics information for supported
        BCI paradigm.

    Examples
    --------
    >>> path = 'MonkeyData\\201907031450.bmi'
    >>> reader = BMIReader(path, verbose=True)
    >>> bmi = reader.read(bin_size=0.1, columns=6, analyze=False)
    >>> reader.analysis(verbose=True)
    >>> masked, index = reader.filter((1, 4))
    """
    # TODO: Add a parameter dictionary instead of this list.
    def __init__(self, neu_binsize, beh_binsize, nbands, method='pmtm', nw=2.5,
                 nfft=1024, bands=[[70, 200]], edge_size=0.0, wave=7, fmin=10,
                 fmax=120, ncontext=0, label_success=3, label_failure=4,
                 is_single_trial=False, njobs=1, verbose=False):
        # ------------------------------
        # Parameters of neural reading
        # ------------------------------
        self.neu_binsize = neu_binsize
        self.nbands = nbands
        # Parameters of PMTM method.
        self.pmtm = {}
        self.pmtm['nw'] = nw
        self.pmtm['nfft'] = nfft
        self.pmtm['band'] = bands
        # Parameters of WAVELET method.
        self.wavelet = {}
        self.wavelet['edge_size'] = edge_size
        self.wavelet['wave'] = wave
        self.wavelet['frange'] = [fmin, fmax]
        self.wavelet['ncontext'] = ncontext

        # ---------------------------------
        # Parameters of behavioral reading
        # ---------------------------------
        # Supported file version of BMI files.
        self.beh_binsize = beh_binsize
        self.label_success = label_success
        self.label_failure = label_failure
        self.support_versions = ['v1.0', 'v1.1', 'v1.2']
        self.reading_function = [self._v1dot0, self._v1dot1, self._v1dot2]
        self._init_header()
        # Statistics dict which contain the Success Rate,
        # Total Number of Trials, Hold Time, Failed Time,
        # Average Time of Each Successed Trial.
        self._init_stats()

        # ---------------------------------
        # Parameters of data properties
        # ---------------------------------
        self.is_single_trial = is_single_trial

        # Number of cores to do the reading task.
        self.njobs = njobs
        self.verbose = verbose

        # Check parameters validation
        self.method = check_params(method, ['pmtm', 'wavelet', 'count'],
                                   'method')
        if self.method == 'pmtm':
            assert nbands == len(bands), "Error number of frequency bands."

    def beh_read(self, file_path=None, target='p', fs_timestamp=30000,
                 analyze=True, save_mat=True):
        """
        Reading the behavioral data from the file path.

        Parameters
        ----------
        filepath : str, optional
            The path of bmi file which will be readed. If it is None, a UI will
            pop out to ask user select one file. Default: None.
        target : str, optional
            The movement target will be generate. 'p' means position, 'v' means
            velocity, and 'a' means acceleration. Default: 'p'.
        fs_timestamp : int, optional
            Sampling rate of time stamp. By default, the time stamp comes from
            recording NSP, which usually equal to 30000.
        analyze : bool, optional
            Analyze the readed BMI raw data by the provided label and paradigm.
            After analysis, statistics parameter will be stored. For example,
            when paradigm is Center-Out, Success Rate, Total Number of Trials,
            Hold Time, Failed Time, Average Time of Each Successed Trial etc.
            will be calculated.
        save_mat : bool, optional
            Choose whether need to save processed LFP as a mat file. Consider
            the time-consuming of calculating LFP from NSx, saving the LFP to
            mat format file will boost loading speed next time.
        """
        ext = (("MATLAB file", "*.mat"), ("BMI files", "*.bmi"),
               ("BMI files", "*.bmi2"), ("all files", "*.*"))
        file_path = check_file(file_path, "Choose a behavioral file...", ext)
        assert os.path.exists(file_path), \
            'The behavioral file \'' + file_path + '\' does not exist!'
        # File extension used as the sambol. If it's 'mat', load the
        # behavioral data directly; if it's 'bmi' or 'bmi2', the data
        # need readed according different file version.
        path, fname = os.path.split(file_path)
        name, ext = os.path.splitext(fname)
        # Reading BMI files.
        if ext == '.mat':
            # Load already processed motion data.
            data = scio.loadmat(file_path)
            self.binned_motion = data['motion']
            self.binned_bstamp = data['stamps'].astype(np.float32)
            self.binned_labels = data['labels']
            self.binned_spikes = data['spikes']
            self.binned_target = data['target']

            # Check if the velocity is needed.
            if 'v' in target:
                velocity = np.zeros((self.binned_motion.shape))
                if self.is_single_trial:
                    velocity[:, 1:] = self.binned_motion[:, 1:] - \
                        self.binned_motion[:, :-1]
                else:
                    velocity[1:] = self.binned_motion[1:] - \
                        self.binned_motion[:-1]
                velocity /= self.beh_binsize
            # Apply the target.
            if target == 'p':
                # NOTE: The raw input of mat file must be position.
                pass
            elif target == 'v':
                self.binned_motion = velocity
            elif target == 'pv':
                self.binned_motion = np.concatenate(
                    (self.binned_motion, velocity), axis=-1
                )
        else:
            tic = time.time()
            beh_filesize = os.path.getsize(file_path)
            with open(file_path, 'rb') as f:
                # Checking the possible version of the file.
                version = check_params(
                    npc_remove(f.read(32), npc=b'\x00'),
                    self.support_versions, 'version'
                )
                # Get the right version of the reading function.
                i = self.support_versions.index(version.lower())
                # Process different version of BMI files.
                raw_data, data_points = \
                    self.reading_function[i](f, beh_filesize)
            # Make sure the timestamp is single increase, not restart.
            i1 = self.header['index_stamps'][0]
            i2 = self.header['index_stamps'][1] + 1
            ts_step = raw_data[1:, i1:i2] - raw_data[:-1, i1:i2]
            # If the timestamp is restart, then the ts_step should
            # contain values smaller than 0. The index of this value
            # is the restart point.
            start_list = np.array(
                [0, *np.where(ts_step < 0)[0].tolist(), data_points]
            )
            # Find the max length without restart.
            i = np.argmax(start_list[1:] - start_list[:-1])
            # Update raw data.
            raw_data = raw_data[start_list[i]:start_list[i + 1]]
            # When velocity required.
            if 'v' in target:
                i1 = self.header['index_motion'][0]
                i2 = self.header['index_motion'][1] + 1
                velocity = np.zeros((raw_data.shape[0], i2 - i1))
                velocity[1:] = raw_data[1:, i1:i2] - raw_data[:-1, i1:i2]
                # Put the velocity to the end column of the raw data.
                raw_data = np.concatenate((raw_data, velocity), axis=-1)
            # Calculate the recording time duration.
            if self.header['fs_motion'] > 0:
                mins = (data_points / self.header['fs_motion']) // 60
                secs = (data_points / self.header['fs_motion']) - mins * 60
            else:
                mins, secs = 0, 0
            # Bin data
            binned_data = self._bin_motion(raw_data, fs_timestamp)
            # Timestamps
            i1 = self.header['index_stamps'][0]
            i2 = self.header['index_stamps'][1] + 1
            self.binned_bstamp = binned_data[:, i1:i2] / fs_timestamp
            # Labels
            i1 = self.header['index_labels'][0]
            i2 = self.header['index_labels'][1] + 1
            self.binned_labels = binned_data[:, i1:i2]
            # Spikes
            i1 = self.header['index_spikes'][0]
            i2 = self.header['index_spikes'][1] + 1
            self.binned_spikes = binned_data[:, i1:i2]
            # Target
            i1 = self.header['index_target'][0]
            i2 = self.header['index_target'][1] + 1
            self.binned_target = binned_data[:, i1:i2]
            # Motion
            i1 = self.header['index_motion'][0]
            i2 = self.header['index_motion'][1] + 1
            if target == 'p':
                self.binned_motion = binned_data[:, i1:i2]
            elif target == 'v':
                self.binned_motion = binned_data[:, (i1 - i2):]
            elif target == 'pv':
                self.binned_motion = np.concatenate(
                    (binned_data[:, i1:i2], binned_data[:, (i1 - i2):]),
                    axis=-1
                )
            toc = time.time()

            if self.verbose:
                self.print(raw_data, path, name, ext, data_points,
                           (mins, secs), (tic, toc), analyze)
            if save_mat:
                scio.savemat(os.path.join(path, name + '.mat'),
                             {'motion': self.binned_motion,
                              'stamps': self.binned_bstamp,
                              'labels': self.binned_labels,
                              'spikes': self.binned_spikes,
                              'target': self.binned_target})
        return self.binned_motion, self.binned_bstamp

    def neu_read(self, file_path, ch_index=None, save_mat=True):
        ext = (("MATLAB file", "*.mat"), ("NSx files", "*.ns3"),
               ("NSx files", "*.ns5"), ("all files", "*.*"))
        file_path = check_file(file_path, "Choose a neural data file...", ext)
        assert os.path.exists(file_path), \
            'The provided file \'' + file_path + '\' does not exist!'
        # File extension used as the sambol. If it's 'mat', load the
        # lfp directly; if it's NSx or bin, the LFP need computed from
        # the raw data, which may cost lots of time.
        path, fname = os.path.split(file_path)
        name, ext = os.path.splitext(fname)
        # Reading neural files.
        if ext == '.mat':
            # Load already processed LFP.
            data = scio.loadmat(file_path)
            self.binned_neural = data['neural']
            self.binned_nstamp = data['stamps']
            self.header['fs_neural'] = data['fs']
            # Reshape the nstamp to equal (T, 1)
            if self.binned_nstamp.shape[0] == 1:
                self.binned_nstamp = self.binned_nstamp.T

            # Make sure the neural data have the same length with time stamps.
            if self.is_single_trial:
                # When data formed by single trial, the 1st dim is number
                # of trials, the 2nd dim is the number of bins of each trial.
                assert self.binned_nstamp.shape[0] == \
                    self.binned_neural.shape[0] * self.binned_neural.shape[1],\
                    "Error shape of readed neural data."
            else:
                # The 1st dim of data is concated by all trials' bins.
                assert self.binned_nstamp.shape[0] == \
                    self.binned_neural.shape[0], \
                    "Error shape of readed neural data."
        else:
            # Need to process first.
            if 'ns' in ext:
                # Indicate the save name is spike data.
                name += '.lfp'
                # Read raw data of NSx
                nsx_file = NsxFile(file_path)
                raw_data = nsx_file.getdata()
                # Close nsx file
                nsx_file.close()
                # Start point of the neural timestamp
                t0 = int(raw_data['data_headers'][0]['Timestamp'])
                # Useful recording parameters of neural data.
                self.header['fs_neural'] = int(raw_data['samp_per_s'])
                # The data part.
                raw_data = raw_data['data'].T
                if ch_index is not None:
                    raw_data = raw_data[:, ch_index]
                # Calculate the timestep of raw data.
                self.binned_nstamp = np.linspace(
                    t0, len(raw_data) - 1, len(raw_data), dtype=int
                ) / self.header['fs_neural']
            elif ext == '.nev':
                # Indicate the save name is spike data.
                name += '.spike'

                tic = time.time()
                # Read raw data of NEV.
                nev_file = NevFile(file_path)
                # Get the wanted channel's data.
                elec_ids = 'all' if ch_index is None else \
                    (ch_index + 1).tolist()
                raw_data = nev_file.getdata(elec_ids, 'noread')['spike_events']
                # Close nev file.
                nev_file.close()
                # Record the time of reading nev file.
                toc = time.time()
                print('The load time was for NEV file was %.2f seconds' %
                      (toc - tic))

                # Check if there are some channels have no activity.
                def append_missed_chn(data: dict, ch: int):
                    data['ChannelID'].append(ch)
                    data['TimeStamps'].append([])
                    data['NEUEVWAV_HeaderIndices'].append(None)
                    data['Classification'].append([])
                    return data

                if len(elec_ids) > len(raw_data['ChannelID']):
                    # Find out the id of missed channel.
                    sorted_id = np.sort(np.asarray(raw_data['ChannelID']))
                    # Check the channels head, tail, and body, respectively.
                    if sorted_id[0] != elec_ids[0]:
                        # Check the head
                        for ch in range(elec_ids[0], sorted_id[0]):
                            raw_data = append_missed_chn(raw_data, ch)
                    if sorted_id[-1] != elec_ids[-1]:
                        # Check the tail
                        for ch in range(sorted_id[-1], elec_ids[-1]):
                            raw_data = append_missed_chn(raw_data, ch + 1)
                    # Check the body.
                    idx = np.where((sorted_id[1:] - sorted_id[:-1]) > 1)[0]
                    if idx.size > 0:
                        # The missed channel at the middle of sequence.
                        missed_ch = (ch_index + 1)[idx + 1]
                        for offset, ch in enumerate(missed_ch):
                            raw_data = append_missed_chn(raw_data, ch + offset)

                # Useful recording parameters of neural data.
                self.header['fs_neural'] = \
                    int(nev_file.basic_header['SampleTimeResolution'])
            elif ext == '.bin':
                with open(file_path, 'rb') as f:
                    # The first element of bin file store the number of
                    # columns of the lfp data.
                    ncol = unpack('B', f.read(1))[0]
                    # The second element of bin file store the sampling
                    # rate of the lfp data.
                    self.header['fs_neural'] = unpack('l', f.read(4))[0]
                    # Read the whole data.
                    raw_data = f.read()
                    raw_data = unpack('f' * (len(raw_data) // 4), raw_data)
                    raw_data = np.array(raw_data).reshape(-1, ncol)
                # End reading. Split the lfp data and timestamp
                self.binned_nstamp = raw_data[:, -1]
                raw_data = raw_data[:, :-1]
            else:
                raise Exception('Unknown file type!')
            # End of reading different file format.

            # Do common avearage reference
            if ext != '.nev':
                raw_data -= np.mean(raw_data, axis=-1, keepdims=True)
            # Computing LFP from the raw data.
            self.binned_neural = self._bin_neural(raw_data)
            # Check if both have same length
            len_neural = self.binned_neural.shape[0]
            len_nstamp = self.binned_nstamp.shape[0]
            if len_neural < len_nstamp:
                self.binned_nstamp = self.binned_nstamp[:len_neural]
            # Because the processing cost time, save the processed LFP as
            # a mat file by default, convenient next time calling.
            if save_mat:
                scio.savemat(os.path.join(path, name + '.mat'),
                             {'neural': self.binned_neural,
                              'stamps': self.binned_nstamp,
                              'fs': self.header['fs_neural']})
        return self.binned_neural, self.binned_nstamp

    def read(self, neu_filepath=None, beh_filepath=None, target='p',
             fs_timestamp=30000, analyze=True, undesired_motion_labels=[],
             ch_index=None, split_trials=False, pad=0, save_mat=True):
        """
        Parameters
        ----------
        neu_filepath : str, optional
            File path to loading the neural data. If it is None, a UI will
            pop out to ask user select one file. Default: None.
        beh_filepath : str, optional
            The path of bmi file which will be readed. If it is None, a UI will
            pop out to ask user select one file. Default: None.
        fs_timestamp : int, optional
            Sampling rate of time stamp. By default, the time stamp comes from
            recording NSP, which usually equal to 30000.
        analyze : bool, optional
            Analyze the readed BMI raw data by the provided label and paradigm.
            After analysis, statistics parameter will be stored. For example,
            when paradigm is Center-Out, Success Rate, Total Number of Trials,
            Hold Time, Failed Time, Average Time of Each Successed Trial etc.
            will be calculated.
        split_trials : bool, optional
            Indicate whether the processed neural and behavior data stored as
            one vector or split in single-trial style. Default: False.
        pad : int, optional
            The padding value when split_trials is True. Only used when
            split_trials is True. Default: 0.
        save_mat : bool, optional
            Choose whether need to save processed LFP as a mat file. Consider
            the time-consuming of calculating LFP from NSx, saving the LFP to
            mat format file will boost loading speed next time.
        """
        # ------------------------------------
        # 1. Read the behavioral data
        # ------------------------------------
        self.beh_read(beh_filepath, target, fs_timestamp, analyze, save_mat)
        # ------------------------------------
        # 2. Read the neural data
        # ------------------------------------
        self.neu_read(neu_filepath, ch_index, save_mat)
        # ------------------------------------
        # 3. Align behavioral and neural data
        # ------------------------------------
        self.align()
        # ------------------------------------
        # 4. Filter out the undesired motion data
        # ------------------------------------
        for label in undesired_motion_labels:
            self.filter(label)
        # ------------------------------------
        # 5. Get single trial data when necessary
        # ------------------------------------
        if split_trials and not self.is_single_trial:
            self.pad_idx = self._get_single_trials(self.label_success, p=pad)
        return self.binned_neural, self.binned_motion

    def analysis(self, raw_data, g, b, verbose=False):
        """
        Trying to analyze the readed BMI file by the provided label
        information and paradigm. After analysis, statistics parameter
        will be calculated. For example, when paradigm is Center-Out,
        Success Rate, Total Number of Trials, Hold Time, Failed Time,
        Average Time of Each Successed Trial etc. will be calculated.

        Parameters
        ----------
        g : int
            A good label means subject's behavior successed satisfied
            the expriment. For example, hold enough time on the target
            when doing Center-out. By default, user doesn't need to
            provide this parameter if the file is supported, otherwise,
            a specified good label can help the analysis perform normally.
        b : int
            A bad label means subject's behavior failed satisfied
            the expriment. For example, ran over the time to the target
            when doing Center-out. By default, user doesn't need to
            provide this parameter if the file is supported, otherwise,
            a specified bad label can help the analysis perform normally.
        verbose : bool, optional
            Print the analyzed statistics on the terminal if verbose set
            to True, otherwise no print.
        """
        # Initialize all stat
        self._init_stats()
        # loacl parameters.
        stats, header = self.stat, self.header
        # Get the labels from the raw data.
        i1 = header['index_labels'][0]
        i2 = header['index_labels'][1] + 1
        labels = raw_data[:, i1:i2]
        # Center-out behavior analysis.
        if header['paradigm'].lower() in ['center-out', 'center out',
                                          'centerout']:
            assert min(labels) <= g <= max(labels), "Invalid success label."
            assert min(labels) <= b <= max(labels), "Invalid failure label."
            # Sum all success labels in data as the total hold time.
            stats['hold_time'] = np.sum(labels == g)
            # Count the statistics behavior of the data
            for i, label in enumerate(labels):
                # Skip the 1st time step
                if i == 0:
                    trial_start = i
                    continue
                # The start of current trial.
                if label != g and label != b and \
                   (labels[i - 1] == g or labels[i - 1] == b):
                    trial_start = i
                # The first finish point of this trial is success label.
                elif label == g and labels[i - 1] != g:
                    stats['num_success'] += 1
                    stats['num_trials'] += 1
                    stats['time_to_target'] += (i - trial_start)
                # The first finish point of this trial is failure label.
                elif label == b and labels[i - 1] != b:
                    stats['num_trials'] += 1
                    stats['failed_time'] += (i - trial_start)
            # Finish checking the whole data one by one
            stats['success_rate'] = stats['num_success'] / stats['num_trials']
            stats['success_rate'] *= 100
            if stats['num_success'] != 0 and header['fs_motion'] != 0:
                t_pt = 1 / header['fs_motion']
                stats['hold_time'] *= (1000 * t_pt) / stats['num_success']
                stats['time_to_target'] *= t_pt / stats['num_success']
            else:
                stats['hold_time'] = float('NaN')
                stats['time_to_target'] = float('NaN')
            # Average failure time of each trial.
            num_fail = stats['num_trials'] - stats['num_success']
            if num_fail != 0:
                stats['failed_time'] *= (1 / header['fs_motion']) / num_fail
            else:
                stats['failed_time'] = float('NaN')
        else:
            # Unsupported pardigm.
            print('\nBMIReader::ERROR: Unsupported paradigm ' +
                  header['paradigm'] + ', can\'t do analyze of it.')

        self.stat = stats

    def filter(self, label):
        """
        Mask out specified trial type from data. The rest trials
        will concatenate one by one to form a new data.

        Parameters
        ----------
        label : {tuple, int}
            The labels of the kind of trial will be masked out. If the
            given labels are tuple, means that it specified one kind of
            trial with its start and end label number. If the given
            label is int, means that this kind of label will be removed
            in all trials, no matter what kind of trial (success, failure)
            it is.

        Returns
        -------
        masked : ndarray
            The data which masked out specified label.
        index : ndarray
            A vector stored the index of masked rows of BMI data.
        """
        if isinstance(label, tuple):
            # The tuple type label indicate trial start and
            # trial end of masked out trail type.
            l1, l2 = label
            index = np.ones(self.binned_motion.shape[0], dtype=np.bool)
            # The sambol represent the start of new trial.
            new_trial = True
            for i, row in enumerate(self.binned_labels):
                if i == 0:
                    trial_start, trial_end = 0, 0
                    continue

                if row == l1 and self.binned_labels[i - 1] != l1:
                    trial_start = i
                    new_trial = True
                elif row == l2 and (i == len(self.binned_labels) - 1 or
                                    self.binned_labels[i + 1] != l2):
                    if new_trial:
                        trial_end = i
                        # End of the trial, set new_trial to False.
                        new_trial = False

                # End of one whole trial
                if trial_start < trial_end:
                    index[trial_start:trial_end + 1] = False
                    # Set trail_start equal to trial_end to avoid between
                    # trial's data be masked.
                    trial_start = trial_end
        elif isinstance(label, int):
            index = ~(self.binned_labels[:, 0] == label)

        self.binned_neural = self.binned_neural[index]
        self.binned_nstamp = self.binned_nstamp[index]
        self.binned_motion = self.binned_motion[index]
        self.binned_bstamp = self.binned_bstamp[index]
        self.binned_labels = self.binned_labels[index]
        self.binned_spikes = self.binned_spikes[index]
        self.binned_target = self.binned_target[index]
        return self.binned_neural, self.binned_motion

    def align(self):
        """
        Trying to align the neural and binned motion data by timestamp.
        """
        T1, T2 = self.binned_neural.shape[0], self.binned_motion.shape[0]
        # Make sure the shape of timestamp x and timestamp y like (T,)
        stamp_x = np.reshape(self.binned_nstamp, (-1,))
        stamp_y = np.reshape(self.binned_bstamp, (-1,))

        # Align neural and behavior data according to time stamps.
        if self.is_single_trial:
            # * For SINGLE TRIAL case ...
            # When data is single trial, T1 and T2 is the number of trials.
            assert T1 == T2, f'ERROR: Neural data have {T1} trials, '
            f'but behavioral data have {T2} trials.'

            # ! NOTE: Not implement here.
            # TODO: Finish single trial data alignment.
        else:
            # *When data is SEQUENCIAL ...
            assert len(stamp_x) == T1, \
                f'Shape error! The length of timestamp x should be {T1}.'
            assert len(stamp_y) == T2, \
                f'Shape error! The length of timestamp y should be {T2}.'

            # First, find the shared part of both timestamp x and timestamp y,
            # calculate the start and end index of shared part in timestamp y.
            start = max(stamp_x[0], stamp_y[0])
            start_index_y = np.argmin(np.abs(stamp_y - start))
            end = min(stamp_x[-1], stamp_y[-1])
            end_index_y = np.argmin(np.abs(stamp_y - end)) + 1

            # Use the start and end index of y data to find aligned x data.
            index_x = []
            for i in range(start_index_y, end_index_y):
                idx = np.argmin(np.abs(stamp_x - stamp_y[i]))
                if np.abs(stamp_x[idx] - stamp_y[i]) >= self.beh_binsize:
                    print('\nWARNING: The difference of timestamp of x and y '
                          f'at index {i} are greater than step size: abs'
                          f'({stamp_x[idx]}-{stamp_y[i]})>={self.beh_binsize}')
                index_x.append(idx)

            # Cut the data x and y
            self.binned_neural = self.binned_neural[index_x]
            self.binned_nstamp = self.binned_nstamp[index_x]
            self.binned_motion = self.binned_motion[start_index_y:end_index_y]
            self.binned_bstamp = self.binned_bstamp[start_index_y:end_index_y]
            self.binned_labels = self.binned_labels[start_index_y:end_index_y]
            self.binned_spikes = self.binned_spikes[start_index_y:end_index_y]
            self.binned_target = self.binned_target[start_index_y:end_index_y]

    def print(self, data, path, name, ext, data_points, duration, t, analyze):
        print('*** FILE INFO **************************')
        print('File Path\t= ' + path)
        print('File Name\t= ' + name)
        print('File Extension\t= ' + ext)
        print('File Version\t= ' + self.header['version'])
        print(f'Duration\t= {duration[0]:.0f}m {duration[1]:.0f}s')
        print('Data Points\t= ' + str(data_points))

        print('*** BASIC HEADER ***********************')
        print('Subject\t\t\t= ' + self.header['subject'])
        print('Experimenter\t\t= ' + self.header['experimenter'])
        print('Paradigm\t\t= ' + self.header['paradigm'])
        print('Date\t\t\t= ' + self.header['date'])
        print('Comment\t\t\t= ' + self.header['comment'])
        print('Sample Frequency\t= ' + str(self.header['fs_motion']))
        i1 = self.header['index_spikes'][0]
        i2 = self.header['index_spikes'][1] + 1
        print('Electrodes Read\t\t= ' + str(i2 - i1))
        i1 = self.header['index_motion'][0]
        i2 = self.header['index_motion'][1] + 1
        print('Kinematic Dimension\t= ' + str(i2 - i1))

        if analyze:
            # Analyze behavior if analyze==True
            self.analysis(data, self.label_success, self.label_failure)
            # Warning the label might missing if bin size greater
            # than hold time
            ht, bt = self.stat['hold_time'], self.beh_binsize * 1000
            if ht < bt:
                print('BMIReader::WARNING: Success label might be missing '
                      f'because bin size={bt}ms > hold time={ht:.1f}ms.\n')
            # Print the behavioral information
            print('*** BEHAVIOR ANALYSIS ******************')
            print(f"Total Number of Trials\t\t= {self.stat['num_trials']}")
            print(f"Number of Success Trials\t= {self.stat['num_success']}")
            print(f"Success Rate\t\t\t= {self.stat['success_rate']:.2f}%")
            print(f"Hold Time\t\t\t= {self.stat['hold_time']:.0f}ms")
            print(f"Ran over failed time\t\t= {self.stat['failed_time']:.2f}s")
            print(f"Time to Target\t\t\t= {self.stat['time_to_target']:.2f}s")

        print(f'The load time for BMI file was {(t[1] - t[0]):.2f} seconds.')

    def _v1dot0(self, f, file_size):
        f.seek(0, 0)
        # Reading file headers.
        self.header['version'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['subject'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['experimenter'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['paradigm'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['date'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['comment'] = npc_remove(f.read(256), npc=b'\x00')
        self.header['fs_motion'] = int.from_bytes(f.read(4), 'little')
        self.header['columns'] = int.from_bytes(f.read(4), 'little')
        self.header['index_stamps'] = array.array('i', f.read(8)).tolist()
        self.header['index_labels'] = array.array('i', f.read(8)).tolist()
        self.header['index_spikes'] = array.array('i', f.read(8)).tolist()
        self.header['index_target'] = array.array('i', f.read(8)).tolist()
        self.header['index_motion'] = array.array('i', f.read(8)).tolist()
        # End of reading the header.
        # Compute the data size, which equal to file_size - header_size
        data_size = file_size - f.tell()
        data_points = data_size // (8 * self.header['columns'])
        error = data_points * 8 * self.header['columns'] - data_size
        assert error == 0, 'Reading BMI file failed. The data size' \
                           'not divisible by the columns.'
        # Reading the file data
        data = array.array('d', f.read(data_size))
        data = np.reshape(data, (data_points, self.header['columns']))
        return data, data_points

    def _v1dot1(self, f, file_size):
        f.seek(0, 0)
        # Reading file headers.
        self.header['version'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['subject'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['experimenter'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['paradigm'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['date'] = npc_remove(f.read(32), npc=b'\x00')
        self.header['comment'] = npc_remove(f.read(256), npc=b'\x00')
        # Time stamp index
        self.header['index_stamps'] = [0, 0]
        # Labels index
        nl = int.from_bytes(f.read(4), 'little')
        self.header['index_labels'] = [1, nl]
        # Spikes index
        ns = int.from_bytes(f.read(4), 'little')
        self.header['index_spikes'] = [self.header['index_labels'][1] + 1,
                                       self.header['index_labels'][1] + ns]
        # Motion & target index
        nk = int.from_bytes(f.read(4), 'little')
        self.header['index_motion'] = [self.header['index_spikes'][1] + 1,
                                       self.header['index_spikes'][1] + nk]
        self.header['index_target'] = [self.header['index_motion'][1] + 1,
                                       self.header['index_motion'][1] + nk]
        self.header['fs_motion'] = int.from_bytes(f.read(4), 'little')
        # This columns represent stamps, labels, spikes, motion, target,
        # external_vel, internal_vel and c
        self.header['columns'] = 1 + nl + ns + nk * 4 + 1
        # End of reading the header.
        # Compute the data size, which equal to file_size - header_size
        data_size = file_size - f.tell()
        data_points = data_size // (8 * self.header['columns'])
        error = data_points * 8 * self.header['columns'] - data_size
        assert error == 0, 'Reading BMI file failed. The data size' \
                           'not divisible by the columns.'
        # Reading the file data
        data = array.array('d', f.read(data_size))
        data = np.reshape(data, (data_points, self.header['columns']))
        return data, data_points

    def _v1dot2(self, f):
        # Reading from file start.
        f.seek(0, 0)
        self._header(f)
        # Check the validation of sampling rate of BMI.
        if self.h['fs'] <= 0:
            print(
                f"BMIReader::WARNING: file sampling rate: {self.h['fs']}."
                " This seems uncorrect, set it to default value: 100."
            )
            self.h['fs'] = 100

        # Compute the data size, which equal to file_size - header_size
        data_size = self.file_size - f.tell()
        # The data columns specified how many data displayed in each row.
        data_cols = 1 + self.h['lbl_dim'] + self.h['spk_dim'] + \
            self.h['kin_dim'] * 4 + 1 + 1

        data_points = data_size // (8 * data_cols)
        assert data_points * 8 * data_cols == data_size, \
            'Reading BMI file \'' + \
            os.path.join(self.path, self.name) + '\' failed.' \
            f'The data size is not divisible by data columns {data_cols}.'

        # Reading the file data
        data = array.array('d', f.read(data_size))
        data = np.reshape(data, (data_points, data_cols))
        return data, data_points

    def _bin_motion(self, raw_data, fs_timestamp):
        """
        Bin the raw data by bin size and check each bin size by time stamp.

        Parameters
        ----------
        raw_data : ndarray
            The readed raw behavioral data, which contain the information
            of timestamps and labels etc.
        fs_timestamp : int
            Sampling rate of time stamp. By default, the time stamp comes from
            recording NSP, which usually equal to 30000.
        """
        # Get the timestamp from the raw data
        i1 = self.header['index_stamps'][0]
        i2 = self.header['index_stamps'][1] + 1
        stamps = raw_data[:, i1:i2]
        # Steps of each bin.
        step = self.beh_binsize * fs_timestamp
        nbins = int((stamps[-1] - stamps[0]) // step) + 1
        # Ideal timestamp array.
        idea_ts = np.linspace(0, nbins - 1, nbins) * step + stamps[0]
        # Check the idea binned timestamp is validate.
        assert idea_ts[-1] <= stamps[-1], 'Invalidate binned timestamp!'
        # Get the binned data according the idea timestamp.
        binned = []
        for ts in idea_ts:
            index = np.argmin(abs(stamps - ts))
            binned.append(raw_data[[index]])
        return np.concatenate(binned)

    def _header(self, f):
        self.h['version'] = npc_remove(f.read(32), npc=b'\x00')
        self.h['subject'] = npc_remove(f.read(32), npc=b'\x00')
        self.h['experimenter'] = npc_remove(f.read(32), npc=b'\x00')
        self.h['paradigm'] = npc_remove(f.read(32), npc=b'\x00')
        self.h['date'] = npc_remove(f.read(32), npc=b'\x00')
        self.h['comment'] = npc_remove(f.read(256), npc=b'\x00')
        self.h['fs'] = int.from_bytes(f.read(4), 'little')
        self.h['columns'] = int.from_bytes(f.read(4), 'little')
        self.h['index_stamps'] = array.array('i', f.read(8)).tolist()
        self.h['index_labels'] = array.array('i', f.read(8)).tolist()
        self.h['index_spikes'] = array.array('i', f.read(8)).tolist()
        self.h['index_target'] = array.array('i', f.read(8)).tolist()
        self.h['index_motion'] = array.array('i', f.read(8)).tolist()

    def _init_header(self):
        self.header = {}
        self.header['version'] = 'Unknown'
        self.header['subject'] = 'Unknown'
        self.header['experimenter'] = 'Unknown'
        self.header['paradigm'] = 'Unknown'
        self.header['date'] = 'Unknown'
        self.header['comment'] = 'Unknown'
        self.header['fs_motion'] = float('NaN')
        self.header['fs_neural'] = float('NaN')
        self.header['columns'] = float('NaN')
        self.header['index_stamps'] = float('NaN')
        self.header['index_labels'] = float('NaN')
        self.header['index_spikes'] = float('NaN')
        self.header['index_target'] = float('NaN')
        self.header['index_motion'] = float('NaN')

    def _init_stats(self):
        """
        Set the statistics dict to initial state.
        """
        self.stat = {}
        self.stat['num_trials'] = 0
        self.stat['num_success'] = 0
        self.stat['success_rate'] = 0
        self.stat['hold_time'] = 0
        self.stat['failed_time'] = 0
        self.stat['time_to_target'] = 0

    def _bin_neural(self, x):
        """
        Computing LFP by different method.

        Parameters
        ----------
        x : ndarray or dict
            The input raw data. If type is ndarray, shape (N, C).
        """
        if self.method == 'pmtm':
            N, C = x.shape  # [Number of samples, Number of Channels]

            step = int(self.header['fs_neural'] * self.neu_binsize)
            nbins = N // step
            # Prepare the input data to pmtm.
            input = [x[n * step:(n + 1) * step] for n in range(nbins)]
            # MTM PSD estimation.
            with Pool(processes=self.njobs) as pool:
                iter = pool.imap(
                    partial(pmtm, NW=self.pmtm['nw'], nfft=self.pmtm['nfft'],
                            fs=self.header['fs_neural']), input
                )
                r = list(tqdm(iter, 'Processing', total=nbins, unit='bin'))
            # Get the correct shape of PSD, [n_bins, channel_count, frequency]
            Pxx, f = zip(*r)
            Pxx, f = np.stack(Pxx, axis=0), f[0]
            # Write specified frequency of Pxx to lfp
            neural = np.zeros((nbins, C, self.nbands))
            for i, freq in enumerate(self.pmtm['band']):
                index = (f >= freq[0]) & (f < freq[1])
                neural[:, :, i] = np.sum(Pxx[:, :, index], axis=-1)
            # Get the timestamp of LFP
            self.binned_nstamp = self.binned_nstamp[::step]
        elif self.method == 'wavelet':
            # Compute tfr scalogram
            neural = tfrscalo(
                x, self.binned_nstamp, self.binned_bstamp,
                self.header['fs_neural'], self.neu_binsize,
                self.wavelet['edge_size'], self.wavelet['wave'],
                self.wavelet['frange'][0], self.wavelet['frange'][1],
                self.nbands, self.wavelet['ncontext'], self.njobs
            )
            self.binned_nstamp = self.binned_bstamp
            # Remove zeros rows from LFP and timestamp
            zeros_idx = \
                np.reshape(neural, (neural.shape[0], -1)).mean(axis=-1) == 0
            neural = neural[~zeros_idx]
            self.binned_nstamp = self.binned_nstamp[~zeros_idx]
            # The shape of LFP is (nbins, nch, nfreq, nlag)
            #  -> permute it to (nbins, nch, nlag, nfreq)
            neural = np.transpose(neural, (0, 1, 3, 2))
            neural = np.reshape(neural, (neural.shape[0], neural.shape[1], -1))
        elif self.method == 'count':
            MAX_UNIT = 5

            channel_id = x['ChannelID']
            timestamps = x['TimeStamps']
            classification = x['Classification']

            # Find out max timestamp of all channels.
            maxt = 0
            for ts in timestamps:
                if ts and maxt < ts[-1]:
                    maxt = ts[-1]
            # Calculate number of bins of neural data.
            nbins = np.ceil(maxt / self.header['fs_neural'] / self.neu_binsize)
            nbins = int(nbins)
            # Get the neural stamps
            self.binned_nstamp = np.linspace(
                    0, nbins - 1, nbins, dtype=int
                ) * self.neu_binsize
            # Initialize the neural data object.
            sorted_neural = np.zeros((nbins, max(channel_id), MAX_UNIT))
            unsorted_neural = np.zeros((nbins, max(channel_id), 1))

            def classify_spikes(nbin, nch, nclass):
                if nclass == 'none':
                    # The class 'none' means unsort spike event.
                    unsorted_neural[nbin, nch - 1] += 1
                elif isinstance(nclass, int):
                    sorted_neural[nbin, nch - 1, nclass - 1] += 1

            # Count the spike data.
            for ch, ts, cs in zip(channel_id, timestamps, classification):
                # 'ts': time sequence for current channel
                # 'cs': class sequence for current channel

                # Convert the event occored time in seconds.
                ts = np.asarray(ts) / self.header['fs_neural']
                # Count spikes for current channel.
                for t, c in zip(ts, cs):
                    # 't': time of spike event occored.
                    # 'c': the class of this spike event.
                    bin = t / (self.neu_binsize + np.finfo(np.float64).eps)
                    classify_spikes(int(bin), ch, c)
            # Return sorted_neural if not all zero.
            is_sua = sorted_neural.sum() != 0
            neural = sorted_neural if is_sua else unsorted_neural
            # Check the useful classes of neural
            if is_sua:
                actived_units = neural.sum(axis=0).sum(axis=0) > 0
                neural = neural[:, :, actived_units]
        return neural

    def _get_single_trials(self, end_label, p=0) -> np.ndarray:
        """
        Assuming this fuction's input data is all passed
        after the filter to mask out the failure trials.

        Parameters
        ----------
        end_label : int
            The end_label should be the labels indicate of trials end.
        p : int
            The padding value to pad the trials which length smaller than
            MAX LENGTH.
        """
        bstamp, nstamp = [], []
        motion, neural, spikes, = [], [], []
        labels, target = [], []

        # Record each trial's index
        self.trial_index = []

        max_len = 0
        # Count the statistics behavior of the data
        for i, label in enumerate(self.binned_labels):
            # Skip the 1st time step
            if i == 0:
                trial_start = i
                continue
            # The start of current trial.
            if label != end_label and self.binned_labels[i - 1] == end_label:
                trial_start = i
            # The first finish point of this trial is success label.
            elif label == end_label and self.binned_labels[i - 1] != end_label:
                bstamp.append(self.binned_bstamp[trial_start:i + 1])
                nstamp.append(self.binned_nstamp[trial_start:i + 1])
                motion.append(self.binned_motion[trial_start:i + 1])
                neural.append(self.binned_neural[trial_start:i + 1])
                spikes.append(self.binned_spikes[trial_start:i + 1])
                labels.append(self.binned_labels[trial_start:i + 1])
                target.append(self.binned_target[trial_start:i + 1])
                # Get the max len
                if max_len < len(bstamp[-1]):
                    max_len = len(bstamp[-1])
                # Current trial index.
                self.trial_index.append([trial_start, i])
        # Stack all trials to one ndarray. Use padding value to pad the trials
        # which lenght smaller than max_len.

        def _w(x):
            """
            Computing the pad width.
            """
            ndim = np.ndim(x)
            ret = ((0, max_len - len(x)),)
            for _ in range(1, ndim):
                ret += ((0, 0),)
            return ret

        self.binned_bstamp = np.stack(
            [np.pad(x, _w(x), constant_values=p) for x in bstamp], axis=0
        )
        self.binned_nstamp = np.stack(
            [np.pad(x, _w(x), constant_values=p) for x in nstamp], axis=0
        )
        self.binned_motion = np.stack(
            [np.pad(x, _w(x), constant_values=p) for x in motion], axis=0
        )
        self.binned_neural = np.stack(
            [np.pad(x, _w(x), constant_values=p) for x in neural], axis=0
        )
        self.binned_spikes = np.stack(
            [np.pad(x, _w(x), constant_values=p) for x in spikes], axis=0
        )
        self.binned_labels = np.stack(
            [np.pad(x, _w(x), constant_values=p) for x in labels], axis=0
        )
        self.binned_target = np.stack(
            [np.pad(x, _w(x), constant_values=p) for x in target], axis=0
        )
        return np.stack([max_len - len(x) for x in motion], axis=0)


if __name__ == '__main__':
    bands = [[0.3, 5], [5, 8], [8, 13], [13, 30],
             [30, 70], [70, 200], [200, 400]]
    reader1 = BMIReader(0.1, 0.1, 7, method='pmtm', nw=2.5, nfft=2048,
                        bands=bands, njobs=6, verbose=True)
    reader1.read()

    reader2 = BMIReader(0.9, 0.05, 10, method='wavelet', edge_size=0.1,
                        wave=7, fmin=10, fmax=200, ncontext=9, njobs=6,
                        verbose=True)
    reader2.read()
