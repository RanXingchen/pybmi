import numpy as np
import os
import scipy.io as scio
import tkinter
import struct

from pybmi.signal.spectrogram import pmtm, tfrscalo
from pybmi.utils.utils import check_params
from joblib import Parallel, delayed
from tkinter import filedialog
from .brPY.brpylib import NsxFile


class LFPReader():
    """
    Use brPY toolkit provided by Blackrock Microsystems. Reading NSx
    neural data files and preprocess it to LFP data. Be note that the
    value the brPY get is in unit of uV. It is 1/4 of the raw values.

    Parameters
    ----------
    binsize : float
        Bin size specified the length of the neural data used to
        compute the LFP. A larger bin size can get more accurate LFP,
        but less online compitiable. Unit: seconds.
    nbands : int
        Number of frequency bands will be generated.
    method : str, optional
        Method used to compute the spectrogram. It can be either
        'pmtm' or 'wavelet'. Default: 'pmtm'.
        When 'pmtm' is used, the additional parameters needed:
            nw : float, optional
                The "time-bandwidth product" for the DPSS used as data windows.
                Typical choices are 2, 5/2, 3, 7/2, or 4.
            nfft : int, optional
                Specifies the FFT length used to calculate the PSD estimates.
            bands : {list, tuple, ndarray}, optional
                The frequency bands desired.
        When 'wavelet' is used, the addtional parameters needed:
            edge_size : float, optional
                The time length used to reduce the edge effects in the
                scalogram calculation. The data length used to compute
                scalogram equals to [binsize + 2 * edge_size]. Unit: second.
            wave : int, optional
                Half length of the Morlet analyzing wavelet at coarsest
                scale. If WAVE = 0, the Mexican hat is used. WAVE can also be
                a vector containing the time samples of any bandpass
                function, at any scale.
            fmin, fmax : float, optional
                Respectively lower and upper frequency bounds of
                the analyzed signal. These parameters fix the equivalent
                frequency bandwidth (expressed in Hz).
                FMIN and FMAX must be > 0 and <= fs / 2.
            ncontext : int, optional
                The number of context for the current time step. Note that only
                history are considered to be the context. Default: 0.
    njobs : int, optional
        NJOBS define how many workers processes parallelly to calculate
        the LFP.

    Examples
    --------
    >>> freq_bands = [[0, 10], [10, 20], [20, 30], [100, 200], [200, 400]]
    >>> bin_size = 0.1
    >>> reader = LFPReader(bin_size, nbands=len(freq_bands), nfft=2048,
    >>>                    bands=freq_bands, njobs=12)
    >>> lfp, timestamp = reader.read()
    """
    def __init__(self, binsize, nbands, method='pmtm', nw=2.5, nfft=1024,
                 bands=None, edge_size=0, wave=7, fmin=10, fmax=120,
                 ncontext=0, njobs=1):
        self.binsize = binsize
        self.nbands = nbands
        self.nw = nw
        self.nfft = nfft
        self.bands = bands
        self.edge_size = edge_size
        self.wave = wave
        self.fmin = fmin
        self.fmax = fmax
        self.ncontext = ncontext
        self.njobs = njobs

        # Check parameters validation
        self.method = check_params(method, ['pmtm', 'wavelet'], 'method')
        if bands is not None and self.method == 'pmtm':
            assert nbands == len(bands), "Error number of frequency bands"
            f" detected. Specified number of bands are {nbands}!"

    def read(self, filepath=None, timeres=30000, save_mat=True):
        """
        Reading the Neural data either MAT format or NSx format.

        Parameters
        ----------
        filepath : string, optional
            File path to loading the neural data. If it is None, a UI will
            pop out to ask user select one file.
        timeres : scalar, optional
            Time resolution of NSP recording. Default: 30000.
        save_mat : bool, optional
            Choose whether need to save processed LFP as a mat file. Consider
            the time-consuming of calculating LFP from NSx, saving the LFP to
            mat format file will boost loading speed next time.

        Returns
        -------
        lfp : ndarray
            The LFP data which computed by PMTM, the shape of LFP is
            [number of bins, channels * number of frequency bands].
            Note that the matrix is stored in 'C' order, means that
            if reshape the LFP to split the channels and number of
            frequency bands, it should be
            [number of bins, channels, number of frequency bands]
        timestamp : ndarray
            The timestamp calculated according the start recording time
            and sampling frequency. The output timestamp is downsamped
            by the bin size.
        """
        if filepath is None:
            tkinter.Tk().withdraw()     # Hidden the main window of Tk.
            # Popup the Open File UI. Get the file name and path.
            filepath = filedialog.askopenfilename(
                title="Choose a neural data file...",
                filetypes=(("MATLAB data file", "*.mat"),
                           ("NS3 files", "*.ns3"),
                           ("all files", "*.*"))
            )
        assert os.path.exists(filepath), \
            'The provided file \'' + filepath + '\' does not exist!'

        # File extension used as the sambol. If it's 'mat', load the
        # lfp directly; if it's NSx or bin, the LFP need computed from
        # the raw data, which may cost lots of time.
        path, fname = os.path.split(filepath)
        name, ext = os.path.splitext(fname)

        if ext == '.mat':
            # Load already processed LFP.
            data = scio.loadmat(filepath)
            lfp, timestamp = data['LFP'], data['timestamp']
            self.fs = data['fs']
        else:
            # Need to process first.
            if 'ns' in ext:
                # Read raw data of NSx
                nsx_file = NsxFile(filepath)
                raw_data = nsx_file.getdata()
                # Close nsx file
                nsx_file.close()
                # The data part and header part.
                data = raw_data['data'].T
                header = raw_data['data_headers'][0]
                # Useful recording parameters of neural data.
                self.fs = int(raw_data['samp_per_s'])
                ts = int(header['Timestamp'])
                # Calculate the timestep of raw data.
                timestamp = np.linspace(
                    ts, data.shape[0] - 1, num=data.shape[0], dtype=np.int
                ) * (timeres / self.fs)
                desire_time = None
            elif ext == '.bin':
                with open(filepath, 'rb') as f:
                    # The first element of bin file store the number of
                    # columns of the lfp data.
                    ncol = struct.unpack('B', f.read(1))[0]
                    # The second element of bin file store the sampling
                    # rate of the lfp data.
                    self.fs = struct.unpack('l', f.read(4))[0]
                    # The third element of bin file store the number of
                    # values of desire_time
                    len_t = struct.unpack('l', f.read(4))[0]
                    desire_time = struct.unpack('f' * len_t, f.read(len_t * 4))
                    desire_time = np.array(desire_time)
                    # Read the whole data.
                    raw_data = f.read()
                    len_d = len(raw_data) // 4
                    raw_data = struct.unpack('f' * len_d, raw_data)
                    raw_data = np.asarray(raw_data).reshape(-1, ncol)
                # End reading. Split the lfp data and timestamp
                data, timestamp = raw_data[:, :-1], raw_data[:, -1]
            else:
                raise Exception('Unknown file type!')
            # End of reading different file format.

            # Do common avearage reference
            data -= np.mean(data, axis=-1, keepdims=True)
            # Computing LFP from the raw data.
            lfp, timestamp = self._compute_lfp(data, timestamp, desire_time)
            # Check if both have same length
            if lfp.shape[0] < timestamp.shape[0]:
                timestamp = timestamp[:lfp.shape[0]]

            # Because the processing cost time, save the processed LFP as
            # a mat file by default, convenient next time calling.
            if save_mat:
                scio.savemat(os.path.join(path, name + '.mat'),
                             {'LFP': lfp,
                              'timestamp': timestamp,
                              'fs': self.fs})
        return lfp, timestamp

    def _compute_lfp(self, x, timestamp, desire_time):
        """
        Computing LFP by different method.

        Parameters
        ----------
        x : ndarray, shape (N, C)
            The input raw data.
        """
        N, C = x.shape  # [Number of samples, Number of Channels]

        if self.method == 'pmtm':
            step = int(self.fs * self.binsize)
            nbins = N // step
            # MTM PSD estimation.
            r = Parallel(n_jobs=self.njobs)(delayed(pmtm)(
                x[n * step:(n + 1) * step], self.nw, self.nfft, self.fs)
                for n in range(nbins)
            )
            # Get the correct shape of PSD, [n_bins, channel_count, frequency]
            Pxx, f = zip(*r)
            Pxx, f = np.stack(Pxx, axis=0), f[0]
            # Write specified frequency of Pxx to lfp
            lfp = np.zeros((nbins, C, self.nbands))
            for i, freq in enumerate(self.bands):
                index = (f >= freq[0]) & (f < freq[1])
                lfp[:, :, i] = np.sum(Pxx[:, :, index], axis=-1)
            # Get the timestamp of LFP
            timestamp = timestamp[::step]
        elif self.method == 'wavelet':
            # Compute tfr scalogram
            lfp = tfrscalo(
                x, timestamp, desire_time, self.fs, self.binsize,
                self.edge_size, self.wave, self.fmin, self.fmax,
                self.nbands, self.ncontext, self.njobs
            )
            timestamp = desire_time
            # Remove zeros rows from LFP and timestamp
            zeros_idx = np.reshape(lfp, (lfp.shape[0], -1)).mean(axis=-1) == 0
            lfp, timestamp = lfp[~zeros_idx], timestamp[~zeros_idx]
            # The shape of LFP is (nbins, nch, nfreq, nlag)
            #  -> permute it to (nbins, nch, nlag, nfreq)
            lfp = np.transpose(lfp, (0, 1, 3, 2))
            lfp = np.reshape(lfp, (lfp.shape[0], lfp.shape[1], -1))
        return lfp, timestamp
