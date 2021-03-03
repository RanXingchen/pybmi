import numpy as np
import os
import scipy.io as scio
import tkinter

from pybmi.signal.spectrogram import pmtm
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
    filepath : string, optional
        File path to loading the neural data. If it is None, a UI will
        pop out to ask user select one file.
    nw : float, optional
        The "time-bandwidth product" for the DPSS used as data windows.
        Typical choices are 2, 5/2, 3, 7/2, or 4.
    nfft : int, optional
        Specifies the FFT length used to calculate the PSD estimates.
    njobs : int, optional
        NJOBS define how many workers processes parallelly to calculate
        the LFP.

    Examples
    --------
    >>> freq_bands = [[0, 10], [10, 20], [20, 30], [100, 200], [200, 400]]
    >>> bin_size = 0.1
    >>> reader = LFPReader(nfft=2048, njobs=12)
    >>> lfp, timestamp = reader.read(freq_bands, bin_size)
    """
    def __init__(self, filepath=None, nw=2.5, nfft=1024, njobs=1):
        self.nw = nw
        self.nfft = nfft
        self.njobs = njobs
        self.filepath = filepath

    def read(self, freq_bands, bin_size, timeres=30000, save_mat=True):
        """
        Reading the Neural data either MAT format or NSx format.

        Parameters
        ----------
        freq_bands : {list, tuple, ndarray}
            The frequency bands desired.
        bin_size : float
            Bin size specified the length of the neural data used to
            compute the LFP. A larger bin size can get more accurate LFP,
            but less online compitiable. Unit: seconds.
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
        if self.filepath is None:
            # Hidden the main window of Tk.
            tkinter.Tk().withdraw()
            # Popup the Open File UI. Get the file name and path.
            self.filepath = filedialog.askopenfilename(
                title="Choose a neural data file...",
                filetypes=(("MATLAB data file", "*.mat"),
                           ("NS3 files", "*.ns3"),
                           ("all files", "*.*"))
            )
        assert os.path.exists(self.filepath), \
            'The provided file \'' + self.filepath + '\' does not exist!'

        path, fname = os.path.split(self.filepath)
        # File extension used as the sambol. If it's 'mat', load the
        # lfp directly; if it's NSx, the LFP need computed from the
        # raw data, which may cost lots of time.
        ext = fname.split('.')[-1]

        if ext == 'mat':
            # Load already processed LFP.
            data = scio.loadmat(self.filepath)
            lfp, timestamp = data['LFP'], data['timestamp']
        elif 'ns' in ext:
            # Read raw data of NSx
            nsx_file = NsxFile(self.filepath)
            raw_data = nsx_file.getdata()
            # Close nsx file
            nsx_file.close()
            # The data part and header part.
            data = raw_data['data'].T
            header = raw_data['data_headers'][0]
            # Useful recording parameters of neural data.
            fs = int(raw_data['samp_per_s'])
            ts = int(header['Timestamp'])
            step_size = int(fs * bin_size)
            # Calculate the timestep of raw data.
            timestamp = np.linspace(
                ts, data.shape[0] - 1, num=data.shape[0], dtype=np.int
            ) * (timeres / fs)

            # Computing LFP from the raw data.
            lfp = self._compute_lfp(data, step_size, freq_bands, fs)
            # Get the timestamp of LFP
            timestamp = timestamp[0::step_size]
            # Check if both have same length
            if lfp.shape[0] < timestamp.shape[0]:
                timestamp = timestamp[:lfp.shape[0]]

            # Because the processing cost time, save the processed LFP as
            # a mat file by default, convenient next time calling.
            if save_mat:
                save_name = ''
                for s in fname.split('.')[:-1]:
                    save_name += (s + '.')
                save_name += 'mat'
                scio.savemat(
                    os.path.join(path, save_name),
                    {'LFP': lfp, 'timestamp': timestamp}
                )
        else:
            raise Exception('Unknown file type!')
        return lfp, timestamp

    def _compute_lfp(self, x, step, freq_bands, fs):
        N, C = x.shape  # [Number of samples, Number of Channel]
        # Length of LFP data.
        n_bins = N // step
        n_bands = len(freq_bands)

        # MTM PSD estimation.
        r = Parallel(n_jobs=self.njobs)(delayed(pmtm.pmtm)(
            x[n * step:(n + 1) * step], NW=self.nw, nfft=self.nfft, fs=fs)
            for n in range(n_bins)
        )
        # Get the correct shape of PSD, [n_bins, channel_count, frequency]
        Pxx, f = zip(*r)
        Pxx, f = np.stack(Pxx, axis=0), f[0]

        # Write specified frequency of Pxx to lfp
        lfp = np.zeros((n_bins, C * n_bands))
        for i, freq in enumerate(freq_bands):
            index = (f >= freq[0]) & (f < freq[1])
            lfp[:, i::n_bands] = np.sum(Pxx[:, :, index], axis=-1)
        return lfp
