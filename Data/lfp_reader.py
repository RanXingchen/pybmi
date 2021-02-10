import matlab.engine
import numpy as np
import os
import scipy.io as scio

from Signal.Spectrogram import pmtm
from joblib import Parallel, delayed


class LFPReader():
    """
    Use MATLAB eigen calling NPMK toolkit provided by Blackrock Microsystems.
    Reading NSx neural data files and preprocess it to LFP data.

    Parameters
    ----------
    matlab_package_path : str, optional
        The file path that contain the NPMK toolkit.
    nw : float, optional
        The "time-bandwidth product" for the DPSS used as data windows.
        Typical choices are 2, 5/2, 3, 7/2, or 4.
    nfft : int, optional
        Specifies the FFT length used to calculate the PSD estimates.
    njobs : int, optional
        NJOBS define how many workers processes parallelly to calculate
        the LFP.
    """
    def __init__(self, matlab_package_path='../../MATLAB',
                 nw=2.5, nfft=1024, njobs=1):
        self.eng = matlab.engine.start_matlab()
        # The MATLAB code package should be contained to read NSx files.
        self.eng.addpath(self.eng.genpath(matlab_package_path))
        self.nw = nw
        self.nfft = nfft
        self.njobs = njobs

    def read(self, freq_bands, bin_size, save_mat=True):
        """
        Reading the Neural data either MAT format or NSx format.

        Parameters
        ----------
        freq_bands : {list, tuple, ndarray}, optional
            The frequency bands desired.
        bin_size : float, optional
            Bin size specified the length of the neural data used to
            compute the LFP. A larger bin size can get more accurate LFP,
            but less online compitiable. Unit: seconds.
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
        """
        # Popup the Open File UI. Get the file name and path.
        fname, path = self.eng.getFile(
            '*.*', 'Choose a neural data file...', nargout=2
        )
        # File extension used as the sambol. If it's 'mat', load the
        # lfp directly; if it's NSx, the LFP need computed from the
        # raw data, which may cost lots of time.
        ext = fname.split('.')[-1]

        if ext == 'mat':
            # Load already processed LFP.
            lfp = scio.loadmat(os.path.join(path, fname))['LFP']
        elif 'ns' in ext:
            # Read raw data of NSx from matlab
            meta_tags, data, raw_data, electrodes_info = \
                self.eng.openNSxPythonWrapper(
                    True, True, os.path.join(path, fname), nargout=4
                )
            # Fast convertion from Matlab matrix to ndarray.
            # https://stackoverflow.com/questions/34155829
            data = np.array(data._data,
                            dtype=np.float64).reshape(data.size, order='F').T
            # Useful recording parameters of neural data.
            fs = int(meta_tags['SamplingFreq'])
            step_size = int(fs * bin_size)

            # Computing LFP from the raw data.
            lfp = self._compute_lfp(data, step_size, freq_bands, fs)

            # Because the processing cost time, save the processed LFP as
            # a mat file by default, convenient next time calling.
            if save_mat:
                save_name = ''
                for s in fname.split('.')[:-1]:
                    save_name += (s + '.')
                save_name += 'mat'
                scio.savemat(os.path.join(path, save_name), {'LFP': lfp})
        else:
            raise Exception('Unknown file type!')
        return lfp

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
