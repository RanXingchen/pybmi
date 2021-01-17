import numpy as np
import math


def compute_psd(Sxx, w, fft_range, nfft, fs=None, esttype='psd'):
    """
    Compute the one-sided or two-sided PSD or Mean-Square.

    Parameters
    ----------
    Sxx : array_like
        Whole power spectrum [Power]; it can be a vector or a matrix.
        For matrices the operation is applied to each row.
    w : array_like
        Frequency vector in rad/sample or in Hz.
    fft_range : str
        Determines if a 'onesided' or a 'twosided' Pxx and Sxx are returned.
    nfft : int
        Number of frequency points.
    fs : int, optional
        Sampling frequency.
    esttype : str, optional
        A string indicating the estimate type: 'psd', or 'ms'. Default: 'psd'.

    Returns
    -------
    Pxx : array_like
        One-sided or two-sided PSD or MEAN-SQUARE (not scaled by fs)
        depending on the input arguments RANGE and TYPE.
    w : array_like
        Frequency vector 0 to 2 * Nyquist or 0 to Nyquist depending on
        range, units will be either rad/sample (if Fs is empty) or Hz
        (otherwise).
    units : str
        Either 'rad/sample' or 'Hz'.
    """

    # Generate the one-sided spectrum [Power] if so wanted
    if fft_range == 'onesided':
        if nfft % 2 == 1:
            end_pt = (nfft + 1) // 2        # ODD
            Sxx_unscaled = Sxx[:, :end_pt]  # Take only [0, pi] or [0, pi)
            # Only DC is a unique point and doesn't get doubled
            Sxx = np.concatenate((Sxx_unscaled[:, :1], 2 * Sxx_unscaled[:, 1:]), axis=-1)
        else:
            end_pt = (nfft // 2) + 1        # EVEN
            Sxx_unscaled = Sxx[:, :end_pt]  # Take only [0, pi] or [0, pi)
            # Don't double unique Nyquist point
            Sxx = np.concatenate(
                (Sxx_unscaled[:, :1], 2 * Sxx_unscaled[:, 1:-1], Sxx_unscaled[:, -1:]),
                axis=-1
            )
        w = w[:end_pt]

    # Compute the PSD [Power/freq]
    if fs is not None:
        Pxx = Sxx / fs              # Scale by the sampling frequency to obtain the psd
        units = 'Hz'
    else:
        Pxx = Sxx / (2 * math.pi)   # Scale the power spectrum by 2*pi to obtain the psd
        units = 'rad/sample'

    if esttype == 'psd':
        return Pxx, w, units
    else:
        return Sxx, w, units
