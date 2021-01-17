import math
import scipy
import numpy as np
from spectrum import dpss, nextpow2
from .psdfreqvec import psdfreqvec
from .computepsd import compute_psd


def pmtm(x, NW=4, nfft=None, fs=None, dlt=True, method='adapt'):
    """
    Power Spectral Density (PSD) estimate via the Thomson multitaper method (MTM).

    Parameters
    ----------
    x : array_like
        A discrete-time signal, the size of x is [C, T], where C means the
        channels of the signal, and T means the time points of samples.
        If x is a vector with [T,] shape, it is converted to [1, T] and treated
        as singal channel.
    NW : float, optional
        The "time-bandwidth product" for the discrete prolate spheroidal sequences
        (or Slepian sequences) used as data windows.
        Typical choices for NW are 2, 5/2, 3, 7/2, or 4. Default: 4
    nfft : int, optional
        Specifies the FFT length used to calculate the PSD estimates.
        For real X, Pxx has (nfft / 2 + 1) columns if nfft is even,
        and (nfft + 1) / 2 columns if nfft is odd. For complex X, Pxx always has length nfft.
        If nfft is specified as empty, nfft is set to either 256 or
        the next power of 2 greater than the length of X, whichever is larger.
    fs : int, optional
        The sampling frequency specified in hertz. If fs is empty, it defaults to 1 Hz.
    dlt : bool, optional
        Specifies whether drop the last taper.
        By default, it's true because the last taper's corresponding eigenvalue is
        significantly smaller than 1. Therefore, The number of tapers used to form
        Pxx is 2 * NW - 1.
    method : str, optional
        The method used for combining the individual spectral estimates:
        'adapt' - Thomson's adaptive non-linear combination (default)
        'unity' - linear combination with unity weights.
        'eigen' - linear combination with eigenvalue weight.

    Returns
    -------
    Pxx : array_like
        The PSD estimate of input x.
        Pxx is computed independently for each channel of x and stored in the
        corresponding row. Pxx is the distribution of power per unit frequency.
        The frequency is expressed in units of radians/sample.
        For real signals, the retured PSD is one-sided; for complex signals,
        it returns the two-sided PSD.
        Note that a one-sided PSD contains the total power of the input signal.
    f : array_like
        f is the vector of frequencies (in hertz) at which the PSD is
        estimated. For real signals, f spans the interval [0, Fs / 2] when nfft
        is even and [0, Fs / 2) when nfft is odd. For complex signals, f always
        spans the interval [0,Fs).

    References
    ----------
    .. [1] Thomson, D.J. "Spectrum estimation and harmonic analysis."
           In Proceedings of the IEEE. Vol. 10 (1982). pp. 1055-1096.
    .. [2] Percival, D.B. and Walden, A.T., "Spectral Analysis For Physical
           Applications", Cambridge University Press, 1993, pp. 368-370.
    """

    assert len(x.shape) < 3, 'Dimension of input array out of range!'

    # Convert 1D vectors to 2D with one channel.
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    # Get the number of channels and sample points.
    C, N = x.shape

    # Compute discrete prolate spheroidal sequences
    [E, V] = dpss(N, NW)
    if dlt:
        assert len(V) > 2, 'Drop the last taper must satisfied the number of tapers larger than 2!'
        E, V = E[:, :-1], V[:-1]
    else:
        assert len(V) > 1, 'The number of tapers should larger than 1!'

    fft_range = 'onesided' if np.all(np.isreal(x)) else 'twosided'

    # Set nfft when it's None
    if nfft is None:
        nfft = max(256, 2 ** nextpow2(N))
    if fs is None:
        fs = 2 * math.pi

    # Compute power spectrum via MTM.
    S = np.concatenate([_mtm_spectrum(i, nfft, E, V, method=method) for i in x])
    # Compute PSD frequency vector
    w = psdfreqvec(npts=nfft, fs=fs)

    # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
    # Also, compute the corresponding freq vector & freq units.
    Pxx, f, _ = compute_psd(S, w, fft_range, nfft, fs=fs, esttype='psd')

    # TODO: Add the confidence level computation at this place...
    return Pxx, f


def _mtm_spectrum(x, NFFT, E, V, method='adapt'):
    """
    Compute the power spectrum via MTM.

    Parameters
    ----------
    x : array_like
        Input data vector.
    nfft : int
        Number of frequency points to evaluate the PSD at.
        The default is max(256, 2^nextpow2(N)).
    E : array_like
        Matrix containing the discrete prolate spheroidal sequences (dpss).
    V : array_like
        Vector containing the concentration of the dpss.
    method: str, optional
        Algorithm used in MTM; default is 'adapt'.

    Returns
    -------
    S : Power spectrum computed via MTM.
    """
    assert method in ['adapt', 'eigen', 'unity']

    N, K = len(x), len(V)

    # Compute DFT using FFT
    Sk_complex = scipy.fft.fft(E.T * x, NFFT)
    Sk = abs(Sk_complex) ** 2   # Shape: [K, N]

    # Compute the MTM spectral estimates, compute the whole spectrum 0:nfft.
    if method in ['eigen', 'unity']:
        # Compute the averaged estimate: simple arithmetic averaging is used.
        # The Sk can also be weighted by the eigenvalues, as in Park et al.
        # Eqn. 9.; note that the eqn. apparently has a typo; as the weights
        # should be V and not 1 / V.
        wt = np.ones((1, K)) if method == 'unity' else V[np.newaxis, :]
        S = np.dot(wt, Sk) / K
    elif method == 'adapt':
        # Set up the iteration to determine the adaptive weights:
        V = V[:, np.newaxis]

        sig2 = np.dot(x, x) / N         # Power
        S = (Sk[[0]] + Sk[[1]]) / 2     # Initial spectrum estimate
        S1 = np.zeros((1, NFFT))

        # The algorithm converges so fast that results are
        # usually 'indistinguishable' after about three iterations.

        # This version uses the equations from [2] (P&W pp 368-370).

        # Set tolerance for acceptance of spectral estimate:
        tol = 0.0005 * sig2
        a = sig2 * (1 - V)

        # Do the iteration:
        while np.sum(np.abs(S - S1)) > tol:
            # calculate weights
            b = S / (np.dot(V, S) + a)
            # calculate new spectral estimate
            wk = (b ** 2) * V
            S1 = np.sum(wk * Sk, axis=0, keepdims=True) / np.sum(wk, axis=0, keepdims=True)
            S, S1 = S1, S  # swap S and S1
    return S
