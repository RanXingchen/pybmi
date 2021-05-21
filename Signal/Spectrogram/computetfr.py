import numpy as np
import math

from .integ2d import integ2d
from scipy.signal import hilbert


def compute_tfr(X, T, wave, fmin, fmax, N, trace=0):
    """
    Computing tfr Scalogram, for Morlet or Mexican hat wavelet.
    This function computes the scalogram (squared magnitude
    of a continuous wavelet transform).

    Parameters
    ----------
    X : ndarray, shape (Nx, 1)
        signal (in time) to be analyzed. Its analytic version is
        used (z=hilbert(real(X))). Note that this function can't
        accept complex type value.
    T : ndarray, shape (Nt, 1)
        Time instant(s) on which the TFR is evaluated (default: 0:Nx-1).
    wave : int
        Half length of the Morlet analyzing wavelet at coarsest
        scale. If WAVE = 0, the Mexican hat is used. WAVE can also be
        a vector containing the time samples of any bandpass
        function, at any scale.
    FMIN, FMAX : float
        Respectively lower and upper frequency bounds of
        the analyzed signal. These parameters fix the equivalent
        frequency bandwidth (expressed in Hz).
        FMIN and FMAX must be >0 and <=0.5.
    N : int
        Number of analyzed voices.
    trace : int, optional
        If nonzero, the progression of the algorithm is shown
        Default: 0.

    Returns
    -------
    TFR : ndarray
        Time-frequency matrix containing the coefficients of the
        decomposition (abscissa correspond to uniformly sampled time,
        and ordinates correspond to a geometrically sampled
        frequency). First row of TFR corresponds to the lowest
        frequency.
    F : ndarray
        Vector of normalized frequencies (geometrically sampled
        from FMIN to FMAX).
    WT : ndarray
        Complex matrix containing the corresponding wavelet
        transform. The scalogram TFR is the square modulus of WT.

    Examples
    --------
    >>> x=np.random.randn(100, 10)
    >>> tfr, f, wt = tfrscalo(x)
    """
    # Do error checking first.
    if X.ndim == 1:
        X = np.reshape(X, (len(X), 1))
    if T.ndim == 1:
        T = np.reshape(T, (1, len(T)))
    assert X.ndim == 2, "X must have only two dimension."
    assert T.ndim == 2, "T must have only two dimension."

    xrow, xcol = X.shape
    trow, tcol = T.shape
    assert 0 < xcol <= 2, "X must have one or two columns."
    assert trow == 1, "T must only have one row."
    assert wave >= 0, "WAVE must be positive."

    # Hilbert transform the X signal.
    s = (X - X.mean(axis=0)).T
    z = hilbert(s)
    if trace:
        print('Scalogram distribution.')

    # Check if fmin and fmax are validate.
    assert fmin < fmax, "FMAX must be greater than FMIN."
    assert fmin > 0 and fmin <= 0.5, "FMIN must be > 0 and <= 0.5."
    assert fmax > 0 and fmax <= 0.5, "FMAX must be > 0 and <= 0.5."
    if trace:
        print(f"Frequency runs from {fmin} to {fmax} with {N} points.")

    f = np.reshape(np.logspace(np.log10(fmin), np.log10(fmax), N), (10, 1))
    a = np.logspace(np.log10(fmax / fmin), np.log10(1), N)

    wt = np.zeros((N, tcol), dtype=np.complex)
    tfr = np.zeros((N, tcol))

    if wave > 0:
        if trace:
            print("Using a Morlet wavelet...")
        for ptr in range(N):
            nha = wave * a[ptr]
            nha_round = int(nha + 0.5)
            tha = np.linspace(-nha_round, nha_round, 2 * nha_round + 1,
                              dtype=np.int)
            ha = np.exp(-(2 * np.log(10) / nha ** 2) * tha ** 2) *\
                np.exp(np.complex(0, 1) * 2 * math.pi * f[ptr] * tha)
            detail = np.convolve(z.flatten(), ha) / np.sqrt(a[ptr])
            detail = detail[nha_round:len(detail) - nha_round]  # careful
            wt[ptr] = detail[T]
            tfr[ptr] = abs(detail[T]) ** 2
    elif wave == 0:
        print('Not implement yet.')
    elif len(wave) > 1:
        print('Not implement yet.')

    # Normalization
    SP = np.fft.fft(z)
    indmin = int(fmin * (xrow - 2) + 0.5)
    indmax = int(fmax * (xrow - 2) + 0.5)
    SPana = SP[:, indmin:indmax + 1]
    tfr = tfr * np.linalg.norm(SPana) ** 2 / integ2d(tfr, T, f) / N
    return tfr, f, wt
