import numpy as np

from joblib import Parallel, delayed
from .computetfr import compute_tfr


def tfrscalo(X, record_time, desire_time, fs, window_size, edge_size,
             wave, fmin, fmax, N, n_context=0, trace=0):
    """
    TFRSCALO Scalogram, for Morlet or Mexican hat wavelet.
    This function computes the scalogram (squared magnitude
    of a continuous wavelet transform) for given time duration.

    Parameters
    ----------
    X : ndarray, shape (length, nch)
        signal (in time) to be analyzed. X should be a 2D matrix that
        rows represent the time points and columns represents the number
        of channels.
    record_time : ndarray, shape (length,)
        The recording time for each X point.
    desire_time : ndarray, shape (N,)
        Specified which time points used as reference to get the window.
    fs : float
        The sampling frequency of the signal.
    window_size : float
        The window size for the tfr computation. Only history used as
        the window sequence. Unit: second.
    edge_size : float
        The time length used to reduce the edge effects in the scalogram
        calculation. The data length used to compute scalogram equals to
        [window_size + 2 * edge_size]. Unit: second.
    wave : int
        Half length of the Morlet analyzing wavelet at coarsest
        scale. If WAVE = 0, the Mexican hat is used. WAVE can also be
        a vector containing the time samples of any bandpass
        function, at any scale.
    FMIN, FMAX : float
        Respectively lower and upper frequency bounds of
        the analyzed signal. These parameters fix the equivalent
        frequency bandwidth (expressed in Hz).
        FMIN and FMAX must be > 0 and <= fs / 2.
    N : int
        Number of analyzed voices.
    n_context : int, optional
        The number of context for the current time step. Note that only
        history are considered to be the context. Default: 0.
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

    Examples
    --------
    >>> x=np.random.randn(100, 10)
    >>> tfr, f = tfrscalo(x, 1.1, 1000, 7, 10, 120, 10)
    """
    if len(X.shape) == 1:
        # Make sure X is a 2D matrix.
        X = np.reshape(X, (len(X), 1))
    assert len(X.shape) == 2, "X must have only two dimension."
    assert len(record_time.shape) == 1, "'record_time' should be a 1D vector."
    assert len(desire_time.shape) == 1, "'desire_time' should be a 1D vector."

    # Compute the number of points for the window and edge.
    n = int(window_size * fs)   # Number of points for the window
    e = int(edge_size * fs)     # Number of points for the edge
    npts = n + 2 * e + 1        # Number of points used for the tfr compute.
    # Time instant(s) on which the TFR is evaluated
    T = np.linspace(0, npts - 1, npts, dtype=int)
    # Make sure the step of context is integer.
    if n_context > 0:
        assert n % n_context == 0, "Invalid number of context."
        step = n // n_context
    else:
        step = n + 1

    # Shape of X: xrow is the total length, xcol is the number of channel.
    xrow, xcol = X.shape

    def _parallel_compute(t):
        # Initialize tfr.
        tfr = np.zeros([xcol, N, n_context + 1])
        # Find the position of t in record_time.
        i = np.argmin(np.abs(record_time - t))
        if i - (n + e) < 0:
            # Current index smaller than window_size + edge_size,
            # the history signal is not recorded. Skip it!
            return tfr
        if i + e > xrow:
            # Current index did not have enough future
            # points for the window_size + edge_size. Breaking loop!
            return tfr
        window = X[i - (n + e):i + e + 1]
        for ch, x in enumerate(window.T):
            r = compute_tfr(x, T, wave, fmin / fs, fmax / fs, N, trace)[0]
            # Remove the edge part, and Compute the context
            tfr[ch, :, :] = np.fliplr(np.fliplr(r)[:, e:-e:step])
        return tfr

    r = Parallel(n_jobs=12)(delayed(_parallel_compute)(t) for t in desire_time)
    # Shape of the output (time, channels, frequencies, context)
    return np.stack(r)
