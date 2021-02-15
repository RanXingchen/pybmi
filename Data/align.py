import numpy as np


def align(x, y, timestamp_x=None, timestamp_y=None, inc=3000):
    """
    Trying to align the neural and behavior data by timestamp.

    Parameters
    ----------
    x : ndarray, shape (T1, N)
        The neural data with the shape (T1, N), where T1 means
        the steps of the neural data, and N means the features.
    y : ndarray, shape (T2, M)
        The behavior data with the shape (T2, M), where T2 means
        the steps of behavior data, and M means the features.
    timestamp_x : ndarray, shape (T1,), optional
        The timestamp of x, specified the occored time of each
        step of neural data. If it is None, means that the timestamp
        is start from 0 by default.
    timestamp_y : ndarray, shape (T2,), optional
        The timestamp of y, specified the occored time of each
        step of behavior data. If it is None, means that the timestamp
        is start from 0 by default.
    inc : int, optional
        Increasement specified the time grows by step. It equals to
        the sample rate of the timestamp multiply bin size. Default
        value: 3000.

    Returns
    -------
    x: ndarray, shape (T, N)
        The aligned neural data with the shape (T, N), where T means
        the steps of the neural data after aligned.
    y : ndarray, shape (T, M)
        The aligned behavior data with the shape (T, M), where T means
        the steps of behavior data after aligned.
    """
    T1, T2 = x.shape[0], y.shape[0]

    # If timestamp x is None, then we assume the timestamp x
    # start at step 0.
    if timestamp_x is None:
        timestamp_x = np.linspace(0, T1 - 1, T1, dtype=np.int) * inc
    if timestamp_y is None:
        timestamp_y = np.linspace(0, T2 - 1, T2, dtype=np.int) * inc

    # Make sure the shape of timestamp x and timestamp y like (T, 1)
    timestamp_x = np.reshape(timestamp_x, (-1, 1))
    timestamp_y = np.reshape(timestamp_y, (-1, 1))

    assert timestamp_x.shape[0] == T1, \
        f'Shape error occured! The length of timestamp x should be {T1}.'
    assert timestamp_y.shape[0] == T2, \
        f'Shape error occured! The length of timestamp y should be {T2}.'

    start_x = int(timestamp_x[0, 0] // inc)
    start_y = int(timestamp_y[0, 0] // inc)

    start = max(start_x, start_y)
    count = min(T1, T2)

    # Cut the data x and y
    x = x[start - start_x:start - start_x + count]
    y = y[start - start_y:start - start_y + count]
    return x, y
