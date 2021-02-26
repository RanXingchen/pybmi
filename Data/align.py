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

    # If timestamp is None, then we assume the timestamp start at step 0.
    if timestamp_x is None:
        timestamp_x = np.linspace(0, T1 - 1, T1, dtype=np.int) * inc
    if timestamp_y is None:
        timestamp_y = np.linspace(0, T2 - 1, T2, dtype=np.int) * inc

    # Make sure the shape of timestamp x and timestamp y like (T,)
    timestamp_x = np.reshape(timestamp_x, (-1,))
    timestamp_y = np.reshape(timestamp_y, (-1,))

    assert len(timestamp_x) == T1, \
        f'Shape error occured! The length of timestamp x should be {T1}.'
    assert len(timestamp_y) == T2, \
        f'Shape error occured! The length of timestamp y should be {T2}.'

    ix = np.round(timestamp_x[0] / inc).astype(np.int)
    iy = np.round(timestamp_y[0] / inc).astype(np.int)

    start = max(ix, iy)
    # Make sure the count is under the bounds of both x and y.
    count = min(min(min(T1, T2), T1 - (start - ix)), T2 - (start - iy))
    # Check if both ends of timestamp of x and y are equal.
    ts_x = (timestamp_x[start - ix], timestamp_x[start - ix + count - 1])
    ts_y = (timestamp_y[start - iy], timestamp_y[start - iy + count - 1])
    if abs(ts_x[0] - ts_y[0]) >= inc:
        print('\nWARNING: The difference of start timestamp of x and y '
              f'are greater than step size: abs({ts_x[0]}-{ts_y[0]})>={inc}')
    if abs(ts_x[1] - ts_y[1]) >= inc:
        print('\nWARNING: The difference of end timestamp of x and y '
              f'are greater than step size: abs({ts_x[1]}-{ts_y[1]})>={inc}')

    # Cut the data x and y
    x = x[start - ix:start - ix + count]
    y = y[start - iy:start - iy + count]
    return x, y
