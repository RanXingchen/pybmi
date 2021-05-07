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

    # First, find the shared part of both timestamp x and timestamp y,
    # calculate the start index and end index of shared part in timestamp y.
    start = max(timestamp_x[0], timestamp_y[0])
    start_index_y = np.argmin(np.abs(timestamp_y - start))
    end = min(timestamp_x[-1], timestamp_y[-1])
    end_index_y = np.argmin(np.abs(timestamp_y - end)) + 1

    # Use the start and end index of y data to find aligned x data.
    index_x = []
    for i in range(start_index_y, end_index_y):
        idx = np.argmin(np.abs(timestamp_x - timestamp_y[i]))
        if np.abs(timestamp_x[idx] - timestamp_y[i]) >= inc:
            print('\nWARNING: The difference of timestamp of x and y at'
                  f'index {i} are greater than step size: '
                  f'abs({timestamp_x[idx]}-{timestamp_y[i]})>={inc}')
        index_x.append(idx)

    # Cut the data x and y
    x = x[index_x]
    y = y[start_index_y:end_index_y]
    return x, y
