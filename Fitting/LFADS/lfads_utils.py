import numpy as np
import matplotlib.pyplot as plt
import pybmi as bmi

from numpy import ndarray


def plot_traces(pred: ndarray, true: ndarray, bin_size, figsize=(8, 8),
                num_traces=12, ncols=2, mode='none', norm=True):
    """
    Plot recnostructed neural signals and compare to the ground truth.

    Parameters
    ----------
    pred : ndarray
        Array of predicted values to plot, shape: [num_steps, num_cells]
    true : ndarray
        Array of true values to plot, shape: [num_steps, num_cells]
    figsize : tuple, optional
        Figure size (width, height) in inches. Default: (8, 8).
    num_traces : int, optional
        Number of traces to plot. Default: 24.
    ncols : int, optional
        Number of columns in figure. Default: 2.
    mode : str, optional
        Mode to select subset of traces. Options: 'activity', 'rand', 'none'.
        'activity' - plots the the most high acitivity neural signals.
        'rand' - randomly choose num_traces cell to plot.
        'none' - choose the first num_traces to plot.
    norm : bool
        Normalize predicted and actual values. Default: True.
    """
    # Check parameters validation.
    mode = bmi.utils.check_params(mode, ['rand', 'activity', 'none'], 'mode')
    # Get the shape of true and prediction data.
    T, N = pred.shape

    if norm:
        true = (true - true.mean()) / true.std()
        pred = (pred - pred.mean()) / pred.std()

    # Choose the index of features to plot.
    if mode == 'rand':
        idxs = np.random.choice(list(range(N)), num_traces, False)
        idxs.sort()
    elif mode == 'activity':
        idxs = true.max(axis=0).argsort()[-num_traces:]
    else:
        idxs = list(range(num_traces))

    # Determine the max and min value of true and pred.
    ymin = min(pred[:, idxs].min(), true[:, idxs].min())
    ymax = max(pred[:, idxs].max(), true[:, idxs].max())

    # time is the x value.
    time = np.linspace(0, T * bin_size, T, endpoint=False)

    # Initialize the figure handle.
    nrows = int(num_traces / ncols)
    fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    axs = np.ravel(axs)
    # Plot
    for ii, (ax, idx) in enumerate(zip(axs, idxs)):
        plt.sca(ax)
        plt.plot(time, pred[:, idx], lw=2, color='#37A1D0')
        plt.plot(time, true[:, idx], lw=2, color='#E84924')
        plt.ylim(ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1)
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if ii >= num_traces - ncols:
            # The last row.
            plt.xlabel('time (s)', fontsize=14)
            plt.xticks(fontsize=12)
            ax.xaxis.set_ticks_position('bottom')
        else:
            plt.xticks([])
            ax.xaxis.set_ticks_position('none')
            ax.spines['bottom'].set_visible(False)

        if ii % ncols == 0:
            # The first column.
            plt.yticks(fontsize=12)
            ax.yaxis.set_ticks_position('left')
        else:
            plt.yticks([])
            ax.yaxis.set_ticks_position('none')
            ax.spines['left'].set_visible(False)
    fig.suptitle('Real vs. Reconstructed')
    fig.legend(['Reconstructed', 'Real'])
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


def plot_factors(factors: ndarray, bin_size, ncols=5, figsize=(8, 8)):
    T, N = factors.shape

    nrows = int(np.ceil(N / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs = np.ravel(axs)

    time = np.linspace(0, T * bin_size, T, endpoint=False)
    fmin = factors.min()
    fmax = factors.max()

    for jx in range(N):
        plt.sca(axs[jx])
        plt.plot(time, factors[:, jx])
        plt.ylim(fmin - 0.1, fmax + 0.1)
        plt.ylabel(f'Activity of components {jx + 1}')

        if jx >= N - ncols:
            plt.xlabel('Time (s)')
        else:
            plt.xlabel('')
            axs[jx].set_xticklabels([])

    fig.suptitle('Factors 1-%i for a sampled trial.' % N)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig
