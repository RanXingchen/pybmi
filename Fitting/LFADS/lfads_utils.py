import numpy as np
import matplotlib.pyplot as plt
import pybmi as bmi

from torch.utils.tensorboard import SummaryWriter
from numpy import ndarray
from sklearn.metrics import r2_score


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

        if jx % ncols == 0:
            plt.ylabel(f'Activity of factor {jx + 1}')
        else:
            plt.ylabel('')
            axs[jx].set_yticklabels([])

        if jx >= N - ncols:
            plt.xlabel('Time (s)')
        else:
            plt.xlabel('')
            axs[jx].set_xticklabels([])

    fig.suptitle('Factors 1-%i for a sampled trial.' % N)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


def plot_rsquared(pred: ndarray, true: ndarray, figsize=(6, 4)):
    # Computing coefficient of determination.
    r2 = r2_score(true, pred)

    fig = plt.figure(figsize=figsize)

    plt.plot(np.ravel(true), np.ravel(pred), '.')
    plt.xlabel('Ground Truth Rates (Hz)')
    plt.ylabel('Inferred Rates (Hz)')
    plt.title('R-squared coefficient = %.3f' % r2)
    return fig


def plot_umean(umean: ndarray, bin_size, fig_width=8, fig_height=4):
    T, N = umean.shape

    figsize = (fig_width, fig_height * N)
    fig, axs = plt.subplots(nrows=N, figsize=figsize)
    fig.suptitle('Input to the generator for a sampled trial')

    time = np.linspace(0, T * bin_size, T, endpoint=False)
    for jx in range(N):
        if N > 1:
            plt.sca(axs[jx])
        else:
            plt.sca(axs)
        plt.plot(time, umean[:, jx])
        plt.xlabel('time (s)')
    return fig


class LFADS_Writer():
    def __init__(self, save_path):
        self.writer = SummaryWriter(save_path)

    def write_loss(self, loss_dict, epoch):
        loss = {
            'Train': loss_dict['train'][-1],
            'Valid': loss_dict['valid'][-1]
        }
        self.writer.add_scalars('1_Loss/1_Total_Loss', loss, epoch)

        loss = {
            'Train': loss_dict['train_rec'][-1],
            'Valid': loss_dict['valid_rec'][-1]
        }
        self.writer.add_scalars('1_Loss/2_Reconstruction_Loss', loss, epoch)

        loss = {
            'Train': loss_dict['train_kl'][-1],
            'Valid': loss_dict['valid_kl'][-1]
        }
        self.writer.add_scalars('1_Loss/3_KL_Loss', loss, epoch)

        self.writer.add_scalar('1_Loss/4_L2_loss', loss_dict['l2'][-1], epoch)

    def write_opt_params(self, lr, w_kl, w_l2, epoch):
        self.writer.add_scalar('2_Optimizer/1_LR', lr, epoch)
        self.writer.add_scalar('2_Optimizer/2_KL_weight', w_kl, epoch)
        self.writer.add_scalar('2_Optimizer/3_L2_weight', w_l2, epoch)

    def plot_examples(self, fig_dict: dict, epoch, label='1_Train'):
        self.writer.add_figure(
            'Examples/' + label, fig_dict['traces'], epoch
        )
        self.writer.add_figure(
            'Factors/' + label, fig_dict['factors'], epoch
        )
        self.writer.add_figure(
            'Inputs/' + label, fig_dict['inputs'], epoch
        )
        if 'truth' in fig_dict:
            self.writer.add_figure(
                'Ground_truth/' + label, fig_dict['truth'], epoch
            )
            self.writer.add_figure(
                'R-squared/' + label, fig_dict['r2'], epoch
            )

    def check_model(self, modules, step):
        """
        Checks the gradient norms for each parameter, what the maximum weight
        is in each weight matrix, and whether any weights have reached NaN.

        Report norm of each weight matrix
        Report norm of each layer activity
        Report norm of each Jacobian

        To report by batch. Look at data that is inducing the blow-ups.

        Create a -Nan report. What went wrong? Create file that shows data
        that preceded blow up,  and norm changes over epochs.

        Notes
        -----
        Theory 1: sparse activity in real data too difficult to encode
            - maybe, but not fixed by augmentation

        Theory 2: Edgeworth approximation ruining everything
            - probably: when switching to order=2 loss does not blow up,
            but validation error is huge
        """
        for i, name in enumerate(modules.keys()):
            if 'gru' in name:
                self.writer.add_scalar(
                    '3_Weight_norms/%ia_%s_ih' % (i + 1, name),
                    modules.get(name).weight_ih.data.norm(), step
                )
                self.writer.add_scalar(
                    '3_Weight_norms/%ib_%s_hh' % (i + 1, name),
                    modules.get(name).weight_hh.data.norm(), step
                )

                if step > 1:
                    self.writer.add_scalar(
                        '4_Gradient_norms/%ia_%s_ih' % (i + 1, name),
                        modules.get(name).weight_ih.grad.data.norm(), step
                    )
                    self.writer.add_scalar(
                        '4_Gradient_norms/%ib_%s_hh' % (i + 1, name),
                        modules.get(name).weight_hh.grad.data.norm(), step
                    )
            elif 'fc' in name or 'conv' in name:
                self.writer.add_scalar(
                    '3_Weight_norms/%i_%s' % (i + 1, name),
                    modules.get(name).weight.data.norm(), step
                )
                if step > 1:
                    self.writer.add_scalar(
                        '4_Gradient_norms/%i_%s' % (i + 1, name),
                        modules.get(name).weight.grad.data.norm(), step
                    )

    def close(self):
        self.writer.close()
