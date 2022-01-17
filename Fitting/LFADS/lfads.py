"""
LFADS - Latent Factor Analysis via Dynamical Systems.

LFADS is an unsupervised method to decompose time series data into various
factors, such as an initial condition, a generative dynamical system, control
inputs to that generator, and a low dimensional description of the observed
data, called factors. Additionally, the observations have a noise model (in
this case Poisson), so a denoised version of the observations is also created
(e.g. underlying rates of a Poisson distribution given the observed event
counts).
"""


import os
import datetime
import pybmi as bmi
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from torch.utils.data import DataLoader

from .lfads_defaultparams import default_hyperparams
from .lfads_utils import plot_traces, plot_factors, plot_rsquared, plot_umean
from .lfads_utils import LFADS_Writer


def weights_init(m: nn.Module):
    """
    The weight initialization is modified from the standard PyTorch,
    which is uniform. Instead, the weights are drawn from a normal
    distribution with mean 0 and std = 1/sqrt(K) where K is the size of
    the input dimension. This helps prevent vanishing/exploding gradients
    by keeping the eigenvalues of the Jacobian close to 1.
    """
    with torch.no_grad():
        for name, p in m.named_parameters():
            if 'weight' in name:
                k = p.shape[1]      # Dimensionality of input.
                # Inplace resetting W ~ N(0, 1/sqrt(K))
                if p.data.numel() > 0:
                    p.data.normal_(std=k ** -0.5)


class LFADS(nn.Module):
    """
    Implemention of LFADS neural network. This code is followed by
    https://github.com/lyprince/lfads_demo.
    And the details of the method is described on the papre:
    C. Pandarinath et al., “Inferring single-trial neural population dynamics
    using sequential auto-encoders,” Nature Methods, vol. 15, no. 10,
    pp. 805-815, Oct. 2018, doi: 10.1038/s41592-018-0109-9.
    """
    def __init__(self, N, ncomponents, dt, hyperparams=None,
                 run_name=''):
        """
        Create an LFADS model.

        Parameters
        ----------
        N : int
            The feature dimensionality of the data.
        ncomponents : int
            Dimensionality of the latent factors.
        dt : float
            The time bin in seconds.
        hyperparameters : dict, optional
            The dictionary of model hyperparameters.
        """
        # Call the nn.Modules constructor
        super(LFADS, self).__init__()

        self.N = N
        self.f_dim = ncomponents
        self.dt = dt
        self.run_name = run_name
        # Get the hyperparameters
        self._update_hyperparams(default_hyperparams, hyperparams)

        # -----------------------
        # NETWORK LAYERS INIT
        # -----------------------

        # Encoder of LFADS
        self.encoder = LFADSEncoder(
            self.N, self.enc_g_dim, self.g_latent_dim,
            self.enc_c_dim, self.dropout, self.clip_val
        )
        # Squeeze g_latent_dim to g_dim
        self.fc_g = LFADSIdentity() if self.g_latent_dim == self.g_dim else \
            nn.Linear(self.g_latent_dim, self.g_dim)

        # The encoder for controller and controller only initialized when
        # enc_c_dim greater than 0.
        if self.enc_c_dim > 0 and self.c_dim > 0 and self.u_dim > 0:
            self.controller = LFADSControllerCell(
                self.enc_c_dim * 2 + self.f_dim, self.c_dim, self.u_dim,
                self.dropout, self.clip_val
            )

        # Generator. Note 'u_dim' must greater than 0.
        self.generator = nn.GRUCell(self.u_dim, self.g_dim)

        # Factors from generator output
        self.fc_factors = nn.Linear(self.g_dim, self.f_dim)
        # Estimate logrates from factors.
        self.fc_logrates = nn.Linear(self.f_dim, self.N)

        self.drop = nn.Dropout(self.dropout)

        # -----------------------
        # WEIGHT INIT
        # -----------------------
        weights_init(self)
        # Row-normalize fc_factors (bullet-point 11 of section 1.9 of online
        # methods). This is used to keep the factors relatively evenly scaled
        # with respect to each other.
        self.fc_factors.weight.data = F.normalize(self.fc_factors.weight.data,
                                                  dim=1)

        # --------------------------
        # LEARNABLE PRIOR PARAMETERS INIT
        # --------------------------
        one_g = torch.ones((self.g_latent_dim,), device='cuda')
        self.g0_prior = {
            'mean': nn.Parameter(one_g * 0.0),
            'logv': nn.Parameter(one_g * np.log(self.kappa))
        }
        if self.enc_c_dim > 0 and self.c_dim > 0 and self.u_dim > 0:
            one_u = torch.ones((self.u_dim,), device='cuda')
            self.u_prior = {
                'mean': nn.Parameter(one_u * 0.0),
                'logv': nn.Parameter(one_u * np.log(self.kappa)),
                'logt': nn.Parameter(one_u * np.log(10))
            }

        # --------------------------
        # Useful training variables INIT
        # --------------------------
        self.current_epoch = 0
        self.current_step = 0
        self.loss_dict = {
            'train': [], 'train_rec': [], 'train_kl': [],
            'valid': [], 'valid_rec': [], 'valid_kl': [],
            'l2': []
        }
        self.last_decay_epoch = 0
        self.best = float('inf')

        # --------------------------
        # Optimizing stuff
        # --------------------------
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, eps=self.epsilon, betas=self.betas
        )
        self.gkl_criterion = bmi.optimization.loss.GaussianKLDivLoss('sum')
        self.rec_criteria = nn.PoissonNLLLoss(
            log_input=False, full=True, reduction='sum'
        )

    def encode(self, x: Tensor, h: tuple, iPad: Tensor = None):
        """
        Function to encode the data with the forward and backward encoders.

        Parameters
        ----------
        x : Tensor
            Tensor of size [batch size, time-steps, input dims]
        h : tuple
            tuple of (h_enc_g, h_enc_c)
            The hidden state of the encoder for generator and controller.
            Shape [2, Batch size, encoder_g/c dim], where the 0 is the forward
            and 1 is the backward.
        """
        # Encode data into forward and backward generator encoders to
        # produce g0 distribution for generator initial conditions.
        g0_mean, g0_logv, o_enc_c = self.encoder(x, h, iPad)
        # Sample initial conditions of generator from posterior distribution
        g0 = self.fc_g(self._sample_gaussian(g0_mean, g0_logv))

        # KL cost for g(0)
        self.kl_loss = self.gkl_criterion(
            self.g0_prior['mean'], self.g0_prior['logv'], g0_mean, g0_logv
        ) / x.shape[0]
        return g0, o_enc_c

    def generate(self, g: Tensor, o_enc_c: Tensor, h_c: Tensor,
                 T: int, iPad: Tensor = None):
        """
        Generates the rates using the controller encoder outputs and the
        sampled initial conditions (g0).

        Parameters
        ----------
        g : Tensor
            Initial condition of generator.
        o_enc_c : Tensor
            output of controller encoder.
        h_c : Tensor
            Hidden state of controller.
        """
        device = g.device
        B = g.shape[0]

        # Prepare the sequence container.
        factors = torch.zeros(B, T, self.f_dim).to(device)
        gen_inp = torch.zeros(B, T, self.u_dim).to(device)
        u_posterior = {
            'mean': torch.zeros(B, T, self.u_dim).to(device),
            'logv': torch.zeros(B, T, self.u_dim).to(device)
        }

        # Get batch padding information.
        batch_info = (~iPad).int().sum(dim=0)

        # Initialize factors by g0.
        f = self.fc_factors(self.drop(g))
        for t in range(T):
            # When input data include padding value, check if current
            # step are all consist of paddings.
            if batch_info[t] == 0:
                # All data is padding, break the loop.
                break
            # Concatenate o_enc_c at time t with factors at time t - 1 as
            # input to controller.
            if self.enc_c_dim > 0 and self.c_dim > 0 and self.u_dim > 0:
                o_enc_c_with_f = torch.cat((o_enc_c[:, t], f), dim=-1)
                # Update controller with controller encoder outputs
                u_mean, u_logv, h_c = self.controller(o_enc_c_with_f, h_c)
                # Calculate posterior distribution parameters for inferred
                # inputs from controller state
                u_posterior['mean'][:, t] = u_mean
                u_posterior['logv'][:, t] = u_logv
                # Sample inputs for generator from u(t) posterior distribution
                u = self._sample_gaussian(u_mean, u_logv)

                # KL cost for u(t)
                self.kl_loss += self.gkl_criterion(
                    self.u_prior['mean'], self.u_prior['logv'],
                    u_mean[~iPad[:, t]], u_logv[~iPad[:, t]]
                ) / batch_info[t]
            else:
                u = torch.empty(B, self.u_dim).to(device)
            # Save input of generator.
            gen_inp[:, t] = u

            # Update generator
            g = self.generator(u, g)
            g = torch.clamp(g, -self.clip_val, self.clip_val)
            # Generate factors from generator state
            f = self.fc_factors(self.drop(g))
            factors[:, t] = f
        # END of running time stamp.

        return factors, gen_inp, u_posterior

    def forward(self, x: Tensor, iPad: Tensor = None):
        """
        Runs a forward pass through the network.

        Parameters
        ----------
        x : Tensor
            Single-trial spike data. shape [batch size, time steps, input dim]
        """
        B, T, N = x.shape

        assert N == self.N, "The input features should be same with N."
        # Update batch size incase the last batch in dataset not enough.
        self.batch_size = B

        # 1.1 Initialize hidden state of encoder for gen and con.
        h_enc_g = self._init_hidden_state((B, self.enc_g_dim), True, x.device)
        h_enc_c = self._init_hidden_state((B, self.enc_c_dim), True, x.device)\
            if self.enc_c_dim > 0 else None
        # 1.2 Encode the input data for generator and controller respectively.
        g0, o_enc_c = self.encode(x, (h_enc_g, h_enc_c), iPad)

        # 2.1 Initialize hidden state for controller.
        h_c = self._init_hidden_state((B, self.c_dim), device=x.device)\
            if self.c_dim > 0 else None
        # 2.2. Generate factors and rates.
        factors, _, u_posterior = self.generate(g0, o_enc_c, h_c, T, iPad)
        # 2.3 Generate rates from factor state
        rates = torch.exp(self.fc_logrates(factors))
        return factors, rates, u_posterior['mean']

    def predict(self, dts, l2_loss=0, pad=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # Create dataloader.
        dl = DataLoader(dts, batch_size=batch_size)

        vloss, vloss_rec, vloss_kl = 0, 0, 0

        # Save factors and rates
        factors, rates, umean = [], [], []

        self.eval()
        for i, batch in enumerate(dl):
            inp = batch[0]
            # Find the padding index
            iPad = bmi.utils.find_padding_index(inp.sum(dim=-1), pad)
            # Replace the padding value to 0.
            inp[iPad] = 0
            with torch.no_grad():
                f, r, u = self(inp, iPad)

            # Save factors of current batch.
            factors.append(f)
            rates.append(r)
            umean.append(u)

            # *Loss computation

            # Reconstruction loss
            rec_loss = self.rec_criteria(r[~iPad] * self.dt,
                                         inp[~iPad]) / inp.shape[0]
            # Sum all the loss
            loss = rec_loss + self.kl_loss + l2_loss

            vloss += loss.item()
            vloss_rec += rec_loss.item()
            vloss_kl += self.kl_loss.item()
        # End of test dataset

        # Concatenate all batches factors and rates.
        factors = torch.cat(factors, dim=0)
        rates = torch.cat(rates, dim=0)
        umean = torch.cat(umean, dim=0)

        vloss /= (i + 1)
        vloss_rec /= (i + 1)
        vloss_kl /= (i + 1)
        return factors, rates, umean, vloss, vloss_rec, vloss_kl

    def fit(self, t_dts, save_path, v_dts=None, pad=float('NaN'),
            use_tensorboard=True, valid_truth=None):
        """
        Fits the LFADS using ADAM optimization.

        Parameters
        ----------
        t_dts : TensorDataset
            Dataset with the training data to fit LFADS model
        save_path : str
            The root save path for all the results.
        v_dts : TensorDataset, optional
            Dataset with validation data to validate LFADS model. When
            Validation set is None, no validate conducted, and the model
            trained all epoches. Default: None.
        use_tensorboard : bool, optional
            Whether to write results to tensorboard. Default: False.
        """
        # Create the training dataloader
        t_dl = DataLoader(t_dts, self.batch_size, True)
        # Initialize tensorboard
        if use_tensorboard:
            tb_folder = os.path.join(save_path, self.run_name + 'tensorboard')
            if not os.path.exists(tb_folder):
                os.mkdir(tb_folder)
            writer = LFADS_Writer(tb_folder)

        # Start training LFADS.
        print('\n=========================')
        print('Beginning training LFADS...')
        print('=========================\n')
        # For each epoch ...
        for epoch in range(self.current_epoch, self.epoch):
            # If minimum learning rate reached, break training loop.
            if self.lr <= self.lr_min:
                break
            # Cumulative training loss for this epoch.
            tloss, tloss_rec, tloss_kl = 0, 0, 0

            # For each step...
            self.train()
            for i, batch in enumerate(t_dl):
                self.current_step += 1
                inp = batch[0]
                # Find the padding index.
                iPad = bmi.utils.find_padding_index(inp.sum(dim=-1), pad)
                # Replace the padding value to 0.
                inp[iPad] = 0

                self.zero_grad()
                # *Forward
                tf, tr, tu = self(inp, iPad)

                # *Loss computation

                # L2 Loss
                l2_g = self._gru_hh_l2_loss(self.generator, self.l2_g_scale)
                if self.enc_c_dim > 0 and self.c_dim > 0 and self.u_dim > 0:
                    l2_c = self._gru_hh_l2_loss(self.controller.controller,
                                                self.l2_c_scale)
                else:
                    l2_c = 0
                l2_loss = l2_g + l2_c
                # Reconstruction loss
                rec_loss = self.rec_criteria(tr[~iPad] * self.dt,
                                             inp[~iPad]) / inp.shape[0]
                # Calculate regularizer weights
                w_kl, w_l2 = self._weight_scheduler(self.current_step)
                # Sum all the loss
                loss = rec_loss + w_kl * self.kl_loss + w_l2 * l2_loss
                # Check if loss is nan
                assert not torch.isnan(loss.data), 'Loss is NaN!'
                # Backward
                loss.backward()
                # clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.parameters(),
                                               max_norm=self.max_norm)
                # update the weights
                self.optimizer.step()
                # Row-normalize fc_factors
                self.fc_factors.weight.data = F.normalize(
                    self.fc_factors.weight.data, dim=1
                )

                # Add batch loss to epoch running loss
                tloss += loss.item()
                tloss_rec += rec_loss.item()
                tloss_kl += self.kl_loss.item()

                if use_tensorboard:
                    writer.check_model(self._modules, self.current_step)
            # End of one epoch training.

            tloss /= (i + 1)
            tloss_rec /= (i + 1)
            tloss_kl /= (i + 1)

            # Do validation.
            vf, vr, vu, vloss, vloss_rec, vloss_kl = \
                self.predict(v_dts, l2_loss, pad)
            # Print Epoch Loss
            print('Epoch: %4d, Step: %5d, train loss: %.3f, valid loss: %.3f'
                  % (epoch + 1, self.current_step, tloss, vloss))
            # Store loss
            self.loss_dict['train'].append(tloss)
            self.loss_dict['train_rec'].append(tloss_rec)
            self.loss_dict['train_kl'].append(tloss_kl)
            self.loss_dict['valid'].append(vloss)
            self.loss_dict['valid_rec'].append(vloss_rec)
            self.loss_dict['valid_kl'].append(vloss_kl)
            self.loss_dict['l2'].append(l2_loss.item())

            # Apply learning rate decay function
            if self.scheduler_on:
                self._apply_decay(epoch)

            # Write results to tensorboard
            if use_tensorboard:
                writer.write_loss(self.loss_dict, epoch)
                writer.write_opt_params(self.lr, w_kl, w_l2, epoch)

            # Save model checkpoint if training error hits a new low and
            # kl and l2 loss weight schedule has completed.
            nkl = self.w_kl_start + self.w_kl_dur
            nl2 = self.w_l2_start + self.w_l2_dur
            if self.current_step >= max(nkl, nl2):
                if self.loss_dict['valid'][-1] < self.best:
                    self.best = self.loss_dict['valid'][-1]
                    # saving checkpoint
                    self.save_checkpoint(save_path)

                    if use_tensorboard:
                        idx = np.random.randint(inp.shape[0])
                        tfigs_dict = self.plot_summary(
                            inp[idx, ~iPad[idx]], tr[idx, ~iPad[idx]],
                            tf[idx, ~iPad[idx]], tu[idx, ~iPad[idx]]
                        )
                        writer.plot_examples(tfigs_dict, epoch, '1_Train')

                        # Get the valid data padding index.
                        v_inp = v_dts.tensors[0]
                        v_iPad = bmi.utils.find_padding_index(
                            v_inp.sum(dim=-1), pad
                        )
                        idx = np.random.randint(vf.shape[0])
                        vfigs_dict = self.plot_summary(
                            v_inp[idx, ~v_iPad[idx]], vr[idx, ~v_iPad[idx]],
                            vf[idx, ~v_iPad[idx]], vu[idx, ~v_iPad[idx]],
                            truth=valid_truth[idx, ~v_iPad[idx]]
                        )
                        writer.plot_examples(vfigs_dict, epoch, '2_Valid')
            self.current_epoch += 1
        # End of all epochs

        if use_tensorboard:
            writer.close()

        df = pd.DataFrame(self.loss_dict)
        df.to_csv(os.path.join(save_path, self.run_name + 'loss.csv'),
                  index_label='epoch')
        # Save a final checkpoint
        self.save_checkpoint(save_path)

        # Print message
        print('...training complete.')

    def plot_summary(self, true, pred, factors, umean, truth=None):
        plt.close()
        figs_dict = {}

        true = true.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        factors = factors.detach().cpu().numpy()
        umean = umean.detach().cpu().numpy()

        figs_dict['traces'] = plot_traces(
            pred, true, self.dt, mode='activity', norm=False
        )
        figs_dict['traces'].suptitle('Spiking Data vs.Inferred Rate')
        figs_dict['traces'].legend(['Inferred Rates', 'Spikes'])

        figs_dict['factors'] = plot_factors(factors, self.dt)

        if torch.is_tensor(truth):
            truth = truth.detach().cpu().numpy()
            figs_dict['truth'] = plot_traces(pred, truth, self.dt, mode='rand')
            figs_dict['truth'].suptitle('Inferred vs. Ground-truth rates')
            figs_dict['truth'].legend(['Inferred', 'Ground-truth'])
            figs_dict['r2'] = plot_rsquared(pred, truth)
        if self.enc_c_dim > 0 and self.c_dim > 0 and self.u_dim > 0:
            figs_dict['inputs'] = plot_umean(umean, self.dt)
        return figs_dict

    def save_checkpoint(self, save_path):
        """
        Save checkpoint of network parameters and optimizer state.

        Parameters
        ----------
        save_path : str
            The path used to save the model.

        Notes
        -----
        Output filename of format [timestamp]_epoch_[epoch]_loss_[valid].pth:
            - timestamp   (YYMMDDhhmm)
            - epoch       (int)
            - loss        (float with decimal point replaced by -)
        """
        model_savepath = os.path.join(save_path, self.run_name + 'models')
        if not os.path.exists(model_savepath):
            os.mkdir(model_savepath)

        # Get current time in YYMMDDhhmm format
        timestamp = datetime.datetime.now().strftime('%y%m%d%H%M')
        # Get epoch_num as string
        epoch = str(self.current_epoch)
        # Get training_error as string
        loss = str(self.loss_dict['valid'][-1]).replace('.', '-')

        model_filename = '%s_epoch_%s_loss_%s.pth' % (timestamp, epoch, loss)

        # Create dictionary of training variables
        train_dict = {
            'best': self.best, 'loss_dict': self.loss_dict,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'last_decay_epoch': self.last_decay_epoch,
            'lr': self.lr
        }
        # Save network parameters, optimizer state, and training variables
        torch.save(
            {'net': self.state_dict(), 'opt': self.optimizer.state_dict(),
             'train': train_dict},
            os.path.join(model_savepath, model_filename)
        )

    def load_checkpoint(self, loadpath, mode='best'):
        """
        Load checkpoint of network parameters and optimizer state.

        Parameters
        ----------
        loadpath : str
            The path where store the model .pth file.
        mode : str, optional
            Path to input file, must have '.pth' extension. It can
            be one of 'best', 'recent', or 'longest'.
            - 'best': checkpoint with lowest saved loss.
            - 'recent': most recent checkpoint.
            - 'longest': checkpoint after most training.
            Default: 'best'.
        """
        mode = bmi.utils.check_params(
            mode, ['best', 'recent', 'longest'], 'mode'
        )

        # If loadpath is not a filename, get checkpoint with specified
        # quality (best, recent, longest).
        if not os.path.isfile(loadpath):
            # The path of model folder.
            model_loadpath = os.path.join(loadpath, self.run_name + 'models')
            # Get checkpoint filenames
            try:
                _, _, filenames = list(os.walk(model_loadpath))[0]
            except IndexError:
                return
            assert len(filenames) > 0, 'No models under ' + model_loadpath

            # Sort in ascending order
            filenames.sort()
            # Split filenames into attributes (date, epoch, loss)
            split_filenames = [os.path.splitext(f)[0].split('_')
                               for f in filenames]
            dates = [att[0] for att in split_filenames]
            epochs = [att[2] for att in split_filenames]
            losses = [att[-1] for att in split_filenames]

            if mode == 'best':
                # Get filename with lowest loss.
                # If conflict, take most recent of subset.
                losses.sort()
                best = losses[0]
                filename = [f for f in filenames if best in f][-1]
            elif mode == 'recent':
                # Get filename with most recent timestamp.
                # If conflict, take first one
                dates.sort()
                recent = dates[-1]
                filename = [f for f in filenames if recent in f][0]
            else:
                # Get filename with most number of epochs run.
                # If conflict, take most recent of subset.
                epochs.sort()
                longest = epochs[-1]
                filename = [f for f in filenames if longest in f][-1]
            # Get the full path filename
            filename = os.path.join(model_loadpath, filename)
        else:
            filename = loadpath
        # END OF loadpath IS FOLDER.

        assert os.path.splitext(filename)[1] == '.pth', \
            'Input filename must have .pth extension'

        print('\nLoading checkpoint ' + filename + '...')

        # Load the specific checkpoint
        state = torch.load(filename)
        # Set network parameters
        self.load_state_dict(state['net'])
        # Set optimizer state
        self.optimizer.load_state_dict(state['opt'])
        # Set training variables
        self.best = state['train']['best']
        self.loss_dict = state['train']['loss_dict']
        self.current_epoch = state['train']['current_epoch']
        self.current_step = state['train']['current_step']
        self.last_decay_epoch = state['train']['last_decay_epoch']
        self.lr = state['train']['lr']

    def _set_params(self, params):
        for k in params.keys():
            self.__setattr__(k, params[k])

    def _update_hyperparams(self, pre_parm: dict, new_parm: dict):
        """
        Update the new hyperparameters through the previous hyperparameters.
        """
        params = bmi.utils.update_dict(pre_parm, new_parm) \
            if new_parm else pre_parm
        self._set_params(params)

    def _weight_scheduler(self, step):
        """
        Calculating KL and L2 regularization weights from current
        training step. Imposes linearly increasing schedule on regularization
        weights to prevent early pathological minimization of KL divergence
        and L2 norm before sufficient data reconstruction improvement.
        See bullet-point 4 of section 1.9 in online methods

        Parameters
        ----------
        step : int
            Current training step number.
        """
        # Get step number of scheduler
        n_kl = max(step - self.w_kl_start, 0)
        n_l2 = max(step - self.w_l2_start, 0)
        # Calculate schedule weight
        kl_weight = min(n_kl / self.w_kl_dur, 1.0)
        l2_weight = min(n_l2 / self.w_l2_dur, 1.0)
        return kl_weight, l2_weight

    def _init_hidden_state(self, shape, bidirectional=False, device='cpu'):
        """
        Initialize the hidden state for GRU Cells.
        """
        hf = torch.zeros(shape)
        if bidirectional:
            # Add backward hidden state.
            hb = torch.zeros(shape)
            h = torch.stack([hf, hb], dim=0).to(device)
        else:
            h = hf.to(device)
        return h

    def _sample_gaussian(self, mean: torch.Tensor, logv: torch.Tensor):
        """
        Sample from a diagonal gaussian with given mean and log-variance.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of diagional gaussian
        logv : torch.Tensor
            Log-variance of diagonal gaussian
        """
        # Generate noise from standard gaussian.
        eps = torch.randn(mean.shape, dtype=mean.dtype).to(mean.device)
        # Scale and shift by mean and standard deviation
        return eps * (logv * 0.5).exp() + mean

    def _apply_decay(self, epoch):
        """
        Decrease the learning rate by a defined factor(lr_decay).
        If loss is greater than the loss in the last 'patience' training
        steps and if the loss has not decreased in the last 'cooldown'
        training steps. See bullet point 8 of section 1.9 in online methods.
        """
        N = self.scheduler_patience
        C = self.scheduler_cooldown

        # index -1 in loss_dict is current loss.
        loss = self.loss_dict['train'][-1]

        # There is N + 1 because the last one is current loss.
        if len(self.loss_dict['train']) >= N + 1:
            if all(loss > torch.tensor(self.loss_dict['train'][-(N + 1):-1])):
                # When current loss is not decreased in last C epoch.
                if epoch >= self.last_decay_epoch + C:
                    # To apply decay, current step must greater than last
                    # step plus scheduler cooldown.
                    self.lr = self.lr * self.lr_decay
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.lr
                    # Update the last decay epoch.
                    self.last_decay_epoch = epoch

                    print('\n\tLearning rate decreased to %.8f' % self.lr)

    def _gru_hh_l2_loss(self, gru: nn.GRUCell, scale: float):
        loss = scale * gru.weight_hh.norm(2) / gru.weight_hh.numel()
        return loss


class LFADSEncoder(nn.Module):
    def __init__(self, inp_size, enc_g_dim, g_latent_dim, enc_c_dim=0,
                 dropout=0.0, clip_val=5.0):
        super(LFADSEncoder, self).__init__()
        self.enc_g_dim = enc_g_dim
        self.g_latent_dim = g_latent_dim
        self.enc_c_dim = enc_c_dim
        self.clip_val = clip_val

        self.dropout = nn.Dropout(dropout)

        self.encoder_g = nn.GRU(inp_size, enc_g_dim, bidirectional=True,
                                batch_first=True)
        self.fc_g0 = nn.Linear(2 * enc_g_dim, 2 * g_latent_dim)

        if enc_c_dim > 0:
            self.encoder_c = nn.GRU(inp_size, enc_c_dim, bidirectional=True,
                                    batch_first=True)

    def forward(self, x: Tensor, h: tuple, iPad: Tensor):
        self.encoder_g.flatten_parameters()
        if self.enc_c_dim > 0:
            self.encoder_c.flatten_parameters()

        # Get the real length of current batch data.
        x_len = (~iPad).long().sum(1).cpu()

        h_enc_g, h_enc_c = h

        # Run bidirectional RNN over data
        x = self.dropout(x)

        x_pack = rnn_utils.pack_padded_sequence(x, x_len, batch_first=True,
                                                enforce_sorted=False)
        _, h_enc_g = self.encoder_g(x_pack, h_enc_g.contiguous())
        h_enc_g = self.dropout(h_enc_g.clamp(-self.clip_val, self.clip_val))
        h_enc_g = torch.cat((h_enc_g[0], h_enc_g[1]), dim=1)

        g0_mean, g0_logv = torch.split(
            self.fc_g0(h_enc_g), self.g_latent_dim, dim=1
        )

        if self.enc_c_dim > 0:
            o_enc_c, _ = self.encoder_c(x_pack, h_enc_c.contiguous())
            o_enc_c, _ = rnn_utils.pad_packed_sequence(o_enc_c, True)
            o_enc_c = torch.clamp(o_enc_c, -self.clip_val, self.clip_val)
        else:
            o_enc_c = None
        return g0_mean, g0_logv, o_enc_c


class LFADSControllerCell(nn.Module):
    def __init__(self, inp_size, hidden_size, u_dim, dropout=0.0,
                 clip_val=5.0):
        super(LFADSControllerCell, self).__init__()
        self.u_dim = u_dim
        self.clip_val = clip_val

        self.dropout = nn.Dropout(dropout)
        self.controller = nn.GRUCell(inp_size, hidden_size)
        self.fc_u = nn.Linear(hidden_size, u_dim * 2)

    def forward(self, x: Tensor, h: Tensor):
        h = self.controller(self.dropout(x), h)
        h = h.clamp(-self.clip_val, self.clip_val)
        u_mean, u_logv = torch.split(self.fc_u(h), self.u_dim, dim=1)
        return u_mean, u_logv, h


class LFADSIdentity(nn.Module):
    def __init__(self):
        super(LFADSIdentity, self).__init__()

    def forward(self, x: Tensor):
        return x
