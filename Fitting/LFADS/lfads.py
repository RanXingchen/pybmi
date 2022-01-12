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
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .lfads_defaultparams import default_hyperparams
from .lfads_utils import plot_traces, plot_factors


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
    def __init__(self, N, ncomponents, dt, model_hyperparams=None,
                 run_name=''):
        """
        Create an LFADS model.

        Parameters
        ----------
        N : int
            The feature dimensionality of the data.
        T : int
            Number of time bins of the data.
        ncomponents : int
            Dimensionality of the latent factors.
        dt : float
            The time bin in seconds.
        hyperparameters : dict, optional
            The dictionary of model hyperparameters.
        """
        # call the nn.Modules constructor
        super(LFADS, self).__init__()

        self.N = N
        self.f_dim = ncomponents
        self.dt = dt
        self.run_name = run_name
        # Get the hyperparameters
        self._update_hyperparams(default_hyperparams, model_hyperparams)

        # -----------------------
        # NETWORK LAYERS INIT
        # -----------------------

        # BiRNN Encoder for Generator
        self.encoder_g = nn.GRU(self.N, self.enc_g_dim, bidirectional=True,
                                batch_first=True)
        # FC layer to estimate posterior of g0.
        self.fc_g0 = nn.Linear(2 * self.enc_g_dim, self.g_dim * 2)
        # BiRNN Encoder for Controller
        if self.enc_c_dim > 0:
            self.encoder_c = nn.GRU(self.N, self.enc_c_dim, bidirectional=True,
                                    batch_first=True)
        # Controller
        if self.enc_c_dim > 0 and self.c_dim > 0 and self.u_dim > 0:
            self.controller = nn.GRUCell(self.enc_c_dim * 2 + self.f_dim,
                                         self.c_dim)
            self.fc_u = nn.Linear(self.c_dim, self.u_dim * 2)
        # Generator
        self.generator = nn.GRUCell(self.u_dim, self.g_dim)

        # FC layers computig mean and log-variance of the posterior
        # distribution for the generator initial conditions (g0 from
        # encoder_g) or the inferred inputs (c from controller).
        # These layers takes input:
        #  - the forward encoder for g/c at time t (h_enc[t])
        #  - the backward encoder for g/c at time 1 (h_enc[1]]
        # Factors from generator output.
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
        one_g = torch.ones((self.g_dim,), device='cuda')
        self.g0_prior = {
            'mean': nn.Parameter(one_g * 0.0),
            'logv': nn.Parameter(one_g * np.log(self.kappa))
        }
        one_u = torch.ones((self.u_dim,), device='cuda')
        self.u_prior_gp = {
            'mean': nn.Parameter(one_u * 0.0),
            'logv': nn.Parameter(one_u * np.log(self.kappa)),
            'logt': nn.Parameter(one_u * np.log(10))
        }

        # --------------------------
        # Useful training variables INIT
        # --------------------------
        self.loss_dict = {
            'train': [], 'train_rec': [], 'train_kl': [], 'train_l2': [],
            'valid': [], 'valid_rec': [], 'valid_kl': []
        }
        self.last_decay_epoch = 0
        self.best = float('inf')
        # Define the optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, eps=self.epsilon, betas=self.betas
        )

    def encode(self, x: Tensor, h: tuple):
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
        # Resets parameter data pointer so that they can use faster code paths.
        self.encoder_g.flatten_parameters()
        if self.enc_c_dim > 0:
            self.encoder_c.flatten_parameters()

        # Get initial hidden state of h_enc_g and h_enc_c
        h_enc_g, h_enc_c = h

        # Dropout some data
        x = self.drop(x)
        # Run bidirectional RNN over data
        _, h_enc_g = self.encoder_g(x, h_enc_g.contiguous())
        h_enc_g = self.drop(h_enc_g.clamp(-self.clip_val, self.clip_val))
        # Concatenate forward/backward hidden state for generator.
        h_enc_g = torch.cat((h_enc_g[0], h_enc_g[1]), dim=-1)

        # Estimating g0 posterior distribution
        g0_mean, g0_logv = torch.split(self.fc_g0(h_enc_g), self.g_dim, dim=-1)
        g0_posterior = {'mean': g0_mean, 'logv': g0_logv}

        # Get output of encoder for controller if exist.
        if self.enc_c_dim > 0:
            out_enc_c, _ = self.encoder_c(x, h_enc_c.contiguous())
            out_enc_c = out_enc_c.clamp(-self.clip_val, self.clip_val)
            return g0_posterior, out_enc_c
        else:
            return g0_posterior, None

    def generate(self, g: Tensor, out_enc_c: Tensor, h_c: Tensor):
        """
        Generates the rates using the controller encoder outputs and the
        sampled initial conditions (g0).
        """
        device = g.device
        T = out_enc_c.shape[1]

        # Prepare the sequence container.
        factors = torch.zeros(g.shape[0], T, self.f_dim).to(device)
        gen_inp = torch.zeros(g.shape[0], T, self.u_dim).to(device)
        u_posterior = {
            'mean': torch.zeros(g.shape[0], T, self.u_dim).to(device),
            'logv': torch.zeros(g.shape[0], T, self.u_dim).to(device)
        }

        # Initialize factors by g0.
        f = self.fc_factors(self.drop(g))
        for t in range(T):
            # Concatenate h_enc_cf and h_enc_cb outputs at time t with factors
            # at time t - 1 as input to controller.
            out_enc_c_with_f = torch.cat((out_enc_c[:, t], f), dim=-1)
            # Update controller with controller encoder outputs
            h_c = self.controller(self.drop(out_enc_c_with_f), h_c)
            h_c = torch.clamp(h_c, min=-self.clip_val, max=self.clip_val)
            # Calculate posterior distribution parameters for inferred inputs
            # from controller state
            u_posterior['mean'][:, t], u_posterior['logv'][:, t] = \
                torch.split(self.fc_u(h_c), self.u_dim, dim=-1)
            # Sample inputs for generator from u(t) posterior distribution
            u = torch.randn(g.shape[0], self.u_dim).to(device) * \
                torch.exp(0.5 * u_posterior['logv'][:, t]) + \
                u_posterior['mean'][:, t]
            gen_inp[:, t] = u
            # Update generator
            g = self.generator(u, g)
            g = torch.clamp(g, -self.clip_val, self.clip_val)
            # Generate factors from generator state
            f = self.fc_factors(self.drop(g))
            factors[:, t] = f
        return u_posterior, factors, gen_inp

    def forward(self, x: Tensor):
        """
        Runs a forward pass through the network.

        Parameters
        ----------
        x : Tensor
            Single-trial spike data. shape [batch size, time-steps, input dim]
        """
        B, _, N = x.shape

        assert N == self.N, "The input features should be same with N."

        # 1. Initialize hidden state of encoder for generator and Controller.
        h_enc_g = self._init_hidden_state((B, self.enc_g_dim), True, x.device)
        h_enc_c = self._init_hidden_state((B, self.enc_c_dim), True, x.device)
        # 2. Encode the input data for generator and controller respectively.
        g0_posterior, out_enc_c = self.encode(x, (h_enc_g, h_enc_c))
        # 3. Sample initial conditions for generator from posterior
        g0 = torch.randn(B, self.g_dim).to(x.device) * \
            (0.5 * g0_posterior['logv']).exp() + g0_posterior['mean']

        # Initialize hidden state for controller.
        h_c = self._init_hidden_state((B, self.c_dim), device=x.device)
        # 4. Generate factors and rates.
        u_posterior, factors, inp_g = self.generate(g0, out_enc_c, h_c)
        # Instantiate AR1 process as mean and variance per time step
        self.u_prior = self._gp_to_normal(self.u_prior_gp, inp_g)

        # 5. Generate rates from factor state
        rates = torch.exp(self.fc_logrates(factors))
        # Generate factors.
        return g0_posterior, u_posterior, factors, rates

    def predict(self, dts, pad=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # Create dataloader.
        dl = DataLoader(dts, batch_size=batch_size)

        # Define the criteria
        rec_criteria = nn.PoissonNLLLoss(log_input=False, full=True,
                                         reduction='sum')
        gkl_criteria = bmi.optimization.loss.GaussianKLDivLoss(reduction='sum')

        loss, loss_rec, loss_kl = 0, 0, 0

        # Save factors and rates
        factors, rates = [], []

        self.eval()
        for i, batch in enumerate(dl):
            inp = batch[0]
            # Find the padding index
            iPad = bmi.utils.find_padding_index(inp.sum(dim=-1), pad)
            # Replace the padding value to 0.
            inp[iPad] = 0
            with torch.no_grad():
                g0_posterior, u_posterior, f, r = self(inp)
                factors.append(f)
                rates.append(r)

            # Loss computation

            # Gaussian KL Div loss
            kl_g = gkl_criteria(self.g0_prior['mean'], self.g0_prior['logv'],
                                g0_posterior['mean'], g0_posterior['logv'])
            kl_g /= inp.shape[0]
            kl_c = 0
            for t in range(inp.shape[1]):
                real_batch_size = (~iPad[:, t]).float().sum()
                if real_batch_size == 0:
                    break
                kl_c += gkl_criteria(
                    self.u_prior['mean'][:, t][~iPad[:, t]],
                    self.u_prior['logv'][:, t][~iPad[:, t]],
                    u_posterior['mean'][:, t][~iPad[:, t]],
                    u_posterior['logv'][:, t][~iPad[:, t]]
                ) / real_batch_size
            kl_loss = kl_g + kl_c
            # Reconstruction loss
            rec_loss = rec_criteria(r[~iPad], inp[~iPad]) / inp.shape[0]
            # Sum all the loss
            all_loss = rec_loss + kl_loss

            loss += all_loss.item()
            loss_rec += rec_loss.item()
            loss_kl += kl_loss.item()
        # End of test dataset

        loss /= (i + 1)
        loss_rec /= (i + 1)
        loss_kl /= (i + 1)

        factors = torch.cat(factors, dim=0)
        rates = torch.cat(rates, dim=0)
        return factors, rates, loss, loss_rec, loss_kl

    def fit(self, train_dts, save_path, valid_dts=None,
            pad=float('NaN'), use_tensorboard=True):
        """
        Fits the LFADS using ADAM optimization.

        Parameters
        ----------
        train_dts : TensorDataset
            Dataset with the training data to fit LFADS model
        save_path : str
            The root save path for all the results.
        valid_dts : TensorDataset, optional
            Dataset with validation data to validate LFADS model. When
            Validation set is None, no validate conducted, and the model
            trained all epoches. Default: None.
        iPad_train : BoolTensor, optional
            index padding of training set. This indicate where the training
            dataset pad with zeros. Default: None.
        iPad_valid : BoolTensor, optional
            index padding of validating set. This indicate where the validating
            dataset pad with zeros. Default: None.
        use_tensorboard : bool, optional
            Whether to write results to tensorboard. Default: False.
        """
        # Create the training dataloader
        t_dl = DataLoader(train_dts, self.batch_size, True, drop_last=True)
        # Define the criteria
        rec_criteria = nn.PoissonNLLLoss(log_input=False, full=True,
                                         reduction='sum')
        gkl_criteria = bmi.optimization.loss.GaussianKLDivLoss(reduction='sum')
        # Initialize tensorboard
        if use_tensorboard:
            tb_folder = os.path.join(save_path, self.run_name + 'tensorboard')
            if not os.path.exists(tb_folder):
                os.mkdir(tb_folder)
            writer = SummaryWriter(tb_folder)

        # Start training LFADS.
        current_step = 0
        print('\n=========================')
        print('Beginning training LFADS...')
        print('=========================\n')
        # For each epoch ...
        for epoch in range(self.epoch):
            # If minimum learning rate reached, break training loop
            if self.lr <= self.lr_min:
                break
            # cumulative training loss for this epoch
            tloss, tloss_rec, tloss_kl, tloss_l2 = 0, 0, 0, 0

            # for each step...
            self.train()
            for i, batch in enumerate(t_dl):
                current_step += 1
                inp = batch[0]
                # Find the padding index.
                iPad = bmi.utils.find_padding_index(inp.sum(dim=-1), pad)
                # Replace the padding value to 0.
                inp[iPad] = 0

                self.zero_grad()
                # Forward
                g0_posterior, u_posterior, t_factors, t_rates = self(inp)

                # Loss computation

                # L2 Loss
                l2_g = self._gru_hh_l2_loss(self.generator, self.l2_g_scale)
                l2_c = self._gru_hh_l2_loss(self.controller, self.l2_c_scale)
                l2_loss = l2_g + l2_c
                # Gaussian KL Div loss
                kl_g = gkl_criteria(
                    self.g0_prior['mean'], self.g0_prior['logv'],
                    g0_posterior['mean'], g0_posterior['logv']
                )
                kl_g /= self.batch_size
                kl_c = 0
                for t in range(inp.shape[1]):
                    real_batch_size = (~iPad[:, t]).float().sum()
                    if real_batch_size == 0:
                        break
                    kl_c += gkl_criteria(
                        self.u_prior['mean'][:, t][~iPad[:, t]],
                        self.u_prior['logv'][:, t][~iPad[:, t]],
                        u_posterior['mean'][:, t][~iPad[:, t]],
                        u_posterior['logv'][:, t][~iPad[:, t]]
                    ) / real_batch_size
                kl_loss = kl_g + kl_c
                # Reconstruction loss
                rec_loss = rec_criteria(t_rates[~iPad], inp[~iPad]) /\
                    inp.shape[0]
                # Calculate regularizer weights
                w_kl, w_l2 = self._weight_scheduler(current_step)
                # Sum all the loss
                loss = rec_loss + w_kl * kl_loss + w_l2 * l2_loss
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
                tloss_kl += kl_loss.item()
                tloss_l2 += l2_loss.item()

                if use_tensorboard:
                    self.health_check(writer, current_step)
            # End of one epoch training.

            tloss /= (i + 1)
            tloss_rec /= (i + 1)
            tloss_kl /= (i + 1)
            tloss_l2 /= (i + 1)

            # Do validation.
            v_factors, v_rates, vloss, vloss_rec, vloss_kl = \
                self.predict(valid_dts, pad)
            # Print Epoch Loss
            print('Epoch: %4d, Step: %5d, train loss: %.3f, '
                  'valid loss: %.3f' % (epoch + 1, current_step, tloss, vloss))
            # Store loss
            self.loss_dict['train'].append(tloss)
            self.loss_dict['train_rec'].append(tloss_rec)
            self.loss_dict['train_kl'].append(tloss_kl)
            self.loss_dict['train_l2'].append(tloss_l2)
            self.loss_dict['valid'].append(vloss)
            self.loss_dict['valid_rec'].append(vloss_rec)
            self.loss_dict['valid_kl'].append(vloss_kl)

            # Apply learning rate decay function
            if self.scheduler_on:
                self._apply_decay(epoch)

            # Write results to tensorboard
            if use_tensorboard:
                writer.add_scalars('1_Loss/1_Total_Loss',
                                   {'Train': tloss, 'Valid': vloss},
                                   epoch)
                writer.add_scalars('1_Loss/2_Reconstruction_Loss',
                                   {'Train':  tloss_rec, 'Valid': vloss_rec},
                                   epoch)
                writer.add_scalars('1_Loss/3_KL_Loss',
                                   {'Train': tloss_kl, 'Valid': vloss_kl},
                                   epoch)
                writer.add_scalar('1_Loss/4_L2_loss', tloss_l2, epoch)

                writer.add_scalar('2_Optimizer/1_Learning_Rate',
                                  self.lr, epoch)
                writer.add_scalar('2_Optimizer/2_KL_weight', w_kl, epoch)
                writer.add_scalar('2_Optimizer/3_L2_weight', w_l2, epoch)

            # Save model checkpoint if training error hits a new low and
            # kl and l2 loss weight schedule has completed.
            nkl = self.w_kl_start + self.w_kl_dur
            nl2 = self.w_l2_start + self.w_l2_dur
            if current_step >= max(nkl, nl2):
                if self.loss_dict['valid'][-1] < self.best:
                    self.best = self.loss_dict['valid'][-1]
                    # saving checkpoint
                    self.save_checkpoint(save_path, epoch + 1)

                    if use_tensorboard:
                        tfigs_dict = self.plot_summary(
                            inp[-1, ~iPad[-1]], t_rates[-1, ~iPad[-1]],
                            t_factors[-1, ~iPad[-1]]
                        )
                        writer.add_figure(
                            'Examples/1_Train', tfigs_dict['traces'], epoch
                        )
                        writer.add_figure(
                            'Factors/1_Train', tfigs_dict['factors'], epoch
                        )

                        # Get the valid data padding index.
                        v_inp = valid_dts.tensors[0]
                        v_iPad = bmi.utils.find_padding_index(
                            v_inp.sum(dim=-1), pad
                        )
                        vfigs_dict = self.plot_summary(
                            v_inp[-1, ~v_iPad[-1]], v_rates[-1, ~v_iPad[-1]],
                            v_factors[-1, ~v_iPad[-1]]
                        )
                        writer.add_figure(
                            'Examples/2_Valid', vfigs_dict['traces'], epoch
                        )
                        writer.add_figure(
                            'Factors/2_Valid', vfigs_dict['factors'], epoch
                        )
        # End of all epochs

        if use_tensorboard:
            writer.close()

        # Save the all loss to csv file.
        df = pd.DataFrame(self.loss_dict)
        df.to_csv(os.path.join(save_path, self.run_name + 'loss.csv'),
                  index_label='epoch')
        # Save a final checkpoint
        self.save_checkpoint(save_path, self.epoch)

        # Print message
        print('...training complete.')

    def plot_summary(self, true, pred, factors):
        plt.close()
        figs_dict = {}

        true = true.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        factors = factors.detach().cpu().numpy()

        figs_dict['traces'] = plot_traces(
            pred, true, self.dt, mode='activity', norm=False
        )
        figs_dict['factors'] = plot_factors(factors, self.dt)
        return figs_dict

    def save_checkpoint(self, save_path, epoch):
        """
        Save checkpoint of network parameters and optimizer state.

        Parameters
        ----------
        purge_limit : int, optional
            Delete previous checkpoint if there have been fewer
            epochs than this limit before saving again.

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
        epoch = str(epoch)
        # Get training_error as string
        loss = str(self.loss_dict['valid'][-1]).replace('.', '-')

        model_filename = '%s_epoch_%s_loss_%s.pth' % (timestamp, epoch, loss)

        # Create dictionary of training variables
        hps_dict = {
            'best': self.best, 'loss_dict': self.loss_dict,
            'last_decay_epoch': self.last_decay_epoch, 'lr': self.lr
        }
        # Save network parameters, optimizer state, and training variables
        torch.save(
            {'net': self.state_dict(), 'opt': self.optimizer.state_dict(),
             'hps': hps_dict},
            os.path.join(model_savepath, model_filename)
        )

    def load_checkpoint(self, loadpath, mode='best'):
        """
        Load checkpoint of network parameters and optimizer state.

        Parameters
        ----------
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
        model_loadpath = os.path.join(loadpath, 'models')

        # Get checkpoint filenames
        try:
            _, _, filenames = list(os.walk(model_loadpath))[0]
        except IndexError:
            return
        assert len(filenames) > 0, 'No model files under ' + model_loadpath

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

        assert os.path.splitext(filename)[1] == '.pth', \
            'Input filename must have .pth extension'

        # Load the specific checkpoint
        state = torch.load(os.path.join(model_loadpath, filename))
        # Set network parameters
        self.load_state_dict(state['net'])
        # Set optimizer state
        self.optimizer.load_state_dict(state['opt'])

        # Set training variables
        self.best = state['hps']['best']
        self.loss_dict = state['hps']['loss_dict']
        self.last_decay_epoch = state['hps']['last_decay_epoch']
        self.lr = state['hps']['lr']

    def health_check(self, writer, current_step):
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
        odict = self._modules

        for i, name in enumerate(odict.keys()):
            if 'gru' in name:
                writer.add_scalar(
                    '3_Weight_norms/%ia_%s_ih' % (i, name),
                    odict.get(name).weight_ih.data.norm(), current_step
                )
                writer.add_scalar(
                    '3_Weight_norms/%ib_%s_hh' % (i, name),
                    odict.get(name).weight_hh.data.norm(), current_step
                )

                if current_step > 1:
                    writer.add_scalar(
                        '4_Gradient_norms/%ia_%s_ih' % (i, name),
                        odict.get(name).weight_ih.grad.data.norm(),
                        current_step
                    )
                    writer.add_scalar(
                        '4_Gradient_norms/%ib_%s_hh' % (i, name),
                        odict.get(name).weight_hh.grad.data.norm(),
                        current_step
                    )
            elif 'fc' in name or 'conv' in name:
                writer.add_scalar(
                    '3_Weight_norms/%i_%s' % (i, name),
                    odict.get(name).weight.data.norm(), current_step
                )
                if current_step > 1:
                    writer.add_scalar(
                        '4_Gradient_norms/%i_%s' % (i, name),
                        odict.get(name).weight.grad.data.norm(), current_step
                    )

    def _gp_to_normal(self, prior, process):
        '''
        Convert gaussian process with given process mean, process log-variance,
        process tau, and realized process to mean and log-variance of diagonal
        Gaussian for each time-step
        '''
        gp_mean, gp_logv, gp_logt = prior['mean'], prior['logv'], prior['logt']
        device = gp_mean.device
        shape = (1, process.shape[0], process.shape[-1])

        tau = torch.exp(-1 / gp_logt.exp())

        mean = gp_mean * torch.ones(shape).to(device)   # (1, B, F)
        m = gp_mean + (process[:, :-1].permute(1, 0, 2) - gp_mean) * tau
        mean = torch.cat((mean, m))

        logv = gp_logv * torch.ones(shape).to(device)
        v = gp_logv * torch.ones(process.shape, device=device)[:, :-1].\
            permute(1, 0, 2)
        logv = torch.cat((logv, torch.log(1 - tau.pow(2)) + v))

        _prior = {}
        _prior['mean'] = mean.permute(1, 0, 2)
        _prior['logv'] = logv.permute(1, 0, 2)
        return _prior

    def _set_params(self, params: dict):
        """
        Register the paramters to the class.
        """
        for key, val in params.items():
            self.__setattr__(key, val)

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

    def _apply_decay(self, epoch):
        """
        Decrease the learning rate by a defined factor(learning_rate_decay).
        If loss is greater than the loss in the last six training steps and
        if the loss has not decreased in the last six training steps.
        See bullet point 8 of section 1.9 in online methods.
        """
        N = self.scheduler_patience
        C = self.scheduler_cooldown

        # index -1 in loss_dict is current loss.
        loss = self.loss_dict['train'][-1]

        if len(self.loss_dict['train']) >= N:
            if all(loss > torch.tensor(self.loss_dict['train'][-N:-1])):
                # When current loss is not decreased in last N epoch.
                if epoch >= self.last_decay_epoch + C:
                    # To apply decay, current step must greater than last
                    # step plus scheduler cooldown.
                    self.lr = self.lr * self.lr_decay
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.lr
                    # Update the last decay epoch.
                    self.last_decay_epoch = epoch

                    print('\n\tLearning rate decreased to %.8f' % self.lr)

    def _get_prior_stats(self, shape, mu, logk, device):
        _mean = torch.ones(shape, device=device) * mu
        _logvar = torch.ones(shape, device=device) * logk
        prior = {'mean': _mean, 'logvar': _logvar}
        return prior

    def _init_hidden_state(self, shape, bidirectional=False, device='cpu'):
        hf = torch.zeros(shape)
        if bidirectional:
            hb = torch.zeros(shape)
            h = torch.stack([hf, hb], dim=0).to(device)
        else:
            h = hf.to(device)
        return h

    def _gru_hh_l2_loss(self, gru: nn.GRUCell, scale: float):
        loss = scale * gru.weight_hh.norm(2) / gru.weight_hh.numel()
        return loss
