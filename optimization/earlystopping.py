import os
import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Early stop the training if validation loss doesn't improve after a
        given patiences.

        Parameters
        ----------
        patience : int, optional
            How long to wait after last time validation loss improved.
            Default: 7
        verbose : bool, optional
            If true, print a message for each validation loss improvement.
            Default: False
        delta : float
            Minimum change in the monitored quantity to qualify
            as an improvement. Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, models, save_path, modelnames='model.pt'):
        """
        Parameters
        ----------
        val_loss : float
            The validation loss of current step.
        models : torch.Module or list or tuple
            The training models. When val_loss smaller than current smallest
            number, the models will be saved. Note that if the models are
            list or tuple, the modelnames should also be list or tuple which
            have same length with models, and each torch.Module in the list
            will be saved under save_path.
        save_path : str
            The path where to save the model.
        modelnames : str or list or tuple, optional
            The name of the model will be saved in current step. If the type
            of modelnames are list or tuple, it need to have same length with
            models.
        """
        score = -val_loss

        if self.best_score is None:
            # Start of recording.
            self.best_score = score
            self._save_checkpoint(val_loss, models, save_path, modelnames)
        elif score < self.best_score + self.delta:
            # When score smaller than best score plus delta, count the
            # stop number.
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: '
                      f'{self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Update the best score and save checkpoint.
            self.best_score = score
            self._save_checkpoint(val_loss, models, save_path, modelnames)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model, save_path, names):
        """
        Save model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} '
                  f'--> {val_loss:.4f}). Saving model ...')
        if save_path is not None and names is not None:
            # Check the type of model and names
            if type(model) in [list, tuple] and type(names) in [list, tuple]:
                m, n = len(model), len(names)
                assert m == n, f'Error of number of model and names: {m}!={n}.'
                for m, n in zip(model, names):
                    torch.save(m.state_dict(), os.path.join(save_path, n))
            else:
                torch.save(model.state_dict(), os.path.join(save_path, names))
        self.val_loss_min = val_loss
