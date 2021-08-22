import numpy as np
import torch
import scipy.io as scio
import os
import torch.nn as nn
import matplotlib.pyplot as plt

from pybmi.optimization import EarlyStopping
from pybmi.stats import corrcoef


def compute_direction(X: torch.Tensor) -> torch.Tensor:
    # Compute the velocity direction.
    direction = X / torch.norm(X, p=2, dim=-1, keepdim=True)
    direction[torch.isnan(direction)] = 0
    return direction


class PopulationVector():
    """
    Kalman filter for brain computer interface (BCI) decoding.

    This Kalman filter is designed for BCI decoding. You can
    use it to predict the target's future location by decoding
    the corresponding neural signal. In this framework, the
    target data (like position, velocity or acceleration, etc.)
    is modeled as the system state and the neural signal is
    modeled as the observation (measurement). Note that to use
    this Kalman filter, it assumes the observation is a linear
    function of the state plus Gaussian noise, and the target
    data at time t is assumed to be a linear function of the
    target data at the previous time instant plus Gaussian noise.

    The Kalman filter algorithm implements a discrete time,
    linear State-Space System described as follows.

        x(t) = A * x(t-1) + r(t-1)           (state equation)
        z(t) = H * x(t) + q(t)               (measurement equation)

    The Kalman filter algorithm involves two steps.
        - Predict: Using the previous states to predict the
                   current state.
        - Update:  Using the current measurement, such as the
                   recorded neural signal, to correct the state.

    Parameters
    ----------
    Z : ndarray or tensor, shape (T, N)
        The measurement matrix which is used to calculate the
        models, where N means the feature dimensions of the
        measurement. In BCI application, it should be the X's
        corresponding neural data.
    X : ndarray or tensor, shape (T, M)
        The state matrix which is used to calculate the models,
        where T means the time steps and M means the feature
        dimensions. In the BCI application, it can be the
        movement or other behavior data.
    eps : float, optional
        In case of during the computation, X'X is not full rank,
        the eps is added on the diagnol of the covariance matrix
        to make sure it always have inverse matrix.

    Attributes
    ----------
    A : tensor, shape (M, M)
        Model describing state transition between time steps.
        Specify the transition of state between times as an M-by-M
        matrix. This attribute cannot be changed once the Kalman
        filter is trained.
    H : tensor, shape (N, M)
        Model describing state to measurement transformation.
        Specify the transition from state to measurement as an N-by-M
        matrix. This attribute cannot be changed once the Kalman
        filter is trained.
    P : tensor, shape (M, M)
        State estimation error covariance. Specify the covariance
        of the state estimation error as an M-by-M diagonal matrix.
    Q : tensor, shape (M, M)
        Process noise covariance. Specify the covariance of process
        noise as an M-by-M matrix.
    R : tensor, shape (N, N)
        Measurement noise covariance. Specify the covariance of
        measurement noise as an N-by-N matrix.

    Examples
    --------

    References
    ----------
    [1] Wu, W., et al. (2002). Inferring hand motion from multi-cell
        recordings in motor cortex using a Kalman filter.
        SAB’02-workshop on motor control in humans and robots: On the
        interplay of real brains and artificial devices.
    """
    def __init__(self, nch, lr=1e-3, early_stop=10, device='cuda'):
        # Initialize the weigths.
        self.W = torch.zeros((nch,), device=device, requires_grad=True)
        # Optimizer used.
        self.optim = torch.optim.Adam([self.W], lr)
        self.erstp = EarlyStopping(early_stop)
        self.crite = nn.MSELoss()

    def train(self, Z, X, max_epoch=5000):
        bias = torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)
        direction = compute_direction(X)
        X_ = torch.cat((direction, bias), dim=-1)
        # Preferred direction of each neuron.
        self.B = [torch.lstsq(Z[:, [i]], X_)[0] for i in range(Z.shape[1])]
        self.B = torch.cat(self.B, dim=1)[:X.shape[1] + 1].T
        # Base line activation of each neuron.
        self.B0 = self.B[:, -1]
        # Normalized PD.
        self.B = self.B[:, :-1]
        self.B /= torch.norm(self.B, p=2, dim=1, keepdim=True)
        self.B[self.B.isnan()] = 0

        # Split the 0.2 of the training data as validation data.
        len_train = int(Z.shape[0] * 0.8)
        train_z, valid_z = Z[:len_train], Z[len_train:]
        train_x, valid_x = X[:len_train], X[len_train:]
        for _ in range(max_epoch):
            self.optim.zero_grad()
            out = self._apply_pd(train_z)
            t_loss = self.crite(out, train_x)
            t_loss.backward()
            self.optim.step()
            with torch.no_grad():
                out = self._apply_pd(valid_z)
                v_loss = self.crite(out, valid_x)
            self.erstp(v_loss, self.W, None, None)
            if self.erstp.early_stop:
                break

    def predict(self, Z):
        with torch.no_grad():
            out = self._apply_pd(Z)
        return out

    def _apply_pd(self, Z):
        return torch.matmul(self.W * (Z - self.B0), self.B)


if __name__ == '__main__':
    # Load the example data.
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(
        os.path.join(cur_path, '..', 'Test/fitting/pv.mat')
    )
    data = scio.loadmat(data_path)
    spikes = torch.tensor(data['spike'], dtype=torch.float).T
    motion = torch.tensor(data['move'], dtype=torch.float).T
    # Normalize the position
    a = torch.min(motion, dim=0)[0]
    b = torch.max(motion, dim=0)[0]
    motion = (motion - a) / (b - a)
    # Compute the velocity of motion data.
    speeds = torch.zeros_like(motion)
    speeds[:-1] = (motion[1:] - motion[:-1]) * 10
    # Split the dataset
    tr_neu, te_neu = spikes[:2250], spikes[2250:]
    tr_beh, te_beh = speeds[:2250], speeds[2250:]
    # Running the pv algorithm.
    pv = PopulationVector(tr_neu.shape[1], device='cpu')
    pv.train(tr_neu, tr_beh)
    # Prediction.
    pred_v = pv.predict(te_neu)
    # Compute the position.
    pred_p = motion[2250:].clone()
    pred_p[1:] = pred_p[:1] + torch.cumsum(pred_v[:-1], dim=0) / 10
    # Compute the correlation coefficient.
    rv = [np.diag(corrcoef(x, y)[0].cpu().numpy(), -1)
          for x, y in zip(pred_v.T, te_beh.T)]
    rp = [np.diag(corrcoef(x, y)[0].cpu().numpy(), -1)
          for x, y in zip(pred_p.T, motion[2250:].T)]
    print(f"Velocity r: {np.concatenate(rv).round(4)}")
    print(f"Position r: {np.concatenate(rp).round(4)}")

    plt.subplot(1, 2, 1)
    plt.plot(motion[2250:, 0], 'r')
    plt.plot(pred_p[:, 0], 'b')
    plt.subplot(1, 2, 2)
    plt.plot(motion[2250:, 1], 'r')
    plt.plot(pred_p[:, 1], 'b')
    plt.show()
