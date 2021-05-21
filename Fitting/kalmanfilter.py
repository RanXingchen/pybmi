import numpy as np
import torch
import pickle


class KalmanFilter():
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
    def __init__(self, Z, X, eps=None):
        assert X.shape[0] == Z.shape[0], \
            'The state and measurement should have same number of time points!'

        # Convert numpy ndarray to torch tensor
        if type(Z) is np.ndarray:
            Z = torch.from_numpy(Z)
        if type(X) is np.ndarray:
            X = torch.from_numpy(X)

        t, m = X.shape
        m_eye = torch.eye(m, device=X.device, dtype=X.dtype)

        if eps is None:
            eps = np.sqrt(torch.finfo(X.dtype).eps) * m_eye

        X1, X2 = X[:-1], X[1:]
        # State Transition Model
        self.A = torch.matmul(
            X2.T.matmul(X1),
            torch.pinverse(X1.T.matmul(X1) + eps)
        )
        # Measurement Model
        self.H = torch.matmul(
            Z.T.matmul(X),
            torch.pinverse(X.T.matmul(X) + eps)
        )
        # Process Noise
        e_sta = X2.T - torch.matmul(self.A, X1.T)
        self.W = torch.matmul(e_sta, e_sta.T) / (t - 1)
        # Measurement Noise
        e_mea = Z.T - torch.matmul(self.H, X.T)
        self.Q = torch.matmul(e_mea, e_mea.T) / t
        # State Covariance
        self.P = m_eye

    def predict(self, Z, x0):
        """
        Predicts the measurement, state, and state estimation error covariance.
        The internal state and covariance of Kalman filter are overwritten
        by the prediction results.

        Parameters
        ----------
        Z : ndarray or tensor, shape (1, N) or (N,) or (T, N)
            The measurement matrix which is used to predict the
            corresponding state, where N means the feature dimensions
            of the measurement. The time step of Z can be 1 or T, if T,
            the predicted state also have T time step.
        x0 : ndarray or tensor, shape (1, M) or (M,)
            The initial state vector used to predict the next time step.

        Returns
        -------
        X : ndarray or tensor, shape (1, M) or (T, M)
            The prediction by the observation model and measurement model.
            It have same length with Z.
        """
        # Make sure Z and X0 have 2 dimensions
        if Z.ndim == 1:
            Z = Z[np.newaxis, :]
        if x0.ndim == 1:
            x0 = x0[np.newaxis, :]

        # Convert numpy ndarray to torch tensor
        isnumpy = False
        if type(Z) is np.ndarray:
            Z = torch.from_numpy(Z)
            isnumpy = True
        if type(x0) is np.ndarray:
            x0 = torch.from_numpy(x0)
            isnumpy = True

        X = []

        # Use one time step of X0 as the initialize state.
        xt = x0[:1]
        Pt = self.P
        for _, z in enumerate(Z):
            zt = torch.reshape(z, (1, -1))
            # Priori estimate of Xt and Pt
            xt = torch.matmul(self.A, xt.T).T
            Pt = torch.matmul(torch.matmul(self.A, Pt), self.A.T) + self.W
            # Calculate Kalman gain of current step
            residual = torch.matmul(self.H.matmul(Pt), self.H.T) + self.Q
            Kt = torch.matmul(Pt.matmul(self.H.T), torch.pinverse(residual))
            # Update the estimation by measurement.
            # * Do not use inplace operation.
            xt = xt + torch.matmul(Kt, zt.T - torch.matmul(self.H, xt.T)).T
            Pt = Pt - torch.matmul(torch.matmul(Kt, self.H), Pt)

            # Push current time step predcting result to X
            X.append(xt)

        X = torch.cat(X, dim=0)
        # Update the state estimation error covariance.
        self.P = Pt
        # convert back to numpy if necessary.
        if isnumpy:
            X = X.numpy()
        return X

    def save(self, path):
        """
        Save the Kalman filter parameters to the specified path.
        The saving parameters including: A, H, P, Q, W

        Parameters
        ----------
        path : str
            The path of the parameters will be saved.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.A, f)
            pickle.dump(self.H, f)
            pickle.dump(self.P, f)
            pickle.dump(self.Q, f)
            pickle.dump(self.W, f)

    def load(self, path):
        """
        Load the Kalman filter parameters to the specified path.
        The loading parameters including: A, H, P, Q, W

        Parameters
        ----------
        path : str
            The path of the parameters will be loaded.
        """
        with open(path, 'rb') as f:
            self.A = pickle.load(f)
            self.H = pickle.load(f)
            self.P = pickle.load(f)
            self.Q = pickle.load(f)
            self.W = pickle.load(f)
