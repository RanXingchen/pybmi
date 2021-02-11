import numpy as np


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
    Z : ndarray, shape (T, N)
        The measurement matrix which is used to calculate the
        models, where N means the feature dimensions of the
        measurement. In BCI application, it should be the X's
        corresponding neural data.
    X : ndarray, shape (T, M)
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
    A : ndarray, shape (M, M)
        Model describing state transition between time steps.
        Specify the transition of state between times as an M-by-M
        matrix. This attribute cannot be changed once the Kalman
        filter is trained.
    H : ndarray, shape (N, M)
        Model describing state to measurement transformation.
        Specify the transition from state to measurement as an N-by-M
        matrix. This attribute cannot be changed once the Kalman
        filter is trained.
    P : ndarray, shape (M, M)
        State estimation error covariance. Specify the covariance
        of the state estimation error as an M-by-M diagonal matrix.
    Q : ndarray, shape (M, M)
        Process noise covariance. Specify the covariance of process
        noise as an M-by-M matrix.
    R : ndarray, shape (N, N)
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

        t, m = X.shape

        X1, X2 = X[:-1], X[1:]
        # State Transition Model
        self.A = np.dot(
            np.dot(X2.T, X1),
            np.linalg.pinv(np.dot(X1.T, X1) + eps * np.eye(m))
        )
        # Measurement Model
        self.H = np.dot(
            np.dot(Z.T, X),
            np.linalg.pinv(np.dot(X.T, X) + eps * np.eye(m))
        )
        # Process Noise
        e_sta = X2.T - np.dot(self.A, X1.T)
        self.W = np.dot(e_sta, e_sta.T) / (t - 1)
        # Measurement Noise
        e_mea = Z.T - np.dot(self.H, X.T)
        self.Q = np.dot(e_mea, e_mea.T) / t
        # State Covariance
        self.P = np.eye(m)

    def predict(self, Z, x0):
        """
        Predicts the measurement, state, and state estimation error covariance.
        The internal state and covariance of Kalman filter are overwritten
        by the prediction results.

        Parameters
        ----------
        Z : ndarray, shape (1, N) or (N,) or (T, N)
            The measurement matrix which is used to predict the
            corresponding state, where N means the feature dimensions
            of the measurement. The time step of Z can be 1 or T, if T,
            the predicted state also have T time step.
        x0 : ndarray, shape (1, M) or (M,)
            The initial state vector used to predict the next time step.

        Returns
        -------
        X : ndarray, shape (1, M) or (T, M)
            The prediction by the observation model and measurement model.
            It have same length with Z.
        """
        # Make sure Z and X0 have 2 dimensions
        if len(Z.shape) == 1:
            Z = Z[np.newaxis, :]
        if len(x0.shape) == 1:
            x0 = x0[np.newaxis, :]

        X = []

        # Use one time step of X0 as the initialize state.
        xt = x0[:1]
        Pt = self.P
        for _, z in enumerate(Z):
            zt = np.reshape(z, (1, -1))
            # Priori estimate of Xt and Pt
            xt = np.dot(self.A, xt.T).T
            Pt = np.dot(np.dot(self.A, Pt), self.A.T) + self.W
            # Calculate Kalman gain of current step
            residual_cov = np.dot(np.dot(self.H, Pt), self.H.T) + self.Q
            Kt = np.dot(np.dot(Pt, self.H.T), np.linalg.pinv(residual_cov))
            # Update the estimation by measurement.
            xt += np.dot(Kt, zt.T - np.dot(self.H, xt.T)).T
            Pt -= np.dot(np.dot(Kt, self.H), Pt)

            # Push current time step predcting result to X
            X.append(xt)

        X = np.concatenate(X, axis=0)
        # Update the state estimation error covariance.
        self.P = Pt
        return X
