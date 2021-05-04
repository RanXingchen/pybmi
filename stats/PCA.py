import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pybmi.utils.utils import check_params
from .statistics import wnanmean, wnanvar


class PCA():
    """
    Principal Component Analysis (PCA) on raw data.

    The input data X should be a 2D matrix with the shape N by P, which N means
    the samples and P means the features of each sample.
    By default, PCA centers the data and uses the singular value decomposition
    algorithm.

    Parameters
    ----------
    rows : str, optional
        Action to take when the data matrix X contains NaN values.
        'complete' - The default action. Observations with NaN values
                     are removed before calculation.
        'pairwise' - If specified, PCA switches 'method' to 'eig'.
                     This option only applies when 'eig' method is used.
                     The (I, J) element of the covariance matrix is
                     computed using rows with no NaN values in columns I
                     or J of X. Please note that the resulting covariance
                     matrix may not be positive definite. In that case,
                     PCA terminates with an error message.
    method : str, optional
        Method that PCA uses to calculate the eigen values/vector. Choices are:
        'svd' - Singular Value Decomposition of X (default)
        'eig' - Eigenvalue Decomposition of the covariance matrix. It is faster
                than SVD when N is greater than P, but less accurate because
                the condition number of the covariance is the square of the
                condition number of X.
    centered : bool, optional
        Indicator for centering the columns of X. Default value is True,
        PCA centers X by subtracting off column means before computing SVD
        or EIG. If X contains NaN missing values, NANMEAN is used to find
        the mean with any data available. If set CENTERED to False, PCA
        does not center the data.

    Attributes
    ----------
    coeff : tensor, shape [num_components, P]
        The principal component coefficients for the N by P data matrix X.
        Each row of COEFF contains coefficients for one principal
        component. The rows are in descending order in terms of component
        variance (LATENT).
    latent : tensor, shape [P,]
        The principal component variances, i.e., the eigenvalues of the
        covariance matrix of X, in LATENT.
    explained : tensor, shape [P,]
        A vector containing the percentage of the total variance explained
        by each principal component.
    mu : tensor, shape [P,]
        The estimated mean of X in each observation, when 'centered' is set to
        true; and all zeros when set to false.
    sigma : tensor, shape [P,]
        When 'variable_weights' is set to 'variance', it is the estimated
        standard deviation of X in each observation; otherwise, it is
        equal to 1 / sqrt(variable_weights).

    Examples
    --------
    >>> X = torch.randn(5000, 96)
    >>> Y = torch.randn(1000, 96)
    >>> pca = PCA(rows='complete', method='svd', centered=True)
    >>> pca.fit(X, n_components=10, variable_weights='variance')
    >>> pca.print()
    >>> score = pca.apply(Y)
    >>> Y_reconstruct = pca.inverse_apply(score)
    """
    def __init__(self, rows='complete', method='svd', centered=True):
        # Validating input argument is legal or not.
        self.rows = check_params(rows, ['complete', 'pairwise'], 'rows')
        self.method = check_params(method, ['svd', 'eig'], 'method')
        # Check the method if it is paired with 'rows'
        if self.rows == 'pairwise' and self.method == 'svd':
            print('WARNING: No pair-wise SVD, switch the \'method\' to EIG')
            self.method = 'eig'

        self.centered = torch.tensor(centered, dtype=int)

    def fit(self, X, n_components=None, weights=None, variable_weights=None):
        """
        Computing the coeff and mu for later use.

        Parameters
        ----------
        X : tensor or ndarray
            The input 2D matrix with the shape N by P.
        n_components : int, optional
            The number of components desired, specified as a scalar integer K
            satisfying 0 < K <= P. When specified, PCA saves the first K
            rows of COEFF, otherwise, PCA saves all P rows of COEFF.
        weights : tensor or ndarray, optional
            Observation weights, a vector of length N containing all
            positive elements.
        variable_weights : tensor or ndarray or str, optional
            Weights of each feature. Two choices are possible for
            variable_weights, one is a vector of length P containing
            all positive elements; another is the string 'variance',
            for this the variable weights are the inverse of sample
            variance. If 'Centered' is set true at the same time,
            the data matrix X is centered and standardized. In this
            case, PCA returns the principal components based on the
            correlation matrix.
        """
        # Convert numpy ndarray to torch tensor
        if type(X) is np.ndarray:
            X = torch.from_numpy(X)
        if type(weights) is np.ndarray:
            weights = torch.from_numpy(weights)
        if type(variable_weights) is np.ndarray:
            variable_weights = torch.from_numpy(variable_weights)

        N, P = X.shape
        # Make sure the number of components desired are resonable.
        # If it's none, all components will be output
        if n_components is None:
            n_components = P
        assert 0 < n_components <= P, \
            f'Wrong number of principal components: {n_components},' \
            f'it should be in range [1, {P}]'

        # Validate weights and variable weights.
        if weights is None:
            # Set the weights to 1 when it is None.
            weights = torch.ones((N, 1), dtype=X.dtype, device=X.device)
        else:
            assert weights.numel() == N, \
                f'Wrong observation weights size: {weights.numel()},' \
                f'correct size should be {N}'
            # Make sure it is a column vector.
            weights = torch.reshape(weights, (N, 1))

        if variable_weights is None:
            # Set the variable weights to 1 when it is None.
            variable_weights = torch.ones(
                (1, P), dtype=X.dtype, device=X.device
            )
        elif isinstance(variable_weights, str):
            check_params(variable_weights, ['variance'], 'variable_weights')
            variable_weights = 1 / wnanvar(X, weights, bias=True, axis=0)
        else:
            assert variable_weights.numel() == P, \
                f'Wrong variable weights size: {variable_weights.numel()},' \
                f'correct size should be {P}'
        # Make sure it is a row vector.
        variable_weights = torch.reshape(variable_weights, (1, P))
        # Sigma, represents the std of X if 'variable_weights'=='variance',
        # otherwise, it's the reciprocal root of user-specified
        # variable_weights.
        self.sigma = 1 / torch.sqrt(variable_weights)

        assert torch.any(weights > 0) and torch.any(variable_weights > 0), \
            'Found none positive weights! Please check the weights vectors.'

        # Check the nan values in X
        nan_idx = torch.isnan(X)
        was_nan = torch.any(nan_idx, dim=1)   # Rows that contain NaN

        # If all X values are NaNs
        if torch.all(nan_idx):
            self.coeff = float('NaN')
            self.latent = float('NaN')
            self.explained = float('NaN')
            self.mu = float('NaN')
            return self
        # If X is scalar value
        if np.isscalar(X):
            self.coeff = 1
            self.latent = (not self.centered) * X ** 2
            self.explained = 100
            self.mu = self.centered * X
            return self

        if self.rows == 'complete':
            # Degrees of freedom (DOF) is M - 1 if centered and M if not
            # centered, where M is the numer of rows without any NaN element.
            D = max(0, N - self.centered - torch.sum(was_nan))
        else:
            # DOF is the maximum number of element pairs without NaNs.
            not_nan = (~nan_idx).type(torch.float)
            nan_cov = torch.matmul(not_nan.T, not_nan) * \
                ~torch.eye(P, dtype=torch.bool, device=not_nan.device)
            D = torch.max(nan_cov) - self.centered

        # Calculate each features mean value across all samples.
        self.mu = wnanmean(X, W=weights, axis=0) \
            if self.centered else torch.zeros(P, device=X.device)
        # Center the data
        _X = X - self.mu

        if self.method == 'eig':
            # Apply observation and variable weights
            _X *= torch.sqrt(weights) / self.sigma
            # Remove NaNs missing data and apply EIG.
            C = self._nancov(_X, D)
            self.latent, self.coeff = torch.eig(C, eigenvectors=True)
            self.latent = self.latent[:, 0]     # Only need the real part.
            # Make the COEFF same order as the method='svd'.
            self.coeff = self.coeff.T

            # Sort the eigen values in descend order.
            idx = self.latent.argsort(descending=True)
            self.coeff = self.coeff[idx] * self.sigma
            self.latent = self.latent[idx]

            # Check if eigvalues are all postive
            assert torch.any(self.latent > 0), \
                'Covariance of X is not positive semi-definite.'
        else:
            # Remove NaNs missing data
            _X = _X[~was_nan]
            # Apply observation and variable weights
            weights = weights[~was_nan]
            omega_sqrt = torch.sqrt(weights)
            _X *= omega_sqrt / self.sigma
            # Apply SVD. NOTE: this coeff is coeff.T in MATLAB.
            U, S, self.coeff = torch.linalg.svd(_X, full_matrices=False)
            U /= omega_sqrt
            self.coeff *= self.sigma
            self.latent = S ** 2 / D

        if D < P:
            # Ignore the the corresponding zero eigenvalues when D < P.
            self.coeff = self.coeff[:D]
            self.latent = self.latent[:D]
        # Calcuate the percentage of the total variance explained by each
        # principal component.
        self.explained = 100 * self.latent / sum(self.latent)
        # Output only the first k principal components
        if n_components < D:
            self.coeff = self.coeff[:n_components]

        # Enforce a sign convention on the coefficients -- the largest element
        # in each row will have a positive sign.
        max_idx = torch.argmax(abs(self.coeff), dim=1)
        d1, d2 = self.coeff.shape
        index = max_idx + torch.linspace(0, (d1 - 1) * d2, d1, dtype=int,
                                         device=X.device)
        rowsign = torch.sign(self.coeff.flatten()[index]).unsqueeze(1)
        self.coeff *= rowsign
        return self

    def apply(self, X):
        """
        Apply PCA based on the computed COEFF and MU and SIGMA.

        Note the COEFF=V*SIGMA, that's make X1=X/SIGMA^2, which
        is the matrix used to mutiply COEFF. Besides, when
        apply COEFF to the X1, the weights of X is ignored, if
        weights are defined in the FIT funciton, you can simply
        apply the weights by mutiply sqrt(weights) to the SCORE.

        Parameters
        ----------
        X : tensor or array_like
            The input matrix with the shape [N, P].

        Returns
        -------
        score : tensor
            The principal component score of X in the principal component
            space. Rows of SCORE correspond to observations, columns to
            components. The centered data can be reconstructed by SCORE*COEFF.
        """
        # Convert numpy ndarray to torch tensor
        if type(X) is np.ndarray:
            X = torch.from_numpy(X)
        # Center the data
        _X = X - self.mu if hasattr(self, 'mu') else X
        # Standarlize the data if necessary
        _X /= self.sigma ** 2 if hasattr(self, 'sigma') else _X
        # The mapped data in new space.
        score = torch.matmul(_X, self.coeff.T) \
            if hasattr(self, 'coeff') else _X
        return score

    def inverse_apply(self, score):
        """
        Inverse apply PCA based on the computed COEFF and MU.
        Reconstructing the data by SCORE*COEFF + MU.

        Parameters
        ----------
        score : tensor or array_like
            The principal component score of X in the principal component
            space. Rows of SCORE correspond to observations, columns to
            components.

        Returns
        -------
        X : tensor
            The original data before apply PCA. Rows of X correspond to
            observations, columns to features in original space.
        """
        # Convert numpy ndarray to torch tensor
        if type(score) is np.ndarray:
            score = torch.from_numpy(score)

        X = torch.matmul(score, self.coeff) \
            if hasattr(self, 'coeff') else score
        X += self.mu if hasattr(self, 'mu') else 0
        return X

    def print(self, plot=False):
        """
        Print the information of Fitted PCA in one table. Including the
        'Number of PCs', 'Eigenvalue', 'Percentage of Variance explained
        by PCs' and 'Cumulative percentage of Variance explained by PCs'.
        """
        if hasattr(self, 'latent'):
            explained_cumsum = np.cumsum(self.explained.cpu().numpy())
            num_pc = list(range(1, self.latent.shape[0] + 1))
            data = np.array(
                [num_pc,
                 self.latent.cpu().numpy(),
                 self.explained.cpu().numpy(),
                 explained_cumsum]
            )
            pca_info = pd.DataFrame(
                data.T,
                columns=[
                    'PC #', 'Eigenvalue', '% of Variance Exp', 'Cumulative %'
                ]
            )
            format_mapping = {
                'PC #': '{:.0f}',
                'Eigenvalue': '{:.3f}',
                '% of Variance Exp': '{:.3f}%',
                'Cumulative %': '{:.3f}%'
            }
            table = pca_info.copy()
            for key, value in format_mapping.items():
                table[key] = pca_info[key].apply(value.format)
            print(table.to_string())

            # plot
            if plot:
                plt.figure(figsize=(6, 6))
                plt.bar(
                    num_pc,
                    explained_cumsum,
                    width=0.5,
                    color='cyan',
                    alpha=0.2,
                    label='Cumulative %'
                )
                plt.plot(
                    num_pc, self.explained, label='% of variance explained'
                )
                plt.plot(num_pc, self.explained, 'ro', label='_nolegend_')

                plt.xlabel('Pricipal Components')
                plt.ylabel('Percentage of variance explained')
                plt.legend(loc='upper left')
                plt.title(
                    'Percentage of variance explained by PCs', fontsize=16
                )
                plt.show()

    def _nancov(self, X, D):
        """
        Computing covariance of X in the situation that X contain NaNs.

        Parameters
        ----------
        X : tensor
            The input matrix that will compute covariance.
        D : int
            Degree of fredom of the samples.

        Returns
        -------
        C : tensor
            The covariance of X, which removed the NaNs from the X,
            if rows='complete'; otherwise, compute paried wise covariance
            in columns i and j, which both columns are not contain NaNs.
        """
        N, P = X.shape
        nan_idx = torch.isnan(X)

        if self.rows == 'complete':
            _X = X[~torch.any(nan_idx, dim=1)]
            C = torch.matmul(_X.T, _X) / D
        elif self.rows == 'pairwise':
            C = torch.zeros((P, P), dtype=X.dtype, device=X.device)
            # Compute the pair-wise covariance of X. Note that we only compute
            # the lower triangular of covariance under the for loop to reduce
            # the computating cost. The upper triangular can get easily by
            # transpose the lower triangular.
            for i in range(P):
                for j in range(i + 1):
                    # Find i and j columns, both have no NaNs rows.
                    non_nan_rows = ~torch.any(nan_idx[:, [i, j]], dim=1)
                    # The DOF of the two columns.
                    denom = max(0, torch.sum(non_nan_rows) - self.centered)
                    # Covariance of i and j column
                    C[i, j] = torch.matmul(X[non_nan_rows, i].T,
                                           X[non_nan_rows, j])
                    C[i, j] /= denom
            C += torch.tril(C, -1).T
        return C
