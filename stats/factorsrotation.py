import numpy as np
from scipy.linalg import null_space

from pybmi.utils.utils import check_params


class RotateFactors():
    """
    Rotation of FA or PCA loadings.

    Rotates the M-by-D loadings matrix A to maximize the varimax criterion,
    and returns the result in B. Rows of A and B correspond to factors and
    columns correspond to variables, e.g., the (i, j)th element of A is the
    coefficient for the j-th variable on the i-th factor. The matrix A
    usually contains principal component coefficients created with PCA,
    **or factor loadings estimated with FACTORAN (not implement yet)**.

    Parameters
    ----------
    method : str, optional
        The method indicates which algorithm will be performed
        to do rotation. Choices are:
            'orthomax'    - rotating A to maximize the varimax criterion.
                            Default method.
            'procrusters' - performs an oblique procrustes rotation of A
                            to the M-by-D target loadings matrix TARGET.
            'pattern'     - performs an oblique rotation of the loadings
                            matrix A to the M-by-D target pattern matrix
                            TARGET, and returns the result in B.
            'promax'      - rotates A to maximize the promax criterion,
                            equivalent to an oblique Procrustes rotation
                            with a target created by an orthomax rotation.
                            Use the four orthomax parameters to control
                            the orthomax rotation used internally by promax.
    gamma : scalar, optional
        GAMMA used when 'method'=='orthomax'. At that, B is the orthogonal
        rotation of A that maximizes sum(D*sum(B^4,1)-GAMMA*sum(B^2,1)^2).
        The default value of 1 for GAMMA corresponds to varimax rotation.
        Other possibilities include GAMMA = 0, M/2, and D*(M-1)/(D+M-2),
        corresponding to quartimax, equamax, and parsimax.
    normalize : bool, optional
        Flag used when 'method'=='orthomax', indicating whether the loadings
        matrix should be column-normalized for rotation. If true (the default),
        column of A are normalized prior to rotation to have unit Euclidean
        norm, and unnormalized after rotation. If false, the raw loadings are
        rotated and returned.
    reltol : scalar, optional
        Relative convergence tolerance in the iterative algorithm used to
        find T, only used when 'method'=='orthomax'. If it's None,
        default set it to sqrt(eps).
    maxit : int, optional
        Iteration limit in the iterative algorithm used to find T, only used
        when 'method'=='orthomax'. Default is 250.
    target : array_like, optional
        A M-by-D target loading matrix used to perform oblique rotation.
        When 'method' is 'pattern', TARGET defines the "restricted" elements
        of B, i.e., elements of B corresponding to zero elements of TARGET
        are constrained to have small magnitude, while elements of B
        corresponding to nonzero elements of TARGET are allowed to take on any
        magnitude. If the loading matrix A has M rows, then for orthogonal
        rotation, the Jth row of TARGET must contain at least M-J zeros; for
        oblique rotation, each row of target must contain at least M-1 zeros.
    cross_type : str, optional
        Type of rotation. If 'orthogonal', the rotation is orthogonal, and
        the factors remain uncorrelated. If 'oblique' (the default), the
        rotation is oblique, and the rotated factor may correlated. Used when
        'method' is 'procrustes' or 'pattern'.
    power : int, optional
        An addtional parameter for 'promax', specified the exponent for
        creating promax target matrix. Must be 1 or greater, default is 4.

    Returns
    -------
    B : array_like
        The rotated loading matrix of A.
    T : array_like
        The rotation matrix T used to create B, i.e., B = A*T.
        inv(T'*T) is the correlation matrix of the rotated factors.
        For orthogonal rotation, this is the identity matrix, while for
        oblique rotation, it has unit diagonal elements but nonzero
        off-diagonal elements.

    Examples
    --------
    >>> X = np.random.rand(100, 10)
    >>> pca = PCA(rows='complete', method='svd', centered=True)
    >>> pca.fit(X, n_components=3, variable_weights='variance')
    >>> m, d = pca.coeff.shape

    Default (normalized varimax) rotation of the components from PCA.
    >>> fr1 = RotateFactors()
    >>> B1, T = fr1.apply(pca.coeff)

    Equamax rotation of the components from a PCA.
    >>> fr2 = RotateFactors(gamma=m/2)
    >>> B2, T = fr2.apply(pca.coeff)

    Promax rotation of the components from a PCA.
    >>> fr3 = RotateFactors(method='promax', power=2)
    >>> B3, T = fr3.apply(pca.coeff)

    Pattern rotation of the components from a PCA.
    >>> target = [[1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 1, 1, 1, 1, 0]]
    >>> fr4 = RotateFactors(method='pattern')
    >>> B3, T = fr3.apply(pca.coeff, target)
    """

    def __init__(self, method='orthomax', gamma=1, normalize=True, reltol=None,
                 maxit=250, corss_type='oblique', power=4):
        self.method = method

        # The orthomax parameters.
        self.gamma = gamma
        self.normalize = normalize
        self.reltol = reltol
        self.maxit = maxit

        # The procrustes and pattern method parameters.
        self.corss_type = corss_type

        # The promax parameters.
        self.power = power

    def apply(self, A, target=None):
        if self.method == 'orthomax':
            return self._orthomax(A)
        elif self.method == 'procrustes':
            return self._procrustes(A, target)
        elif self.method == 'pattern':
            return self._pattern(A, target)
        elif self.method == 'promax':
            return self._promax(A)

    def _orthomax(self, A):
        """
        Orthogonal rotation of FA or PCA loadings.
        Default choose varimax rotation.
        """
        assert self.gamma >= 0, \
            f'Wrong coefficient for orthogonal rotation: {self.gamma}, '\
            'required larger than -1.'

        m, d = A.shape

        if self.reltol is None:
            self.reltol = np.sqrt(np.finfo(A.dtype).eps)

        # Normalize the factor loadings.
        if self.normalize:
            h = np.sqrt(np.sum(A ** 2, axis=0))
            A = A / h

        # Initialize the rotation matrix
        T = np.eye(m)
        B = np.dot(A.T, T).T

        converged = False
        if 0 <= self.gamma <= 1:
            # Use Lawley and Maxwell's fast version

            # Choose a random rotation matrix if identity rotation
            # makes an obviously bad start.
            criterion = d * B ** 3 - \
                self.gamma * np.dot(np.diag(np.sum(B ** 2, axis=1)), B)
            L, _, M = np.linalg.svd(np.dot(A, criterion.T))
            T = np.dot(L, M)
            if np.linalg.norm(T - np.eye(m)) < self.reltol:
                # Using identity as the initial rotation matrix, the first
                # iteration does not move the loadings enough to escape the
                # the convergence criteria. Therefore, pick an initial rotation
                # matrix at random.
                T, _ = np.linalg.qr(np.random.randn((m, m)))
                B = np.dot(A.T, T).T

            D = 0
            for k in range(self.maxit):
                Dold = D
                criterion = d * B ** 3 - \
                    self.gamma * np.dot(np.diag(np.sum(B ** 2, axis=1)), B)
                L, D, M = np.linalg.svd(np.dot(A, criterion.T))
                T = np.dot(L, M)
                D = np.sum(D)
                B = np.dot(A.T, T).T
                if abs(D - Dold) / D < self.reltol:
                    converged = True
                    break
        else:
            # Use a sequence of bivariate rotations
            for k in range(self.maxit):
                max_theta = 0.0
                for i in range(m - 1):
                    for j in range(i + 1, m):
                        u = B[i] * B[i] - B[j] * B[j]
                        v = 2 * B[i] * B[j]
                        usum, vsum = np.sum(u), np.sum(v)
                        numer = 2 * np.dot(u, v.T) - \
                            2 * self.gamma * usum * vsum / d
                        denom = np.dot(u, u.T) - np.dot(v, v.T) - \
                            self.gamma * (usum ** 2 - vsum ** 2) / d
                        theta = np.arctan2(numer, denom) / 4.0
                        max_theta = max(max_theta, np.abs(theta))
                        Tij = [[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]]
                        B[[i, j]] = np.dot(B[[i, j]].T, Tij).T
                        T[[i, j]] = np.dot(T[[i, j]].T, Tij).T
                if max_theta < self.reltol:
                    converged = True
                    T = T.T
                    break

        assert converged, \
            f"Rotate factors failed. Iteration limitation to {self.maxit}."

        # Unnormalize the rotated loadings
        if self.normalize:
            B = B * h
        return B, T

    def _procrustes(self, A, target):
        """
        Procrustes rotation of FA or PCA loadings.
        """
        m, d = A.shape
        assert target is not None, 'The procrustes method required TARGET!'
        assert target.shape == (m, d), \
            'The shape of input TARGET mismatch to the shape of A!'

        self.corss_type = check_params(
            self.corss_type, ['oblique', 'orthogonal'], 'cross_type'
        )

        if self.corss_type == 'orthogonal':
            # Orthogonal rotation to target
            L, _, M = np.linalg.svd(np.dot(target, A.T))
            T = np.dot(L, M).T
        elif self.corss_type == 'oblique':
            # Oblique rotation to target. LS, then normalize
            T = np.dot(target, np.linalg.pinv(A)).T
            T = np.dot(
                T, np.diag(np.sqrt(np.diag(
                    np.dot(np.linalg.pinv(np.dot(T.T, T)), np.eye(m)))))
            )

        B = np.dot(A.T, T).T
        return B, T

    def _pattern(self, A, target):
        """
        Rotation of FA or PCA loadings to a target pattern.

        In the context of Factor Analysis, Lawley and Maxwell describe a
        variation of this rotation where the rotation matrix is computed using
        a loadings matrix whose rows have been weighted by the inverse sqrt of
        the specific variances.  This can be done as

            W = diag(1./sqrt(Psi));
            [L,T] = PATTERN(W*L0); L = diag(sqrt(Psi))*L.

        or equivalently,

            [L,T] = PATTERN(W*L0); L = L0*T.
        """
        m, d = A.shape

        assert target is not None, 'The pattern method required TARGET!'
        assert target.shape == (m, d), \
            'The shape of input TARGET mismatch to the shape of A!'

        self.corss_type = check_params(
            self.corss_type, ['oblique', 'orthogonal'], 'cross_type'
        )

        if self.corss_type == 'orthogonal':
            assert ~np.any(np.sum(target == 0) < m - np.linspace(1, m, m)),\
                'Target checking failed.'\
                'The target should contain at least M-J zeros in Jth row.'

            T = np.eye(m)
            B = np.zeros_like(A)
            for j in range(m - 1):
                _, R = np.linalg.qr(A.T)
                A0 = A.copy()
                A0[:, target[j] == 0] = 0
                _, _, V = np.linalg.svd(
                    np.dot(A0.T, np.linalg.pinv(R)), full_matrices=False
                )
                U = np.dot(np.linalg.pinv(R), V.T[:, [0]])
                Tj = np.concatenate(
                    (U / np.linalg.norm(U), null_space(U.T)), axis=1
                )
                T[:, j:m] = np.dot(T[:, j:m], Tj)
                B[j:m] = np.dot(A.T, Tj).T
                A = B[j + 1:m]
        elif self.corss_type == 'oblique':
            assert ~np.any(np.sum(target == 0) < m - 1),\
                'Target checking failed.'\
                'The target should contain at least M-1 zeros in each row.'

            T = np.zeros((m, m))
            _, R = np.linalg.qr(A.T)
            for j in range(m):
                A0 = A.copy()
                A0[:, target[j] == 0] = 0
                _, _, V = np.linalg.svd(
                    np.dot(A0.T, np.linalg.pinv(R)), full_matrices=False
                )
                # 1st eigenvector of inv(A'*A)*(A0'*A0)
                T[:, [j]] = np.dot(np.linalg.pinv(R), V.T[:, [0]])
            # Normalize inv(T)
            T = np.dot(
                T,
                np.diag(np.sqrt(np.diag(
                    np.dot(np.linalg.pinv(np.dot(T.T, T)), np.eye(m)))))
            )
            B = np.dot(A.T, T).T

            # Make the largest element in each row of B positive.
            idx = np.argmax(abs(B), axis=1) + \
                np.linspace(0, (m - 1), m, dtype=np.int32) * d
            signer = np.diag(np.sign(B.reshape(-1)[idx]))
            B = np.dot(signer, B)
            T = np.dot(T, signer)
        return B, T

    def _promax(self, A):
        """
        Promax oblique rotation of FA or PCA loadings.
        """
        assert self.power >= 1, 'The promax method needs power >=1!'

        # Create target matrix from orthomax (defaults to varimax) solution
        B0 = self._orthomax(A)[0]
        # Keep it real, respect sign
        target = np.sign(B0) * abs(B0) ** self.power

        # Oblique rotation to target
        self.corss_type = 'oblique'
        return self._procrustes(A, target)
