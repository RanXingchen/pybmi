def CG(A, b, x0, P=None, max_iter=50, tol=5e-4, eps=1e-8):
    """
    Minimizes the linear system x^T*A*x - x^T*b using the conjugate
    gradient method.

    Parameters
    ----------
    A : callable
        An abstract linear operator implementing the product A*x.
        A must represent a hermitian, positive definite matrix.
    b : torch.Tensor
        The vector b.
    x0 : torch.Tensor
        An initial guess for x.
    P : callable, optional
        An abstract linear operator implementing the product of the
        preconditioner (for A) matrix with a vector. Default: None.
    max_iter : int, optional
        The maximum number of iteration. Default: 50.
    tol : float, optional
        Tolerance for convergence. Default: 1.2e-6.
    eps : float, optional
        term added to the denominator to improve numerical stability.
        Default: 1e-8
    """
    x = [x0]

    r = A(x[0]) - b
    y = P(r) if P is not None else r
    p = -y

    m = []

    numerator = r @ y
    for i in range(max_iter):
        # *Stop condition: Martens' Relative Progress (Section 20.4)
        m.append(0.5 * (r - b) @ x[i])

        k = max(10, int(i / 10))
        if i > k:
            stop = (m[i] - m[i - k]) / (m[i] + eps)
            if m[i] < 0 and stop < k * tol:
                break

        # CG iteration.
        Ap = A(p)
        alpha = numerator / ((p @ Ap) + eps)

        x.append(x[i] + alpha * p)
        r = r + alpha * Ap
        y = P(r) if P is not None else r

        numerator_new = r @ y
        beta = numerator_new / (numerator + eps)
        p = -y + beta * p

        # Update numerator
        numerator = numerator_new
    return x, m
