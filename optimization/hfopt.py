import torch

from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.nn.utils.convert_parameters import parameters_to_vector
from .conjgrad import CG


class HessianFree(torch.optim.Optimizer):
    """
    Implements the Hessian-free algorithm presented in
    `Training Deep and Recurrent Networks with Hessian-Free Optimization`.
    https://doi.org/10.1007/978-3-642-35289-8_27

    Parameters
    ----------
    params :iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, optional
        Learning rate, must not greater than 1. Default: 1
    damping : float, optional
        Initial value of the Tikhonov damping coefficient. Default: 0.5.
    mu : float, optional
        Parameters used for structural damping.
    zeta : float, optional
        Decay of the previous result of computing delta with CG for the
        initialization of the next CG iteration (See details in Section 20.10).
        Default: 0.95
    cg_max_iter : int, optional
        Maximum number of Conjugate-Gradient iterations. Default: 50.
    cg_tol : float, optional
        The tolerance of the CG iteration. Default: 5e-4.
    eps :float, optional
        Term added to the denominator to improve numerical stability.
        Default: 1e-8.
    verbose : bool, optional
        Print statements (debugging). Default: False.
    """
    def __init__(self, params, lr=1, damping=0.5, mu=0.01, zeta=0.95,
                 cg_max_iter=30, cg_tol=5e-4, eps=1e-8, verbose=False):
        if not (0.0 < lr <= 1):
            raise ValueError("Invalid lr: {}".format(lr))

        if not (0.0 < damping <= 1):
            raise ValueError("Invalid damping: {}".format(damping))

        if not cg_max_iter > 0:
            raise ValueError("Invalid cg_max_iter: {}".format(cg_max_iter))

        defaults = dict(alpha=lr, damping=damping, mu=mu, zeta=zeta,
                        cg_max_iter=cg_max_iter, cg_tol=cg_tol,
                        eps=eps, verbose=verbose)
        super(HessianFree, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("HessianFree doesn't support per-parameter "
                             "options (parameter groups)")

    def step(self, closure, P=None, Gv=None):
        """
        Performs a single optimization step.
        Parameters
        ----------
        closure : callable
            A closure that re-evaluates the model and returns a tuple of
            the loss and the output.
        P : Tensor, optional
            The preconditioner of diag(GGN).
        Gv : callable, optional
            Provide the callable Gv function instead of auto differential.
        """
        assert len(self.param_groups) == 1
        group = self.param_groups[0]

        # Compute the loss before optimization.
        if Gv is None:
            with torch.backends.cudnn.flags(enabled=False):
                loss, output = closure()
            # Compute the Hessian of loss w.r.t output
            dLdz = torch.autograd.grad(loss, output, create_graph=True)
            H = torch.autograd.grad(
                dLdz, output, torch.ones_like(output), retain_graph=True
            )
            H = parameters_to_vector(H)

            # The jacobian with vector.
            _v = torch.zeros_like(output, requires_grad=True)
            _Jv = torch.autograd.grad(output, self.param_groups[0]['params'],
                                      grad_outputs=_v, create_graph=True)
            _Jv = parameters_to_vector(_Jv)
        else:
            loss, output = closure()

        # Gather current parameters and gradients to a vector
        theta = parameters_to_vector(group['params'])
        b = parameters_to_vector(
            torch.autograd.grad(loss, group['params'], retain_graph=True)
        ).detach()

        # Just one state needed.
        state = self.state[group['params'][0]]
        # Lazy state initialization
        if len(state) == 0:
            # Initial guesses x0 for x of CG
            state['x0'] = torch.zeros_like(theta)

        # Define linear operator: Generalized Gauss-Newton vector product
        def A(x):
            if Gv is None:
                return self._Gv(_Jv, _v, H, output, x, group['damping'])
            else:
                return Gv(x, group['mu'], group['damping'])

        if P is not None:
            # Preconditioner recipe (Section 20.13)
            m = (P + group['damping']) ** 0.75

            def M(x):
                return m * x
        else:
            M = None

        # Initializing Conjugate-Gradient (Section 20.10)
        x0 = group['zeta'] * state['x0']
        # Conjugate-Gradient
        deltas, ms = CG(A, b.neg(), x0, P=M, max_iter=group['cg_max_iter'],
                        tol=group['cg_tol'], eps=group['eps'])

        # Update parameters
        delta = state['x0'] = deltas[-1]
        vector_to_parameters(theta + delta, group['params'])
        m = ms[-1]

        # Evaluate the loss after CG.
        with torch.no_grad():
            loss_now = closure()[0]
        if group['verbose']:
            print("Loss before CG: {}".format(float(loss)))
            print("Loss before BT: {}".format(float(loss_now)))

        # Conjugate-Gradient backtracking (Section 20.8.7)
        for (d, _m) in zip(reversed(deltas[:-1][::2]), reversed(ms[:-1][::2])):
            vector_to_parameters(theta + d, group['params'])
            with torch.no_grad():
                loss_prev = closure()[0]
            if float(loss_prev) > float(loss_now):
                break
            delta = d
            m = _m
            loss_now = loss_prev

        if group['verbose']:
            print("Loss after BT:  {}".format(float(loss_now)))

        # The Levenberg-Marquardt Heuristic (Section 20.8.5)
        rho = (float(loss_now) - float(loss)) / m if m != 0 else 1

        if rho < 0.25:
            group['damping'] *= 3 / 2
        elif rho > 0.75:
            group['damping'] *= 2 / 3
        if rho < 0:
            group['x0'] = 0

        # Line Searching (Section 20.8.8)
        alpha = group['alpha']
        beta = 0.8
        c = 1e-2
        min_improve = min(c * alpha * torch.dot(b, delta), 0)

        for _ in range(60):
            if float(loss_now) <= float(loss) + min_improve:
                break

            alpha *= beta
            vector_to_parameters(theta + alpha * delta, group['params'])
            with torch.no_grad():
                loss_now = closure()[0]
        else:  # No good update found
            alpha = 0.0
            loss_now = loss

        # Update the parameters (this time fo real)
        vector_to_parameters(theta + alpha * delta, group['params'])

        if group['verbose']:
            print("Loss after LS:  {0} (lr: {1:.3f})".format(
                float(loss_now), alpha))
            print("Tikhonov damping: {0:.3f} (reduction ratio: {1:.3f})".
                  format(group['damping'], rho), end='\n\n')

        return loss_now

    def _Gv(self, _Jv, _v, H, z, vec, damping):
        """
        Computes the generalized Gauss-Newton vector product.
        """
        Jv = torch.autograd.grad(_Jv, _v, grad_outputs=vec, retain_graph=True)

        HJv = H * Jv[0].reshape(-1)

        JHJv = torch.autograd.grad(z.reshape(-1),
                                   self.param_groups[0]['params'],
                                   grad_outputs=HJv,
                                   retain_graph=True)

        # Tikhonov damping (Section 20.8.1)
        return parameters_to_vector(JHJv).detach() + damping * vec
