import numpy as np
import scipy
import torch

from cmr.methods.abstract_estimation_method import AbstractEstimationMethod
from cmr.utils.sieve_basis import CardinalPolynomialSplineBasis
from cmr.utils.torch_utils import np_to_tensor


class SMDIdentity(AbstractEstimationMethod):
    def __init__(self, model, moment_function, optimization=None, verbose=0, **kwargs):
        super().__init__(
            model=model,
            moment_function=moment_function,
            optimization=optimization,
            verbose=verbose,
            **kwargs,
        )
        self.basis = CardinalPolynomialSplineBasis(degree=3, num_knots=3)

    def _train_internal(self, x, z, x_val, z_val, debugging):
        n_sample = x[0].shape[0]
        # Basis expansion on CPU
        z_np = z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z
        f_z = self.basis.fit_transform(z_np)

        # Calculate omega_inv on CPU
        f_z_f_z = (f_z.T @ f_z) / n_sample
        omega_inv = scipy.linalg.inv(f_z_f_z)

        # Move to tensors on correct device
        f_z = self._to_tensor_and_device(f_z)
        omega_inv = self._to_tensor_and_device(omega_inv)

        self._fit_theta(x, f_z, omega_inv)

    def _fit_theta(self, x, f_z, omega_inv):
        optimizer = torch.optim.LBFGS(
            self.model.parameters(), line_search_fn="strong_wolfe"
        )

        # Pre-process f_z for matmul
        f_z_t = f_z.t()  # (J, n)

        def closure():
            optimizer.zero_grad()
            psi = self.moment_function(x)  # (n, k)

            # Use torch.matmul which handles broadcasting
            # f_z_t is (J, n)
            # psi is (n, k)
            # Result (J, k)
            mom = torch.matmul(f_z_t, psi) / x[0].shape[0]

            # Quadratic form: mom.T @ omega_inv @ mom
            # omega_inv is (J, J)

            if mom.shape[1] == 1:
                # (1, J) @ (J, J) @ (J, 1) -> (1, 1)
                loss = torch.matmul(mom.t(), torch.matmul(omega_inv, mom))
            else:
                loss = 0
                for k in range(mom.shape[1]):
                    m_k = mom[:, k]
                    loss += m_k @ omega_inv @ m_k

            loss.backward()
            return loss

        optimizer.step(closure)
