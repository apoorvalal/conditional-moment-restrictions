import torch
from cmr.methods.abstract_estimation_method import AbstractEstimationMethod
from cmr.default_config import mmr_kwargs


class MMR(AbstractEstimationMethod):
    def __init__(self, model, moment_function, verbose=0, **kwargs):
        # Merge default kwargs with user kwargs carefully to preserve 'optimization' object
        # which might be in kwargs
        defaults = mmr_kwargs.copy()
        defaults.update(kwargs)

        super().__init__(
            model=model, moment_function=moment_function, verbose=verbose, **defaults
        )

    def _train_internal(self, x, z, x_val, z_val, debugging):
        n_sample = x[0].shape[0]
        self._set_kernel_z(z, z_val)

        optimizer = torch.optim.LBFGS(
            self.model.parameters(), line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            psi = self.moment_function(x)
            # kernel_z is already on self.device from _set_kernel_z
            loss = torch.einsum("ir, ij, jr -> ", psi, self.kernel_z, psi) / (
                n_sample**2
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        if self.verbose:
            print(f"MMR Loss: {closure().item()}")


if __name__ == "__main__":
    from experiments.tests import test_cmr_estimator

    test_cmr_estimator(estimation_method="MMR", n_runs=5)
