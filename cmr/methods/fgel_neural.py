import torch

from cmr.methods.generalized_el import GeneralizedEL
from cmr.utils.torch_utils import ModularMLPModel
from cmr.config import FGELConfig, NetworkConfig, OptimizationConfig


class NeuralFGEL(GeneralizedEL):
    def __init__(
        self,
        model,
        moment_function,
        fgel_config: FGELConfig = None,
        optimization: OptimizationConfig = None,
        verbose=0,
        **kwargs,
    ):
        self.fgel_config = fgel_config or FGELConfig()

        # Map legacy kwargs
        if "divergence" in kwargs:
            self.fgel_config.divergence = kwargs["divergence"]
        if "reg_param" in kwargs:
            self.fgel_config.regularization = kwargs["reg_param"]

        super().__init__(
            model=model,
            moment_function=moment_function,
            optimization=optimization,
            verbose=verbose,
            divergence=self.fgel_config.divergence,
            reg_param=self.fgel_config.regularization,
            **kwargs,
        )

        self.dual_func_network_kwargs_custom = kwargs.get(
            "dual_func_network_kwargs", None
        )

    def _init_dual_params(self):
        dual_func_network_kwargs = self._update_default_dual_func_network_kwargs(
            self.dual_func_network_kwargs_custom
        )
        # Update input dim
        dual_func_network_kwargs["input_dim"] = self.dim_z
        dual_func_network_kwargs["num_out"] = self.dim_psi

        self.dual_moment_func = ModularMLPModel(**dual_func_network_kwargs)
        self.dual_moment_func = self.dual_moment_func.to(self.device)
        self.all_dual_params = list(self.dual_moment_func.parameters())

    def _update_default_dual_func_network_kwargs(self, dual_func_network_kwargs):
        dual_func_network_kwargs_default = {
            "layer_widths": [50, 20],
            "activation": torch.nn.LeakyReLU,
        }
        if dual_func_network_kwargs is not None:
            dual_func_network_kwargs_default.update(dual_func_network_kwargs)
        return dual_func_network_kwargs_default

    def _eval_dual_moment_func(self, z):
        return self.dual_moment_func(z)

    def _objective(self, x, z, *args, **kwargs):
        objective, _ = super()._objective(x, z, *args)
        if self.reg_param > 0:
            l_reg = self.reg_param * torch.mean(self.dual_moment_func(z) ** 2)
        else:
            l_reg = 0
        return objective, -objective + l_reg


if __name__ == "__main__":
    from experiments.tests import test_cmr_estimator

    pass
