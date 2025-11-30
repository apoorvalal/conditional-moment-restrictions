import cvxpy as cvx
import numpy as np
import torch
import rff
from sklearn.neighbors import KernelDensity

from cmr.utils.rkhs_utils import get_rbf_kernel, calc_sq_dist
from cmr.utils.torch_utils import Parameter, np_to_tensor, tensor_to_np
from cmr.methods.generalized_el import GeneralizedEL
from cmr.config import KMMConfig, NetworkConfig, OptimizationConfig, KernelConfig

cvx_solver = cvx.MOSEK


class KMMNeural(GeneralizedEL):
    """
    Kernel Method of Moments with Neural Network dual function.
    """

    def __init__(
        self,
        model,
        moment_function,
        kmm_config: KMMConfig = None,
        dual_network: NetworkConfig = None,
        optimization: OptimizationConfig = None,
        verbose=0,
        **kwargs,
    ):
        # Config defaults
        self.kmm_config = kmm_config or KMMConfig()
        self.dual_network = dual_network or NetworkConfig()

        # Legacy mapping for kwargs
        if "n_random_features" in kwargs:
            self.kmm_config.n_random_features = kwargs["n_random_features"]
        if "n_reference_samples" in kwargs:
            self.kmm_config.n_reference_samples = kwargs["n_reference_samples"]
        if "entropy_reg_param" in kwargs:
            self.kmm_config.entropy_regularization = kwargs["entropy_reg_param"]

        super().__init__(
            model=model,
            moment_function=moment_function,
            optimization=optimization,
            verbose=verbose,
            divergence="kl",
            reg_param=self.dual_network.regularization,
            **kwargs,
        )

        self.entropy_reg_param = self.kmm_config.entropy_regularization
        self.rkhs_reg_param = self.kmm_config.rkhs_regularization
        self.n_rff = self.kmm_config.n_random_features
        self.n_reference_samples = self.kmm_config.n_reference_samples
        self.kde_bw = self.kmm_config.kde_bandwidth
        self.rkhs_func_z_dependent = self.kmm_config.reference_depends_on_z
        self.t_as_instrument = self.kmm_config.t_as_instrument

        # Kernel args for X space (usually defaults are fine)
        self.kernel_x_kwargs = kwargs.get("kernel_x_kwargs", {})

        # Do not use self.dim_psi here as it is not initialized yet
        self.dual_func_network_kwargs = self._update_default_dual_func_network_kwargs(
            kwargs.get("dual_func_network_kwargs", None)
        )

        # Override dual network config from NetworkConfig if provided
        if dual_network:
            self.dual_func_network_kwargs.update(
                {
                    "layer_widths": dual_network.hidden_layers,
                    # 'activation': dual_network.activation
                }
            )

    def _update_default_dual_func_network_kwargs(self, dual_func_network_kwargs):
        dual_func_network_kwargs_default = {
            # Input/Output dims will be set in _init_dual_params
            "layer_widths": [50, 20],
            "activation": torch.nn.LeakyReLU,
        }
        if dual_func_network_kwargs is not None:
            dual_func_network_kwargs_default.update(dual_func_network_kwargs)
        return dual_func_network_kwargs_default

    def _init_dual_params(self):
        from cmr.utils.torch_utils import ModularMLPModel

        # Re-update input_dim and num_out now that we know dimensions
        self.dual_func_network_kwargs["input_dim"] = self.dim_z
        self.dual_func_network_kwargs["num_out"] = self.dim_psi

        self.dual_moment_func = ModularMLPModel(**self.dual_func_network_kwargs)
        self.dual_moment_func = self.dual_moment_func.to(self.device)

        self.dual_normalization = Parameter(shape=(1, 1)).to(self.device)

        if self.n_rff:
            self.rkhs_func = Parameter(shape=(self.n_rff, 1)).to(self.device)
        else:
            # Full kernel version
            self.rkhs_func = None

        self.all_dual_params = list(self.dual_moment_func.parameters()) + list(
            self.dual_normalization.parameters()
        )
        if self.rkhs_func is not None:
            self.all_dual_params += list(self.rkhs_func.parameters())

    def _init_rff(self, x, z):
        x_np, z_np = tensor_to_np(x), tensor_to_np(z)
        # Calculate sigmas on CPU
        sigma_t = np.sqrt(0.5 * np.median(calc_sq_dist(x_np[0], x_np[0], numpy=True)))
        sigma_y = np.sqrt(0.5 * np.median(calc_sq_dist(x_np[1], x_np[1], numpy=True)))
        sigma_z = (
            np.sqrt(0.5 * np.median(calc_sq_dist(z_np, z_np, numpy=True)))
            if z_np is not None
            else 1.0
        )

        self._eval_rff_t = rff.layers.GaussianEncoding(
            sigma=sigma_t, input_size=x[0].shape[1], encoded_size=self.n_rff // 2
        ).to(self.device)
        self._eval_rff_y = rff.layers.GaussianEncoding(
            sigma=sigma_y, input_size=x[1].shape[1], encoded_size=self.n_rff // 2
        ).to(self.device)

        if z is not None and self.rkhs_func_z_dependent:
            self._eval_rff_z = rff.layers.GaussianEncoding(
                sigma=sigma_z, input_size=z.shape[1], encoded_size=self.n_rff // 2
            ).to(self.device)
        else:
            self._eval_rff_z = lambda arg: 1.0

    def eval_rff(self, x, z):
        rff_t = self._eval_rff_t(x[0])
        rff_y = self._eval_rff_y(x[1])
        rff_z = self._eval_rff_z(z)
        return rff_t * rff_y * rff_z

    def eval_rkhs_func(self, x, z):
        if self.n_rff:
            return torch.einsum(
                "ij, ki -> kj", self.rkhs_func.params, self.eval_rff(x, z)
            )
        else:
            return torch.einsum("ij, ki -> kj", self.rkhs_func.params, self.kernel_x)

    def rkhs_norm_sq(self):
        if self.n_rff:
            return torch.einsum(
                "ij, ij ->", self.rkhs_func.params, self.rkhs_func.params
            )
        else:
            return torch.einsum(
                "ir, ij, jr ->",
                self.rkhs_func.params,
                self.kernel_x,
                self.rkhs_func.params,
            )

    def _eval_dual_moment_func(self, z):
        return self.dual_moment_func(z)

    def init_estimator(self, x_tensor, z_tensor):
        self._init_reference_distribution(x_tensor, z_tensor)
        if self.n_rff:
            self._init_rff(x_tensor, z_tensor)
        else:
            # Full kernel support - strict limitations
            print("Using full batch kernel Gram matrix version ...")
            assert self.batch_size is None, (
                "Cannot use mini-batch optimization with representer theorem version."
            )
            assert self.n_reference_samples is None or self.n_reference_samples == 0, (
                "Cannot use reference samples with representer theorem version."
            )
            self._set_kernel_x(x_tensor, z_tensor)
            # Init rkhs_func here for full kernel
            self.rkhs_func = Parameter(shape=(self.kernel_x.shape[0], 1)).to(
                self.device
            )
            self.all_dual_params += list(self.rkhs_func.parameters())

        super().init_estimator(x_tensor, z_tensor)

    def _set_kernel_x(self, x, z):
        x_np, z_np = tensor_to_np(x), tensor_to_np(z)
        kernel_t, _ = get_rbf_kernel(x_np[0], x_np[0], **self.kernel_x_kwargs)
        kernel_y, _ = get_rbf_kernel(x_np[1], x_np[1], **self.kernel_x_kwargs)
        kernel_z, _ = (
            get_rbf_kernel(z_np, z_np, **self.kernel_z_kwargs)
            if z_np is not None
            else (1.0, 1.0)
        )
        self.kernel_x = torch.Tensor(kernel_t * kernel_y * kernel_z).to(self.device)

    def _init_reference_distribution(self, x, z, sample_weight=None):
        x_np, z_np = tensor_to_np(x), tensor_to_np(z)
        if z_np is not None:
            xz = np.concatenate([x_np[0], x_np[1], z_np], axis=1)
        else:
            xz = np.concatenate(x_np, axis=1)
        self.kde = KernelDensity(bandwidth=self.kde_bw)
        self.kde.fit(xz, sample_weight=sample_weight)

    def append_reference_samples(self, x, z):
        if self.n_reference_samples is None or self.n_reference_samples == 0:
            return x, z

        # Sample on CPU
        xz_sampled = self.kde.sample(n_samples=self.n_reference_samples)

        # Convert to tensor
        t_sampled = np_to_tensor(xz_sampled[:, : self.dim_t])
        y_sampled = np_to_tensor(xz_sampled[:, self.dim_t : (self.dim_t + self.dim_y)])

        if self.t_as_instrument:
            z_sampled = t_sampled
        else:
            z_sampled = np_to_tensor(xz_sampled[:, (self.dim_t + self.dim_y) :])

        # Move to device
        t_sampled, y_sampled, z_sampled = (
            t_sampled.to(self.device),
            y_sampled.to(self.device),
            z_sampled.to(self.device),
        )

        t_total = torch.concat((x[0], t_sampled), dim=0)
        y_total = torch.concat((x[1], y_sampled), dim=0)
        z_total = torch.concat((z, z_sampled), dim=0)
        return [t_total, y_total], z_total

    def objective(self, x, z, which_obj="both", *args, **kwargs):
        """Modifies `objective` of base class to include sampling from reference distribution"""
        self.check_init()
        assert which_obj in ["both", "theta", "dual"]

        # Append reference samples if we are training dual variables or both
        x_reference, z_reference = self.append_reference_samples(x, z)
        return self._objective(x, z, x_ref=x_reference, z_ref=z_reference)

    def _objective(self, x, z, x_ref=None, z_ref=None, *args, **kwargs):
        # Empirical rkhs function on batch
        rkhs_func_empirical = self.eval_rkhs_func(x, z)

        # Reference rkhs function on batch+reference
        rkhs_func_reference = self.eval_rkhs_func(x_ref, z_ref)

        # Dual function on reference
        dual_moments = self._eval_dual_moment_func(z_ref)
        model_moments = self.moment_function(x_ref)

        conj_div_arg = (
            rkhs_func_reference
            + self.dual_normalization.params
            - torch.sum(dual_moments * model_moments, dim=1, keepdim=True)
        )

        objective = (
            torch.mean(rkhs_func_empirical)
            + self.dual_normalization.params
            - 1 / 2 * self.rkhs_reg_param * self.rkhs_norm_sq()
            - self.entropy_reg_param
            * torch.mean(
                self.conj_divergence(1 / self.entropy_reg_param * conj_div_arg)
            )
        )

        # Add neural network L2 reg
        if self.reg_param > 0:
            l2_reg = self.reg_param * torch.mean(dual_moments**2)
            return objective, -objective + l2_reg

        return objective, -objective


if __name__ == "__main__":
    from experiments.tests import test_cmr_estimator

    pass
