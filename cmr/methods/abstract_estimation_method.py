from cmr.utils.rkhs_utils import get_rbf_kernel, compute_cholesky_factor, hsic
from cmr.utils.torch_utils import np_to_tensor, to_device
from cmr.config import OptimizationConfig
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Literal


class AbstractEstimationMethod:
    def __init__(
        self,
        model: torch.nn.Module,
        moment_function,
        optimization: Optional[OptimizationConfig] = None,
        val_loss_func=None,
        verbose=0,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        gpu=False,  # Deprecated
        **kwargs,
    ):
        self.model = ModelWrapper(model)
        self.moment_function = self._wrap_moment_function(moment_function)

        # Configuration
        self.optimization = optimization or OptimizationConfig()

        # Handle optimization kwargs legacy overrides
        if "theta_optim_args" in kwargs:
            # Basic mapping for legacy support - users should migrate to Config
            pass

        self.is_trained = False
        self.train_stats = {}
        self._val_loss_func = val_loss_func
        self.verbose = verbose

        # Device handling
        if device == "auto":
            # Check legacy gpu flag if device is auto (default)
            if gpu:
                device = "cuda"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        if self.device == "cuda" and self.verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Set by `_init_data_dependent_attributes`
        self._dim_psi = None
        self._dim_z = None
        self._is_init = False
        self.dim_t = None
        self.dim_y = None

        # For validation purposes by default all methods for CMR use the MMR loss and therefore require the kernel Gram matrices
        try:
            self.kernel_z_kwargs = kwargs.get("kernel_z_kwargs", {})
        except KeyError:
            self.kernel_z_kwargs = {}
        self.kernel_z = None
        self.kernel_z_cholesky = None
        self.kernel_z_val = None

    @property
    def dim_psi(self):
        if self._dim_psi is None:
            raise AttributeError(
                "Data dependent attributes have not been set. "
                "First use method `init_estimator(x, z)`"
            )
        return self._dim_psi

    @property
    def dim_z(self):
        # `dim_z` is allowed to be `None` for unconditional moment restrictions
        return self._dim_z

    def _wrap_moment_function(self, moment_function):
        model = self.model

        def eval_moment_function(x):
            t, y = torch.Tensor(x[0]), torch.Tensor(x[1])
            # Ensure inputs are on the correct device
            if next(model.parameters()).is_cuda:
                t = t.to(next(model.parameters()).device)
                y = y.to(next(model.parameters()).device)
            return moment_function(model(t), y)

        return eval_moment_function

    def init_estimator(self, x, z):
        self._init_data_dependent_attributes(x, z)
        self._is_init = True

    def check_init(self):
        if not self._is_init:
            raise AttributeError(
                "Called method requires running method `init_estimator(x,z)` first."
            )

    def objective(self, x, z, which_obj="both", *args, **kwargs):
        self.check_init()
        assert which_obj in ["both", "theta", "dual"]
        return self._objective(x=x, z=z, which_obj=which_obj, *args, **kwargs)

    def _objective(self, x, z, *args, **kwargs):
        raise NotImplementedError(
            "Method `objective` needs to be implemented in child class."
        )

    def _init_data_dependent_attributes(self, x, z):
        if not self._is_init:
            if z is None:
                self._dim_z = None
            else:
                self._dim_z = z.shape[1]

            # Eval moment function once on a single sample to get its dimension
            # Ensure sample is on cpu for initial check to avoid device issues if model is not moved yet
            single_sample = [x[0][0:1], x[1][0:1]]
            # Temporary wrap to avoid device mismatch during init
            try:
                self._dim_psi = self.moment_function(single_sample).shape[1]
            except Exception:
                # If model is already on GPU, we might need to move sample
                t_s = torch.Tensor(single_sample[0]).to(self.device)
                y_s = torch.Tensor(single_sample[1]).to(self.device)
                self.model = self.model.to(self.device)  # Ensure model is on device
                self._dim_psi = self.moment_function([t_s, y_s]).shape[1]

            self.dim_t = x[0].shape[1]
            self.dim_y = x[1].shape[1]

    def train(self, train_data, val_data=None, debugging=False):
        x_train = [train_data["t"], train_data["y"]]
        z_train = train_data["z"]

        if val_data is None:
            x_val = x_train
            z_val = z_train
        else:
            x_val = [val_data["t"], val_data["y"]]
            z_val = val_data["z"]

        x_train, z_train = (
            self._to_tensor_and_device(x_train),
            self._to_tensor_and_device(z_train),
        )
        x_val, z_val = (
            self._to_tensor_and_device(x_val),
            self._to_tensor_and_device(z_val),
        )
        self.model = self.model.to(self.device)

        if not self._is_init:
            self.init_estimator(x_train, z_train)

        if self.device == "cuda":
            print("Starting training on GPU ...")

        self._train_internal(x_train, z_train, x_val, z_val, debugging=debugging)

        # Move model back to CPU after training for safety/portability unless user wants it there?
        # Usually it's better to keep it where it is, but the original code did .cpu().
        # Let's respect the device choice of the user. If they asked for CUDA, leave it on CUDA?
        # The original code forced .cpu(). Let's keep it on device if specified, or maybe .cpu() is safer for eval.
        # Let's follow the standard pattern: Keep it on device.
        # self.model.cpu()
        self.is_trained = True

    def get_trained_parameters(self):
        if not self.is_trained:
            raise RuntimeError("Need to fit model before getting fitted params")
        return self.model.get_parameters()

    def _set_kernel_z(self, z=None, z_val=None):
        if self.kernel_z is None and z is not None:
            # Calculate kernel on whatever device z is on
            self.kernel_z, _ = get_rbf_kernel(z, z, **self.kernel_z_kwargs)
            self.kernel_z = self.kernel_z.type(torch.float32).to(self.device)

            # Cholesky needs to happen on CPU for numpy, or we use torch.cholesky
            # The original code converted to numpy. Let's fix the device transfer.
            kernel_z_cpu = self.kernel_z.detach().cpu().numpy()
            self.kernel_z_cholesky = torch.tensor(
                np.transpose(compute_cholesky_factor(kernel_z_cpu))
            ).to(self.device)

        if self.kernel_z_val is None and z_val is not None:
            self.kernel_z_val, _ = get_rbf_kernel(z_val, z_val, **self.kernel_z_kwargs)
            self.kernel_z_val = self.kernel_z_val.type(torch.float32).to(self.device)

    def _calc_val_hsic(self, x_val, z_val):
        assert z_val is not None, (
            "HSIC cannot be used as validation loss without specifying instrument data `z`."
        )
        max_num_val_data = 4001
        self._set_kernel_z(z_val=z_val[:max_num_val_data])

        # Ensure inputs are on correct device
        x_in = x_val[0][:max_num_val_data].to(self.device)
        y_in = x_val[1][:max_num_val_data].to(self.device)

        residue = self.model(x_in) - y_in
        kernel_residue, _ = get_rbf_kernel(residue, residue)
        return (
            float(
                hsic(kernel_matrix_x=kernel_residue, kernel_matrix_y=self.kernel_z_val)
                .detach()
                .cpu()
                .numpy()
            )
            * 100
        )

    def _calc_val_mmr(self, x_val, z_val):
        assert z_val is not None, (
            "MMR cannot be used as validation loss without specifying instrument data `z`."
        )
        max_num_val_data = 4001
        self._set_kernel_z(z_val=z_val[:max_num_val_data])

        x_in = [
            x_val[0][:max_num_val_data].to(self.device),
            x_val[1][:max_num_val_data].to(self.device),
        ]

        psi = self.moment_function(x_in)

        # Kernel matrix is already on device
        loss = torch.einsum("ir, ij, jr -> ", psi, self.kernel_z_val, psi) / (
            z_val[:max_num_val_data].shape[0] ** 2
        )
        return float(loss.detach().cpu().numpy())

    def _calc_val_moment_violation(self, x_val, z_val=None):
        # inputs to moment_function are wrapped to handle device transfer, but let's be explicit
        x_in = [x_val[0].to(self.device), x_val[1].to(self.device)]
        psi = self.moment_function(x_in)
        mse_moment_violation = torch.sum(torch.square(psi)) / psi.shape[0]
        return float(mse_moment_violation.detach().cpu().numpy())

    def calc_validation_metric(self, x_val, z_val):
        if self._val_loss_func is None:
            return None
        elif self._val_loss_func == "mmr":
            return self._calc_val_mmr(x_val, z_val)
        elif self._val_loss_func == "moment_violation":
            return self._calc_val_moment_violation(x_val, z_val)
        elif self._val_loss_func == "hsic":
            return self._calc_val_hsic(x_val, z_val)
        elif not isinstance(self._val_loss_func, str):
            val_data = {"t": x_val[0], "y": x_val[1], "z": z_val}
            return self._val_loss_func(self.model, val_data)
        else:
            raise NotImplementedError(
                f"Input `val_loss_func`` must be either a function or in "
                f"['mmr', 'moment_violation']. Currently is {self._val_loss_func}"
            )

    @staticmethod
    def _to_tensor(data_array):
        return np_to_tensor(data_array)

    def _to_tensor_and_device(self, data_array):
        if data_array is None:
            return None
        return to_device(self._to_tensor(data_array), device=self.device)

    def _train_internal(self, x, z, x_val, z_val, debugging):
        raise NotImplementedError()

    def _pretrain_theta(self, x, z, mmr=True):
        optimizer = torch.optim.LBFGS(
            self.model.parameters(), line_search_fn="strong_wolfe"
        )

        if mmr and z is not None and z.shape[0] < 5000:
            self._set_kernel_z(z=z)

            def closure():
                optimizer.zero_grad()
                psi = self.moment_function(x)
                loss = torch.einsum("ir, ij, jr -> ", psi, self.kernel_z, psi) / (
                    x[0].shape[0] ** 2
                )
                loss.backward()
                return loss
        else:

            def closure():
                optimizer.zero_grad()
                psi = self.moment_function(x)
                loss = (psi**2).mean()
                loss.backward()
                return loss

        optimizer.step(closure)


class ModelWrapper(nn.Module):
    def __init__(self, model):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.Tensor(t)
        # Device transfer is handled by wrapping logic in Estimator,
        # but if called directly, we might need check.
        # Generally model parameters determine device.
        if next(self.model.parameters()).is_cuda and not t.is_cuda:
            t = t.to(next(self.model.parameters()).device)
        return self.model(t)

    def get_parameters(self):
        try:
            return self.model.get_parameters()
        except AttributeError:
            param_tensor = list(self.model.parameters())
            return [p.detach().cpu().numpy() for p in param_tensor]

    def is_finite(self):
        params = self.get_parameters()
        isnan = bool(sum([np.sum(np.isnan(p)) for p in params]))
        isinf = bool(sum([np.sum(np.isinf(p)) for p in params]))
        return (not isnan) and (not isinf)

    def initialize(self):
        try:
            self.model.initialize()
        except AttributeError:
            pass
