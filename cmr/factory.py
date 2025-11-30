from typing import Callable, Optional, Literal, Union, Dict, Any
import torch
from cmr.config import (
    OptimizationConfig,
    KMMConfig,
    NetworkConfig,
    FGELConfig,
    KernelConfig,
    MMRConfig,
    VMMConfig,
)
from cmr.methods.kmm_neural import KMMNeural
from cmr.methods.fgel_neural import NeuralFGEL
from cmr.methods.mmr import MMR
from cmr.methods.fgel_kernel import KernelFGEL
from cmr.methods.vmm_kernel import KernelVMM


def create_estimator(
    method: str,
    model: torch.nn.Module,
    moment_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_data_size: int,
    device: Literal["cpu", "cuda", "auto"] = "auto",
    verbose: bool = False,
    # Overrides
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_epochs: Optional[int] = None,
    regularization: Optional[float] = None,
    entropy_regularization: Optional[float] = None,
    **kwargs,
):
    """
    Factory function to create a CMR estimator with smart defaults.
    """

    # 1. Determine optimal optimization config
    opt_config = OptimizationConfig()

    # Auto-scale batch size and optimizer
    if train_data_size < 1000:
        opt_config.batch_size = None  # Full batch
        opt_config.optimizer = "lbfgs"
        opt_config.max_epochs = 500  # LBFGS converges fast
    elif train_data_size < 10000:
        opt_config.batch_size = 256
        opt_config.optimizer = "oadam_gda"
        opt_config.max_epochs = 2000
    else:
        opt_config.batch_size = 1024
        opt_config.optimizer = "oadam_gda"
        opt_config.max_epochs = 3000

    # Apply overrides
    if learning_rate is not None:
        opt_config.learning_rate = learning_rate
    if batch_size is not None:
        opt_config.batch_size = batch_size
    if max_epochs is not None:
        opt_config.max_epochs = max_epochs

    # 2. Method selection
    if method == "KMM-neural":
        kmm_config = KMMConfig()
        if entropy_regularization is not None:
            kmm_config.entropy_regularization = entropy_regularization

        # Apply any specific kwargs to KMMConfig if they match
        for k, v in kwargs.items():
            if hasattr(kmm_config, k):
                setattr(kmm_config, k, v)

        network_config = NetworkConfig()
        if regularization is not None:
            network_config.regularization = regularization

        estimator = KMMNeural(
            model=model,
            moment_function=moment_function,
            kmm_config=kmm_config,
            dual_network=network_config,
            optimization=opt_config,
            device=device,
            verbose=verbose,
        )

    elif method == "FGEL-neural":
        fgel_config = FGELConfig()
        if regularization is not None:
            fgel_config.regularization = regularization

        # Apply kwargs
        for k, v in kwargs.items():
            if hasattr(fgel_config, k):
                setattr(fgel_config, k, v)

        estimator = NeuralFGEL(
            model=model,
            moment_function=moment_function,
            fgel_config=fgel_config,
            optimization=opt_config,
            device=device,
            verbose=verbose,
        )

    elif method == "MMR":
        estimator = MMR(
            model=model,
            moment_function=moment_function,
            optimization=opt_config,
            device=device,
            verbose=verbose,
            **kwargs,
        )

    elif method == "FGEL-kernel":
        estimator = KernelFGEL(
            model=model,
            moment_function=moment_function,
            optimization=opt_config,
            device=device,
            verbose=verbose,
            reg_param=regularization if regularization is not None else 1e-4,
            **kwargs,
        )

    elif method == "VMM-kernel":
        estimator = KernelVMM(
            model=model,
            moment_function=moment_function,
            optimization=opt_config,
            device=device,
            verbose=verbose,
            reg_param=regularization if regularization is not None else 1e-4,
            **kwargs,
        )

    else:
        raise NotImplementedError(
            f"Method {method} not yet fully supported in new API factory. Use direct class initialization."
        )

    return estimator
