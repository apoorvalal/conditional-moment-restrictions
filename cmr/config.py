from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union, Callable, Any


@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters."""

    optimizer: Literal["adam", "oadam", "lbfgs", "oadam_gda", "sgd"] = "oadam_gda"
    learning_rate: float = 5e-4
    dual_optimizer: Optional[Literal["adam", "oadam", "lbfgs", "oadam_gda", "sgd"]] = (
        None
    )
    dual_learning_rate: Optional[float] = None
    max_epochs: int = 3000
    batch_size: Optional[int] = None  # None means full batch
    early_stopping_patience: int = 5
    burn_in_epochs: int = 5
    eval_frequency: int = 100
    # Additional optim args specific to certain optimizers (like betas for Adam)
    optimizer_kwargs: dict = field(default_factory=dict)


@dataclass
class NetworkConfig:
    """Configuration for neural networks (dual functions)."""

    hidden_layers: List[int] = field(default_factory=lambda: [50, 30, 20])
    activation: str = "leaky_relu"  # string representation, mapped later
    regularization: float = 1e-2


@dataclass
class KernelConfig:
    """Configuration for kernel methods."""

    kernel_type: Literal["rbf", "polynomial", "linear"] = "rbf"
    bandwidth: Optional[float] = None  # None = auto (median heuristic)


@dataclass
class KMMConfig:
    """Configuration specific to KMM method."""

    entropy_regularization: float = 10.0
    rkhs_regularization: float = 1.0
    n_random_features: int = 2000
    n_reference_samples: int = 200
    kde_bandwidth: float = 0.3
    reference_depends_on_z: bool = True
    t_as_instrument: bool = False


@dataclass
class FGELConfig:
    """Configuration specific to FGEL method."""

    divergence: Literal["chi2", "kl", "log"] = "chi2"
    regularization: float = 1e-2


@dataclass
class MMRConfig:
    """Configuration specific to MMR method."""

    # MMR is usually parameter-free besides the kernel, but we might want to add options here.
    pass


@dataclass
class VMMConfig:
    """Configuration specific to VMM method."""

    regularization: float = 1e-2
    num_iter: int = 20
