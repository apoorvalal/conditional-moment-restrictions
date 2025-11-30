# API Design Proposal for CMR Library

This document outlines a comprehensive plan to improve the API of the Conditional Moment Restrictions (CMR) library. The goal is to address the issues identified in `demo_quick.py` (parameter explosion, broken GPU support, inconsistent defaults, etc.) and provide a modern, type-safe, and user-friendly interface.

## 1. Core Principles

1.  **Type Safety**: Use Python type hints strictly.
2.  **Structured Configuration**: Replace loose `**kwargs` with typed dataclasses.
3.  **Smart Defaults**: Auto-detect hardware (GPU/CPU) and suggest hyperparameters based on data size.
4.  **Hierarchical Design**: Separate method-specific parameters from optimization and architectural parameters.

## 2. Configuration Objects

Instead of passing dozens of arguments to the estimator constructor, we will use structured configuration objects.

```python
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union, Callable
import torch

@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    optimizer: Literal['adam', 'oadam', 'lbfgs', 'oadam_gda', 'sgd'] = 'oadam_gda'
    learning_rate: float = 5e-4
    max_epochs: int = 3000
    batch_size: Optional[int] = None  # None means full batch
    early_stopping_patience: int = 5
    burn_in_epochs: int = 5
    eval_frequency: int = 100

@dataclass
class NetworkConfig:
    """Configuration for neural networks (dual functions)."""
    hidden_layers: List[int] = field(default_factory=lambda: [50, 30, 20])
    activation: str = 'leaky_relu'
    regularization: float = 1e-2

@dataclass
class KernelConfig:
    """Configuration for kernel methods."""
    kernel_type: Literal['rbf', 'polynomial', 'linear'] = 'rbf'
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

@dataclass
class FGELConfig:
    """Configuration specific to FGEL method."""
    divergence: Literal['chi2', 'kl', 'log'] = 'chi2'
    regularization: float = 1e-2
```

## 3. Unified Estimator Interface

The base `AbstractEstimationMethod` will be updated to accept these config objects.

```python
class AbstractEstimationMethod:
    def __init__(
        self,
        model: torch.nn.Module,
        moment_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimization: Optional[OptimizationConfig] = None,
        device: Literal['cpu', 'cuda', 'auto'] = 'auto',
        verbose: bool = False,
        **kwargs  # For backward compatibility only
    ):
        self.model = model
        self.moment_function = moment_function
        self.optimization = optimization or OptimizationConfig()
        
        # Smart device detection
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cuda':
                print(f"Auto-detected GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = device
            
        self.verbose = verbose
        # ... initialization ...
```

## 4. Method-Specific Implementations

Each method will have a clean `__init__` that takes specific configs.

```python
class KMMNeural(AbstractEstimationMethod):
    def __init__(
        self,
        model: torch.nn.Module,
        moment_function: Callable,
        kmm_config: Optional[KMMConfig] = None,
        dual_network: Optional[NetworkConfig] = None,
        optimization: Optional[OptimizationConfig] = None,
        device: Literal['cpu', 'cuda', 'auto'] = 'auto',
        verbose: bool = False
    ):
        super().__init__(model, moment_function, optimization, device, verbose)
        self.config = kmm_config or KMMConfig()
        self.dual_network = dual_network or NetworkConfig()
        
        # ... validation ...
        if self.config.n_random_features < 1:
            raise ValueError("n_random_features must be positive")
```

## 5. Factory and Auto-Configuration

A factory function will simplify usage for users who don't want to configure everything manually.

```python
def create_estimator(
    method: str,
    model: torch.nn.Module,
    moment_function: Callable,
    train_data_size: int,
    device: str = 'auto',
    **overrides
):
    """Factory to create an estimator with smart defaults."""
    
    # 1. Determine optimal configuration based on data size
    opt_config = OptimizationConfig()
    if train_data_size < 1000:
        opt_config.batch_size = None  # Full batch
        opt_config.optimizer = 'lbfgs'
    elif train_data_size > 10000:
        opt_config.batch_size = 500
        opt_config.optimizer = 'oadam_gda'
        
    # 2. Create specific configs based on method
    if method == 'KMM-neural':
        est = KMMNeural(
            model=model,
            moment_function=moment_function,
            optimization=opt_config,
            device=device,
            # Apply overrides
            kmm_config=KMMConfig(**{k: v for k,v in overrides.items() if hasattr(KMMConfig, k)})
        )
    # ... handle other methods ...
    
    return est
```

## 6. Implementation Plan

### Phase 1: Fix Critical Bugs (Immediate)
-   **GPU Support**: Audit all `tensor.numpy()` calls and insert `.detach().cpu()` beforehand.
-   **Device Handling**: Ensure all created tensors (kernels, dual parameters) are created on `self.device`.

### Phase 2: Configuration Objects (Short-term)
-   Define the dataclasses in `cmr/config.py`.
-   Update `AbstractEstimationMethod` to use them (keeping `**kwargs` for compatibility).

### Phase 3: Refactor Estimators (Medium-term)
-   Update `NeuralFGEL`, `KMMNeural`, `MMR`, etc., to accept config objects.
-   Add proper type hints to all `__init__` methods.

### Phase 4: User-Facing API (Long-term)
-   Implement `create_estimator` factory.
-   Deprecate the old flat `**kwargs` style with warnings.
-   Update documentation and examples.

## 7. Migration Guide

Users will transition from:

```python
# Old Style
est = KMMNeural(model, fn, entropy_reg_param=10, batch_size=100, gpu=True)
```

To:

```python
# New Style (Explicit)
est = KMMNeural(
    model, fn,
    kmm_config=KMMConfig(entropy_regularization=10),
    optimization=OptimizationConfig(batch_size=100),
    device='cuda'
)

# OR New Style (Factory)
est = create_estimator('KMM-neural', model, fn, 
                       entropy_regularization=10, batch_size=100)
```

This ensures both power users and beginners have a suitable interface.