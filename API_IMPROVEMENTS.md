# API Improvements for Conditional Moment Restrictions Library

Based on the demonstration in [demo_quick.py](demo_quick.py), this document outlines critical improvements needed for the library's API.

## Current Issues Demonstrated

### 1. Parameter Explosion
**Problem**: Method complexity ranges from 2 parameters (OLS) to 20+ parameters (KMM-neural)

**Example**:
```python
# Simple OLS
OrdinaryLeastSquares(model=model, moment_function=fn)

# Complex KMM-neural
KMMNeural(
    model=model, moment_function=fn,
    divergence='kl', entropy_reg_param=10.0, reg_param=1e-2, rkhs_reg_param=1.0,
    n_random_features=500, n_reference_samples=50, kde_bandwidth=0.3,
    kernel_x_kwargs={}, kernel_z_kwargs={}, dual_func_network_kwargs={...},
    pretrain=True, rkhs_func_z_dependent=True,
    theta_optim_args={...}, dual_optim_args={...},
    max_num_epochs=500, batch_size=100, eval_freq=50,
    max_no_improve=3, burn_in_cycles=2, gpu=False, verbose=False
)
```

### 2. Broken GPU Support
**Problem**: Multiple locations missing `.cpu()` before `.numpy()` conversion

**Error**:
```
TypeError: can't convert cuda:0 device type tensor to numpy.
Use Tensor.cpu() to copy the tensor to host memory first.
```

**Location**: `abstract_estimation_method.py:131` in `_set_kernel_z()`

**Additional Issues**:
- GPU usage is opt-in (must pass `gpu=True` to every estimator)
- No auto-detection or warning when GPU is available but unused
- Should default to GPU when available

### 3. No Clear Parameter Organization
**Problem**: Parameters are mixed without clear categorization

**Current mess**:
- Method-specific: `entropy_reg_param`, `rkhs_reg_param`, `divergence`
- Optimization: `theta_optim_args`, `dual_optim_args`
- Algorithmic: `n_random_features`, `n_reference_samples`
- Training: `max_num_epochs`, `batch_size`, `eval_freq`
- All at the same level with no structure

### 4. Inconsistent Defaults Across Methods
**Problem**: What works for one method breaks another

| Method | batch_size | pretrain | optimizer |
|--------|-----------|----------|-----------|
| MMR | `None` | Not applicable | LBFGS |
| FGEL-kernel | `None` | `True` | LBFGS |
| FGEL-neural | Required (e.g., 200) | `True` | oadam_gda |
| KMM-neural | Required (e.g., 200) | `True` | oadam_gda |

Users must know these differences or face cryptic errors.

### 5. Nested Dictionaries Without Documentation
**Problem**: Hard to know what goes in each dict

**Examples**:
```python
kernel_z_kwargs = {}  # What can go here? No docs!
dual_func_network_kwargs = {'layer_widths': [30, 20]}  # What else is valid?
theta_optim_args = {'optimizer': 'oadam_gda', 'lr': 5e-4}  # What optimizers exist?
```

### 6. No Type Hints or Validation
**Problem**: Easy to pass wrong types, get unhelpful errors

**Current**:
```python
def __init__(self, model, moment_function, **kwargs):
    # No type hints, no validation
```

**Should be**:
```python
def __init__(
    self,
    model: torch.nn.Module,
    moment_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reg_param: float = 1e-4,
    divergence: Literal['chi2', 'kl', 'log'] = 'chi2',
    ...
) -> None:
```

### 7. Method Selection Unclear
**Problem**: No guidance on which method to use or why

Users must:
- Read multiple papers to understand differences
- Trial-and-error to find appropriate hyperparameters
- Guess which method suits their problem

---

## Proposed Solutions

### Solution 1: Hierarchical Configuration Objects

```python
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class OptimizationConfig:
    """Configuration for optimization"""
    optimizer: Literal['adam', 'oadam', 'lbfgs', 'oadam_gda'] = 'oadam_gda'
    learning_rate: float = 5e-4
    max_epochs: int = 3000
    batch_size: Optional[int] = None
    early_stopping_patience: int = 5
    burn_in_epochs: int = 5
    eval_frequency: int = 100

@dataclass
class DualFunctionConfig:
    """Configuration for dual function (neural or kernel)"""
    type: Literal['neural', 'kernel', 'none'] = 'neural'
    hidden_layers: list[int] = (50, 30, 20)
    activation: str = 'leaky_relu'
    regularization: float = 1e-2

@dataclass
class KernelConfig:
    """Configuration for kernel methods"""
    kernel_type: Literal['rbf', 'polynomial', 'linear'] = 'rbf'
    bandwidth: Optional[float] = None  # None = auto (median heuristic)

@dataclass
class KMMConfig:
    """KMM-specific configuration"""
    entropy_regularization: float = 10.0
    rkhs_regularization: float = 1.0
    n_random_features: int = 2000
    n_reference_samples: int = 200
    kde_bandwidth: float = 0.3
    reference_depends_on_z: bool = True

# Usage becomes much cleaner:
estimator = KMMNeural(
    model=model,
    moment_function=fn,
    optimization=OptimizationConfig(learning_rate=1e-3, max_epochs=5000),
    dual_function=DualFunctionConfig(hidden_layers=[30, 20]),
    kernel=KernelConfig(),
    kmm_config=KMMConfig(entropy_regularization=5.0),
    device='cuda'  # or 'auto' to auto-detect
)
```

### Solution 2: Smart Defaults Based on Data Size

```python
def auto_configure(
    method: str,
    n_samples: int,
    problem_type: Literal['conditional', 'unconditional']
) -> dict:
    """Automatically configure estimator based on data size and problem"""

    if n_samples < 1000:
        # Small data: use kernel methods, no batching
        return {
            'batch_size': None,
            'n_random_features': None,  # Use exact kernel
            'optimizer': 'lbfgs'
        }
    elif n_samples < 5000:
        # Medium data: moderate RFF, small batches
        return {
            'batch_size': 200,
            'n_random_features': 1000,
            'optimizer': 'oadam_gda'
        }
    else:
        # Large data: large RFF, larger batches
        return {
            'batch_size': 500,
            'n_random_features': 2000,
            'optimizer': 'oadam_gda'
        }

# Usage:
config = auto_configure('KMM-neural', n_samples=len(train_data['t']), problem_type='conditional')
estimator = KMMNeural(model=model, moment_function=fn, **config)
```

### Solution 3: Unified Estimator Factory

```python
from cmr import create_estimator

# Simple interface that handles all configuration
estimator = create_estimator(
    method='KMM-neural',  # or 'FGEL-neural', 'MMR', etc.
    model=model,
    moment_function=fn,
    data_size=len(train_data['t']),
    # Optional overrides
    learning_rate=1e-3,
    regularization=1e-2,
    device='auto'  # Auto-detect GPU
)

estimator.train(train_data, val_data)
```

### Solution 4: Fix GPU Support

**Changes needed in `abstract_estimation_method.py`**:

```python
# Line 131: Fix CUDA tensor conversion
def _set_kernel_z(self, z=None, z_val=None):
    if self.kernel_z is None and z is not None:
        self.kernel_z, _ = get_rbf_kernel(z, z, **self.kernel_z_kwargs)
        self.kernel_z = self.kernel_z.type(torch.float32)
        # FIX: Add .cpu() before .numpy()
        kernel_z_np = self.kernel_z.detach().cpu().numpy()
        self.kernel_z_cholesky = torch.tensor(
            np.transpose(compute_cholesky_factor(kernel_z_np))
        )
```

**Auto-detect GPU by default**:

```python
class AbstractEstimationMethod:
    def __init__(
        self,
        model,
        moment_function,
        device: Literal['cpu', 'cuda', 'auto'] = 'auto',  # Changed from gpu=False
        **kwargs
    ):
        if device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, using CPU")
            self.device = 'cpu'
        else:
            self.device = device
```

### Solution 5: Type Hints and Validation

```python
from typing import Callable, Literal, Optional
import torch

class KMMNeural:
    def __init__(
        self,
        model: torch.nn.Module,
        moment_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        divergence: Literal['chi2', 'kl', 'log'] = 'kl',
        entropy_reg_param: float = 10.0,
        reg_param: float = 1e-2,
        n_random_features: int = 2000,
        device: Literal['cpu', 'cuda', 'auto'] = 'auto',
        verbose: bool = False
    ) -> None:
        # Validate inputs
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model must be torch.nn.Module, got {type(model)}")
        if entropy_reg_param <= 0:
            raise ValueError(f"entropy_reg_param must be positive, got {entropy_reg_param}")
        if n_random_features < 1:
            raise ValueError(f"n_random_features must be >= 1, got {n_random_features}")

        # ... rest of init
```

### Solution 6: Method Selection Guide

```python
def recommend_method(
    n_samples: int,
    has_instruments: bool,
    is_nonlinear: bool,
    computational_budget: Literal['low', 'medium', 'high'] = 'medium'
) -> str:
    """
    Recommend an estimation method based on problem characteristics.

    Args:
        n_samples: Number of training samples
        has_instruments: Whether instrumental variables are available
        is_nonlinear: Whether structural function is likely nonlinear
        computational_budget: Available computation (affects kernel vs neural)

    Returns:
        Recommended method name
    """
    if not has_instruments:
        return 'OLS'  # or 'GMM' if you want robustness

    if n_samples < 500:
        # Small data: use kernel methods
        if computational_budget == 'low':
            return 'MMR'  # Simplest conditional method
        else:
            return 'FGEL-kernel'  # More sophisticated

    elif n_samples < 5000:
        # Medium data: neural methods start to shine
        if is_nonlinear:
            return 'FGEL-neural'
        else:
            return 'VMM-neural'

    else:
        # Large data: use neural with RFF
        if computational_budget == 'high':
            return 'KMM-neural'  # Best theoretical properties
        else:
            return 'FGEL-neural'  # Faster to train

# Usage:
method = recommend_method(
    n_samples=1000,
    has_instruments=True,
    is_nonlinear=True,
    computational_budget='medium'
)
print(f"Recommended method: {method}")
```

### Solution 7: Better Error Messages

```python
# Current: cryptic error
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

# Improved: helpful error
class DeviceMismatchError(Exception):
    def __init__(self, tensor1_device, tensor2_device):
        message = (
            f"Device mismatch detected:\n"
            f"  Tensor 1 is on: {tensor1_device}\n"
            f"  Tensor 2 is on: {tensor2_device}\n\n"
            f"This usually happens when gpu=True but some tensors weren't moved to GPU.\n"
            f"Try:\n"
            f"  1. Set gpu=False to use CPU\n"
            f"  2. File a bug report - this shouldn't happen!"
        )
        super().__init__(message)
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
1. Fix GPU support (add `.cpu()` calls)
2. Change `gpu=False` default to `device='auto'`
3. Add basic type hints to all __init__ methods

### Phase 2: API Cleanup (Weeks 2-3)
1. Create configuration dataclasses
2. Implement auto-configuration based on data size
3. Add method recommendation function

### Phase 3: User Experience (Week 4)
1. Add helpful error messages
2. Create unified `create_estimator()` factory
3. Write comprehensive examples

### Phase 4: Testing & Documentation (Week 5)
1. Test GPU support thoroughly
2. Add type checking with mypy
3. Update all documentation
4. Create tutorial notebooks

---

## Backward Compatibility

To maintain backward compatibility during transition:

```python
class KMMNeural:
    def __init__(
        self,
        model,
        moment_function,
        # New API
        config: Optional[KMMConfig] = None,
        device: Literal['cpu', 'cuda', 'auto'] = 'auto',
        # Legacy API (deprecated)
        gpu: Optional[bool] = None,
        **kwargs
    ):
        # Handle legacy gpu parameter
        if gpu is not None:
            warnings.warn(
                "gpu parameter is deprecated, use device='cuda' instead",
                DeprecationWarning
            )
            device = 'cuda' if gpu else 'cpu'

        # Handle legacy kwargs
        if kwargs and config is None:
            warnings.warn(
                "Passing parameters as kwargs is deprecated, "
                "use config object instead",
                DeprecationWarning
            )
            config = KMMConfig(**kwargs)
```

---

## Summary

The current API suffers from:
1. **Parameter explosion** (2 to 20+ parameters)
2. **Broken GPU support** (missing `.cpu()` calls)
3. **Poor organization** (flat parameter space)
4. **Inconsistent defaults** (what works for one method breaks another)
5. **Undocumented nested dicts** (kernel_kwargs, optim_args, etc.)
6. **No type safety** (easy to make mistakes)
7. **No guidance** (hard to choose method or hyperparameters)

The proposed solutions provide:
- **Structured configuration** via dataclasses
- **Auto-detection** of computational resources
- **Smart defaults** based on data characteristics
- **Type safety** with hints and validation
- **Better errors** with actionable messages
- **Guided selection** of methods
- **Backward compatibility** during migration

These improvements will make the library significantly more user-friendly while maintaining its powerful capabilities.
