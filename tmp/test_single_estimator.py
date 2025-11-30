"""
Quick test script for testing individual CMR estimators in isolation.

This allows rapid iteration on single methods without waiting for the full demo
chain of all 11 estimators to complete.

Usage:
    python tmp/test_single_estimator.py --method MMR
    python tmp/test_single_estimator.py --method KMM-neural
    python tmp/test_single_estimator.py --method FGEL-neural
"""

import argparse
import numpy as np
import torch
from typing import Dict

# Import config and estimator classes
from cmr.config import OptimizationConfig, KMMConfig, NetworkConfig, FGELConfig
from cmr.methods.least_squares import OrdinaryLeastSquares
from cmr.methods.gmm import GMM
from cmr.methods.generalized_el import GeneralizedEL
from cmr.methods.kmm import KMM
from cmr.methods.sieve_minimum_distance import SMDIdentity
from cmr.methods.vmm_neural import NeuralVMM
from cmr.methods.fgel_neural import NeuralFGEL
from cmr.methods.kmm_neural import KMMNeural
from cmr.methods.mmr import MMR
from cmr.methods.fgel_kernel import KernelFGEL
from cmr.methods.vmm_kernel import KernelVMM
from cmr.factory import create_estimator


def generate_nonparametric_iv_data(n: int, seed: int = None) -> Dict[str, np.ndarray]:
    """Generate data from nonparametric IV model with confounding."""
    if seed is not None:
        np.random.seed(seed)

    # Unobserved confounder
    U = np.random.normal(0, 1.0, size=(n, 1))

    # Instrument (exogenous)
    Z = np.random.uniform(-3, 3, size=(n, 1))

    # Treatment (endogenous due to U)
    epsilon_x = np.random.normal(0, 0.1, size=(n, 1))
    X = Z + U + epsilon_x

    # Outcome (depends on X nonlinearly and U)
    epsilon_y = np.random.normal(0, 0.1, size=(n, 1))
    Y = 2 * np.log(np.abs(X) + 0.5) + U + epsilon_y

    return {"t": X, "y": Y, "z": Z, "u": U}


def true_structural_function(x: np.ndarray) -> np.ndarray:
    """True structural function: y = 2*log(|x| + 0.5)"""
    return 2 * np.log(np.abs(x) + 0.5)


def get_neural_network(input_dim: int = 1, output_dim: int = 1) -> torch.nn.Module:
    """Create a flexible neural network for nonparametric estimation."""
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 50),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(50, 30),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(30, 20),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(20, output_dim),
    )


def iv_moment_function(model_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard IV moment function: psi(x,y;theta) = f_theta(x) - y"""
    return model_pred - y


def evaluate_estimator(model: torch.nn.Module, test_data: Dict[str, np.ndarray], name: str):
    """Evaluate estimator on test data."""
    model = model.cpu()
    with torch.no_grad():
        X_test = torch.Tensor(test_data["t"])
        Y_true = true_structural_function(test_data["t"])
        Y_pred = model(X_test).detach().numpy()

        mse = np.mean((Y_pred - Y_true) ** 2)
        bias = np.mean(Y_pred - Y_true)

        print(f"\n{name:30s} | MSE: {mse:8.4f} | Bias: {bias:8.4f}")
        return mse, bias


def test_estimator(method: str, n_train: int = 500, use_factory: bool = False):
    """Test a single estimator."""
    print("=" * 80)
    print(f"TESTING: {method}")
    print("=" * 80)

    # Generate data
    print(f"\nGenerating data (n_train={n_train})...")
    train_data = generate_nonparametric_iv_data(n=n_train, seed=42)
    val_data = generate_nonparametric_iv_data(n=n_train, seed=43)
    test_data = generate_nonparametric_iv_data(n=2000, seed=44)

    # Create model
    model = get_neural_network()

    # Train estimator
    print(f"\nTraining {method}...")

    if use_factory:
        # Use factory function
        print("Using factory function...")
        estimator = create_estimator(
            method=method,
            model=model,
            moment_function=iv_moment_function,
            train_data_size=n_train,
            verbose=True,
        )
    else:
        # Direct instantiation
        print("Using direct instantiation...")

        if method == "OLS":
            opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
            estimator = OrdinaryLeastSquares(
                model=model, moment_function=iv_moment_function, optimization=opt_config
            )
        elif method == "GMM":
            opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
            estimator = GMM(
                model=model,
                moment_function=iv_moment_function,
                optimization=opt_config,
                reg_param=1e-4,
            )
        elif method == "GEL":
            opt_config = OptimizationConfig(optimizer="lbfgs", learning_rate=5e-4, max_epochs=10)
            estimator = GeneralizedEL(
                model=model,
                moment_function=iv_moment_function,
                optimization=opt_config,
                divergence="chi2",
                reg_param=1e-6,
            )
        elif method == "MMR":
            opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
            estimator = MMR(
                model=model,
                moment_function=iv_moment_function,
                optimization=opt_config,
                verbose=True,
            )
        elif method == "SMD":
            opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
            estimator = SMDIdentity(
                model=model, moment_function=iv_moment_function, optimization=opt_config
            )
        elif method == "VMM-kernel":
            opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
            estimator = KernelVMM(
                model=model,
                moment_function=iv_moment_function,
                optimization=opt_config,
                divergence="chi2",
                reg_param=1e-4,
                verbose=True,
            )
        elif method == "VMM-neural":
            opt_config = OptimizationConfig(
                optimizer="oadam_gda", learning_rate=5e-4, max_epochs=1000, batch_size=200
            )
            estimator = NeuralVMM(
                model=model,
                moment_function=iv_moment_function,
                optimization=opt_config,
                divergence="chi2",
                reg_param=1e-2,
            )
        elif method == "FGEL-kernel":
            opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
            estimator = KernelFGEL(
                model=model,
                moment_function=iv_moment_function,
                optimization=opt_config,
                divergence="chi2",
                reg_param=1e-3,
                verbose=True,
            )
        elif method == "FGEL-neural":
            opt_config = OptimizationConfig(
                optimizer="oadam_gda", learning_rate=5e-4, max_epochs=2000, batch_size=200
            )
            fgel_config = FGELConfig(divergence="chi2", regularization=1e-2)
            estimator = NeuralFGEL(
                model=model,
                moment_function=iv_moment_function,
                optimization=opt_config,
                fgel_config=fgel_config,
                verbose=True,
            )
        elif method == "KMM-neural":
            opt_config = OptimizationConfig(
                optimizer="oadam_gda", learning_rate=5e-4, max_epochs=2000, batch_size=200
            )
            kmm_config = KMMConfig(
                entropy_regularization=10.0,
                rkhs_regularization=1e-2,
                n_random_features=2000,
            )
            dual_network = NetworkConfig(hidden_layers=[50, 30, 20], regularization=1e-2)
            estimator = KMMNeural(
                model=model,
                moment_function=iv_moment_function,
                optimization=opt_config,
                kmm_config=kmm_config,
                dual_network=dual_network,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    # Train
    if method in ["VMM-neural", "FGEL-neural", "KMM-neural"]:
        estimator.train(train_data, val_data)
    else:
        estimator.train(train_data)

    # Evaluate
    print("\nEvaluation:")
    print("-" * 80)
    evaluate_estimator(estimator.model, test_data, method)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a single CMR estimator")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "OLS",
            "GMM",
            "GEL",
            "MMR",
            "SMD",
            "VMM-kernel",
            "VMM-neural",
            "FGEL-kernel",
            "FGEL-neural",
            "KMM-neural",
        ],
        help="Estimator to test",
    )
    parser.add_argument(
        "--n_train", type=int, default=500, help="Number of training samples"
    )
    parser.add_argument(
        "--factory",
        action="store_true",
        help="Use factory function instead of direct instantiation",
    )

    args = parser.parse_args()

    test_estimator(args.method, n_train=args.n_train, use_factory=args.factory)
