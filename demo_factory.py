"""
Demonstration of CMR estimators using the FACTORY FUNCTION on a nonparametric IV problem.

This script demonstrates the new factory API with smart defaults based on data size.
Currently, the factory supports 5 methods:
1. MMR
2. KMM-neural
3. FGEL-neural
4. FGEL-kernel
5. VMM-kernel

The remaining methods (OLS, GMM, GEL, KMM, SMD, VMM-neural) require direct instantiation.

DGP: Nonparametric IV with unobserved confounding
- True structural function: y = 2*log(x)
- Unobserved confounder U affects both X and Y
- Instrument Z affects X but not Y directly
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Import factory function
from cmr.factory import create_estimator


def generate_nonparametric_iv_data(n: int, seed: int = None) -> Dict[str, np.ndarray]:
    """
    Generate data from nonparametric IV model with confounding.
    """
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


def evaluate_estimator(
    model: torch.nn.Module, test_data: Dict[str, np.ndarray], name: str
) -> Tuple[float, float]:
    """
    Evaluate estimator on test data.
    """
    # Ensure model is on CPU for evaluation
    model = model.cpu()
    with torch.no_grad():
        X_test = torch.Tensor(test_data["t"])
        Y_true = true_structural_function(test_data["t"])

        Y_pred = model(X_test).detach().numpy()

        mse = np.mean((Y_pred - Y_true) ** 2)
        bias = np.mean(Y_pred - Y_true)

        print(f"{name:30s} | MSE: {mse:8.4f} | Bias: {bias:8.4f}")
        return mse, bias


def plot_results(results: Dict[str, torch.nn.Module], test_data: Dict[str, np.ndarray]):
    """Plot predictions from all methods."""
    X_test = test_data["t"]
    Y_true = true_structural_function(X_test)

    # Sort for plotting
    sort_idx = np.argsort(X_test.flatten())
    X_sorted = X_test[sort_idx]
    Y_sorted = Y_true[sort_idx]

    plt.figure(figsize=(14, 10))

    # Plot true function
    plt.scatter(
        test_data["t"], test_data["y"], alpha=0.3, s=10, label="Data", color="gray"
    )
    plt.plot(X_sorted, Y_sorted, "k-", linewidth=3, label="True function", zorder=100)

    # Plot predictions from each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, model), color in zip(results.items(), colors):
        model = model.cpu()
        with torch.no_grad():
            Y_pred = model(torch.Tensor(X_sorted)).detach().numpy()
        plt.plot(
            X_sorted, Y_pred, "--", linewidth=2, label=name, color=color, alpha=0.8
        )

    plt.xlabel("X (Treatment)", fontsize=12)
    plt.ylabel("Y (Outcome)", fontsize=12)
    plt.title(
        "Nonparametric IV: Factory API Demo\nTrue function: y = 2*log(|x| + 0.5)",
        fontsize=14,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("factory_demo_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'factory_demo_comparison.png'")


def main():
    print("=" * 80)
    print("NONPARAMETRIC IV ESTIMATION: FACTORY API DEMO")
    print("=" * 80)

    # Generate data
    print("\nGenerating data...")
    n_train = 500
    train_data = generate_nonparametric_iv_data(n=n_train, seed=42)
    val_data = generate_nonparametric_iv_data(n=n_train, seed=43)
    test_data = generate_nonparametric_iv_data(n=2000, seed=44)

    results = {}

    print(f"\n{'Method':<30s} | {'MSE':<8s} | {'Bias':<8s}")
    print("-" * 55)

    # =========================================================================
    # 1. MMR (Factory supported)
    # =========================================================================
    print("\n[1/5] Training MMR using factory...")
    model_mmr = get_neural_network()
    estimator = create_estimator(
        method="MMR",
        model=model_mmr,
        moment_function=iv_moment_function,
        train_data_size=n_train,
        verbose=False,
    )
    estimator.train(train_data)
    results["MMR"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "MMR")

    # =========================================================================
    # 2. VMM-KERNEL (Factory supported)
    # =========================================================================
    print("\n[2/5] Training VMM-kernel using factory...")
    model_vmm_k = get_neural_network()
    estimator = create_estimator(
        method="VMM-kernel",
        model=model_vmm_k,
        moment_function=iv_moment_function,
        train_data_size=n_train,
        regularization=1e-4,
        verbose=False,
    )
    estimator.train(train_data)
    results["VMM-kernel"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "VMM-kernel")

    # =========================================================================
    # 3. FGEL-KERNEL (Factory supported)
    # =========================================================================
    print("\n[3/5] Training FGEL-kernel using factory...")
    model_fgel_k = get_neural_network()
    estimator = create_estimator(
        method="FGEL-kernel",
        model=model_fgel_k,
        moment_function=iv_moment_function,
        train_data_size=n_train,
        regularization=1e-3,
        verbose=False,
    )
    estimator.train(train_data)
    results["FGEL-kernel"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "FGEL-kernel")

    # =========================================================================
    # 4. FGEL-NEURAL (Factory supported)
    # =========================================================================
    print("\n[4/5] Training FGEL-neural using factory...")
    model_fgel_n = get_neural_network()
    # Factory auto-selects oadam_gda and batch_size=256 for n=500
    estimator = create_estimator(
        method="FGEL-neural",
        model=model_fgel_n,
        moment_function=iv_moment_function,
        train_data_size=n_train,
        regularization=1e-2,
        max_epochs=2000,
        batch_size=200,
        verbose=False,
    )
    estimator.train(train_data, val_data)
    results["FGEL-neural"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "FGEL-neural")

    # =========================================================================
    # 5. KMM-NEURAL (Factory supported)
    # =========================================================================
    print("\n[5/5] Training KMM-neural using factory...")
    model_kmm_n = get_neural_network()
    # Factory provides smart defaults
    estimator = create_estimator(
        method="KMM-neural",
        model=model_kmm_n,
        moment_function=iv_moment_function,
        train_data_size=n_train,
        regularization=1e-2,
        entropy_regularization=10.0,
        max_epochs=2000,
        batch_size=200,
        verbose=False,
    )
    estimator.train(train_data, val_data)
    results["KMM-neural"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "KMM-neural")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nFactory API successfully demonstrated for 5 methods.")
    print("Remaining methods (OLS, GMM, GEL, KMM, SMD, VMM-neural) require direct")
    print("instantiation. See demo_all_methods.py for examples.")
    print("\nGenerating comparison plot...")
    plot_results(results, test_data)
    print("\nDemo complete. All methods created via factory function.")


if __name__ == "__main__":
    main()
