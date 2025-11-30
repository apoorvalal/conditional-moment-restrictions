"""
Demonstration of all CMR estimators on a nonparametric IV problem.

This script demonstrates:
1. Why naive OLS fails (endogeneity bias)
2. Why linear IV/GMM fails (model misspecification)
3. How each nonparametric method performs using DIRECT CLASS INSTANTIATION.

DGP: Nonparametric IV with unobserved confounding
- True structural function: y = 2*log(x)
- Unobserved confounder U affects both X and Y
- Instrument Z affects X but not Y directly
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Import all estimator classes directly
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
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
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
        "Nonparametric IV: All Methods Comparison\nTrue function: y = 2*log(|x| + 0.5)",
        fontsize=14,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("all_methods_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved as 'all_methods_comparison.png'")


def main():
    print("=" * 80)
    print("NONPARAMETRIC IV ESTIMATION: ALL METHODS DEMO (DIRECT INSTANTIATION)")
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
    # 1. ORDINARY LEAST SQUARES
    # =========================================================================
    print("\n[1/11] Training OLS...")
    model_ols = get_neural_network()
    opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
    estimator = OrdinaryLeastSquares(
        model=model_ols, moment_function=iv_moment_function, optimization=opt_config
    )
    estimator.train(train_data)
    results["OLS (Biased)"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "OLS (Biased)")

    # =========================================================================
    # 2. GMM
    # =========================================================================
    print("\n[2/11] Training GMM...")
    model_gmm = get_neural_network()
    opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
    estimator = GMM(
        model=model_gmm,
        moment_function=iv_moment_function,
        optimization=opt_config,
        reg_param=1e-4,
    )
    estimator.train(train_data)
    results["GMM (Unconditional)"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "GMM (Unconditional)")

    # =========================================================================
    # 3. GENERALIZED EL
    # =========================================================================
    print("\n[3/11] Training GEL...")
    model_gel = get_neural_network()
    opt_config = OptimizationConfig(
        optimizer="lbfgs", learning_rate=5e-4, max_epochs=10
    )
    estimator = GeneralizedEL(
        model=model_gel,
        moment_function=iv_moment_function,
        optimization=opt_config,
        divergence="chi2",
        reg_param=1e-6,
    )
    estimator.train(train_data)
    results["GEL (Unconditional)"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "GEL (Unconditional)")

    # =========================================================================
    # 4. KMM (Unconditional)
    # =========================================================================
    # print("\n[4/11] Training KMM (unconditional)...")
    # model_kmm = get_neural_network()
    # opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
    # estimator = KMM(
    #     model=model_kmm,
    #     moment_function=iv_moment_function,
    #     optimization=opt_config,
    #     entropy_reg_param=10.0,
    #     n_random_features=1000,
    #     verbose=False,
    # )
    # estimator.train(train_data)
    # results["KMM (Unconditional)"] = estimator.model
    # evaluate_estimator(estimator.model, test_data, "KMM (Unconditional)")

    # =========================================================================
    # 5. MMR
    # =========================================================================
    print("\n[5/11] Training MMR...")
    model_mmr = get_neural_network()
    opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
    estimator = MMR(
        model=model_mmr,
        moment_function=iv_moment_function,
        optimization=opt_config,
        verbose=False,
    )
    estimator.train(train_data)
    results["MMR"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "MMR")

    # =========================================================================
    # 6. SIEVE MINIMUM DISTANCE
    # =========================================================================
    print("\n[6/11] Training SMD...")
    model_smd = get_neural_network()
    opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
    estimator = SMDIdentity(
        model=model_smd, moment_function=iv_moment_function, optimization=opt_config
    )
    estimator.train(train_data)
    results["SMD"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "SMD")

    # =========================================================================
    # 7. VMM-KERNEL
    # =========================================================================
    print("\n[7/11] Training VMM-kernel...")
    model_vmm_k = get_neural_network()
    opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
    estimator = KernelVMM(
        model=model_vmm_k,
        moment_function=iv_moment_function,
        optimization=opt_config,
        divergence="chi2",
        reg_param=1e-4,
        verbose=False,
    )
    estimator.train(train_data)
    results["VMM-kernel"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "VMM-kernel")

    # =========================================================================
    # 8. VMM-NEURAL
    # =========================================================================
    print("\n[8/11] Training VMM-neural...")
    model_vmm_n = get_neural_network()
    opt_config = OptimizationConfig(
        optimizer="oadam_gda", learning_rate=5e-4, max_epochs=1000, batch_size=200
    )
    estimator = NeuralVMM(
        model=model_vmm_n,
        moment_function=iv_moment_function,
        optimization=opt_config,
        divergence="chi2",
        reg_param=1e-2,
    )
    estimator.train(train_data, val_data)
    results["VMM-neural"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "VMM-neural")

    # =========================================================================
    # 9. FGEL-KERNEL
    # =========================================================================
    print("\n[9/11] Training FGEL-kernel...")
    model_fgel_k = get_neural_network()
    opt_config = OptimizationConfig(optimizer="lbfgs", max_epochs=10)
    estimator = KernelFGEL(
        model=model_fgel_k,
        moment_function=iv_moment_function,
        optimization=opt_config,
        divergence="chi2",
        reg_param=1e-3,
        verbose=False,
    )
    estimator.train(train_data)
    results["FGEL-kernel"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "FGEL-kernel")

    # =========================================================================
    # 10. FGEL-NEURAL
    # =========================================================================
    print("\n[10/11] Training FGEL-neural...")
    model_fgel_n = get_neural_network()

    # Using explicit Config objects
    opt_config = OptimizationConfig(
        optimizer="oadam_gda", learning_rate=5e-4, max_epochs=2000, batch_size=200
    )
    fgel_config = FGELConfig(divergence="chi2", regularization=1e-2)

    estimator = NeuralFGEL(
        model=model_fgel_n,
        moment_function=iv_moment_function,
        optimization=opt_config,
        fgel_config=fgel_config,
        verbose=False,
    )
    estimator.train(train_data, val_data)
    results["FGEL-neural"] = estimator.model
    evaluate_estimator(estimator.model, test_data, "FGEL-neural")

    # =========================================================================
    # 11. KMM-NEURAL
    # =========================================================================
    print("\n[11/11] Training KMM-neural...")
    model_kmm_n = get_neural_network()

    # Using explicit Config objects
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
        model=model_kmm_n,
        moment_function=iv_moment_function,
        optimization=opt_config,
        kmm_config=kmm_config,
        dual_network=dual_network,
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
    print("\nGenerating comparison plot...")
    plot_results(results, test_data)
    print("\nDemo complete. All methods use direct class instantiation (no factory).")


if __name__ == "__main__":
    main()
