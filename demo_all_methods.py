"""
Demonstration of all CMR estimators on a nonparametric IV problem.

This script demonstrates:
1. Why naive OLS fails (endogeneity bias)
2. Why linear IV/GMM fails (model misspecification)
3. How each nonparametric method performs

DGP: Nonparametric IV with unobserved confounding
- True structural function: y = 2*log(x)
- Unobserved confounder U affects both X and Y
- Instrument Z affects X but not Y directly
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Import all estimator classes
from cmr.methods.least_squares import OrdinaryLeastSquares
from cmr.methods.gmm import GMM
from cmr.methods.generalized_el import GeneralizedEL
from cmr.methods.kmm import KMM
from cmr.methods.mmr import MMR
from cmr.methods.sieve_minimum_distance import SMDIdentity
from cmr.methods.vmm_kernel import KernelVMM
from cmr.methods.vmm_neural import NeuralVMM
from cmr.methods.fgel_kernel import KernelFGEL
from cmr.methods.fgel_neural import NeuralFGEL
from cmr.methods.kmm_neural import KMMNeural


def generate_nonparametric_iv_data(n: int, seed: int = None) -> Dict[str, np.ndarray]:
    """
    Generate data from nonparametric IV model with confounding.

    Structural equations:
        U ~ N(0, 1)                    [unobserved confounder]
        Z ~ Uniform(-3, 3)             [instrument]
        X = Z + U + epsilon_x          [treatment, confounded]
        Y = 2*log(|X| + 0.5) + U + epsilon_y  [outcome, nonlinear]

    Key features:
    - True function is y = 2*log(|x| + 0.5)
    - U confounds X and Y (endogeneity)
    - Z is a valid instrument (affects X, not Y directly)
    - Nonlinear, so linear IV is misspecified
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

    return {'t': X, 'y': Y, 'z': Z, 'u': U}


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
        torch.nn.Linear(20, output_dim)
    )


def iv_moment_function(model_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard IV moment function: psi(x,y;theta) = f_theta(x) - y"""
    return model_pred - y


def evaluate_estimator(
    model: torch.nn.Module,
    test_data: Dict[str, np.ndarray],
    name: str
) -> Tuple[float, float]:
    """
    Evaluate estimator on test data.

    Returns:
        mse: Mean squared error
        bias: Average bias in predictions
    """
    with torch.no_grad():
        X_test = torch.Tensor(test_data['t'])
        Y_test = test_data['y']
        Y_true = true_structural_function(test_data['t'])

        Y_pred = model(X_test).detach().numpy()

        mse = np.mean((Y_pred - Y_true)**2)
        bias = np.mean(Y_pred - Y_true)

        print(f"{name:30s} | MSE: {mse:8.4f} | Bias: {bias:8.4f}")
        return mse, bias


def plot_results(results: Dict[str, torch.nn.Module], test_data: Dict[str, np.ndarray]):
    """Plot predictions from all methods."""
    X_test = test_data['t']
    Y_true = true_structural_function(X_test)

    # Sort for plotting
    sort_idx = np.argsort(X_test.flatten())
    X_sorted = X_test[sort_idx]
    Y_sorted = Y_true[sort_idx]

    plt.figure(figsize=(14, 10))

    # Plot true function
    plt.scatter(test_data['t'], test_data['y'], alpha=0.3, s=10, label='Data', color='gray')
    plt.plot(X_sorted, Y_sorted, 'k-', linewidth=3, label='True function', zorder=100)

    # Plot predictions from each method
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    for (name, model), color in zip(results.items(), colors):
        with torch.no_grad():
            Y_pred = model(torch.Tensor(X_sorted)).detach().numpy()
        plt.plot(X_sorted, Y_pred, '--', linewidth=2, label=name, color=color, alpha=0.8)

    plt.xlabel('X (Treatment)', fontsize=12)
    plt.ylabel('Y (Outcome)', fontsize=12)
    plt.title('Nonparametric IV: All Methods Comparison\nTrue function: y = 2*log(|x| + 0.5)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('all_methods_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'all_methods_comparison.png'")


def main():
    print("=" * 80)
    print("NONPARAMETRIC IV ESTIMATION: ALL METHODS DEMO")
    print("=" * 80)
    print("\nDGP: Y = 2*log(|X| + 0.5) + U + noise")
    print("     X = Z + U + noise")
    print("     U ~ N(0,1) is unobserved confounder")
    print("     Z ~ Uniform(-3, 3) is instrument")
    print("\n" + "=" * 80)

    # Generate data
    print("\nGenerating data...")
    train_data = generate_nonparametric_iv_data(n=500, seed=42)
    val_data = generate_nonparametric_iv_data(n=500, seed=43)
    test_data = generate_nonparametric_iv_data(n=2000, seed=44)

    # Store results
    results = {}

    print("\n" + "=" * 80)
    print("TRAINING ESTIMATORS")
    print("=" * 80)
    print(f"\n{'Method':<30s} | {'MSE':<8s} | {'Bias':<8s}")
    print("-" * 55)

    # =========================================================================
    # 1. ORDINARY LEAST SQUARES (will fail due to endogeneity)
    # =========================================================================
    print("\n[1/11] Training OLS (will be biased due to endogeneity)...")
    model_ols = get_neural_network()
    estimator = OrdinaryLeastSquares(
        model=model_ols,
        moment_function=iv_moment_function
    )
    estimator.train(train_data)
    results['OLS (Biased)'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'OLS (Biased)')

    # =========================================================================
    # 2. GMM (unconditional, will fail due to endogeneity)
    # =========================================================================
    print("\n[2/11] Training GMM (unconditional, will be biased)...")
    model_gmm = get_neural_network()
    estimator = GMM(
        model=model_gmm,
        moment_function=iv_moment_function,
        reg_param=1e-4,
        num_iter=2,
        pretrain=True,
        verbose=False
    )
    estimator.train(train_data)
    results['GMM (Unconditional)'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'GMM (Unconditional)')

    # =========================================================================
    # 3. GENERALIZED EL (unconditional, will fail)
    # =========================================================================
    print("\n[3/11] Training GEL (unconditional, will be biased)...")
    model_gel = get_neural_network()
    estimator = GeneralizedEL(
        model=model_gel,
        moment_function=iv_moment_function,
        divergence='chi2',
        reg_param=1e-6,
        pretrain=True,
        theta_optim_args={'optimizer': 'lbfgs', 'lr': 5e-4},
        dual_optim_args={'optimizer': 'lbfgs', 'lr': 5e-4, 'inneriters': 100},
        max_num_epochs=10,
        batch_size=None,
        eval_freq=5,
        max_no_improve=3,
        burn_in_cycles=2,
        verbose=False
    )
    estimator.train(train_data)
    results['GEL (Unconditional)'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'GEL (Unconditional)')

    # =========================================================================
    # 4. KMM (unconditional, will fail)
    # =========================================================================
    print("\n[4/11] Training KMM (unconditional, will be biased)...")
    model_kmm = get_neural_network()
    estimator = KMM(
        model=model_kmm,
        moment_function=iv_moment_function,
        entropy_reg_param=10.0,
        n_random_features=1000,
        n_reference_samples=100,
        kde_bandwidth=0.3,
        pretrain=True,
        verbose=False
    )
    estimator.train(train_data)
    results['KMM (Unconditional)'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'KMM (Unconditional)')

    # =========================================================================
    # 5. MMR (conditional, uses instruments)
    # =========================================================================
    print("\n[5/11] Training MMR (conditional, uses instruments)...")
    model_mmr = get_neural_network()
    estimator = MMR(
        model=model_mmr,
        moment_function=iv_moment_function,
        kernel_z_kwargs={},
        verbose=False
    )
    estimator.train(train_data)
    results['MMR'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'MMR')

    # =========================================================================
    # 6. SIEVE MINIMUM DISTANCE (conditional)
    # =========================================================================
    print("\n[6/11] Training Sieve Minimum Distance...")
    model_smd = get_neural_network()
    estimator = SMDIdentity(
        model=model_smd,
        moment_function=iv_moment_function,
        verbose=False
    )
    estimator.train(train_data)
    results['SMD'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'SMD')

    # =========================================================================
    # 7. VMM-KERNEL (conditional)
    # =========================================================================
    print("\n[7/11] Training VMM-kernel...")
    model_vmm_k = get_neural_network()
    estimator = KernelVMM(
        model=model_vmm_k,
        moment_function=iv_moment_function,
        reg_param=1e-4,
        num_iter=2,
        kernel_z_kwargs={},
        verbose=False
    )
    estimator.train(train_data)
    results['VMM-kernel'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'VMM-kernel')

    # =========================================================================
    # 8. VMM-NEURAL (conditional)
    # =========================================================================
    print("\n[8/11] Training VMM-neural...")
    model_vmm_n = get_neural_network()
    estimator = NeuralVMM(
        model=model_vmm_n,
        moment_function=iv_moment_function,
        divergence='chi2',
        reg_param=1e-2,
        reg_param_rkhs_norm=0.0,
        dual_func_network_kwargs={'layer_widths': [30, 20]},
        pretrain=True,
        theta_optim_args={'optimizer': 'oadam_gda', 'lr': 5e-4},
        dual_optim_args={'optimizer': 'oadam_gda', 'lr': 5e-4},
        max_num_epochs=3000,
        batch_size=200,
        eval_freq=100,
        max_no_improve=5,
        burn_in_cycles=5,
        verbose=False
    )
    estimator.train(train_data, val_data)
    results['VMM-neural'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'VMM-neural')

    # =========================================================================
    # 9. FGEL-KERNEL (conditional)
    # =========================================================================
    print("\n[9/11] Training FGEL-kernel...")
    model_fgel_k = get_neural_network()
    estimator = KernelFGEL(
        model=model_fgel_k,
        moment_function=iv_moment_function,
        divergence='chi2',
        reg_param=1e-3,
        kernel_z_kwargs={},
        pretrain=True,
        theta_optim_args={'optimizer': 'lbfgs', 'lr': 5e-4},
        dual_optim_args={'optimizer': 'lbfgs', 'lr': 5e-4},
        max_num_epochs=10,
        batch_size=None,
        eval_freq=5,
        max_no_improve=3,
        burn_in_cycles=2,
        verbose=False
    )
    estimator.train(train_data)
    results['FGEL-kernel'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'FGEL-kernel')

    # =========================================================================
    # 10. FGEL-NEURAL (conditional)
    # =========================================================================
    print("\n[10/11] Training FGEL-neural...")
    model_fgel_n = get_neural_network()
    estimator = NeuralFGEL(
        model=model_fgel_n,
        moment_function=iv_moment_function,
        divergence='chi2',
        reg_param=1e-2,
        dual_func_network_kwargs={'layer_widths': [30, 20]},
        pretrain=True,
        theta_optim_args={'optimizer': 'oadam_gda', 'lr': 5e-4},
        dual_optim_args={'optimizer': 'oadam_gda', 'lr': 5e-4},
        max_num_epochs=3000,
        batch_size=200,
        eval_freq=100,
        max_no_improve=5,
        burn_in_cycles=5,
        verbose=False
    )
    estimator.train(train_data, val_data)
    results['FGEL-neural'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'FGEL-neural')

    # =========================================================================
    # 11. KMM-NEURAL (conditional, flagship method)
    # =========================================================================
    print("\n[11/11] Training KMM-neural (flagship method)...")
    model_kmm_n = get_neural_network()
    estimator = KMMNeural(
        model=model_kmm_n,
        moment_function=iv_moment_function,
        divergence='kl',
        entropy_reg_param=10.0,
        reg_param=1e-2,
        rkhs_reg_param=1.0,
        n_random_features=2000,
        n_reference_samples=200,
        kde_bandwidth=0.3,
        kernel_x_kwargs={},
        kernel_z_kwargs={},
        dual_func_network_kwargs={'layer_widths': [30, 20]},
        pretrain=True,
        rkhs_func_z_dependent=True,
        theta_optim_args={'optimizer': 'oadam_gda', 'lr': 5e-4},
        dual_optim_args={'optimizer': 'oadam_gda', 'lr': 1e-4},
        max_num_epochs=3000,
        batch_size=200,
        eval_freq=100,
        max_no_improve=5,
        burn_in_cycles=5,
        verbose=False
    )
    estimator.train(train_data, val_data)
    results['KMM-neural'] = estimator.model
    evaluate_estimator(estimator.model, test_data, 'KMM-neural')

    # =========================================================================
    # SUMMARY AND VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey Observations:")
    print("1. OLS, GMM, GEL, KMM (unconditional) are all biased due to endogeneity")
    print("2. Methods using instruments (MMR, SMD, VMM, FGEL, KMM-neural) can")
    print("   recover the true nonlinear function")
    print("3. Neural versions (VMM-neural, FGEL-neural, KMM-neural) are most")
    print("   flexible for complex nonlinear functions")

    # Create visualization
    print("\nGenerating comparison plot...")
    plot_results(results, test_data)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)

    # Commentary on API issues
    print("\n" + "=" * 80)
    print("API ISSUES DEMONSTRATED")
    print("=" * 80)
    print("""
The current API has several confusing aspects:

1. INCONSISTENT PARAMETERS: Each estimator has different required kwargs
   - Some need 'divergence', others don't
   - Some need 'entropy_reg_param', others need 'reg_param'
   - Some need 'dual_func_network_kwargs', others don't

2. DIFFERENT OPTIMIZATION SPECS:
   - Kernel methods use LBFGS typically
   - Neural methods use GDA (gradient descent-ascent)
   - Each has different optim_args structures

3. UNCLEAR DEFAULTS:
   - Some methods need 'pretrain=True' to work well
   - Batch size should be None for kernel methods, but set for neural
   - Early stopping params (max_no_improve, burn_in_cycles) vary by method

4. CONDITIONAL vs UNCONDITIONAL:
   - Not clear from class names which methods support instruments
   - Same moment function works for both, but behavior differs drastically

5. HYPERPARAMETER SENSITIVITY:
   - Neural methods very sensitive to learning rates
   - Kernel methods sensitive to regularization
   - Reference sample sizes matter a lot for KMM

SUGGESTED IMPROVEMENTS:
- Unified base class with clearer method categorization
- Sensible defaults that work for each method type
- Clear documentation of which parameters matter for each method
- Helper functions to auto-configure based on data size and problem type
- Better error messages when using incompatible parameters
    """)


if __name__ == "__main__":
    main()
