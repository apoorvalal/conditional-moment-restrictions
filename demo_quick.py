"""
Quick demo of API issues with a few representative methods.
"""

import numpy as np
import torch

from cmr.methods.least_squares import OrdinaryLeastSquares
from cmr.methods.mmr import MMR
from cmr.methods.fgel_neural import NeuralFGEL
from cmr.methods.kmm_neural import KMMNeural


def generate_data(n: int, seed: int = None):
    """Generate nonparametric IV data: y = 2*log(|x| + 0.5)"""
    if seed:
        np.random.seed(seed)
    U = np.random.normal(0, 1.0, size=(n, 1))
    Z = np.random.uniform(-3, 3, size=(n, 1))
    X = Z + U + np.random.normal(0, 0.1, size=(n, 1))
    Y = 2 * np.log(np.abs(X) + 0.5) + U + np.random.normal(0, 0.1, size=(n, 1))
    return {'t': X, 'y': Y, 'z': Z}


def get_model():
    return torch.nn.Sequential(
        torch.nn.Linear(1, 30),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(30, 20),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(20, 1)
    )


def moment_fn(pred, y):
    return pred - y


def evaluate(model, test_data):
    Y_true = 2 * np.log(np.abs(test_data['t']) + 0.5)
    Y_pred = model(torch.Tensor(test_data['t'])).detach().numpy()
    mse = np.mean((Y_pred - Y_true)**2)
    return mse


print("=" * 80)
print("QUICK DEMO: API Issues in Conditional Moment Restrictions Library")
print("=" * 80)

# Check GPU
if torch.cuda.is_available():
    print(f"\nGPU Available: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("BUT: Library has BROKEN GPU support (tensor.cpu() missing in multiple places)!")
    print("AND: Even if it worked, you'd need to pass gpu=True to EVERY estimator!")
    print("Running on CPU for now...")
    USE_GPU = False
else:
    print("\nNo GPU available, using CPU")
    USE_GPU = False

train_data = generate_data(200, seed=42)
test_data = generate_data(1000, seed=43)

print("\n1. OLS (Biased - no instruments used)")
print("-" * 80)
model1 = get_model()
est = OrdinaryLeastSquares(model=model1, moment_function=moment_fn, gpu=USE_GPU)
print("   Parameters needed: model, moment_function (+ gpu=True for GPU!)")
print("   Training...")
est.train(train_data)
mse1 = evaluate(est.model, test_data)
print(f"   MSE: {mse1:.4f} (BIASED due to endogeneity)")

print("\n2. MMR (Simple conditional method)")
print("-" * 80)
model2 = get_model()
est = MMR(model=model2, moment_function=moment_fn, kernel_z_kwargs={}, gpu=USE_GPU)
print("   Parameters needed: model, moment_function, kernel_z_kwargs (+ gpu=True)")
print("   Training...")
est.train(train_data)
mse2 = evaluate(est.model, test_data)
print(f"   MSE: {mse2:.4f}")

print("\n3. FGEL-neural (Complex conditional method)")
print("-" * 80)
model3 = get_model()
print("   Parameters needed:")
print("     - model, moment_function")
print("     - divergence (str): type of f-divergence")
print("     - reg_param (float): regularization")
print("     - dual_func_network_kwargs (dict): dual network architecture")
print("     - pretrain (bool): whether to pretrain")
print("     - theta_optim_args (dict): optimizer config for model")
print("     - dual_optim_args (dict): optimizer config for dual")
print("     - max_num_epochs, batch_size, eval_freq, max_no_improve, burn_in_cycles")
print("   Training...")
est = NeuralFGEL(
    model=model3,
    moment_function=moment_fn,
    divergence='chi2',
    reg_param=1e-2,
    dual_func_network_kwargs={'layer_widths': [20, 10]},
    pretrain=True,
    theta_optim_args={'optimizer': 'oadam_gda', 'lr': 5e-4},
    dual_optim_args={'optimizer': 'oadam_gda', 'lr': 5e-4},
    max_num_epochs=500,
    batch_size=100,
    eval_freq=50,
    max_no_improve=3,
    burn_in_cycles=2,
    gpu=USE_GPU,
    verbose=False
)
est.train(train_data, train_data)
mse3 = evaluate(est.model, test_data)
print(f"   MSE: {mse3:.4f}")

print("\n4. KMM-neural (Flagship method, even more complex)")
print("-" * 80)
model4 = get_model()
print("   Parameters needed:")
print("     - All the FGEL parameters PLUS:")
print("     - entropy_reg_param (float): entropy regularization")
print("     - rkhs_reg_param (float): RKHS regularization")
print("     - n_random_features (int): RFF dimension")
print("     - n_reference_samples (int): reference distribution samples")
print("     - kde_bandwidth (float): KDE bandwidth")
print("     - kernel_x_kwargs (dict): kernel for X,Y,Z space")
print("     - kernel_z_kwargs (dict): kernel for Z space")
print("     - rkhs_func_z_dependent (bool): whether RKHS function depends on Z")
print("   Training...")
est = KMMNeural(
    model=model4,
    moment_function=moment_fn,
    divergence='kl',
    entropy_reg_param=10.0,
    reg_param=1e-2,
    rkhs_reg_param=1.0,
    n_random_features=500,
    n_reference_samples=50,
    kde_bandwidth=0.3,
    kernel_x_kwargs={},
    kernel_z_kwargs={},
    dual_func_network_kwargs={'layer_widths': [20, 10]},
    pretrain=True,
    rkhs_func_z_dependent=True,
    theta_optim_args={'optimizer': 'oadam_gda', 'lr': 5e-4},
    dual_optim_args={'optimizer': 'oadam_gda', 'lr': 1e-4},
    max_num_epochs=500,
    batch_size=100,
    eval_freq=50,
    max_no_improve=3,
    burn_in_cycles=2,
    gpu=USE_GPU,
    verbose=False
)
est.train(train_data, train_data)
mse4 = evaluate(est.model, test_data)
print(f"   MSE: {mse4:.4f}")

print("\n" + "=" * 80)
print("SUMMARY OF API ISSUES")
print("=" * 80)
print("""
1. PARAMETER EXPLOSION: Methods go from 2 params (OLS) to 20+ params (KMM-neural)

2. NO CLEAR ORGANIZATION:
   - Some params are method-specific (entropy_reg_param, rkhs_reg_param)
   - Some are shared but named differently (reg_param means different things)
   - Some are optimization-related (theta_optim_args, dual_optim_args)
   - Some are algorithmic (n_random_features, n_reference_samples)

3. INCONSISTENT DEFAULTS:
   - Some methods work with batch_size=None, others need it set
   - Some need pretrain=True, others don't
   - Learning rates vary wildly between methods

4. NESTED DICTS:
   - kernel_x_kwargs, kernel_z_kwargs, dual_func_network_kwargs
   - theta_optim_args, dual_optim_args
   - Hard to know what goes in each dict

5. NO TYPE HINTS OR VALIDATION:
   - Easy to pass wrong types
   - No helpful error messages

6. METHOD SELECTION UNCLEAR:
   - Not obvious which method to use for a given problem
   - No guidance on hyperparameter selection

7. GPU SUPPORT IS BROKEN:
   - Multiple places missing .cpu() before .numpy() conversion
   - Results in RuntimeError when gpu=True is used
   - Even if it worked, gpu=True must be passed to EVERY estimator
   - Should auto-detect and use GPU by default
   - No warning if GPU is available but not being used
""")

print("\nResults:")
print(f"OLS (biased):      MSE = {mse1:.4f}")
print(f"MMR:               MSE = {mse2:.4f}")
print(f"FGEL-neural:       MSE = {mse3:.4f}")
print(f"KMM-neural:        MSE = {mse4:.4f}")
print("\nAll conditional methods (MMR, FGEL, KMM) should outperform biased OLS.")
