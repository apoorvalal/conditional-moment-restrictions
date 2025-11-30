import argparse
import numpy as np
import torch
import sys
import os

# Ensure cmr is in path
sys.path.append(os.getcwd())

from cmr.factory import create_estimator
from cmr.config import OptimizationConfig

# Import experiments
from experiments.exp_heteroskedastic import HeteroskedasticNoiseExperiment
# Add others as needed


def run_experiment(experiment_name, method, n_train, n_runs, gpu):
    print(f"Running experiment: {experiment_name} with method: {method}")

    # Initialize experiment
    if experiment_name == "heteroskedastic":
        exp = HeteroskedasticNoiseExperiment(
            theta=[1.7], noise=1.0, heteroskedastic=True
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    results = {"mse": [], "test_risk": []}

    for i in range(n_runs):
        print(f"\nRun {i + 1}/{n_runs}")

        # Prepare data
        # Note: exp.prepare_dataset generates random data each time usually
        exp.prepare_dataset(n_train=n_train, n_val=min(n_train, 1000), n_test=10000)

        model = exp.get_model()

        # Create estimator with new API
        try:
            device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
            estimator = create_estimator(
                method=method,
                model=model,
                moment_function=exp.moment_function,
                train_data_size=n_train,
                device=device,
                verbose=True,
                # Optional overrides
                max_epochs=200,
                batch_size=128 if n_train > 500 else None,
            )

            # Train
            estimator.train(exp.train_data, exp.val_data)

            # Evaluate
            trained_model = estimator.model

            # MSE of parameters (specific to this experiment)
            theta_pred = np.squeeze(trained_model.get_parameters())
            mse = np.mean(np.square(theta_pred - exp.theta0))
            results["mse"].append(mse)

            # Test risk
            # Note: eval_risk might need model on CPU? AbstractExperiment usually expects CPU tensors/numpy
            # But trained_model might be on GPU if we didn't move it back.
            # Wrapper handles forward on correct device, but eval_risk logic in experiment might assume something.
            # exp_heteroskedastic.eval_risk uses:
            # y_test = np_to_tensor(data['y']) (CPU)
            # y_pred = model.forward(np_to_tensor(data['t']))
            # If model is on GPU, input must be on GPU.
            # The ModelWrapper I modified handles this check in forward!
            # So it should be fine.

            test_risk = exp.eval_risk(trained_model, exp.test_data)
            results["test_risk"].append(test_risk)

            print(f"Run {i + 1} MSE: {mse:.4f}, Risk: {test_risk:.4f}")

        except Exception as e:
            print(f"Run {i + 1} failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\nSummary:")
    print(f"Mean MSE: {np.mean(results['mse']):.4f} +/- {np.std(results['mse']):.4f}")
    print(
        f"Mean Risk: {np.mean(results['test_risk']):.4f} +/- {np.std(results['test_risk']):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CMR experiments with new API")
    parser.add_argument(
        "--experiment", type=str, default="heteroskedastic", help="Experiment name"
    )
    parser.add_argument(
        "--method", type=str, default="KMM-neural", help="Estimation method"
    )
    parser.add_argument(
        "--n_train", type=int, default=200, help="Number of training samples"
    )
    parser.add_argument("--n_runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")

    args = parser.parse_args()
    run_experiment(args.experiment, args.method, args.n_train, args.n_runs, args.gpu)
