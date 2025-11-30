import torch


class AbstractExperiment:
    def __init__(self, dim_psi, dim_theta, dim_z):
        self.dim_psi = dim_psi
        self.dim_theta = dim_theta
        self.dim_z = dim_z
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def get_true_parameters(self):
        """If method not specified with signature `get_true_parameters(self) -> np.array` then assume there are no true
        parameters and the model we want to train is non-parametric or a NN"""
        return None

    def generate_data(self, num_data, **kwargs):
        raise NotImplementedError

    def prepare_dataset(self, n_train, n_val=None, n_test=None):
        self.train_data = self.generate_data(n_train)
        self.val_data = self.generate_data(n_val)
        self.test_data = self.generate_data(n_test)

    def eval_risk(self, model, data):
        # Default implementation (can be overridden)
        y_test = np_to_tensor(data["y"])
        t_test = np_to_tensor(data["t"])

        # Handle device if model is on GPU
        if next(model.parameters()).is_cuda:
            t_test = t_test.to(next(model.parameters()).device)

        y_pred = model(t_test)

        # Move back to CPU for comparison
        y_pred = y_pred.detach().cpu()

        return float(torch.mean((y_test - y_pred) ** 2))

    def get_model(self):
        raise NotImplementedError
