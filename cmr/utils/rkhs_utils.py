import numpy as np
import scipy.linalg
import torch


def calc_sq_dist(x, y, numpy=False):
    """
    Calculate the pairwise squared euclidean distance between two tensor batches.

    Parameters
    ----------
    x: tensor of shape (n, d)
    y: tensor of shape (m, d)
    numpy: bool
        If true, returns numpy array, else torch tensor

    Returns
    -------
    sq_dist: tensor of shape (n, m)
    """
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]

    if numpy:
        # Numpy implementation
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        x = np.expand_dims(x, 1)  # (n, 1, d)
        y = np.expand_dims(y, 0)  # (1, m, d)
        sq_dist = np.sum((x - y) ** 2, axis=2)
    else:
        # Torch implementation
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        # Ensure devices match if one is on GPU
        if x.is_cuda and not y.is_cuda:
            y = y.to(x.device)
        if y.is_cuda and not x.is_cuda:
            x = x.to(y.device)

        x = torch.unsqueeze(x, 1)
        y = torch.unsqueeze(y, 0)
        sq_dist = torch.sum((x - y) ** 2, dim=2)
    return sq_dist


def get_rbf_kernel(x, y, bandwidth=None, **kwargs):
    """
    Calculate the RBF kernel matrix between two tensor batches.

    Parameters
    ----------
    x: tensor of shape (n, d)
    y: tensor of shape (m, d)
    bandwidth: float
        If None, use the median heuristic

    Returns
    -------
    kernel: tensor of shape (n, m)
    bandwidth: float
    """
    # Check if inputs are numpy or torch
    is_numpy = isinstance(x, np.ndarray) or (isinstance(y, np.ndarray))

    if is_numpy:
        sq_dist = calc_sq_dist(x, y, numpy=True)
        if bandwidth is None:
            bandwidth = np.sqrt(0.5 * np.median(sq_dist))
        kernel = np.exp(-sq_dist / (2 * bandwidth**2))
    else:
        sq_dist = calc_sq_dist(x, y, numpy=False)
        if bandwidth is None:
            bandwidth = torch.sqrt(0.5 * torch.median(sq_dist))
        kernel = torch.exp(-sq_dist / (2 * bandwidth**2))
    return kernel, bandwidth


def compute_cholesky_factor(kernel_matrix, jitter=1e-6):
    """
    Compute the Cholesky factor of a kernel matrix.

    Parameters
    ----------
    kernel_matrix: numpy array of shape (n, n)
    jitter: float

    Returns
    -------
    L: numpy array of shape (n, n)
    """
    # Ensure numpy
    if isinstance(kernel_matrix, torch.Tensor):
        kernel_matrix = kernel_matrix.detach().cpu().numpy()

    try:
        L = scipy.linalg.cholesky(kernel_matrix, lower=True)
    except scipy.linalg.LinAlgError:
        return compute_cholesky_factor(
            kernel_matrix + jitter * np.eye(kernel_matrix.shape[0]), jitter * 10
        )
    return L


def hsic(kernel_matrix_x, kernel_matrix_y):
    """
    Calculate the Hilbert-Schmidt Independence Criterion (HSIC) between two kernel matrices.

    Parameters
    ----------
    kernel_matrix_x: tensor of shape (n, n)
    kernel_matrix_y: tensor of shape (n, n)

    Returns
    -------
    hsic: float
    """
    n = kernel_matrix_x.shape[0]
    device = kernel_matrix_x.device

    # Create centering matrix on the correct device
    centering_matrix = (
        torch.eye(n, device=device) - torch.ones((n, n), device=device) / n
    )

    # Ensure kernel_matrix_y is on the same device
    if kernel_matrix_y.device != device:
        kernel_matrix_y = kernel_matrix_y.to(device)

    hsic_val = torch.trace(
        torch.mm(
            torch.mm(centering_matrix, kernel_matrix_x),
            torch.mm(centering_matrix, kernel_matrix_y),
        )
    ) / ((n - 1) ** 2)
    return hsic_val
