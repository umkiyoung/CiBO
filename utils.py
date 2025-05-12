import random
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os

def set_seed(seed, torch_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    
def adaptive_temp_v2(scores_np, q=None):
    """Calculate an adaptive temperature value based on the
    statistics of the scores array

    Args:

    scores_np: np.ndarray
        an array that represents the vectorized scores per data point

    Returns:

    temp: np.ndarray
        the scalar 90th percentile of scores in the dataset
    """

    inverse_arr = scores_np
    max_score = inverse_arr.max()
    scores_new = inverse_arr - max_score
    if q is None:
        quantile_ninety = np.quantile(scores_new, q=0.9)
    else:
        quantile_ninety = np.quantile(scores_new, q=q)
    return np.maximum(np.abs(quantile_ninety), 0.001)

def softmax(arr, temp=1.0):
    """Calculate the softmax using numpy by normalizing a vector
    to have entries that sum to one

    Args:

    arr: np.ndarray
        the array which will be normalized using a tempered softmax
    temp: float
        a temperature parameter for the softmax

    Returns:

    normalized: np.ndarray
        the normalized input array which sums to one
    """

    max_arr = arr.max()
    arr_new = arr - max_arr
    exp_arr = np.exp(arr_new / temp)
    return exp_arr / np.sum(exp_arr)

def get_value_based_weights(scores, temp="90"):
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective

    Args:

    scores: np.ndarray
        scores which correspond to the value of data points in the dataset

    Returns:

    weights: np.ndarray
        an array with the same shape as scores that reweights samples
    """

    scores_np = scores[:, 0]
    hist, bin_edges = np.histogram(scores_np, bins=20)
    hist = hist / np.sum(hist)

    # if base_temp is None:
    if temp == "90":
        base_temp = adaptive_temp_v2(scores_np, q=0.9)
    elif temp == "75":
        base_temp = adaptive_temp_v2(scores_np, q=0.75)
    elif temp == "50":
        base_temp = adaptive_temp_v2(scores_np, q=0.5)
    else:
        raise RuntimeError("Invalid temperature")
    softmin_prob = softmax(bin_edges[1:], temp=base_temp)

    provable_dist = softmin_prob * (hist / (hist + 1e-3))
    provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)
    # print(provable_dist)

    bin_indices = np.digitize(scores_np, bin_edges[1:])
    hist_prob = hist[np.minimum(bin_indices, 19)]

    weights = provable_dist[np.minimum(bin_indices, 19)] / (hist_prob + 1e-7)
    weights = np.clip(weights, a_min=0.0, a_max=5.0)
    return weights.astype(np.float64)[:, np.newaxis]

def get_rank_based_weights(scores):
    ranks = np.argsort(np.argsort(-scores))
    weights = 1.0 / (1e-2 * len(scores) + ranks)
    return weights

def get_max_value_per_rounds(Total_Y, n_init, max_evals, batch_size):
    max_values = []
    initial_max = Total_Y[:n_init].max().item()
    max_values.append(initial_max)
    num_rounds = (max_evals - n_init) // batch_size
    for round in range(num_rounds):
        max_values.append(Total_Y[:n_init + (round + 1) * batch_size].max().item())
    return max_values


def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X

def sample_visual(flow, sample_size, dim, step_size):
    """
    File from the flow matching tutorial codes.
    Visualizes the evolution of samples through a flow model over time steps.
    Parameters:
        flow (object): A flow model object that implements a `step` method to transform samples.
        sample_size (int): The number of samples to generate and visualize.
        dim (int): The dimensionality of the samples.
        step_size (int): The number of time steps for the flow transformation.
    Returns:
        None: Displays a matplotlib figure showing the progression of samples at each time step.
    """

    x = torch.randn(sample_size, dim)
    fig, axes = plt.subplots(1, step_size + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, step_size + 1)

    axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
    axes[0].set_title(f't = {time_steps[0]:.2f}')
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)

    for i in range(step_size):
        x = flow.step(x, time_steps[i], time_steps[i + 1])
        axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
        axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

    plt.tight_layout()
    plt.show()
    
def save_numpy_array(path, array, file_name):
    string = path.split('/')
    path_parent = '/'.join(string[:-1])
    if not os.path.exists(path_parent):
        os.makedirs(path_parent)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, file_name), array)
    print(f"Array saved to '{os.path.join(path, file_name)}'.")
    
def get_lagrangian(pred, constraints, lamb=None, indicator=None):
    """Calculate the Lagrangian value based on the predicted values and constraints.""" 
    if indicator == True:
        return pred - lamb * torch.sum(constraints.clamp(min=0, max=1), dim=-1)    
    return pred - lamb * torch.sum(constraints.clamp(min=0), dim=-1) 

