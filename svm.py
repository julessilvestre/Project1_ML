"""Linear SVM with L2 regularization, trained by full-batch subgradient descent.

Adapted from the course exercise ex06 (hinge loss + lambda/2 * ||w||^2).
The exercise uses SGD with one sample per step and recomputes the full cost
at every step, which is far too slow for 300k rows. Here every step uses all
training points at once (vectorized), like reg_logistic_regression.

Two additions make it stable on this dataset:
- a decreasing step size gamma / sqrt(t + 1) and averaging of the weights
  over the iterations: the hinge loss has kinks, so with a constant step the
  last weight vector keeps jumping around instead of settling;
- an optional weight pos_weight on the positive class: with only 9% positives
  the plain hinge loss pushes almost every score below 0.

Labels must be in {-1, 1}.
"""

import numpy as np


def svm_loss(y, tx, w):
    """Mean hinge loss, without the regularization term.

    Args:
        y: labels in {-1, 1}, shape (N,)
        tx: feature matrix, shape (N, D)
        w: weights, shape (D,)

    Returns:
        scalar, the average of max(0, 1 - y_n * x_n @ w)
    """
    margins = y * (tx @ w)
    return np.mean(np.maximum(0, 1 - margins))


def svm_subgradient(y, tx, w, lambda_, sample_weights):
    """Subgradient of: weighted mean hinge loss + lambda/2 * ||w||^2.

    Only points with margin < 1 (the "support" points, wrongly classified or
    too close to the boundary) contribute -c_n * y_n * x_n; the others give 0.
    """
    margins = y * (tx @ w)
    is_support = margins < 1
    grad_hinge = -(tx.T @ (sample_weights * y * is_support)) / np.sum(sample_weights)
    return grad_hinge + lambda_ * w


def svm_l2(y, tx, lambda_, initial_w, max_iters, gamma, pos_weight=1.0):
    """Train an L2-regularized linear SVM with averaged subgradient descent.

    Args:
        y: labels in {-1, 1}, shape (N,)
        tx: feature matrix, shape (N, D)
        lambda_: regularization strength (objective uses lambda/2 * ||w||^2)
        initial_w: starting weights, shape (D,)
        max_iters: number of full-batch steps
        gamma: initial step size, decreased as gamma / sqrt(t + 1)
        pos_weight: weight of positive examples in the loss (1 = unweighted)

    Returns:
        (w, loss): averaged weights and their mean hinge loss (without
        penalty), the same convention as the other project functions.
    """
    sample_weights = np.where(y == 1, pos_weight, 1.0)
    w = initial_w.astype(float)
    w_avg = np.zeros_like(w)
    for it in range(max_iters):
        grad = svm_subgradient(y, tx, w, lambda_, sample_weights)
        w = w - gamma / np.sqrt(it + 1) * grad
        w_avg = w_avg + (w - w_avg) / (it + 1)  # running mean of all iterates
    loss = svm_loss(y, tx, w_avg)
    return w_avg, loss
