"""Linear-algebra and regressor construction helpers."""

from copy import copy

import numpy as np
import scipy.linalg
from loguru import logger


def tikhonov_regularization(matrix, lam=0.1):
    return matrix + lam * np.identity(matrix.shape[0])


def SVD_dim_reduction(matrix):
    """Reduce a matrix to its numerical rank using singular vectors."""
    u, singular_values, v = np.linalg.svd(matrix, full_matrices=True)
    rank = np.linalg.matrix_rank(matrix)
    logger.info(f"Before dimension reduction, shape: {matrix.shape}, rank: {rank}")
    reduced = u[:, :rank] @ np.diag(singular_values[:rank]) @ v[:rank, :rank]
    logger.info(f"After dimension reduction, shape: {reduced.shape}, rank: {rank}")
    return reduced


def QR_dim_reduction_backup(matrix):
    """Legacy QR dimension-reduction implementation."""
    _, r = scipy.linalg.qr(matrix, mode="economic")
    column_count = matrix.shape[1]
    rank = np.linalg.matrix_rank(r)

    diagonal = abs(np.diagonal(r))
    columns = sorted(np.arange(column_count), key=lambda index: diagonal[index])
    deleted_columns = sorted(columns[: column_count - rank])
    reduced = np.delete(copy(r), deleted_columns, 1)
    return reduced, np.linalg.cond(reduced)


def QR_dim_reduction(matrix):
    """Reduce dependent QR columns and return condition information."""
    _, r = scipy.linalg.qr(matrix, mode="economic")
    column_count = matrix.shape[1]

    diagonal = np.abs(np.diagonal(r))
    tolerance = diagonal.max() * max(r.shape) * np.finfo(r.dtype).eps
    rank = int(np.sum(diagonal > tolerance))

    columns = np.argsort(diagonal)
    deleted_columns = np.sort(columns[: column_count - rank])
    reduced = np.delete(r, deleted_columns, axis=1)

    singular_values = np.linalg.svd(reduced, compute_uv=False)
    condition = singular_values[0] / singular_values[-1]
    return reduced, condition, singular_values


def feature2regressor(list_of_features, n_datapoints, njoints):
    """Expand joint-wise features into a block-diagonal regressor."""
    check_features(list_of_features, n_datapoints, njoints)
    expanded_features = [
        np.repeat(feature, njoints, axis=0) for feature in list_of_features
    ]
    identity_filter = np.tile(np.eye(njoints), (n_datapoints, 1))
    regressors = [feature * identity_filter for feature in expanded_features]
    return np.hstack(regressors)


def check_features(list_of_features, n_datapoints, njoints):
    for index, feature in enumerate(list_of_features):
        assert feature.shape == (n_datapoints, njoints), (
            f"Feature{index} is not compatible with the given dataset."
        )
