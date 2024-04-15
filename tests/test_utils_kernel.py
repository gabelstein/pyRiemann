import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest

from pyriemann.utils.base import logm

from pyriemann.utils.kernel import *

from pyriemann.utils.kernel import (
    _euclid,
    _logeuclid,
    _log,
    _riemann,
)

from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.test import is_sym_pos_semi_def as is_spsd

rker_str = ['euclid', 'logeuclid', 'riemann']
rker_fct = [kernel_euclid, kernel_logeuclid, kernel_riemann]
feature_maps = [_euclid, _logeuclid, _riemann, _log]
rker_types = kernel_types.keys()

@pytest.mark.parametrize("ker", rker_str)
@pytest.mark.parametrize("ker_type", rker_types)
def test_kernel_x_x(ker, ker_type, get_mats):
    """Test kernel build"""
    n_matrices, n_channels = 7, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel(X, metric=ker, ktype=ker_type)
    assert K.shape == (n_matrices, n_matrices)
    assert_array_almost_equal(K, K.T, decimal=15)
    assert_array_almost_equal(K, kernel(X, X, metric=ker, ktype=ker_type))


@pytest.mark.parametrize("ker", rker_str)
def test_kernel_cref(ker, get_mats):
    """Test kernel reference"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    cref = mean_covariance(X, metric=ker)
    K = kernel(X, X, metric=ker)
    K1 = kernel(X, X, Cref=cref, metric=ker)
    assert_array_equal(K, K1)


@pytest.mark.parametrize("ker", rker_str)
@pytest.mark.parametrize("ker_type", rker_types)
def test_kernel_x_y(ker, ker_type, get_mats):
    """Test kernel for different X and Y"""
    n_matrices_X, n_matrices_Y, n_channels = 6, 5, 3
    X = get_mats(n_matrices_X, n_channels, "spd")
    Y = get_mats(n_matrices_Y, n_channels, "spd")
    K = kernel(X, Y, metric=ker, ktype=ker_type)
    assert K.shape == (n_matrices_X, n_matrices_Y)


@pytest.mark.parametrize("ker", rker_str)
def test_metric_string(ker, get_mats):
    """Test generic kernel function"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = globals()[f'kernel_{ker}'](X)
    K1 = kernel(X, metric=ker)
    assert_array_equal(K, K1)


@pytest.mark.parametrize("ker", rker_types)
def test_kernel_type(ker, get_mats):
    """Test generic kernel function"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = globals()[f'kernel_{ker}'](X, reg=0)
    K1 = kernel(X, ktype=ker, reg=0)
    assert_array_equal(K, K1)


def test_metric_string_error(get_mats):
    """Test generic kernel function error raise"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    with pytest.raises(ValueError):
        kernel(X, metric='foo')


@pytest.mark.parametrize("ker", rker_str)
def test_input_dimension_error(ker, get_mats):
    """Test errors for incorrect dimension"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels + 1, "spd")
    cref = get_mats(1, n_channels + 1, "spd")[0]
    if ker == 'riemann':
        with pytest.raises(AssertionError):
            kernel(X, Cref=cref, metric=ker)
    with pytest.raises(AssertionError):
        kernel(X, Y, metric=ker)


@pytest.mark.parametrize("n_dim0, n_dim1", [(4, 4), (4, 5), (5, 4)])
def test_frobenius(n_dim0, n_dim1, rndstate):
    """Test Euclidean kernel for generic matrices"""
    n_matrices_X, n_matrices_Y = 2, 3
    X = rndstate.randn(n_matrices_X, n_dim0, n_dim1)
    Y = rndstate.randn(n_matrices_Y, n_dim0, n_dim1)
    K = kernel_frobenius(X, Y)
    assert K.shape == (n_matrices_X, n_matrices_Y)

    K1 = np.empty((n_matrices_X, n_matrices_Y))
    K2 = np.empty((n_matrices_X, n_matrices_Y))
    for i in range(n_matrices_X):
        for j in range(n_matrices_Y):
            K1[i, j] = np.trace(X[i].T @ Y[j])
            K2[i, j] = np.dot(X[i].flatten(), Y[j].flatten())
    assert_array_almost_equal(K, K1)
    assert_array_almost_equal(K, K2)


def test_riemann_correctness(get_mats):
    """Test Riemannian kernel correctness"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel_riemann(X, Cref=np.eye(n_channels), reg=0)

    log_X = logm(X)
    tensor = np.tensordot(log_X, log_X.T, axes=1)
    K1 = np.trace(tensor, axis1=1, axis2=2)
    assert_array_almost_equal(K, K1)


@pytest.mark.parametrize("feature_map", feature_maps)
def test_feature_map(feature_map, get_mats):
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = feature_map(X, Cref=np.eye(n_channels))
    assert K.shape == (n_matrices, n_channels, n_channels)
