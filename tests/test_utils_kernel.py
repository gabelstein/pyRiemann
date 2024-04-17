import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
from sklearn.gaussian_process.kernels import RationalQuadratic

from sklearn.metrics.pairwise import (laplacian_kernel,
                                      sigmoid_kernel,
                                      polynomial_kernel,
                                      rbf_kernel,
                                      )

from pyriemann.utils.kernel import (
    kernel,
    kernel_euclid,
    kernel_logeuclid,
    kernel_riemann,
    kernel_frobenius,
    kernel_gaussian,
    kernel_laplacian,
    kernel_sigmoid,
    kernel_polynomial,
    kernel_rational_quadratic,
    Gram,
    kernel_types,
    kernel_canonical,
    kernel_exponential,
    kernel_inverse_multiquadratic,
    kernel_multiquadratic,
    kernel_stein,
    kernel_logfrobenius
)

from pyriemann.utils.distance import distance_functions
from pyriemann.utils.base import logm

from pyriemann.utils.kernel import (
    _euclid,
    _logeuclid,
    _log,
    _riemann,
)

from pyriemann.utils.mean import mean_covariance

rker_str = ['euclid', 'logeuclid', 'riemann']
rker_fct = [kernel_euclid, kernel_logeuclid, kernel_riemann]
feature_maps = [_euclid, _logeuclid, _riemann, _log]
rker_types = kernel_types.keys()
metrics = distance_functions.keys()
distance_kernels = ['gaussian',
                    'laplacian',
                    'sigmoid',
                    'rational_quadratic',
                    'inverse_multiquadratic',
                    'multiquadratic']


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
    K = kernel_types[ker](X, reg=0)
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


@pytest.mark.parametrize("ker_type", distance_kernels)
@pytest.mark.parametrize("metric", metrics)
def test_gram_matrix(ker_type, metric, get_mats):
    """Test gram matrix"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = Gram(metric=metric, kernel_fct=kernel_types[ker_type])
    X_ = K.fit_transform(X)
    assert X_.shape == (n_matrices, n_matrices)
    assert_array_almost_equal(X_, X_.T)


@pytest.mark.parametrize("ker_type", distance_kernels)
def test_gram_matrix_kernel_params(ker_type, get_mats):
    """Test gram matrix"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = Gram(metric='logeuclid', kernel_fct=kernel_types[ker_type],
             kernel_params={'reg': 0.1, 'l': 0.1, 'sigma': 0.1})
    X_ = K.fit_transform(X)
    assert X_.shape == (n_matrices, n_matrices)
    assert_array_almost_equal(X_, X_.T)


def test_gaussian_kernel_correctness(get_mats):
    """Test gaussian kernel correctness"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel_gaussian(X, gamma=0.5, metric='euclid')
    K1 = rbf_kernel(X.reshape(n_matrices, -1), gamma=0.5)
    assert_array_almost_equal(K, K1)


def test_rq_kernel_correctness(get_mats):
    """Test rational quadratic kernel correctness"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel_rational_quadratic(X, alpha=2, s=0.5, metric='euclid')
    K1 = RationalQuadratic(0.5, 2)(X.reshape(n_matrices, -1))
    assert_array_almost_equal(K, K1)


def test_laplacian_kernel_correctness(get_mats):
    """Test laplacian kernel correctness"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel_laplacian(X, gamma=0.5, metric='euclid')
    K1 = laplacian_kernel(X.reshape(n_matrices, -1), gamma=0.5)
    assert_array_almost_equal(K, K1)


def test_sigmoid_kernel_correctness(get_mats):
    """Test sigmoid kernel correctness"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel_sigmoid(X, gamma=0.5, r=1, metric='euclid',
                       Cref=np.zeros((n_channels, n_channels)))
    K1 = sigmoid_kernel(X.reshape(n_matrices, -1), gamma=0.5)
    assert_array_almost_equal(K, K1)


def test_polynomial_kernel_correctness(get_mats):
    """Test polynomial kernel correctness"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    K = kernel_polynomial(X, r=1, s=2, gamma=0.5, metric='euclid',
                          Cref=np.zeros((n_channels, n_channels)))
    K1 = polynomial_kernel(X.reshape(n_matrices, -1), coef0=1, degree=2,
                           gamma=.5)
    assert_array_almost_equal(K, K1)
