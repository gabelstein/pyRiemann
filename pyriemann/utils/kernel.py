"""Kernels for SPD matrices."""

import numpy as np

from .base import invsqrtm, logm
from .mean import mean_riemann, mean_covariance
from .distance import pairwise_distance
from functools import partial, reduce

###############################################################################


def kernel_euclid(X, Y=None, *, reg=1e-10, **kwargs):
    r"""Euclidean kernel between two sets of matrices.

    Calculates the Euclidean kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of matrices in
    :math:`\mathbb{R}^{n \times m}` by calculating pairwise products:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}(\mathbf{X}_i^T \mathbf{Y}_j)

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, m)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, m), default=None
        Second set of matrices. If None, Y is set to X.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Euclidean kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel
    """

    return _apply_matrix_kernel(_euclid, X, Y, reg=reg)


def kernel_log(X, Y=None, *, reg=1e-10, **kwargs):
    r"""Log-Euclidean kernel between two sets of SPD matrices.

    Calculates the Log-Euclidean kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products [1]_:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}(\log(\mathbf{X}_i) \log(\mathbf{Y}_j))

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Log-Euclidean kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel

    References
    ----------
    .. [1] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    """

    K = _apply_matrix_kernel(_log, X, Y, reg=reg)
    return K


def kernel_logeuclid(X, Y=None, *, Cref=None, reg=1e-10, **kwargs):
    r"""Log-Euclidean kernel between two sets of SPD matrices.

    Calculates the Log-Euclidean kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products [1]_:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}(\log(\mathbf{X}_i) \log(\mathbf{Y}_j))

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Log-Euclidean kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel

    References
    ----------
    .. [1] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    """

    return _apply_matrix_kernel(_logeuclid, X, Y, Cref=Cref, reg=reg)


def kernel_riemann(X, Y=None, *, Cref=None, reg=1e-10, **kwargs):
    r"""Affine-invariant Riemannian kernel between two sets of SPD matrices.

    Calculates the affine-invariant Riemannian kernel matrix :math:`\mathbf{K}`
    of inner products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of
    SPD matrices in :math:`\mathbb{R}^{n \times n}` on tangent space at
    :math:`\mathbf{C}_\text{ref}` by calculating pairwise products [1]_:

    .. math::
        \mathbf{K}_{i,j} = \text{tr}( \log( \mathbf{C}_\text{ref}^{-1/2}
        \mathbf{X}_i \mathbf{C}_\text{ref}^{-1/2} )
        \log( \mathbf{C}_\text{ref}^{-1/2} \mathbf{Y}_j
        \mathbf{C}_\text{ref}^{-1/2}) )

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, n), default=None
        Reference point for the tangent space and inner product calculation.
        If None, Cref is calculated as the Riemannian mean of X.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The affine-invariant Riemannian kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel

    References
    ----------
    .. [1] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    """

    return _apply_matrix_kernel(_riemann, X, Y, Cref=Cref, reg=reg)


def kernel_determinant(X, Y=None, *, reg=1e-10, k_type='canonical', **kwargs):
    r"""Determinant kernel between two sets of SPD matrices.

    Calculates the determinant kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products:

    .. math::
        \mathbf{K}_{i,j} = \text{det}(\mathbf{X}_i \mathbf{Y}_j)

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The determinant kernel matrix

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel
    """

    K = _apply_matrix_kernel(_det, X, Y, reg=reg)
    return K


def kernel_gaussian(X, Y=None, *, metric='riemann', gamma=1):
    r"""Gaussian kernel between two sets of SPD matrices.

    Calculates the Gaussian kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
        \mathbf{K}_{i,j} = \exp(-\gamma \text{dist}(\mathbf{X}_i, \mathbf{Y}_j)^2)

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    metric : {'riemann', 'logeuclid', 'euclid'}, default='riemann'
        The type of metric used for pairwise distances.
    gamma : float, default=1
        Kernel parameter.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Gaussian kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel
    """
    K = _distance2(X, Y, metric=metric)
    K = _exponential(K, gamma=-gamma)
    return K


def kernel_laplacian(X, Y=None, *, metric='riemann', gamma=1):
    """
    Laplacian kernel between two sets of SPD matrices.

    Calculates the Laplacian kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
        \mathbf{K}_{i,j} = \exp(-\gamma \text{dist}(\mathbf{X}_i, \mathbf{Y}_j))

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    metric : {'riemann', 'logeuclid', 'euclid'}, default='riemann'
        The type of metric used for pairwise distances.
    gamma : float, default=1
        Kernel parameter.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Laplacian kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel
    """
    K = _distance(X, Y, metric=metric)
    K = _exponential(K, gamma=gamma)
    return K


def kernel_periodic(X, Y=None, *, metric='riemann', gamma=1):
    """
    Periodic kernel between two sets of SPD matrices.

    Calculates the periodic kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
        \mathbf{K}_{i,j} = \exp(-2 \sin^2(\pi \text{dist}(\mathbf{X}_i, \mathbf{Y}_j) / \gamma))


    """
    K = _distance(X, Y, metric=metric)
    K = _periodic(K, gamma=gamma)
    return K


def kernel_rational_quadratic(X, Y=None, *, metric='riemann', alpha=1):

    K = _distance(X, Y, metric=metric, squared=True)
    K = _rationalquadratic(K, alpha=alpha)
    return K


def kernel_polynomial(X, Y=None, *, Cref=None, reg=10e-10, metric='riemann', r=0, s=1):

    feature_map = globals()[f'_{metric}']
    K = _apply_matrix_kernel(feature_map, X, Y, Cref=Cref, reg=reg)
    K = _polynomial(K, r=r, s=s)
    return K


def kernel_exponential(X, Y=None, *,
                       Cref=None, reg=10e-10, metric='riemann', gamma=1):

    feature_map = globals()[f'_{metric}']
    K = _apply_matrix_kernel(feature_map, X, Y, Cref=Cref, reg=reg)
    K = _exponential(K, gamma=gamma)
    return K


def kernel_sigmoid(X, Y=None, *,
                   Cref=None, reg=10e-10, metric='riemann', gamma=1, r=0):

    feature_map = globals()[f'_{metric}']
    K = _apply_matrix_kernel(feature_map, X, Y, Cref=Cref, reg=reg)
    K = _sigmoid(K, gamma=gamma, r=r)
    return K


###############################################################################

def _log(X, Cref):
    """Feature map for Log-Euclidean kernel."""
    X_ = logm(X)
    return X_


def _logeuclid(X, Cref):
    """Feature map for Log-Euclidean kernel."""
    if Cref is None:
        Cref = mean_covariance(X, metric='logeuclid')

    X_ = logm(X) - logm(Cref)
    return X_


def _riemann(X, Cref):
    """Feature map for affine-invariant Riemannian kernel."""
    if Cref is None:
        Cref = mean_covariance(X, metric='riemann')

    C_invsq = invsqrtm(Cref)
    X_ = logm(C_invsq @ X @ C_invsq)
    return X_


def _euclid(X, Cref):
    """Feature map for Euclidean kernel."""
    return X


def _det(X, Cref):
    """Feature map for determinant kernel."""
    # TODO: determinant of block diagonal matrix is product of determinants
    return np.linalg.det(X)


###############################################################################
'''Inner product kernels'''


def _polynomial(K, r=0, s=1):
    """Polynomial function."""
    return (K + r) ** s


def _exponential(K, gamma=1):
    """Exponential function."""
    return np.exp(K * gamma)


def _sigmoid(K, gamma=1, r=0):
    """Sigmoid function."""
    return np.tanh(gamma * K + r)


###############################################################################
'''Distance kernels'''


# might be wrong
def _periodic(K, gamma=1, l=1):
    """Periodic function."""
    return np.exp(-2 * np.sin(np.pi * K / gamma) ** 2/l**2)


def _rationalquadratic(K, alpha=1, l=1):
    """Rational quadratic function."""
    return (1 + K / (2 * alpha*l**2)) ** (-alpha)


###############################################################################

def _distance(X, Y=None, *, metric='riemann', squared=False, **kwargs):
    distances = pairwise_distance(X, Y, metric=metric, squared=squared)
    return distances


_distance2 = partial(_distance, squared=True)


###############################################################################

def _check_dimensions(X, Y, Cref):
    """Check for matching dimensions in X, Y and Cref."""
    if not isinstance(Y, type(None)):
        assert Y.shape[1:] == X.shape[1:], f"Dimension of matrices in Y must "\
                                           f"match dimension of matrices in " \
                                           f"X. Expected {X.shape[1:]}, got " \
                                           f"{Y.shape[1:]}."

    if not isinstance(Cref, type(None)):
        assert Cref.shape == X.shape[1:], f"Dimension of Cref must match " \
                                          f"dimension of matrices in X. " \
                                          f"Expected {X.shape[1:]}, got " \
                                          f"{Cref.shape}."


def _apply_matrix_kernel(kernel_fct, X, Y=None, *, Cref=None, reg=1e-10):
    """Apply a matrix kernel function."""
    _check_dimensions(X, Y, Cref)
    n_matrices_X, n, n = X.shape


    if isinstance(Y, type(None)) or np.array_equal(X, Y):
        X_ = kernel_fct(X, Cref)
        Y_ = X_

    else:
        Y_ = kernel_fct(Y, Cref)

    # calculate scalar products: K[i,j] = np.trace(X_[i]^T @ Y_[j])
    X_T = X_.transpose((0, 2, 1))
    K = np.einsum('acb,dbc->ad', X_T, Y_, optimize=True)

    # regularization due to numerical errors
    if np.array_equal(X_, Y_):
        K.flat[:: n_matrices_X + 1] += reg

    return K


def _canonical(metric='riemann', **kwargs):
    return _apply_matrix_kernel(kernel_fct=globals()[f'_{metric}'], **kwargs)


def kernel(X, Y=None, *,
           Cref=None,
           metric='riemann',
           ktype='canonical',
           reg=1e-10,
           **kwargs):
    """Kernel matrix between matrices according to a specified metric.

    Calculates the kernel matrix K of inner products of two sets X and Y of
    matrices on the tangent space at Cref according to a specified metric.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, n), default=None
        Reference point for the tangent space and inner product
        calculation. Only used if metric='riemann'.
    metric : {'euclid', 'logeuclid', 'riemann'}, default='riemann'
        The type of metric used for tangent space and mean estimation.

    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation, to provide a positive-definite kernel matrix.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    kernel_euclid
    kernel_logeuclid
    kernel_riemann
    """



    if ktype is 'canonical':
        ktype = {'canonical': {'metric': metric,
                               'Cref': Cref,
                               'Y': Y,
                               'reg': reg}}

    call_stack = _make_kernel_call_stack(ktype)
    K = reduce(lambda x, f: f(x), call_stack, {'X': X,
                                               'Y': Y,
                                               'Cref': Cref,
                                               'metric': metric,
                                               **kwargs})

    return K


class Gram(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Gram matrix transformer.

    Parameters
    ----------
    metric : str
        The metric to use to compute the mean. See
        :func:`pyriemann.utils.mean.mean_covariance` for available options.
    kernel : str
        The kernel to use to compute the gram matrix. See
        :func:`pyriemann.utils.kernel.kernel` for available options.

    Attributes
    ----------
    data_ : ndarray, shape (n_trials, n_channels, n_channels)
        The data used to compute the mean covariance matrix.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference covariance matrix.

    See Also
    --------
    pyriemann.utils.mean.mean_covariance
    pyriemann.utils.kernel.kernel

    """

    def __init__(self, metric, kernel):
        self.metric = metric
        self.kernel = kernel

    def fit(self, X, y=None):
        self.data_ = X
        self.Cref = pr.utils.mean.mean_covariance(X, metric=self.metric)
        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'data_'):
            self.data_ = X
            self.Cref = mean_covariance(self.data_, metric=self.metric)
        gram = self.kernel(X, self.data_, Cref=self.Cref)
        return gram

    def fit_transform(self, X, y=None):
        gram = self.fit(X, y).transform(X, y)

        return gram

def _make_kernel_call_stack(call_dict={}):
    call_stack = [partial(globals()[f'_{key}'], **kwargs)
                  for key, kwargs in call_dict.items()]
    return call_stack


def _execute_call_stack(call_stack, input):
    result = input
    for func in call_stack:
        result = func(result)

    return result

