"""Kernels for SPD matrices."""

import numpy as np

from .base import invsqrtm, logm
from .mean import mean_covariance, mean_functions
from .distance import pairwise_distance
from sklearn.base import BaseEstimator, TransformerMixin
from .utils import check_function

###############################################################################
'''Canonical Kernels'''


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

    return _apply_matrix_kernel(_euclid, X, Y, reg=reg, metric='euclid')


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

    return _apply_matrix_kernel(_logeuclid, X, Y, Cref=Cref, reg=reg,
                                metric='logeuclid')


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

    return _apply_matrix_kernel(_riemann, X, Y, Cref=Cref, reg=reg,
                                metric='riemann')


def kernel_canonical(X, Y=None, *,
                     metric='riemann', Cref=None, reg=1e-10, **kwargs):
    r"""Canonical kernel between two sets of SPD matrices.

        Calculates the canonical kernel matrix :math:`\mathbf{K}` of inner
        products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
        matrices in :math:`\mathbb{R}^{n \times n}` according to the metric.

        Parameters
        ----------
        X : ndarray, shape (n_matrices_X, n, n)
            First set of SPD matrices.
        Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
            Second set of SPD matrices. If None, Y is set to X.
        metric : {'euclid', 'logeuclid', 'riemann'}, default='riemann'
            The metric used for kernel estimation.
        Cref : None | ndarray, shape (n, n), default=None
            Reference point for tangent space and inner product calculation.
            If None, Cref is calculated as the geometric mean of X according to
            the metric.
        reg : float, default=1e-10
            Regularization parameter to mitigate numerical errors in kernel
            matrix estimation.

        Returns
        -------
        K : ndarray, shape (n_matrices_X, n_matrices_Y)
            The canonical kernel matrix between X and Y.

        Notes
        -----
        .. versionadded:: 0.6

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
    kfunc = check_function(metric, _canonical_kernels)
    return kfunc(X, Y, Cref=Cref, reg=reg)


_canonical_kernels = {
    'euclid': kernel_euclid,
    'logeuclid': kernel_logeuclid,
    'riemann': kernel_riemann
}

###############################################################################
'''Distance Kernels.'''


def _distance_kernel(func, squared=False):
    def wrapper(X, Y=None, *, metric='riemann', reg=1e-10, **kwargs):
        K = pairwise_distance(X, Y, metric=metric, squared=squared)
        K = func(K, **kwargs)
        K = _regularize_kernel(K, reg=reg)
        return K
    return wrapper


def kernel_gaussian(X, Y=None, *, metric='riemann', reg=0, gamma=1, **kwargs):
    r"""Gaussian kernel between two sets of SPD matrices.

    Calculates the Gaussian kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
        \mathbf{K}_{i,j} = \exp(-\gamma \text{dist}(\mathbf{X}_i,
        \mathbf{Y}_j)^2)

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
    K = _distance_kernel(_exponential, squared=True)(X, Y,
                                                     metric=metric,
                                                     gamma=-gamma,
                                                     reg=reg)
    return K


def kernel_laplacian(X, Y=None, *, metric='riemann', gamma=1, reg=0, **kwargs):
    r"""Laplacian kernel between two sets of SPD matrices.

    Calculates the Laplacian kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
        \mathbf{K}_{i,j} = \exp(-\gamma \text{dist}(\mathbf{X}_i,
        \mathbf{Y}_j))

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
    K = _distance_kernel(_exponential, squared=False)(X, Y,
                                                      metric=metric,
                                                      gamma=-gamma,
                                                      reg=reg)
    return K


def kernel_rational_quadratic(X, Y=None, *,
                              metric='riemann', alpha=1, gamma=1, reg=0,
                              **kwargs):
    r"""Rational quadratic kernel between two sets of SPD matrices.

    Calculates the rational quadratic kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products:

    .. math::
        \mathbf{K}_{i,j} = \left( 1 + \frac{\text{dist}(\mathbf{X}_i,
        \mathbf{Y}_j)^2}{2 \alpha l^2} \right)^{-\alpha}

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    metric : string or callable, default='riemann'
        Metric to calculate the pairwise distances. If metric is a string, it
        must be one of 'euclid', 'harmonic', 'kullback', 'kullback_right',
        'kullback_sym', 'logdet', 'logeuclid', 'riemann', 'wasserstein'.
        If metric is a callable, it must take two arguments and return a float.
    alpha : float, default=1
        Kernel parameter.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.
    s : float, default=1
        Kernel parameter.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The rational quadratic kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel
    """
    K = _distance_kernel(_rational_quadratic, squared=True)(X, Y,
                                                            metric=metric,
                                                            alpha=alpha,
                                                            reg=reg,
                                                            gamma=gamma)
    return K


def kernel_multiquadratic(X, Y=None, *,
                          metric='riemann', beta=1, sigma=1, reg=0, **kwargs):
    r"""Multiquadratic kernel between two sets of SPD matrices.

    Calculates the  multiquadratic kernel matrix :math:`\mathbf{K}` of
    inner products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products:

    .. math::
        \mathbf{K}_{i,j} = \left( \sigma + \text{dist}(\mathbf{X}_i,
        \mathbf{Y}_j)^2 \right)^{\beta}

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    metric : string or callable, default='riemann'
        Metric to calculate the pairwise distances. If metric is a string, it
        must be one of 'euclid', 'harmonic', 'kullback', 'kullback_right',
        'kullback_sym', 'logdet', 'logeuclid', 'riemann', 'wasserstein'.
        If metric is a callable, it must take two arguments and return a float.
    beta : float, default=1
        Kernel parameter.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The inverse multiquadratic kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel
    """

    K = _distance_kernel(_multiquadratic,
                         squared=True)(X, Y,
                                       metric=metric,
                                       beta=beta,
                                       reg=reg,
                                       sigma=sigma)
    return K


def kernel_inverse_multiquadratic(X, Y=None, *,
                                  metric='riemann',
                                  beta=1,
                                  sigma=1,
                                  reg=0,
                                  **kwargs):
    r"""Inverse multiquadratic kernel between two sets of SPD matrices.

    Calculates the inverse multiquadratic kernel matrix :math:`\mathbf{K}` of
    inner products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products:

    .. math::
        \mathbf{K}_{i,j} = \left( \sigma + \text{dist}(\mathbf{X}_i,
        \mathbf{Y}_j)^2 \right)^{-\beta}

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    metric : string or callable, default='riemann'
        Metric to calculate the pairwise distances. If metric is a string, it
        must be one of 'euclid', 'harmonic', 'kullback', 'kullback_right',
        'kullback_sym', 'logdet', 'logeuclid', 'riemann', 'wasserstein'.
        If metric is a callable, it must take two arguments and return a float.
    beta : float, default=1
        Kernel parameter.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The inverse multiquadratic kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel
    """
    K = _distance_kernel(_inverse_multiquadratic,
                         squared=True)(X, Y,
                                       metric=metric,
                                       beta=beta,
                                       reg=reg,
                                       sigma=sigma)
    return K


###############################################################################
'''Inner Product Kernels'''


def _inner_product_kernel(func):
    def wrapper(X, Y=None, *,
                Cref=None, reg=1e-10, metric='riemann', **kwargs):
        feature_map = check_function(metric, _feature_maps)
        K = _apply_matrix_kernel(feature_map, X, Y,
                                 Cref=Cref, reg=0, metric=metric)
        K = func(K, **kwargs)
        K = _regularize_kernel(K, reg=reg)
        return K
    return wrapper


def kernel_polynomial(X, Y=None, *,
                      Cref=None,
                      reg=10e-10,
                      metric='riemann',
                      r=0,
                      s=1,
                      gamma=1,
                      **kwargs):
    r"""Polynomial kernel between two sets of SPD matrices.

    Calculates the polynomial kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products:

    .. math::
        \mathbf{K}_{i,j} = (<\mathbf{X}_i, \mathbf{Y}_j>_* + r)^s

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
    metric : {'euclid', 'logeuclid', 'riemann'}, default='riemann'
        The type of metric used for tangent space and mean estimation.
    r : float, default=0
        Kernel parameter.
    s : float, default=1
        Kernel parameter.
    gamma : float, default=1
        Kernel parameter.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The polynomial kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel

    """
    K = _inner_product_kernel(_polynomial)(X, Y,
                                           Cref=Cref, reg=reg, metric=metric,
                                           r=r, s=s, gamma=gamma)

    return K


def kernel_exponential(X, Y=None, *,
                       Cref=None, reg=0, metric='riemann', gamma=1, **kwargs):
    r"""Exponential kernel between two sets of SPD matrices.

    Calculates the exponential kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products:

    .. math::
      \mathbf{K}_{i,j} = \exp(\gamma <\mathbf{X}_i, \mathbf{Y}_j>_*)

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
    metric : {'euclid', 'logeuclid', 'riemann'}, default='riemann'
      The type of metric used for tangent space and mean estimation.
    gamma : float, default=1
      Kernel parameter.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
      The exponential kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel

    """
    K = _inner_product_kernel(_exponential)(X, Y,
                                            Cref=Cref, reg=reg,
                                            metric=metric, gamma=gamma)
    return K


def kernel_sigmoid(X, Y=None, *,
                   Cref=None, reg=10e-10, metric='riemann', gamma=1, r=0,
                   **kwargs):
    r"""Sigmoid kernel between two sets of SPD matrices.

    Calculates the sigmoid kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
         \mathbf{K}_{i,j} = \tanh(\gamma <\mathbf{X}_i, \mathbf{Y}_j>_*) + r)

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
    metric : {'euclid', 'logeuclid', 'riemann'}, default='riemann'
        The type of metric used for tangent space and mean estimation.
    gamma : float, default=1
        Kernel parameter.
    r : float, default=0
        Kernel parameter.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The sigmoid kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel

    """
    K = _inner_product_kernel(_sigmoid)(X, Y,
                                        Cref=Cref, reg=reg,
                                        metric=metric, gamma=gamma, r=r)
    return K


###############################################################################
'''Other Kernels'''


def kernel_frobenius(X, Y=None, *, reg=1e-10, **kwargs):
    r"""Frobenius inner product kernel between two sets of matrices.

    Calculates the Frobenius inner product kernel matrix :math:`\mathbf{K}` of
    inner products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of
    matrices in :math:`\mathbb{R}^{n \times m}` by calculating pairwise
    products:

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
        The Frobenius inner product kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel
    """
    K = _apply_matrix_kernel(_euclid, X, Y,
                             reg=reg, Cref=np.zeros(X.shape[-2:]))
    return K


def kernel_logfrobenius(X, Y=None, *, reg=1e-10, **kwargs):
    r"""Log-Frobenius kernel between two sets of SPD matrices.

    Calculates the Log-Frobenius kernel matrix :math:`\mathbf{K}` of inner
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
        The Log-Frobenius kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.7

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


def kernel_stein(X, Y=None, *, reg=1e-10, beta=1, c=1, **kwargs):
    r"""Stein kernel between two sets of SPD matrices.

    Calculates the Stein kernel matrix :math:`\mathbf{K}` of inner products of
    two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products [1]_:

    .. math::
        \mathbf{K}_{i,j} = 2^{c \beta} \left( \frac{\det(\mathbf{X}_i)^{\beta}
        \det(\mathbf{Y}_j)^{\beta}}{\det(\mathbf{X}_i + \mathbf{Y}_j)^{\beta}}
        \right)^{1/2}

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of SPD matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of SPD matrices. If None, Y is set to X.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.
    beta : float, default=1
        Kernel parameter.
    c : float, default=1
        Kernel parameter.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The Stein kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.7

    See Also
    --------
    kernel

    References
    ----------
    .. [1] `Sparse coding and dictionary learning for symmetric positive
        definite matrices: A kernel approach
        <https://link.springer.com/chapter/10.1007/978-3-642-33709-3_16>`_
        M. T. Harandi, C. Sanderson, R. Hartley, and B. C. Lovell, ECCV, 2012,
        pp. 216–229
    """

    if Y is None or np.array_equal(X, Y):
        X_ = _det(X)
        Y_ = X_
    else:
        X_, Y_ = _det(X), _det(Y)

    frac = np.sqrt((X_[:, None] * Y_) ** beta) / (X_[:, None] + Y_)**beta
    K = 2**(c*beta) * frac
    K = _regularize_kernel(K, reg=reg)
    return K


###############################################################################
'''Feature Maps.'''


def _log(X, Cref=None):
    """Feature map for Log kernel."""
    return logm(X)


def _logeuclid(X, Cref):
    """Feature map for Log-Euclidean kernel."""
    return logm(X) - logm(Cref)


def _riemann(X, Cref):
    """Feature map for affine-invariant Riemannian kernel."""
    C_invsq = invsqrtm(Cref)
    X_ = logm(C_invsq @ X @ C_invsq)
    return X_


def _euclid(X, Cref):
    """Feature map for Euclidean kernel."""
    return X - Cref


def _det(X, Cref=None):
    """Feature map for determinant kernel."""
    return np.linalg.det(X)


_feature_maps = {
    'log': _log,
    'logeuclid': _logeuclid,
    'riemann': _riemann,
    'euclid': _euclid,
    'det': _det,
}

###############################################################################
'''Kernel functions.'''


def _polynomial(K, r=1, s=2, gamma=1):
    """Polynomial function."""
    return (gamma*K + r) ** s


def _exponential(K, gamma=1):
    """Exponential function."""
    return np.exp(K * gamma)


def _sigmoid(K, gamma=1, r=0):
    """Sigmoid function."""
    return np.tanh(gamma * K + r)


def _rational_quadratic(K, alpha=1, gamma=1):
    """Rational quadratic function."""
    return (1 + K * gamma / alpha) ** (-alpha)


def _multiquadratic(K, beta=1, sigma=1):
    """Inverse multiquadratic function."""
    K = (sigma + K) ** beta
    return K


def _inverse_multiquadratic(K, beta=1, sigma=1):
    """Inverse multiquadratic function."""
    return _multiquadratic(K, beta=-beta, sigma=sigma)


###############################################################################

def _check_dimensions(X, Y, Cref):
    """Check for matching dimensions in X, Y and Cref."""
    if not isinstance(Y, type(None)):
        msg = f"Dimension of matrices in Y must match dimension of matrices "\
              f"in X. Expected {X.shape[1:]}, got {Y.shape[1:]}."
        assert Y.shape[1:] == X.shape[1:], msg

    if not isinstance(Cref, type(None)):
        msg = f"Dimension of Cref must match dimension of matrices in X. "\
              f"Expected {X.shape[1:]}, got {Cref.shape}."
        assert Cref.shape == X.shape[1:], msg


def _regularize_kernel(K, reg=1e-10):
    """Regularize kernel matrix."""
    if np.array_equal(K, K.T):
        K.flat[:: K.shape[0] + 1] += reg
    return K


def _apply_matrix_kernel(feature_map, X, Y=None, *,
                         Cref=None, reg=1e-10, metric='euclid', **kwargs):

    """Apply a matrix kernel function."""
    _check_dimensions(X, Y, Cref)
    if Y is None or np.array_equal(X, Y):
        if Cref is None:
            Cref = mean_covariance(X, metric=metric)
        X_ = feature_map(X, Cref)
        Y_ = X_
    else:
        if Cref is None:
            Cref = mean_covariance(Y, metric=metric)
        Y_ = feature_map(Y, Cref)
        X_ = feature_map(X, Cref)

    # calculate scalar products: K[i,j] = np.trace(X_[i]^T @ Y_[j])
    X_T = X_.transpose((0, 2, 1))
    K = np.einsum('acb,dbc->ad', X_T, Y_, optimize=True)

    # regularization due to numerical errors
    _regularize_kernel(K, reg=reg)

    return K


kernel_functions = {
    "euclid": kernel_euclid,
    "logeuclid": kernel_logeuclid,
    "riemann": kernel_riemann,
}


def kernel(X, Y=None, *,
           Cref=None,
           metric='riemann',
           ktype='canonical',
           reg=1e-10,
           **kwargs):
    """Kernel matrix between two sets of matrices.

    Calculates the kernel matrix K of two sets X and Y of matrices.
    The kernel function is specified by the user and can be any of the
    available kernel functions in :mod:`pyriemann.utils.kernel` or a custom
    function.

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)
        First set of matrices.
    Y : None | ndarray, shape (n_matrices_Y, n, n), default=None
        Second set of matrices. If None, Y is set to X.
    Cref : None | ndarray, shape (n, n), default=None
        Reference point for the tangent space and inner product
        calculation. If None, Cref is calculated as the Riemannian mean of X
        according to the specified metric.
    metric : string, default='riemann'
        The type of metric used for tangent space and mean estimation or
        pairwise distances. If metric is a string, it must be one of 'euclid',
        'harmonic', 'kullback', 'kullback_right', 'kullback_sym', 'logdet',
        'logeuclid', 'riemann', 'wasserstein'.
    ktype : string | callable, default='canonical'
        The type of kernel to use. can be: "canonical", "determinant",
        "gaussian", "laplacian", "polynomial", "rational_quadratic",
        "exponential", "sigmoid", "log", "row_feature",
        "inverse_multiquadratic", "stein", "multiquadratic" or a callable
        function. If a callable function is provided, it must take the
        arguments X, Y, Cref, reg and metric and return a kernel matrix.
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

    kernel_function = check_function(ktype, kernel_types)
    return kernel_function(X, Y, Cref=Cref, reg=reg, metric=metric, **kwargs)


class Gram(BaseEstimator, TransformerMixin):
    r"""Gram matrix transformer for kernel functions.

    This transformer computes the Gram matrix between two sets of SPD matrices
    using a kernel function. The kernel function is used to compute the inner
    product between the matrices in the two sets. The kernel function is
    specified by the user and can be any of the available kernel functions in
    :mod:`pyriemann.utils.kernel` or a custom function.
    The Gram matrix of a kernel function ``k`` between two sets of SPD matrices
    X and Y is defined as:

    .. math::
        \mathbf{K}_{i,j} = \text{k}(\mathbf{X}_i, \mathbf{Y}_j)

    Parameters
    ----------
    metric : str
        The metric to use to compute the mean. See
        :func:`pyriemann.utils.mean.mean_covariance` for available options.
    kernel_fct : callable
        The kernel to use to compute the gram matrix. See
        :func:`pyriemann.utils.kernel.kernel` for available options.
    kernel_params : dict
        Parameters to pass to the kernel function.

    Attributes
    ----------
    data_ : ndarray, shape (n_trials, n_channels, n_channels)
        The data used to compute the mean covariance matrix.
    Cref : ndarray, shape (n_channels, n_channels)
        The reference covariance matrix.

    See Also
    --------
    pyriemann.utils.kernel.kernel

    """

    def __init__(self, metric, kernel_fct, kernel_params=None, Cref=None):
        self.metric = metric
        self.kernel_fct = kernel_fct
        self.kernel_params = kernel_params
        self.Cref = Cref

    def fit(self, X, y=None):
        self.data_ = X
        if self.Cref is None and self.metric in mean_functions.keys():
            self.Cref = mean_covariance(X, metric=self.metric)
        if self.kernel_params is None:
            self.kernel_params = {}
        return self

    def transform(self, X, y=None):
        gram = self.kernel_fct(X, self.data_,
                               Cref=self.Cref,
                               **self.kernel_params)
        return gram


kernel_types = {
    'canonical': kernel_canonical,
    'gaussian': kernel_gaussian,
    'laplacian': kernel_laplacian,
    'polynomial': kernel_polynomial,
    'rational_quadratic': kernel_rational_quadratic,
    'exponential': kernel_exponential,
    'sigmoid': kernel_sigmoid,
    'logfrobenius': kernel_logfrobenius,
    'inverse_multiquadratic': kernel_inverse_multiquadratic,
    'stein': kernel_stein,
    'multiquadratic': kernel_multiquadratic
}
