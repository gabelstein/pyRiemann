"""Kernels for SPD matrices."""
from functools import wraps

import numpy as np

from .base import invsqrtm, logm
from .mean import mean_covariance
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
            Reference point for the tangent space and inner product calculation.
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
    try:
        return globals()[f'kernel_{metric}'](X, Y, Cref=Cref, reg=reg)
    except KeyError:
        raise ValueError(
            "Kernel metric must be 'euclid', 'logeuclid', or 'riemann' for"
            f" canonical kernel. Got {metric}.")


###############################################################################
'''Distance Kernels.'''


def _distance_kernel(func, squared=False):
    def wrapper(X, Y=None, *, metric='riemann', reg=1e-10, **kwargs):
        K = pairwise_distance(X, Y, metric=metric, squared=squared)
        K = func(K, **kwargs)
        K = _regularize_kernel(K, reg=reg)
        return K
    return wrapper


def kernel_gaussian(X, Y=None, *, metric='riemann', reg=0, gamma=1):
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
    K = _distance_kernel(_exponential, squared=True)(X, Y, metric=metric,
                                                     gamma=-gamma, reg=reg)
    return K


def kernel_laplacian(X, Y=None, *, metric='riemann', gamma=1, reg=0):
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
    K = _distance_kernel(_exponential, squared=False)(X, Y, metric=metric,
                                                      gamma=-gamma, reg=reg)
    return K


def kernel_periodic(X, Y=None, *, metric='riemann', gamma=1, reg=0):
    """
    Periodic kernel between two sets of SPD matrices.

    Calculates the periodic kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
        \mathbf{K}_{i,j} = \exp(-2 \sin^2(\pi
        \text{dist}(\mathbf{X}_i, \mathbf{Y}_j) / \gamma))

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
    gamma : float, default=1
        Kernel parameter.
    reg : float, default=1e-10
        Regularization parameter to mitigate numerical errors in kernel
        matrix estimation.

    Returns
    -------
    K : ndarray, shape (n_matrices_X, n_matrices_Y)
        The periodic kernel matrix between X and Y.

    Notes
    -----
    .. versionadded:: 0.6

    See Also
    --------
    kernel
    """
    K = _distance_kernel(_periodic, squared=False)(X, Y, metric=metric,
                                                   gamma=gamma, reg=reg)
    return K


def kernel_rational_quadratic(X, Y=None, *, metric='riemann', alpha=1, l=1,
                              reg=0):
    """
    Rational quadratic kernel between two sets of SPD matrices.

    Calculates the rational quadratic kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products:

    .. math::
        \mathbf{K}_{i,j} = \left( 1 + \frac{\text{dist}(\mathbf{X}_i,
        \mathbf{Y}_j)^2}{2 \alpha} \right)^{-\alpha}

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
    K = _distance_kernel(_rational_quadratic, squared=True)(X, Y, metric=metric,
                                                            alpha=alpha,
                                                            reg=reg,
                                                            l=l)
    return K


def kernel_multiquadratic(X, Y=None, *,
                                metric='riemann', beta=1, sigma=1, reg=0):
    """
    Multiquadratic kernel between two sets of SPD matrices.

    Calculates the  multiquadratic kernel matrix :math:`\mathbf{K}` of
    inner products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products:

    .. math::
        \mathbf{K}_{i,j} = \left( 1 + \text{dist}(\mathbf{X}_i,
        \mathbf{Y}_j)^2 \right)^{\beta / 2}

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
                                  metric='riemann', beta=1, sigma=1, reg=0):
        """
        Inverse multiquadratic kernel between two sets of SPD matrices.

        Calculates the inverse multiquadratic kernel matrix :math:`\mathbf{K}` of
        inner products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
        matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
        products:

        .. math::
            \mathbf{K}_{i,j} = \left( 1 + \text{dist}(\mathbf{X}_i,
            \mathbf{Y}_j)^2 \right)^{-\beta / 2}

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
    def wrapper(X, Y=None, *, Cref=None, reg=1e-10, metric='riemann', **kwargs):
        feature_map = globals()[f'_{metric}']
        K = _apply_matrix_kernel(feature_map, X, Y,
                                 Cref=Cref, reg=0, metric=metric)
        K = func(K, **kwargs)
        K = _regularize_kernel(K, reg=reg)
        return K
    return wrapper


def kernel_polynomial(X, Y=None, *,
                      Cref=None, reg=10e-10, metric='riemann', r=0, s=1):
    """Polynomial kernel between two sets of SPD matrices.

    Calculates the polynomial kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
        \mathbf{K}_{i,j} = (\text{tr}(\mathbf{X}_i^T \mathbf{Y}_j) + r)^s

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
                                           Cref=Cref, reg=reg,
                                           metric=metric, r=r, s=s)

    return K


def kernel_exponential(X, Y=None, *,
                       Cref=None, reg=0, metric='riemann', gamma=1):
    """Exponential kernel between two sets of SPD matrices.

    Calculates the exponential kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products:

    .. math::
      \mathbf{K}_{i,j} = \exp(-\gamma \text{tr}(\mathbf{X}_i^T \mathbf{Y}_j))

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


def kernel_sigmoid(X, Y=None, *, Cref=None, reg=10e-10, metric='riemann',
                     gamma=1, r=0):
    """Sigmoid kernel between two sets of SPD matrices.

    Calculates the sigmoid kernel matrix :math:`\mathbf{K}` of inner products
    of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD matrices in
    :math:`\mathbb{R}^{n \times n}` by calculating pairwise products:

    .. math::
         \mathbf{K}_{i,j} = \tanh(\gamma \text{tr}(\mathbf{X}_i^T \mathbf{Y}_j)
         + r)

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


def kernel_determinant(X, Y=None, *, reg=1e-10, **kwargs):
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


def kernel_row_feature(X, Y=None, *,
                      Cref=None,
                       kernel_fct=np.exp,
                       kernel_parameters=None,
                       **kwargs):
    """Row feature kernel between two sets of SPD matrices.

    Calculates the row feature kernel matrix :math:`\mathbf{K}` of inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}` of SPD
    matrices in :math:`\mathbb{R}^{n \times n}` by calculating pairwise
    products [1]_:

    .. math::
        \mathbf{K}_{i,j} = \sum_{k=1}^n \exp(-\gamma \text{dist}(\mathbf{X}_{i,k},
        \mathbf{Y}_{j,k})^2)

    Parameters
    ----------
    X : ndarray, shape (n_matrices_X, n, n)

    """

    n_matrices_X, n, n = X.shape
    C12inv = invsqrtm(Cref)
    X = C12inv @ X @ C12inv
    if Y is None:
        Y = X
    else:
        Y = C12inv @ Y @ C12inv

    n_matrices_Y, n, n = Y.shape

    full_res = np.zeros((n_matrices_X, n_matrices_Y))

    for i, dat_ in enumerate(X):
        res = Y - dat_
        res = np.linalg.norm(res, axis=-1) ** 2
        res = kernel_fct(res, **kernel_parameters)
        res = np.sum(res, axis=-1)
        full_res[i] = res

    return full_res


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


###############################################################################
'''Kernel functions.'''


def _polynomial(K, r=1, s=2):
    """Polynomial function."""
    return (K + r) ** s


def _exponential(K, gamma=1):
    """Exponential function."""
    return np.exp(K * gamma)


def _sigmoid(K, gamma=1, r=0):
    """Sigmoid function."""
    return np.tanh(gamma * K + r)


# might be wrong
def _periodic(K, gamma=1, l=1):
    """Periodic function."""
    return np.exp(-2 * np.sin(np.pi * K / gamma) ** 2/l**2)


def _rational_quadratic(K, alpha=1, l=1):
    """Rational quadratic function."""
    return (1 + K / (2 * alpha*l**2)) ** (-alpha)


def _multiquadratic(K, beta=1, sigma=1):
    """Inverse multiquadratic function."""
    K = (sigma**2 + K) ** beta
    return K


def _inverse_multiquadratic(K, beta=1, sigma=1):
    """Inverse multiquadratic function."""
    return _multiquadratic(K, beta=-beta, sigma=sigma)


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


def _regularize_kernel(K, reg=1e-10):
    """Regularize kernel matrix."""
    if np.array_equal(K, K.T):
        K.flat[:: K.shape[0] + 1] += reg
    return K


def _apply_matrix_kernel(feature_map, X, Y=None, *,
                         Cref=None, reg=1e-10, metric='euclid', **kwargs):

    """Apply a matrix kernel function."""
    _check_dimensions(X, Y, Cref)
    n_matrices_X, n, n = X.shape
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
    ktype : {'canonical', 'determinant', 'gaussian', 'laplacian', 'periodic',
                'polynomial', 'rational_quadratic'}, default='canonical'
        The type of kernel to use.
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
    msg = f"Kernel type must be in {list(kernel_types.keys())}. Got {ktype}."

    assert ktype in kernel_types.keys(), msg

    kernel_function = kernel_types[ktype]
    return kernel_function(X, Y, Cref=Cref, reg=reg, metric=metric, **kwargs)


class Gram(BaseEstimator, TransformerMixin):
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

    def __init__(self, metric, kernel_fct):
        self.metric = metric
        self.kernel_fct = kernel_fct

    def fit(self, X, y=None):
        self.data_ = X
        self.Cref = mean_covariance(X, metric=self.metric)
        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'data_'):
            self.data_ = X
            self.Cref = mean_covariance(self.data_, metric=self.metric)
        gram = self.kernel_fct(X, self.data_, Cref=self.Cref)
        return gram

    def fit_transform(self, X, y=None):
        gram = self.fit(X, y).transform(X, y)

        return gram


kernel_types = {
    'canonical': kernel_canonical,
    'determinant': kernel_determinant,
    'gaussian': kernel_gaussian,
    'laplacian': kernel_laplacian,
    'periodic': kernel_periodic,
    'polynomial': kernel_polynomial,
    'rational_quadratic': kernel_rational_quadratic,
    'exponential': kernel_exponential,
    'sigmoid': kernel_sigmoid,
    'log': kernel_log,
    'row_feature': kernel_row_feature,
    'inverse_multiquadratic': kernel_inverse_multiquadratic,

}

