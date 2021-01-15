import numpy
from numba import njit


@njit
def _check_SPD(eigvals):
    if not numpy.positive(eigvals).all():
        raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error.")


@njit
def _matrix_operator(Ci, operator, check_SPD=True):
    """matrix equivalent of an operator."""
    try:
        eigvals, eigvects = numpy.linalg.eigh(Ci)
    except Exception:
        raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error.")

    if check_SPD:
        _check_SPD(eigvals)
    eigvals = numpy.diag(operator(eigvals))
    Out = numpy.dot(numpy.dot(eigvects, eigvals), eigvects.T)
    return Out


@njit
def sqrtm(Ci):
    """Return the matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix square root

    """
    return _matrix_operator(Ci, numpy.sqrt)


@njit
def logm(Ci):
    """Return the matrix logarithm of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \log{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix logarithm

    """
    return _matrix_operator(Ci, numpy.log)


@njit
def expm(Ci):
    """Return the matrix exponential of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \exp{(\mathbf{\Lambda})} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the matrix exponential

    """
    return _matrix_operator(Ci, numpy.exp, False)


@njit
def invsqrtm(Ci):
    """Return the inverse matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root

    """
    eigvals, eigvects = numpy.linalg.eigh(Ci)
    _check_SPD(eigvals)
    eigvals = numpy.diag(1./numpy.sqrt(eigvals))
    Out = numpy.dot(numpy.dot(eigvects, eigvals), eigvects.T)
    return Out

@njit
def invm(Ci):
    """Return the inverse matrix square root of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root

    """
    eigvals, eigvects = numpy.linalg.eigh(Ci)
    _check_SPD(eigvals)
    eigvals = numpy.diag(1./eigvals)
    Out = numpy.dot(numpy.dot(eigvects, eigvals), eigvects.T)
    return Out


@njit
def powm(Ci, alpha):
    """Return the matrix power :math:`\\alpha` of a covariance matrix defined by :

    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{\\alpha} \mathbf{V}^T

    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`

    :param Ci: the coavriance matrix
    :param alpha: the power to apply
    :returns: the matrix power

    """
    eigvals, eigvects = numpy.linalg.eigh(Ci)
    _check_SPD(eigvals)
    eigvals = numpy.diag(eigvals**alpha)
    Out = numpy.dot(numpy.dot(eigvects, eigvals), eigvects.T)
    return Out