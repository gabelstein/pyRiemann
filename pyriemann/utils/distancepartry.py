"""Distance utils."""
import numpy
from numba import njit, prange
from .base import logm, sqrtm, invsqrtm


@njit
def distance_kullback(A, B):
    """Kullback leibler divergence between two covariance matrices A and B.

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Kullback leibler divergence between A and B

    """
    dim = A.shape[0]
    logdet = numpy.log(numpy.linalg.det(B) / numpy.linalg.det(A))
    kl = numpy.trace(numpy.dot(numpy.linalg.inv(B), A)) - dim + logdet
    return 0.5 * kl


@njit
def distance_kullback_right(A, B):
    """wrapper for right kullblack leibler div."""
    return distance_kullback(B, A)


@njit
def distance_kullback_sym(A, B):
    """Symetrized kullback leibler divergence."""
    return distance_kullback(A, B) + distance_kullback_right(A, B)


@njit
def distance_euclid(A, B):
    """Euclidean distance between two covariance matrices A and B.

    The Euclidean distance is defined by the Froebenius norm between the two
    matrices.

    .. math::
            d = \Vert \mathbf{A} - \mathbf{B} \Vert_F

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Eclidean distance between A and B

    """
    return numpy.linalg.norm(A - B)


@njit
def distance_logeuclid(A, B):
    """Log Euclidean distance between two covariance matrices A and B.

    .. math::
            d = \Vert \log(\mathbf{A}) - \log(\mathbf{B}) \Vert_F

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Eclidean distance between A and B

    """
    return distance_euclid(logm(A), logm(B))


@njit
def distance_riemann(A, B):
    Bm12 = invsqrtm(B)
    eigvals = numpy.linalg.eigvalsh(numpy.dot(numpy.dot(Bm12, A), Bm12))
    log = numpy.log(eigvals) ** 2
    sqrtsum = numpy.sqrt((log).sum())
    return sqrtsum


@njit
def distance_logdet(A, B):
    """Log-det distance between two covariance matrices A and B.

    .. math::
            d = \sqrt{\left(\log(\det(\\frac{\mathbf{A}+\mathbf{B}}{2})) - 0.5 \\times \log(\det(\mathbf{A}) \det(\mathbf{B}))\\right)}  # noqa

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Log-Euclid distance between A and B

    """
    return numpy.sqrt(numpy.log(numpy.linalg.det(
        (A + B) / 2.0)) - 0.5 *
                      numpy.log(numpy.linalg.det(A) * numpy.linalg.det(B)))


@njit
def distance_wasserstein(A, B):
    """Wasserstein distance between two covariances matrices.

    .. math::
        d = \left( {tr(A + B - 2(A^{1/2}BA^{1/2})^{1/2})}\\right )^{1/2}

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Wasserstein distance between A and B

    """
    B12 = sqrtm(B)
    C = sqrtm(numpy.dot(numpy.dot(B12, A), B12))
    return numpy.sqrt(numpy.trace(A + B - 2 * C))


@njit
def distance(A, B, metric='riemann'):
    """Distance between two covariance matrices A and B according to the metric.

    :param A: First covariance matrix
    :param B: Second covariance matrix
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
    'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
    'kullback_sym'.
    :returns: the distance between A and B

    """
    if metric == 'riemann':
        return distance_riemann(A, B)
    elif metric == 'logeuclid':
        return distance_logeuclid(A, B)
    elif metric == 'euclid':
        return distance_euclid(A, B)
    elif metric == 'logdet':
        return distance_logdet(A, B)
    elif metric == 'wasserstein':
        return distance_wasserstein(A, B)
    elif metric == 'kullback_right':
        return distance_kullback_right(A, B)
    elif metric == 'kullback':
        return distance_kullback(A, B)
    elif metric == 'kullback_sym':
        return distance_kullback_sym(A, B)
    else:
        raise ValueError('Unknown distance method')


@njit(parallel=True)
def pairwise_distance_self(X, metric='riemann'):
    """Pairwise distance matrix

    :param A: fist Covariances instance
    :param B: second Covariances instance (optional)
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
    'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
    'kullback_sym'.
    :returns: the distances between pairs of elements of X or between elements
    of X and Y.

    """
    Ntx, _, _ = X.shape
    dist = numpy.zeros((Ntx, Ntx))
    for i in prange(Ntx):
        for j in prange(i + 1, Ntx):
            dist[i][j] = distance(X[i], X[j], metric)
    return dist


@njit(parallel=True)
def pairwise_distance(X, Y, metric='riemann'):
    """Pairwise distance matrix

    :param A: fist Covariances instance
    :param B: second Covariances instance (optional)
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
    'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
    'kullback_sym'.
    :returns: the distances between pairs of elements of X or between elements
    of X and Y.

    """
    Ntx, _, _ = X.shape
    Nty, _, _ = Y.shape
    dist = numpy.empty((Ntx, Nty))
    for i in prange(Ntx):
        for j in prange(Nty):
            dist[i][j] = distance(X[i], Y[j], metric)
    return dist

@njit(parallel=True)
def pairwise_distance_single(X, y, metric='riemann'):
    """Pairwise distance matrix

    :param A: fist Covariances instance
    :param B: second Covariances instance (optional)
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' ,
    'logeuclid' , 'euclid' , 'logdet', 'kullback', 'kullback_right',
    'kullback_sym'.
    :returns: the distances between pairs of elements of X or between elements
    of X and Y.

    """
    Ntx, _, _ = X.shape
    dist = numpy.empty((Ntx))
    for i in prange(Ntx):
        dist[i] = distance(X[i], y, metric)
    return dist


def _check_distance_method(metric):
    """checks methods """
    if not metric in ['riemann', 'logeuclid', 'euclid', 'logdet', 'wasserstein', 'kullback', 'kullback_right',
                      'kullback_sym']:
        raise ValueError('Unknown distance method')

