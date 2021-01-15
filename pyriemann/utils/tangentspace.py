import numpy
from numba import njit, prange

from .base import sqrtm, invsqrtm, logm, expm
from .mean import mean_covariance
###############################################################
# Tangent Space
###############################################################


def tangent_space(covmats, Cref):
    """Project a set of covariance matrices in the tangent space. according to
    the reference point Cref

    :param covmats: np.ndarray
        Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)

    """
    Nt, Ne, Ne = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = numpy.triu_indices_from(Cref)
    Nf = int(Ne * (Ne + 1) / 2)
    T = numpy.empty((Nt, Nf))
    coeffs = (numpy.sqrt(2) * numpy.triu(numpy.ones((Ne, Ne)), 1) +
              numpy.eye(Ne))[idx]
    for index in range(Nt):
        tmp = numpy.dot(numpy.dot(Cm12, covmats[index]), Cm12)
        tmp = logm(tmp)
        T[index] = numpy.multiply(coeffs, tmp[idx])
    return T

@njit
def tangent_space(covmats, Cref):
    """Project a set of covariance matrices in the tangent space. according to
    the reference point Cref

    :param covmats: np.ndarray
        Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        the Tangent space , a matrix of Ntrials X (Nchannels*(Nchannels+1)/2)

    """
    Nt, Ne, Ne = covmats.shape
    Cm12 = invsqrtm(Cref)
    idx = numpy.triu_indices_from(Cref)
    flatidx = idx[0]*Ne + idx[1] #because numba cant handle double indexing
    Nf = int(Ne * (Ne + 1) / 2)
    coeffs = (numpy.sqrt(2) * numpy.triu(numpy.ones((Ne, Ne)), 1) +
              numpy.eye(Ne)).flatten()[flatidx]

    T = make_T(covmats, Cm12, Nt, Nf, coeffs, flatidx)

    return T


@njit(parallel=True)
def make_T(covmats, Cm12, Nt, Nf, coeffs, flatidx):
    T = numpy.empty((Nt, Nf))

    for index in prange(Nt):
        tmp = numpy.dot(numpy.dot(Cm12, covmats[index]), Cm12)
        tmp = logm(tmp).flatten()[flatidx]
        T[index] = numpy.multiply(coeffs, tmp)
    return T


def untangent_space(T, Cref):
    """Project a set of Tangent space vectors back to the manifold.

    :param T: np.ndarray
        the Tangent space , a matrix of Ntrials X (channels * (channels + 1)/2)
    :param Cref: np.ndarray
        The reference covariance matrix
    :returns: np.ndarray
        A set of Covariance matrix, Ntrials X Nchannels X Nchannels

    """
    Nt, Nd = T.shape
    Ne = int((numpy.sqrt(1 + 8 * Nd) - 1) / 2)
    C12 = sqrtm(Cref)

    idx = numpy.triu_indices_from(Cref)
    covmats = numpy.empty((Nt, Ne, Ne))
    covmats[:, idx[0], idx[1]] = T
    for i in range(Nt):
        triuc = numpy.triu(covmats[i], 1) / numpy.sqrt(2)
        covmats[i] = (numpy.diag(numpy.diag(covmats[i])) + triuc + triuc.T)
        covmats[i] = expm(covmats[i])
        covmats[i] = numpy.dot(numpy.dot(C12, covmats[i]), C12)

    return covmats

@njit(parallel=True)
def transport(Covs, Cref, metric='riemann'):
    """Parallel transport of two set of covariance matrix.

    """
    C = mean_covariance(Covs, metric=metric)
    iC = invsqrtm(C)
    E = sqrtm(numpy.dot(numpy.dot(iC, Cref), iC))
    out = numpy.zeros((Covs.shape))
    for i in prange(Covs.shape[0]):
        out[i] = numpy.dot(numpy.dot(E, Covs[i]), E.T)
    return out
