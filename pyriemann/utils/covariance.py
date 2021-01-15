import numpy
np = numpy
import multiprocessing
from joblib import Parallel, delayed
from sklearn.covariance import oas, ledoit_wolf, fast_mcd, empirical_covariance
from numba import njit, prange


@njit
def oas_par(X):
    """Estimate covariance with the Oracle Approximating Shrinkage algorithm.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.
    assume_centered : bool, default=False
      If True, data will not be centered before computation.
      Useful to work with data whose mean is significantly equal to
      zero but is not exactly zero.
      If False, data will be centered before computation.
    Returns
    -------
    shrunk_cov : array-like of shape (n_features, n_features)
        Shrunk covariance.
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.
    Notes
    -----
    The regularised (shrunk) covariance is:
    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
    where mu = trace(cov) / n_features
    The formula we used to implement the OAS is slightly modified compared
    to the one given in the article. See :class:`OAS` for more details.
    """
    X = np.asarray(X).T

    for i in range(X.shape[0]):
        X[i] = X[i] - np.mean(X[i])

    n_features, n_samples = X.shape
    emp_cov = np.dot(X, X.T) / n_samples
    mu = np.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = np.mean(emp_cov ** 2)
    num = alpha + mu ** 2
    den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)

    shrinkage = 1. if den == 0 else min(num / den, 1.)
    shrunk_cov = (1. - shrinkage) * emp_cov

    add_shrink_mu = np.zeros((n_features**2))
    add_shrink_mu[::n_features + 1] += shrinkage * mu
    add_shrink_mu = add_shrink_mu.reshape((n_features,n_features))
    shrunk_cov += add_shrink_mu
    """
    shrunk_cov.flat[::n_features + 1] += shrinkage * mu
    """
    return shrunk_cov


def _lwf(X):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = ledoit_wolf(X.T)
    return C


def _oas(X):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = oas(X.T)
    return C


def _scm(X):
    """Wrapper for sklearn sample covariance estimator"""
    return empirical_covariance(X.T)


def _mcd(X):
    """Wrapper for sklearn mcd covariance estimator"""
    _, C, _, _ = fast_mcd(X.T)
    return C


def _check_est(est):
    """Check if a given estimator is valid"""

    # Check estimator exist and return the correct function
    estimators = {
        'cov': numpy.cov,
        'scm': _scm,
        'lwf': _lwf,
        'oas': _oas,
        'mcd': _mcd,
        'corr': numpy.corrcoef
    }

    if callable(est):
        # All good (cross your fingers)
        pass
    elif est in estimators.keys():
        # Map the corresponding estimator
        est = estimators[est]
    else:
        # raise an error
        raise ValueError(
            """%s is not an valid estimator ! Valid estimators are : %s or a
             callable function""" % (est, (' , ').join(estimators.keys())))
    return est


def covariances(X, estimator='cov'):
    """Special form covariance matrix."""
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    covmats = numpy.zeros((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = est(X[i])
    return covmats

@njit(parallel=True)
def covariances_par(X):
    """Estimation of covariance matrix."""
    Nt, Ne, Ns = X.shape
    covmats = numpy.zeros((Nt, Ne, Ne))
    for i in prange(Nt):
        covmats[i] = oas_par(X[i].T)
    return covmats


def covariances_EP(X, P, estimator='cov'):
    """Special form covariance matrix."""
    est = _check_est(estimator)
    Nt, Ne, Ns = X.shape
    Np, Ns = P.shape
    covmats = numpy.zeros((Nt, Ne + Np, Ne + Np))
    for i in range(Nt):
        covmats[i] = est(numpy.concatenate((P, X[i]), axis=0))
    return covmats


def eegtocov(sig, window=128, overlapp=0.5, padding=True, estimator='cov'):
    """Convert EEG signal to covariance using sliding window"""
    est = _check_est(estimator)
    X = []
    if padding:
        padd = numpy.zeros((int(window / 2), sig.shape[1]))
        sig = numpy.concatenate((padd, sig, padd), axis=0)

    Ns, Ne = sig.shape
    jump = int(window * overlapp)
    ix = 0
    while (ix + window < Ns):
        X.append(est(sig[ix:ix + window].T))
        ix = ix + jump

    return numpy.array(X)


def coherence(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute coherence."""
    cosp = cospectrum(X, window, overlap, fmin, fmax, fs)
    coh = numpy.zeros_like(cosp)
    for f in range(cosp.shape[-1]):
        psd = numpy.sqrt(numpy.diag(cosp[..., f]))
        coh[..., f] = cosp[..., f] / numpy.outer(psd, psd)
    return coh


def cospectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute Cospectrum."""
    Ne, Ns = X.shape
    number_freqs = int(window / 2)

    step = int((1.0 - overlap) * window)
    step = max(1, step)

    number_windows = int((Ns - window) / step + 1)
    # pre-allocation of memory
    fdata = numpy.zeros((number_windows, Ne, number_freqs), dtype=complex)
    win = numpy.hanning(window)

    # Loop on all frequencies
    for window_ix in range(int(number_windows)):

        # time markers to select the data
        # marker of the beginning of the time window
        t1 = int(window_ix * step)
        # marker of the end of the time window
        t2 = int(t1 + window)
        # select current window and apodize it
        cdata = X[:, t1:t2] * win

        # FFT calculation
        fdata[window_ix] = numpy.fft.fft(
            cdata, n=window, axis=1)[:, 0:number_freqs]

    # Adjust Frequency range to specified range (in case it is a parameter)
    if fmin is not None:
        f = numpy.arange(0, 1, 1.0 / number_freqs) * (fs / 2.0)
        Fix = (f >= fmin) & (f <= fmax)
        fdata = fdata[:, :, Fix]

    # fdata = fdata.real
    Nf = fdata.shape[2]
    S = numpy.zeros((Ne, Ne, Nf), dtype=complex)
    normval = numpy.linalg.norm(win)**2
    for i in range(Nf):
        S[:, :, i] = numpy.dot(fdata[:, :, i].conj().T, fdata[:, :, i]) / (
            number_windows * normval)

    return numpy.abs(S)**2
