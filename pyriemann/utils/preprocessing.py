import numpy as np
import pyriemann as pr
from pyriemann.tangentspace import TangentSpace
import mne
from scipy.fft import fft, ifft
import scipy


def make_Xy(data, label, intlength=200, step_size=20, adjust_class_size=True):
    """
    This function creates sliding windows for multivariate data.
    ----------
    data : ndarray
        multivariate data to turn into sliding windows based on the labels provided.
        Dimension: nSamples x nChannels
    label : ndarray
        labels for the data.
    intlength : int,
        lenght of the sliding windows. The default is 200.
    step_size : int,
        size of step until next sliding window is calculated. The default is 20. Only produces exact overlapping windows
        if intlength is an integer multiple of step_size.

    Returns
    -------
    X,y : ndarray, ndarray
        data turned into sliding windows and corresponding label for each data point.
        Dimension: nSamples x nChannels x intlength
    """
    classes,class_counts =  np.unique(label, return_counts=True)
    min_class = np.min(class_counts)
    class_ratios = class_counts/min_class
    ratio_dict = dict(zip(classes,class_ratios))

    splitter = np.argwhere(np.diff(label) != 0)[:, 0] + 1
    datasplit = np.split(data, splitter)
    labelsplit = np.split(label, splitter)

    X = [[] for i in range(len(labelsplit))]
    y = [[] for i in range(len(labelsplit))]

    for i, datachunk in enumerate(datasplit):
        if adjust_class_size:
          step = step_size*ratio_dict[labelsplit[i][0]].astype(int)
          print(step)
        else:
          step = step_size
        X[i] = np.array([datachunk[i*step:i*step+intlength].T
                         for i in range(datachunk.shape[0]//step-intlength//step)]
                        ).reshape((-1, datachunk.shape[1], intlength))

        y[i] = (np.ones(X[i].shape[0]) * labelsplit[i][0])
    X = np.concatenate(X)
    y = np.concatenate(y)

    return X, y


def make_cov(X, estimator='scm'):
    cov = pr.estimation.Covariances(estimator).fit_transform(X)
    return cov


def ts_projection(cov_tr, cov_te=None, metric="riemann"):
    ts = TangentSpace(metric=metric)
    ref_fitter = ts.fit(cov_tr)

    if cov_te is None:
        ts_tr = ref_fitter.transform(cov_tr)
        return ts_tr

    else:
        ts_tr = ref_fitter.transform(cov_tr)
        ts_te = ref_fitter.transform(cov_te)

        return ts_tr, ts_te


def make_sklearn_cv_idx(Xarr):
    cv = len(Xarr)
    fullidx = np.arange(np.sum(Xarr))
    train = [[] for i in range(cv)]
    test = [[] for i in range(cv)]
    start = 0
    end = 0
    for i, data in enumerate(Xarr):
        end += data
        test[i] = fullidx[start:end]
        train[i] = np.delete(fullidx, test[i])
        start += data
    return np.array([train, test])


def cv_split_by_labels(labels, splitter, cv):
    new_labels = np.sum(np.array(np.array(labels).astype("str")).astype(object), axis=0)
    new_labels_unique = np.unique(new_labels)
    idx = np.zeros((cv, splitter.size))

    for label in new_labels_unique:
        split_labels = np.unique(splitter[new_labels == label])
        for i, split in enumerate(split_labels):
            tmp = np.array(((new_labels == label) * (splitter == split))).flatten()
            idx[i % cv] += tmp

    return idx.astype(bool)


def calc_band_filters(f_ranges, sample_rate, filter_len=2001, l_trans_bandwidth=4, h_trans_bandwidth=4, joined=True):
    """
    This function returns for the given frequency band ranges filter coefficients with with length "filter_len"
    Thus the filters can be sequentially used for band power estimation
    Parameters
    ----------
    f_ranges : TYPE
        DESCRIPTION.
    sample_rate : float
        sampling frequency.
    filter_len : int,
        lenght of the filter. The default is 1001.
    l_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.
    h_trans_bandwidth : TYPE, optional
        DESCRIPTION. The default is 4.
    Returns
    -------
    filter_fun : array
        filter coefficients stored in rows.
    """
    filter_fun = np.zeros([len(f_ranges), filter_len])

    for a, f_range in enumerate(f_ranges):
        h = mne.filter.create_filter(None, sample_rate, l_freq=f_range[0], h_freq=f_range[1],
                                     fir_design='firwin', l_trans_bandwidth=l_trans_bandwidth,
                                     h_trans_bandwidth=h_trans_bandwidth, filter_length='1000ms')

        filter_fun[a, :] = h

    if joined:
        ffts = [fft(filt) for filt in filter_fun]

        sum_fft = np.sum(ffts, axis=0)
        filter_joined = np.real(ifft(sum_fft))

        return np.array([filter_joined])

    else:
        return filter_fun


def apply_filter(dat_, sample_rate, filter_fun, line_noise, variance=False, seglengths=None):
    """
    For a given channel, apply 4 notch line filters and apply previously calculated filters

    Parameters
    ----------
    dat_ : array (ns,)
        segment of data at a given channel and downsample index.
    sample_rate : float
        sampling frequency.
    filter_fun : array
        output of calc_band_filters.
    line_noise : int|float
        (in Hz) the line noise frequency.
    seglengths : list
        list of ints with the leght to which variance is calculated.
        Used only if variance is set to True.
    variance : bool,
        If True, return the variance of the filtered signal, else
        the filtered signal is returned.
    Returns
    -------
    filtered : array
        if variance is set to True: (nfb,) array with the resulted variance
        at each frequency band, where nfb is the number of filter bands used to decompose the signal
        if variance is set to False: (nfb, filter_len) array with the filtered signal
        at each freq band, where nfb is the number of filter bands used to decompose the signal
    """


    filtered = []

    for data in dat_:
        dat_noth_filtered = mne.filter.notch_filter(x=data, Fs=sample_rate, trans_bandwidth=7,
                                                    freqs=np.arange(line_noise, 4 * line_noise, line_noise),
                                                    fir_design='firwin', notch_widths=1,
                                                    filter_length=data.shape[0] - 1)
        filtered.append(scipy.signal.convolve(dat_noth_filtered, filter_fun[0, :], mode='same'))
    return np.array(filtered)