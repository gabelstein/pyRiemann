
import time

import pyriemann as pr
from pyriemann.tangentspace import TangentSpace
import numpy as np


def make_Xy(data, label, intlength=200, step_size=20):
    labels = np.unique(label)
    n_labels = labels.size

    splitter = np.argwhere(np.diff(label) != 0)[:, 0] + 1
    datasplit = np.split(data, splitter)
    labelsplit = np.split(label, splitter)

    X = [[] for i in range(len(labelsplit))]
    y = [[] for i in range(len(labelsplit))]

    for i, datachunk in enumerate(datasplit):
        if datachunk.shape[0] < intlength:
            X[i] = np.array([]).reshape((0, data.shape[1], intlength))
            continue
        else:
            length = datachunk.shape[0]
            steps = int(np.ceil(min(intlength, length - intlength + 1) / step_size))
            result = [[] for i in range(0, steps)]
            for k in range(0, steps):
                split = np.split(datachunk, range(k * step_size, length, intlength), axis=0)
                if len(split) > 1:
                    if split[0].shape[0] < intlength:
                        split = split[1:]
                    if split[-1].shape[0] < intlength:
                        split = split[:-1]
                    result[k] = np.array(split).swapaxes(1, 2)

                else:
                    print(split)
                    print(array)

            X[i] = np.vstack(result)
            y[i] = (np.ones(X[i].shape[0]) * labelsplit[i][0])
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y


def make_cov(X):
    startcov = time.time()
    cov = pr.estimation.Covariances().fit_transform(X)
    endcov = time.time()

    return cov


def make_cov_reg(X, regularizer='oas'):
    startcov = time.time()
    cov = pr.estimation.Covariances(regularizer).fit_transform(X)
    endcov = time.time()
    # print(f"covariance estimation in {(endcov - startcov):.2f} seconds.")
    return cov


def ts_projection(cov_tr, cov_te=None, metric="riemann"):
    startcov = time.time()
    ts = TangentSpace(metric=metric)
    ref_fitter = ts.fit(cov_tr)

    if cov_te is None:
        ts_tr = ref_fitter.transform(cov_tr)
        endcov = time.time()
        return ts_tr

    else:
        ts_tr = ref_fitter.transform(cov_tr)
        ts_te = ref_fitter.transform(cov_te)

        endcov = time.time()
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

