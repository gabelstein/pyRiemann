def load_data(pname, directory):
    '''
    Gets a participants code and return the data.

    Parameters
    ----------
    pname: str
        code for participant (e.g. S010)

    Returns
    ----------
    tuple of dictionaries
        dictionaries with full data and auxillary data

    '''
    full_data = sio.loadmat(f"{directory}{pname}")

    emg = full_data["emg"]
    grasp = full_data["grasp"].flatten()
    grasp_rep = full_data["grasprepetition"].flatten()
    obj = full_data["object"].flatten()
    objpart = full_data["objectpart"].flatten()
    position = full_data["position"].flatten()
    dynamic = full_data["dynamic"].flatten()
    return emg, grasp, grasp_rep, obj, objpart, position, dynamic


def save_data(pname, dictionary):
    '''
    Gets a participants code and dictionary and saves the data.

    Parameters
    ----------
    pname: str
        code for participant (e.g. S010)
    dictionary: dict
        dictionary for data

    '''
    sio.savemat(f"/Volumes/Steam Library/LR_Data/Dataset 2/{pname}_short.mat", dictionary)


def classify_MDM(X, y):
    '''
    Classifies a given data-set using the Riemannian manifold classifier.

    Parameters
    ----------
    X: ndarray
        EEG data, in format Ntrials x Nchannels X Nsamples
    y: ndarray
        corresponding labels for X

    Returns
    ----------
    float
        accuracy for the classification (5-fold crossvalidation)
    '''
    startcov = time.time()
    cov = pr.estimation.Covariances().fit_transform(X)
    endcov = time.time()
    print(f"estimated covariances in {(endcov - startcov):.2f} seconds.")
    startcov = time.time()

    mdm = pr.classification.MDM(n_jobs=2)
    accuracy = cross_val_score(mdm, cov, y)
    endcov = time.time()
    print(f"classify with MDM in {(endcov - startcov):.2f} seconds.")

    return accuracy


def classify(X, y, classifier_params):
    '''
    Classifies a given data-set on Riemannian manifold.

    Parameters
    ----------
    X: ndarray
        EEG data, in format Ntrials x Nchannels X Nsamples
    y: ndarray
        corresponding labels for X

    Returns
    ----------
    float
        accuracy for the classification (5-fold crossvalidation)
    '''

    classifier = classifier_params["classifier"]
    params = classifier_params["params"]

    startcov = time.time()

    accuracy = cross_val_score(classifier(**params), X, y, n_jobs=2).mean()

    endcov = time.time()
    print(f"classify with {classifier.__name__} in {(endcov - startcov):.2f} seconds. result: {accuracy:.2f}")

    return accuracy


'''
def classify_on_tangentspace(cov_ts, y, classifier, classifier_params):
    """
    Classifies a given data-set using the Riemannian manifold classifier.

    Parameters
    ----------
    X: ndarray
        EEG data, in format Ntrials x Nchannels X Nsamples
    y: ndarray
        corresponding labels for X

    Returns
    ----------
    float
        accuracy for the classification (5-fold crossvalidation)
    """

    startcov = time.time()
    accuracy = cross_val_score(classifier(**classifier_params), cov_ts, y, n_jobs=2).mean()
    endcov = time.time()
    print(f"classify with {classifier.__name__} in {(endcov - startcov):.2f} seconds. result: {accuracy:.2f}")

    return accuracy
'''


def make_sliding_subarrays(array, window_size, step_size):
    length = array.shape[0]
    steps = int(np.ceil(min(window_size, length - window_size + 1) / step_size))
    result = [[] for i in range(0, steps)]
    for i in range(0, steps):
        split = np.split(array, range(i * step_size, length, window_size), axis=0)
        if len(split) > 1:
            if split[0].shape[0] < window_size:
                split = split[1:]
            if split[-1].shape[0] < window_size:
                split = split[:-1]
            result[i] = np.array(split).swapaxes(1, 2)

        else:
            print(split)
            print(array)
    return np.vstack(result)


def sliding_windows_X(emg, grasp, grasprepetition, intlength=1000, step_size=100):
    valid_reps = grasprepetition >= 0
    grasps = np.unique(grasp)
    graspreps = np.unique(grasprepetition)

    n_grasps = grasps.size
    n_reps = graspreps.size
    X = [[] for i in range(n_grasps)]
    y = [[] for i in range(n_grasps)]

    for i, g in enumerate(grasps):
        for k, gr in enumerate(graspreps):
            true_array = (valid_reps * (grasp == g) * (grasprepetition == gr))

            data = emg[true_array, :]
            if data.shape[0] < 400:
                continue
            else:
                X[i].append(make_sliding_subarrays(data, intlength, step_size))
                y[i].append(np.ones(X[i][-1].shape[0]) * g)
        X[i] = np.concatenate(X[i])
        y[i] = np.concatenate(y[i])
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y


def make_Xy(emg, grasp, grasp_rep, intlength=400, step_size=200):
    start = time.time()
    X, y = sliding_windows_X(emg, grasp, grasp_rep, intlength=intlength, step_size=step_size)
    end = time.time()
    # print(f"X,y sliding window time: {(end - start):.2f}s for {X.shape}")
    return X, y


def make_cov(X):
    startcov = time.time()
    cov = pr.estimation.Covariances().fit_transform(X)
    endcov = time.time()

    return cov


def make_cov_reg(X, regularizer='lwf'):
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


def write_json(in_dict, filename):
    js = json.dumps(in_dict)
    f = open(f"/content/drive/My Drive/emgData/{filename}.json", "w")
    f.write(js)
    f.close()
    f = open(f"/content/drive/My Drive/emgData/{filename}{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w")
    f.write(js)
    f.close()


def write_pickle(filename, data, path="/content/drive/My Drive/emgData/results/"):
    with open(f"{path}{filename}.pk", "wb") as fp:  # Pickling
        pickle.dump(data, fp)


def append_to_pickle(filename, data, path="/content/drive/My Drive/emgData/results/"):
    with open(f"{path}{filename}.pk", "rb") as fp:  # Unpickling
        b = pickle.load(fp)
    with open(f"{path}{filename}{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pk", "wb") as fp:  # Pickling
        pickle.dump(b, fp)

    b.append(data)
    with open(f"{path}{filename}.pk", "wb") as fp:  # Pickling
        pickle.dump(b, fp)


def make_resdict(participant, intlength, stepsize, classifier_name, classifier_params, accuracy):
    return {"participant": participant,
            "intlength": intlength,
            "stepsize": stepsize,
            "classifier_name": classifier_name,
            "classifier_params": classifier_params,
            "accuracy": accuracy}


def runjob(subjects, mf_classifiers, ts_classifiers, intlength, stepsize, filename):
    for subject in subjects:
        print(f"participant {subject} (intlength: {intlength}, stepsize: {stepsize})")
        emg, grasp, grasp_rep = load_data(subject)
        X, y = make_Xy(emg, grasp, grasp_rep, intlength, stepsize)
        cov = make_cov(X)
        for mfc in mf_classifiers:
            acc = classify(cov, y, mfc)
            print(f"{mfc['classifier'].__name__} with {mfc['params']} - accuracy: {acc}")
            resdict = make_resdict(subject, intlength, stepsize, mfc["classifier"].__name__, mfc["params"], acc)
            append_to_pickle(filename, resdict)

        ts_proj = ts_projection(cov)

        for tsc in ts_classifiers:
            acc = classify(ts_proj, y, tsc)
            print(f"{tsc['classifier'].__name__} with {tsc['params']} - accuracy: {acc}")
            resdict = make_resdict(subject, intlength, stepsize, tsc["classifier"].__name__, tsc["params"], acc)
            append_to_pickle(filename, resdict)


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


def evaluate_subjects_n_fold_cv(subj_list, subj_dir, classification, folds=4):
    for subj in subj_list:
        startsubj = time.time()
        emg, grasp, grasp_rep, obj, objpart, position, dynamic = load_data(subj, subj_dir)
        valid_reps = (grasp_rep >= 0)
        emg, grasp, grasp_rep, obj, objpart, position, dynamic = emg[valid_reps], grasp[valid_reps], grasp_rep[
            valid_reps], obj[valid_reps], objpart[valid_reps], position[valid_reps], dynamic[valid_reps]
        labels = [grasp + 500, obj + 500, position + 500, dynamic + 500, objpart + 500]

        idx = cv_split_by_labels(labels, grasp_rep, folds)

        Xy = np.array([make_Xy(emg[part], grasp[part], grasp_rep[part], 400, 40) for part in idx])
        X = Xy[:, 0]
        y = Xy[:, 1]
        covs = [make_cov(split) for split in X]
        train, test = make_sklearn_cv_idx([len(cov) for cov in covs])
        data = np.concatenate(covs)
        labels = np.concatenate(y)
        print(data.shape)
        print(labels.shape)

        result = classification(data, labels, train, test)

        end_subj = time.time()
        print(f"{subj}: {result} in {end_subj - startsubj}s")

        # print(f"{subj} {clf.cv_results_} in {end_subj-startsubj}s")
        """
        for k in range(folds):
    
          print(f"start cpu {k}")
          starttime = time.time()
    
          cov_train = np.concatenate(np.delete(covs,k,axis=0))
          cov_test = covs[k]
    
          y_train = np.concatenate(np.delete(y,k,axis=0))
          y_test = y[k]
    
          if on_ts:
            tr_data, te_data = ts_projection(cov_train, cov_test)
            del cov_train, cov_test
    
          else:
            tr_data = cov_train
            te_data = cov_test
    
    
          endtime = time.time()
          print(f"done cpu {k} in {endtime-starttime}")
          clf = sklearn.model_selection.GridSearchCV(classifier, classifier_params)
          clf.fit(tr_data, y_train)
    
          scores.append(clf.best_score_(te_data, y_test))
          print(f"subj {subj} leaveout {k} score {scores[k]}")
        """
        # del emg, grasp, grasp_rep, tr_data, te_data
        # end_subj = time.time()
        # print(f"{classifier} {np.mean(scores)} in {end_subj-startsubj}s")


def cv_split_on_raw_data(grasp, folds=5):
    unique = np.unique(grasp)
    idx = np.array([np.array_split(np.where(grasp == g)[0], folds) for g in unique]).T
    idx = np.array([np.concatenate(a) for a in idx])
    return idx


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def cv_split_by_labels(labels, splitter, cv):
    new_labels = np.sum(np.array(np.array(labels).astype("str")).astype(object), axis=0)
    new_labels_unique = np.unique(new_labels)
    idx = np.zeros((cv, splitter.size))

    for label in new_labels_unique:
        split_labels = np.unique(splitter[new_labels == label])
        for i, split in enumerate(split_labels):
            tmp = np.array(((new_labels == label) * (splitter == split))).flatten()
            idx[i % cv] += tmp

    print(np.unique(idx))

    return idx.astype(bool)


def riemann_inner_product(A, B, G=[]):
    '''
    G must be inverse of geometric mean we want to use
    '''
    A = A.reshape((12, 12))
    B = B.reshape((12, 12))

    if G == []:
        return np.trace(A @ B)
    return np.trace(G @ A @ G @ B)


def proxy_kernel(X, Y, K):
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = K(x, y)
    return gram_matrix


def riemann_kernel(X, Y=None):
    if Y is None:
        res = [[] for i in range(X.shape[0])]
        for i, cov in enumerate(X):
            A = np.matmul(X, cov, dtype=np.float32)
            res[i] = A
            del A
        print("kernel loop done")
        return res
        # return np.array([np.matmul(cov,X) for cov in X])

    return np.array([np.matmul(cov, X) for cov in Y])


def gridsearch_classification(X, y, train, test, classifier, classifier_params):
    # pipeline = sklearn.pipeline.make_pipeline(TangentSpace(metric="riemann"),classifier)

    clf = sklearn.model_selection.GridSearchCV(classifier, classifier_params, cv=zip(train, test), n_jobs=-1, verbose=2)
    clf.fit(X, y)

    return clf.best_score_


def paramsearch_classification(X, y, train, test, classifier, classifier_params):
    # pipeline = sklearn.pipeline.make_pipeline(TangentSpace(metric="riemann"),classifier)

    clf = sklearn.model_selection.GridSearchCV(classifier, classifier_params, cv=zip(train, test), n_jobs=-1, verbose=2)
    clf.fit(X, y)

    return clf.best_score_


def single_param_classification(X, y, train, test, classifier, classifier_params):
    # pipeline = sklearn.pipeline.make_pipeline(TangentSpace(metric="riemann"),classifier(**classifier_params))

    cv_results = sklearn.model_selection.cross_val_score(classifier(**classifier_params), X.reshape((-1, 144)), y,
                                                         cv=zip(train, test), n_jobs=-1, verbose=2)
    return cv_results


def riemannsvm_cv(X, y, train, test, classifier, classifier_params):
    scores = []
    for i, fold in enumerate(zip(train, test)):
        X_train, X_test = X[fold[0]], X[fold[1]]
        y_train, y_test = y[fold[0]], y[fold[1]]
        print(X_train.shape)
        print(X_test.shape)
        G_mean = pr.utils.mean.mean_riemann(X_train)
        Ginv = np.linalg.inv(G_mean)

        X_train = np.matmul(X_train, Ginv)
        X_test = np.matmul(X_test, Ginv)
        print("X_train,X_test done")
        K_train = riemann_kernel(X_train)
        print("K_train done")
        K_test = riemann_kernel(X_test, X_train)
        print("K_test done")
        del X_train
        del X_test

        clf = classifier(kernel="precomputed", **classifier_params)
        clf.fit(K_train, y_train)

        scores.append(clf.score(K_test, y_test))
    return np.mean(scores)


def chisquaredsvm_cv(X, y, train, test, classifier, classifier_params):
    scores = []
    for i, fold in enumerate(zip(train, test)):
        X_train, X_test = X[fold[0]], X[fold[1]]
        y_train, y_test = y[fold[0]], y[fold[1]]
        print(X_train.shape)
        print(X_test.shape)
        ts_train, ts_test = np.abs(ts_projection(X_train, X_test))

        clf = classifier(kernel='precomputed', **classifier_params)
        K_train = chi2_kernel(ts_train, gamma=.5)

        clf.fit(K_train, y_train)
        K_test = chi2_kernel(ts_test, ts_train, gamma=.5)

        scores.append(clf.score(K_test, y_test))
    return np.mean(scores)
