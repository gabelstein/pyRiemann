"""Module for classification function."""
import numpy

from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed

from .utils.mean import mean_covariance, _check_mean_method
from .utils.distance import distance, _check_distance_method, pairwise_distance_single
from .tangentspace import FGDA, TangentSpace


class MDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean.
    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid.
    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    Attributes
    ----------
    covmeans_ : list
        the class centroids.
    classes_ : list
        list of classes.
    See Also
    --------
    Kmeans
    FgMDM
    KNearestNeighbor
    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry," in IEEE
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.
    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.
    """

    def __init__(self, metric='riemann'):
        """Init."""
        # store params for cloning purpose
        self.metric = metric
        _check_mean_method(metric)
        _check_distance_method(metric)
        self.metric_mean = metric
        self.metric_dist = metric

    def fit(self, X, y, sample_weight=numpy.empty(0)):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample. if None, each sample is treated with
            equal weights.
        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        self.classes_ = numpy.unique(y)

        if sample_weight.size == 0:
            sample_weight = numpy.ones(X.shape[0])
        Nc = self.classes_.size
        Nt, Ne, Ne = X.shape
        self.covmeans_ = numpy.zeros((Nc, Ne, Ne))
        for l in range(Nc):
            self.covmeans_[l] = mean_covariance(X[y == self.classes_[l]], metric=self.metric_mean,
                            sample_weight=sample_weight[y == self.classes_[l]])
        return self

    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""

        Nc = len(self.classes_)
        Nt, Ne, Ne = covtest.shape
        dist = numpy.zeros((Nc, Nt))
        for m in range(Nc):
            dist[m] = pairwise_distance_single(covtest, self.covmeans_[m], self.metric_dist)

        return dist

    def predict(self, covtest):
        """get the predictions.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        return self.classes_[dist.argmin(axis=0)]

    def transform(self, X):
        """get the distance to each centroid.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(-self._predict_distances(X))


class FgMDM(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean with geodesic filtering.
    Apply geodesic filtering described in [1], and classify using MDM algorithm
    The geodesic filtering is achieved in tangent space with a Linear
    Discriminant Analysis, then data are projected back to the manifold and
    classifier with a regular mdm.
    This is basically a pipeline of FGDA and MDM
    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    See Also
    --------
    MDM
    FGDA
    TangentSpace
    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.
    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of
    covariance matrices using a Riemannian-based kernel for BCI applications",
    in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self, metric='riemann', tsupdate=False):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate

        if isinstance(metric, str):
            self.metric_mean = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y):
        """Fit FgMDM.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : FgMDM instance
            The FgMDM instance.
        """
        self._mdm = MDM(metric=self.metric)
        self._fgda = FGDA(metric=self.metric_mean, tsupdate=self.tsupdate)
        cov = self._fgda.fit_transform(X, y)
        self._mdm.fit(cov, y)
        return self

    def predict(self, X):
        """get the predictions after FGDA filtering.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict(cov)

    def predict_proba(self, X):
        """Predict proba using softmax after FGDA filtering.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict_proba(cov)

    def transform(self, X):
        """get the distance to each centroid after FGDA filtering.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        dist : ndarray, shape (n_trials, n_cluster)
            the distance to each centroid according to the metric.
        """
        cov = self._fgda.transform(X)
        return self._mdm.transform(cov)


class TSclassifier(BaseEstimator, ClassifierMixin):
    """Classification in the tangent space.
    Project data in the tangent space and apply a classifier on the projected
    data. This is a simple helper to pipeline the tangent space projection and
    a classifier. Default classifier is LogisticRegression
    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    clf: sklearn classifier (default LogisticRegression)
        The classifier to apply in the tangent space
    See Also
    --------
    TangentSpace
    Notes
    -----
    .. versionadded:: 0.2.4
    """

    def __init__(self, metric='riemann', tsupdate=False,
                 clf=LogisticRegression()):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate
        self.clf = clf

        if not isinstance(clf, ClassifierMixin):
            raise TypeError('clf must be a ClassifierMixin')

    def fit(self, X, y):
        """Fit TSclassifier.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : TSclassifier. instance
            The TSclassifier. instance.
        """
        ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        self._pipe = make_pipeline(ts, self.clf)
        self._pipe.fit(X, y)
        return self

    def predict(self, X):
        """get the predictions.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict(X)

    def predict_proba(self, X):
        """get the probability.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of ifloat, shape (n_trials, n_classes)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict_proba(X)


class KNearestNeighbor(MDM):
    """Classification by K-NearestNeighbor.
    Classification by nearest Neighbors. For each point of the test set, the
    pairwise distance to each element of the training set is estimated. The
    class is affected according to the majority class of the k nearest
    neighbors.
    Parameters
    ----------
    n_neighbors : int, (default: 5)
        Number of neighbors.
    metric : string | dict (default: 'riemann')
        The type of metric used for distance estimation.
        see `distance` for the list of supported metric.
    Attributes
    ----------
    classes_ : list
        list of classes.
    See Also
    --------
    Kmeans
    MDM
    """

    def __init__(self, n_neighbors=5, metric='riemann'):
        """Init."""
        # store params for cloning purpose
        self.n_neighbors = n_neighbors
        MDM.__init__(self, metric=metric)

    def fit(self, X, y):
        """Fit (store the training data).
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : NearestNeighbor instance
            The NearestNeighbor instance.
        """
        self.classes_ = y
        self.covmeans_ = X

        return self

    def predict(self, covtest):
        """get the predictions.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        neighbors_classes = self.classes_[numpy.argsort(dist)]
        out, _ = stats.mode(neighbors_classes[:, 0:self.n_neighbors], axis=1)
        return out.ravel()


class rSVC(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid.

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.

    Attributes
    ----------
    covmeans_ : list
        the class centroids.
    classes_ : list
        list of classes.

    See Also
    --------
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry," in IEEE
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.
    """

    def __init__(self, svc_clf, metric='riemann'):
        """Init."""
        # store params for cloning purpose
        self.metric = metric
        self.svc_clf = svc_clf(kernel="precomputed")

        if isinstance(metric, str):
            self.metric_mean = metric
            self.metric_dist = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'distance']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "distance"')

            self.metric_mean = metric['mean']
            self.metric_dist = metric['distance']

        else:
            raise TypeError('metric must be dict or str')

    def fit(self, X, y, sample_weight=numpy.empty(0)):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample. if None, each sample is treated with
            equal weights.

        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        self.classes_ = numpy.unique(y)

        if sample_weight.size == 0:
            sample_weight = numpy.ones(X.shape[0])

        self.covmeans_ = [mean_covariance(X[y == l], metric=self.metric_mean,
                                          sample_weight=sample_weight[y == l])
                          for l in self.classes_]


        return self

    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        Nc = len(self.covmeans_)

        dist = [distance(covtest, self.covmeans_[m], self.metric_dist)
                for m in range(Nc)]
        dist = numpy.concatenate(dist, axis=1)
        return dist

    def predict(self, covtest):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(-self._predict_distances(X))


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
            res[i] = np.trace(A, axis1=1, axis2=2)
            del A
        print("kernel loop done")
        return res
        # return np.array([np.matmul(cov,X) for cov in X])

    return np.array([np.trace(np.matmul(cov, X), axis1=1, axis2=2) for cov in Y])
