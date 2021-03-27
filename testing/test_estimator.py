import unittest

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from scitime import RuntimeEstimator


class TestEstimate(unittest.TestCase):
    inputs, outputs = None, None

    def setUp(self):
        self.estimator_metarf = RuntimeEstimator(meta_algo='RF', verbose=0)
        self.estimator_metann = RuntimeEstimator(meta_algo='NN', verbose=0)

    def test_fetch_name_metarf(self):
        rf = RandomForestRegressor()
        svc = SVC()
        kmeans = KMeans()
        rf_name = self.estimator_metarf._fetch_algo_metadata(rf)['name']
        svc_name = self.estimator_metarf._fetch_algo_metadata(svc)['name']
        kmeans_name = \
            self.estimator_metarf._fetch_algo_metadata(kmeans)['name']

        assert rf_name == 'RandomForestRegressor'
        assert svc_name == 'SVC'
        assert kmeans_name == 'KMeans'

    def test_estimate_duration_regression_metarf(self):
        rf = RandomForestRegressor()
        X, y_continuous = np.random.rand(10000, 10), np.random.rand(10000, 1)
        rf_duration = self.estimator_metarf.time(rf, X, y_continuous)
        assert type(rf_duration[0]) == np.float64
        assert type(rf_duration[1]) == np.float64
        assert type(rf_duration[2]) == np.float64
        assert rf_duration[1] <= rf_duration[0]
        assert rf_duration[0] <= rf_duration[2]

    def test_estimate_duration_classification_metarf(self):
        svc = SVC()
        X, y_class = np.random.rand(10000, 10), np.random.randint(0, 4, 10000)
        svc_duration = self.estimator_metarf.time(svc, X, y_class)
        assert type(svc_duration[0]) == np.float64
        assert type(svc_duration[1]) == np.float64
        assert type(svc_duration[2]) == np.float64
        assert svc_duration[1] <= svc_duration[0]
        assert svc_duration[0] <= svc_duration[2]

    def test_estimate_duration_unsupervised_metarf(self):
        kmeans = KMeans()
        X = np.random.rand(10000, 10)
        kmeans_duration = self.estimator_metarf.time(kmeans, X)
        assert type(kmeans_duration[0]) == np.float64
        assert type(kmeans_duration[1]) == np.float64
        assert type(kmeans_duration[2]) == np.float64
        assert kmeans_duration[1] <= kmeans_duration[0]
        assert kmeans_duration[0] <= kmeans_duration[2]

    def test_fetch_name_metann(self):
        rf = RandomForestRegressor()
        svc = SVC()
        kmeans = KMeans()
        rf_name = self.estimator_metann._fetch_algo_metadata(rf)['name']
        svc_name = self.estimator_metann._fetch_algo_metadata(svc)['name']
        kmeans_name = \
            self.estimator_metann._fetch_algo_metadata(kmeans)['name']

        assert rf_name == 'RandomForestRegressor'
        assert svc_name == 'SVC'
        assert kmeans_name == 'KMeans'

    def test_estimate_duration_regression_metann(self):
        rf = RandomForestRegressor()
        X, y_continuous = np.random.rand(10000, 10), np.random.rand(10000, 1)
        rf_duration = self.estimator_metann.time(rf, X, y_continuous)
        assert type(rf_duration[0]) == np.float64

    def test_estimate_duration_classification_metann(self):
        svc = SVC()
        X, y_class = np.random.rand(10000, 10), np.random.randint(0, 4, 10000)
        svc_duration = self.estimator_metann.time(svc, X, y_class)
        assert type(svc_duration[0]) == np.float64

    def test_estimate_duration_unsupervised_metann(self):
        kmeans = KMeans()
        X = np.random.rand(10000, 10)
        kmeans_duration = self.estimator_metann.time(kmeans, X)
        assert type(kmeans_duration[0]) == np.float64


if __name__ == '__main__':
    unittest.main()
