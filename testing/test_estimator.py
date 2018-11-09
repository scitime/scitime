import unittest

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from scikest.estimate import Estimator


class TestEstimate(unittest.TestCase):
    inputs, outputs = None, None

    def setUp(self):
        self.estimator = Estimator(meta_algo='RF', verbose=0)

    def test_fetch_name(self):
        rf = RandomForestRegressor()
        svc = SVC()
        kmeans = KMeans()
        rf_name = self.estimator._fetch_name(rf)
        svc_name = self.estimator._fetch_name(svc)
        kmeans_name = self.estimator._fetch_name(kmeans)
        assert rf_name == 'RandomForestRegressor'
        assert svc_name == 'SVC'
        assert kmeans_name == 'KMeans'


    def test_estimate_duration_regression(self):
        rf = RandomForestRegressor()
        X, y_continuous = np.random.rand(10000, 10), np.random.rand(10000, 1)
        rf_duration = self.estimator.estimate_duration(rf, X, y_continuous)
        assert type(rf_duration[0]) == np.float64
        assert type(rf_duration[1]) == np.float64
        assert type(rf_duration[2]) == np.float64
        assert rf_duration[1] <= rf_duration[0]
        assert rf_duration[0] <= rf_duration[2]


    def test_estimate_duration_classification(self):
        svc = SVC()
        X, y_class = np.random.rand(10000, 10), np.random.randint(0, 4, 10000)
        svc_duration = self.estimator.estimate_duration(svc, X, y_class)
        assert type(svc_duration[0]) == np.float64
        assert type(svc_duration[1]) == np.float64
        assert type(svc_duration[2]) == np.float64
        assert svc_duration[1] <= svc_duration[0]
        assert svc_duration[0] <= svc_duration[2]

    def test_estimate_duration_unsupervised(self):
        kmeans = KMeans()
        X = np.random.rand(10000, 10)
        kmeans_duration = self.estimator.estimate_duration(kmeans, X)
        assert type(kmeans_duration[0]) == np.float64
        assert type(kmeans_duration[1]) == np.float64
        assert type(kmeans_duration[2]) == np.float64
        assert kmeans_duration[1] <= kmeans_duration[0]
        assert kmeans_duration[0] <= kmeans_duration[2]


if __name__ == '__main__':
    unittest.main()
