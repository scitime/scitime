import unittest

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from scikest.estimate import Estimator


class TestEstimate(unittest.TestCase):
    inputs, outputs = None, None

    def setUp(self):
        self.estimator = Estimator(meta_algo='RF', verbose=0)

    def test_fetch_name(self):
        rf = RandomForestRegressor()
        svc = SVC()
        kmeans = KMeans()
        name = self.estimator._fetch_name(rf)
        assert name == 'RandomForestRegressor'


    def test_estimate_duration(self):
        rf = RandomForestRegressor()
        X, y = np.random.rand(100000, 10), np.random.rand(100000, 1)
        # run the estimation
        duration = self.estimator.estimate_duration(rf, X, y)

        assert type(duration[0]) == np.float64


if __name__ == '__main__':
    unittest.main()
