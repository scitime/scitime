import unittest

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from scikest.estimate import Estimator


class TestEstimate(unittest.TestCase):
    inputs, outputs = None, None

    def setUp(self):
        self.estimator = Estimator(meta_algo='RF', verbose=0)

    def test_fetch_name(self):
        rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=200,
                                   max_features=10, max_leaf_nodes=10, min_impurity_decrease=10,
                                   min_impurity_split=10, min_samples_leaf=10,
                                   min_samples_split=10, min_weight_fraction_leaf=0.5,
                                   n_estimators=100, n_jobs=10, oob_score=False, random_state=None,
                                   verbose=2, warm_start=False)
        name = self.estimator._fetch_name(rf)
        assert name == 'RandomForestRegressor'


    def test_estimate_duration(self):
        rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=200,
                                   max_features=10, max_leaf_nodes=10, min_impurity_decrease=10,
                                   min_impurity_split=10, min_samples_leaf=10,
                                   min_samples_split=10, min_weight_fraction_leaf=0.5,
                                   n_estimators=100, n_jobs=10, oob_score=False, random_state=None,
                                   verbose=2, warm_start=False)
        X, y = np.random.rand(100000, 10), np.random.rand(100000, 1)
        # run the estimation
        duration = self.estimator.estimate_duration(rf, X, y)

        assert type(duration[0]) == np.float64


if __name__ == '__main__':
    unittest.main()
