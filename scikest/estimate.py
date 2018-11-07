"""Class used to instantiate an object that will estimate the running time of the user's model"""
import os
import psutil

import joblib
import pandas as pd
import numpy as np
import time

import warnings

warnings.simplefilter("ignore")

from scikest.utils import LogMixin, get_path, config, timeout


class Estimator(LogMixin):
    # default meta-algorithm
    META_ALGO = 'RF'

    def __init__(self, meta_algo=META_ALGO, verbose=0):
        self.meta_algo = meta_algo
        self.verbose = verbose

    @property
    def num_cpu(self):
        return os.cpu_count()

    @property
    def memory(self):
        return psutil.virtual_memory()

    @timeout(1)
    def _fit_start(self, algo, X, y=None):
        """
        starts fitting the model to make sure the fit is legit, throws error if error happens before 1 sec
        raises a TimeoutError if no other exception is raised before
        used in the estimate_duration function

        :param algo: algo used
        :param X: inputs for the algo
        :param y: outputs for the algo
        """
        algo.verbose = 0
        algo_name = self._fetch_name(algo)
        params = config(algo_name)
        algo_type = params['type']
        if algo_type == 'unsupervised':
            algo.fit(X)
        else:
            algo.fit(X, y)
        time.sleep(1)

    @staticmethod
    def _fetch_name(algo):
        """
        retrieves algo name from sklearn model

        :param algo: sklearn model
        :return: algo name
        :rtype: str
        """

        return type(algo).__name__

    @staticmethod
    def _fetch_inputs(params):
        """
        retrieves estimation inputs (made dummy)

        :param params: dict data
        :return: list of inputs
        """
        return params['other_params'] + [i for i in params['external_params'].keys()] \
               + [i for i in params['internal_params'].keys() if
                  i not in params['dummy_inputs']] \
               + [i + '_' + str(k) for i in params['internal_params'].keys() if
                  i in params['dummy_inputs'] for k in params['internal_params'][i]]

    def _estimate_intervals(self, estimator, X, percentile=95):
        """
        estimate the prediction intervals for one data-point
        #inputs
        :return: low and high values of the percentile-confidence interval
        :rtype: tuple

        """

        preds = []
        for pred in estimator.estimators_:
            preds.append(pred.predict(X[x])[0])
        err_down = np.percentile(preds, (100 - percentile) / 2. )
        err_up = np.percentile(preds, 100 - (100 - percentile) / 2.)
        return err_down, err_up

    def _estimate(self, algo, X, y=None, percentile=95):
        """
        estimates the model's training time given that the fit starts

        :param X: np.array of inputs to be trained
        :param y: np.array of outputs to be trained (set to None if unsupervised algo)
        :param algo: algo whose runtime the user wants to predict
        :param percentile: prediction interval percentile
        :return: predicted runtime, low and high values of the percentile-confidence interval
        :rtype: tuple
        """
        # fetching sklearn model of the end user
        algo_name = self._fetch_name(algo)
        if algo_name not in config("supported_algos"):
            raise ValueError(f'{algo_name} not currently supported by this package')

        if self.meta_algo not in config('supported_meta_algos'):
            raise ValueError(f'meta algo {self.meta_algo} currently not supported')

        params = config(algo_name)
        estimation_inputs = self._fetch_inputs(params)

        if self.verbose >= 2:
            self.logger.info(f'Fetching estimator: {self.meta_algo}_{algo_name}_estimator.pkl')
        path = f'{get_path("models")}/{self.meta_algo}_{algo_name}_estimator.pkl'
        estimator = joblib.load(path)

        n = X.shape[0]
        p = X.shape[1]
        # retrieving all parameters of interest
        inputs = [self.memory.total, self.memory.available, self.num_cpu, n, p]

        if params["type"] == "classification":
            num_cat = len(np.unique(y))
            inputs.append(num_cat)

        algo_params = algo.get_params()
        param_list = params['other_params'] + list(params['external_params'].keys()) + list(
            params['internal_params'].keys())

        for i in params['internal_params'].keys():
            # handling n_jobs=-1 case
            if i == 'n_jobs':
                if algo_params[i] == -1:
                    inputs.append(self.num_cpu)
                else:
                    inputs.append(algo_params[i])

            else:
                if i in params['dummy_inputs']:
                    # to make dummy
                    inputs.append(str(algo_params[i]))
                else:
                    inputs.append(algo_params[i])

        # making dummy
        dic = dict(zip(param_list, [[i] for i in inputs]))
        if self.verbose >= 2:
            self.logger.info(f'Training your model for these params: {dic}')

        df = pd.DataFrame(dic, columns=param_list)
        df = pd.get_dummies(df)

        # adding 0 columns for columns that are not in the dataset, assuming it s only dummy columns
        inputs_to_fill = list(set(list(estimation_inputs)) - set(list((df.columns))))
        missing_inputs = list(set(list(df.columns)) - set(list((estimation_inputs))))
        if self.verbose >= 1 and (len(missing_inputs) > 0):
            self.logger.warning(f'Parameters {missing_inputs} will not be accounted for')
        for i in inputs_to_fill:
            df[i] = 0
        df = df[estimation_inputs]

        X = (df[estimation_inputs]
                 ._get_numeric_data()
                 .dropna(axis=0, how='any')
                 .as_matrix())
        prediction = estimator.predict(X)

        errors = self._estimate_intervals(estimator, X, percentile)

        if self.verbose >= 2:
            self.logger.info(f'Training your model should take ~ {prediction[0]} seconds')
            self.logger.info(f'The prediction interval is ~ {prediction[0]} seconds')
        return prediction, errors[0], errors[1]

    def estimate_duration(self, algo, X, y=None):
        """
        predicts training runtime for a given training

        :param X: np.array of inputs to be trained
        :param y: np.array of outputs to be trained (set to None is unsupervised algo)
        :param algo: algo whose runtime the user wants to predict
        :return: predicted runtime
        :rtype: float
        """
        try:
            self._fit_start(algo=algo, X=X, y=y)
        except Exception as e:
            if e.__class__.__name__ != 'TimeoutError':
                # this means that the sklearn fit has raised a natural exception before we artificially raised a timeout
                raise e
            else:
                if self.verbose >= 2:
                    self.logger.info('The model would fit. Moving on')
                return self._estimate(algo, X, y)
