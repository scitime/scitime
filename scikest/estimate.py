import os
import psutil

import joblib
import json
import pandas as pd
import numpy as np
import time

import warnings
warnings.simplefilter("ignore")

from scikest.utils import LogMixin, get_path, config, timeout
from scikest.train import Trainer


class Estimator(Trainer, LogMixin):
    """
    This class is used to instantiate an object that will estimate the running time of the user's model
    """
    ALGO_ESTIMATOR = 'RF' # This is the meta-algorithm

    def __init__(self, algo_estimator=ALGO_ESTIMATOR, verbose=True):
        super().__init__(verbose=verbose, algo_estimator=algo_estimator)
        self.algo_estimator = algo_estimator  # This is the meta-algorithm
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
        Starts fitting the model to make sure the fit is legit, throws error if error happens before 1 sec
        Raises a TimeoutError if no other exception is raised before
        Used in the estimate_duration function

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

        return str(algo).split('(')[0]

    # @nathan same function as estimation_inputs in train.py? refactor?
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


    def _estimate(self, algo, X, y=None):
        """
        estimates the model's training time given that the fit starts

        :param X: np.array of inputs to be trained
        :param y: np.array of outputs to be trained (set to None if unsupervised algo)
        :param algo: algo whose runtime the user wants to predict
        :return: predicted runtime
        :rtype: float
        """
        algo_name = self._fetch_name(algo) #this is the sklearn model of the user
        if self._fetch_name(algo_name) not in config("supported_algos"):
            raise ValueError(f'{algo_name} not currently supported by this package')

        params = config(algo_name)
        estimation_inputs = self._fetch_inputs(params)


        #@nathan  want to deprecate 'LR'?
        if self.algo_estimator == 'LR':
            if self.verbose:
                self.logger.info('Loading LR coefs from json file')
            with open('coefs/lr_coefs.json', 'r') as f:
                coefs = json.load(f)
        else:
            if self.verbose:
                self.logger.info(f'Fetching estimator: {self.algo_estimator}_{algo_name}_estimator.pkl')
            path = f'{get_path("models")}/{self.algo_estimator}_{algo_name}_estimator.pkl'
            estimator = joblib.load(path)

        # Retrieving all parameters of interest
        inputs = []
        inputs.append(self.memory.total)
        inputs.append(self.memory.available)
        inputs.append(self.num_cpu)
        n = X.shape[0]
        inputs.append(n)
        p = X.shape[1]
        inputs.append(p)
        if params["type"] == "classification":
            num_cat = len(np.unique(y))
            inputs.append(num_cat)

        algo_params = algo.get_params()
        param_list = params['other_params'] + list(params['external_params'].keys()) + list(params['internal_params'].keys())

        for i in params['internal_params'].keys():
            # Handling n_jobs=-1 case
            if i == 'n_jobs':
                if algo_params[i] == -1:
                    inputs.append(self.num_cpu)
                else:
                    inputs.append(algo_params[i])

            else:
                if i in self.params['dummy_inputs']:
                    # To make dummy
                    inputs.append(str(algo_params[i]))
                else:
                    inputs.append(algo_params[i])

        # Making dummy
        dic = dict(zip(param_list, [[i] for i in inputs]))
        if self.verbose:
            self.logger.info(f'Training your model for these params: {dic}')

        df = pd.DataFrame(dic, columns=param_list)
        df = pd.get_dummies(df)

        # adding 0 columns for columns that are not in the dataset, assuming it s only dummy columns
        inputs_to_fill = list(set(list(estimation_inputs)) - set(list((df.columns))))
        missing_inputs = list(set(list(df.columns)) - set(list((estimation_inputs))))
        if self.verbose and (len(missing_inputs) > 0):
            self.logger.warning(f'Parameters {missing_inputs} will not be accounted for')
        for i in inputs_to_fill:
            df[i] = 0
        df = df[estimation_inputs]
        if self.algo_estimator == 'LR':
            prediction = coefs[0]
            for i in range(df.shape[1]):
                prediction += df.ix[0, i] * coefs[i + 1]
        else:
            X = (df[estimation_inputs]
                 ._get_numeric_data()
                 .dropna(axis=0, how='any')
                 .as_matrix())
            prediction = estimator.predict(X)

        if self.verbose:
            self.logger.info(f'Training your model should take ~ {prediction[0]} seconds')
        return prediction

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
                #this means that the sklearn fit has raised a natural exception before we artificailly raised a timeout
                raise e
            else:
                if self.verbose:
                    self.logger.info('The model would fit. Moving on')
                return self._estimate(algo, X, y)
