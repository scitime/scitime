"""Class used to instantiate an object that will estimate the running time of the user's model"""
import os
import psutil

import joblib
import pandas as pd
import numpy as np
import time
import json

import warnings

warnings.simplefilter("ignore")

from scikest.utils import get_path, config, timeout
from scikest.log import LogMixin

class Estimator(LogMixin):
    # default meta-algorithm
    META_ALGO = 'RF'

    def __init__(self, meta_algo=META_ALGO, verbose=3):
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
        algo_name = self._fetch_algo_metadata(algo)[0]
        params = config(algo_name)
        algo_type = params['type']
        if algo_type == 'unsupervised':
            algo.fit(X)
        else:
            algo.fit(X, y)
        time.sleep(1)

    @staticmethod
    def _fetch_algo_metadata(algo):
        """
        retrieves algo name from sklearn model

        :param algo: sklearn model
        :return: algo name and parameters
        :rtype: str, dict
        """
        algo_name = type(algo).__name__
        algo_params = algo.get_params()
        params = config(algo_name)

        return algo_name, algo_params, params

    @staticmethod
    def _fetch_inputs(json_path):
        """
        retrieves estimation inputs (made dummy)

        :param json_path: list of columns in json fils
        :return: list of inputs
        """
        return json.load(open(get_path(json_path)))

    def _add_semi_dummy(self, df, semi_dummy_inputs):
        """
        add columns for semi dummy inputs (continuous except for one or a few values)

        :param df: df before made dummy
        :param semi_dummy_inputs: from meta params
        :return: df transformed with added columns
        """
        if self.verbose >= 2:
            self.logger.info(f'Transforming dataset for semi dummy features')

        for key in semi_dummy_inputs:
            for sub_key in semi_dummy_inputs[key]:
                df[f'{key}_{sub_key}'] = df[f'{key}'].apply(lambda x: x == sub_key)
                df[f'{key}'] = df[f'{key}'].apply(lambda x: None if x == sub_key else x)

        return df

    def _fetch_params(self, algo, X, y):
        """
        builds a dataframe of the params of the estimated model

        :param X: np.array of inputs to be trained
        :param y: np.array of outputs to be trained (set to None if unsupervised algo)
        :param algo: algo whose runtime the user wants to predict
        :return: dataframe of all inputed parameters
        :rtype: pandas dataframe
        """
        algo_name, algo_params, params = self._fetch_algo_metadata(algo)

        n = X.shape[0]
        p = X.shape[1]
        inputs = [self.memory.total, self.memory.available, self.num_cpu, n, p]

        if params["type"] == "classification":
            num_cat = len(np.unique(y))
            inputs.append(num_cat)

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

        return df  
        
    def _tranform_params(self, algo, df):
        """
        builds a dataframe of the params of the estimated model

        :param df: dataframe of all inputed parameters
        :param algo: algo whose runtime the user wants to predict
        :return: matrix of all relevant algo parameters and system features used to estimate algo training time
        :rtype: pandas matrix object  
        """
        algo_name, algo_params, params = self._fetch_algo_metadata(algo)

        json_path = f'{get_path("models")}/{self.meta_algo}_{algo_name}_estimator.json'
        estimation_inputs = self._fetch_inputs(json_path)['dummy']
        estimation_original_inputs = self._fetch_inputs(json_path)['original']

        # first we transform semi dummy features
        semi_dummy_inputs = params['semi_dummy_inputs']
        # we add columns for each semi dummy features (*number of potential dummy values)
        df = self._add_semi_dummy(df, semi_dummy_inputs)

        forgotten_inputs = list(set(list(estimation_original_inputs)) - set(list((df.columns))))

        if len(forgotten_inputs) > 0:
            raise ValueError(f'{forgotten_inputs} parameters missing')

        df = pd.get_dummies(df.fillna(-1))

        # adding 0 columns for columns that are not in the dataset
        dummy_inputs_to_fill = list(set(list(estimation_inputs)) - set(list((df.columns))))
        missing_inputs = list(set(list(algo_params.keys())) - set(list((params['internal_params'].keys()))))

        if self.verbose >= 1 and (len(missing_inputs) > 0):
            self.logger.warning(f'Parameters {missing_inputs} will not be accounted for')

        for i in dummy_inputs_to_fill:
            df[i] = 0

        df = df[estimation_inputs]

        X = (df[estimation_inputs]
                 ._get_numeric_data()
                 .dropna(axis=0, how='any')
                 .as_matrix())

        return X

    def _estimate_interval(self, meta_estimator, X, percentile=95):
        """
        estimate the prediction intervals for one data-point
        :param meta_estimator: the fitted random-forest meta-algo
        :param X: parameters, the same as the ones fed in the meta-algo
        :return: low and high values of the percentile-confidence interval
        :rtype: tuple
        """

        if self.meta_algo == 'RF':
            preds = []
            for pred in meta_estimator.estimators_:
                preds.append(pred.predict(X)[0])
            lower_bound = np.percentile(preds, (100 - percentile) / 2. )
            upper_bound = np.percentile(preds, 100 - (100 - percentile) / 2.)
            return lower_bound, upper_bound
        else:
            #To be completed when/if we change the meta-algo
            pass

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
        algo_name, algo_params, params = self._fetch_algo_metadata(algo)
        
        if algo_name not in config("supported_algos"):
            raise ValueError(f'{algo_name} not currently supported by this package')

        if self.meta_algo not in config('supported_meta_algos'):
            raise ValueError(f'meta algo {self.meta_algo} currently not supported')

        if self.verbose >= 2:
            self.logger.info(f'Fetching estimator: {self.meta_algo}_{algo_name}_estimator.pkl')
        model_path = f'{get_path("models")}/{self.meta_algo}_{algo_name}_estimator.pkl'
        meta_estimator = joblib.load(model_path)

        # retrieving all parameters of interest:
        df = self._fetch_params(algo, X, y)

        # Transforming the inputs:
        X = self._tranform_params(algo, df)

        prediction = meta_estimator.predict(X)
        lower_bound, upper_bound = self._estimate_interval(meta_estimator, X, percentile)

        if self.verbose >= 2:
            self.logger.info('Training your model should take ~ {:.2} seconds'.format(prediction[0]))
            self.logger.info('The {}% prediction interval is [{:.2}, {:.2}] seconds'.format(percentile, lower_bound, upper_bound))
        return prediction[0], lower_bound, upper_bound

    def estimate_duration(self, algo, X, y=None):
        """
        predicts training runtime for a given training

        :param X: np.array of inputs to be trained
        :param y: np.array of outputs to be trained (set to None is unsupervised algo)
        :param algo: algo whose runtime the user wants to predict
        :return: predicted runtime, low and high values of the percentile-confidence interval
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
