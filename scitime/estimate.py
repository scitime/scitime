"""Class used to instantiate an object
that will estimate the running time of the user's model"""
from scitime._utils import get_path, config, timeout
from scitime._log import LogMixin

import os
import psutil

import joblib
import pandas as pd
import numpy as np
import time
import json
from scipy import stats

import warnings

warnings.simplefilter("ignore")


class Estimator(LogMixin):
    """
    Estimator class arguments

    :param meta_algo: meta algorithm (RF or NN)
    :param verbose: log output (0, 1, 2 or 3)
    :param confidence: confidence for intervals
    """
    # default meta-algorithm
    META_ALGO = 'RF'
    # bins to consider for computing confindence intervals (for NN meta algo)
    BINS = [(1, 5), (5, 30), (30, 60), (60, 10 * 60)]
    BINS_VERBOSE = ['less than 1s',
                    'between 1s and 5s',
                    'between 5s and 30s',
                    'between 30s and 1m',
                    'between 1m and 10m',
                    'more than 10m']

    def __init__(self, meta_algo=META_ALGO, verbose=3, confidence=0.95):
        self.meta_algo = meta_algo
        self.verbose = verbose
        self.confidence = confidence
        self.bins = self.BINS, self.BINS_VERBOSE

    @property
    def num_cpu(self):
        return os.cpu_count()

    @property
    def memory(self):
        return psutil.virtual_memory()

    @timeout(1)
    def _fit_start(self, algo, X, y=None):
        """
        starts fitting the model on a small subset of the data
        to make sure the fit is legit, throws error if error happens before
        1 sec raises or a TimeoutError if no other exception is raised
        before used in the .time function

        :param algo: algo used
        :param X: inputs for the algo
        :param y: outputs for the algo
        """
        algo.verbose = 0
        param_dic = self._fetch_algo_metadata(algo)
        params = param_dic['config']
        algo_type = params['type']

        if X.shape[0] > 10:
            X = X[:10, :]
            if y is not None:
                y = y[:10]

        if algo_type == 'unsupervised':
            algo.fit(X)
        else:
            algo.fit(X, y)
        time.sleep(1.1)

    @staticmethod
    def _clean_output(seconds):
        """
        from seconds to cleaner format (minutes and hours when relevant)

        :param seconds: output of meta prediction
        :return: cleaned output (string)
        """
        full_minutes = seconds // 60
        seconds_rest = seconds % 60

        if seconds_rest == 1:
            sec_unit = f'second'

        else:
            sec_unit = f'seconds'

        if full_minutes == 0:
            return f'{seconds:.0f} {sec_unit}'

        else:
            full_hours = full_minutes // 60
            minutes_rest = full_minutes % 60

            if minutes_rest == 1:
                min_unit = f'minute'
            else:
                min_unit = f'minutes'

            if full_hours == 0:
                return f'''{full_minutes:.0f} {min_unit}
                and {seconds_rest:.0f} {sec_unit}'''

            else:
                if full_hours == 1:
                    h_unit = f'hour'
                else:
                    h_unit = f'hours'

                return f'''{full_hours:.0f} {h_unit} and {minutes_rest:.0f}
                {min_unit} and {seconds_rest:.0f} {sec_unit}'''

    @staticmethod
    def _fetch_algo_metadata(algo):
        """
        retrieves algo name, algo params and meta params from sklearn model

        :param algo: sklearn model
        :return: dictionary
        :rtype: dict
        """
        algo_name = type(algo).__name__
        algo_params = algo.get_params()
        params = config(algo_name)

        param_dic = {'name': algo_name,
                     'params': algo_params, 'config': params}

        return param_dic

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
        add columns for semi dummy inputs (continuous except
        for one or a few values) columns that are either
        continuous or categorical for a discrete number of
        outputs (usually 1) are treated as continuous and
        we add a binary column for the categorical output.
        As an example, max_depth (for RandomForest) can be
        either an integer or 'None', we treat this column
        as continuous and add a boolean column is_max_depth_None

        :param df: df before made dummy
        :param semi_dummy_inputs: from meta params
        :return: df transformed with added columns
        """
        if self.verbose >= 2:
            self.logger.info(f'Transforming dataset for semi dummy features')

        for key in semi_dummy_inputs:
            for sub_key in semi_dummy_inputs[key]:
                df[f'{key}_{sub_key}'] = (df[f'{key}']
                                          .apply(lambda x: x == sub_key))
                df[f'{key}'] = (df[f'{key}']
                                .apply(lambda x: None if x == sub_key else x))

        return df

    def _fetch_params(self, algo, X, y):
        """
        builds a dataframe of the params of the estimated model

        :param X: np.array of inputs to be trained
        :param y: np.array of outputs to be trained
        (set to None if unsupervised algo)
        :param algo: algo whose runtime the user wants to predict
        :return: dataframe of all inputed parameters
        :rtype: pandas dataframe
        """
        param_dic = self._fetch_algo_metadata(algo)
        params = param_dic['config']
        algo_params = param_dic['params']

        n = X.shape[0]
        p = X.shape[1]
        inputs = [self.memory.total, self.memory.available, self.num_cpu, n, p]

        if params["type"] == "classification":
            num_cat = len(np.unique(y))
            inputs.append(num_cat)

        param_list = params['other_params'] + \
            list(params['external_params'].keys()) + \
            list(params['internal_params'].keys())

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

        dic = dict(zip(param_list, [[i] for i in inputs]))

        if self.verbose >= 2:
            self.logger.info(f'Training your model for these params: {dic}')

        df = pd.DataFrame(dic, columns=param_list)

        return df

    def _transform_params(self, algo, df, scaled=False):
        """
        transforms the dataframe of the params of the estimated
        model before predicting runtime

        :param df: dataframe of all inputed parameters
        :param algo: algo whose runtime the user wants to predict
        :param scaled: scaling the input if set to True
        :return: np array of all relevant algo parameters
        and system features used to estimate algo training time
        :rtype: pandas matrix object
        """
        param_dic = self._fetch_algo_metadata(algo)
        algo_name = param_dic['name']
        algo_params = param_dic['params']
        params = param_dic['config']

        json_path = f'''{get_path("models")}/{self.meta_algo}_{algo_name}_estimator.json'''
        estimation_inputs = self._fetch_inputs(json_path)['dummy']
        estimation_original_inputs = self._fetch_inputs(json_path)['original']

        # first we transform semi dummy features
        semi_dummy_inputs = params['semi_dummy_inputs']
        # we add columns for each semi dummy features
        # (times the number of potential dummy values)
        df = self._add_semi_dummy(df, semi_dummy_inputs)

        forgotten_inputs = list(set(list(estimation_original_inputs)) - set(list((df.columns))))

        if len(forgotten_inputs) > 0:
            # if some params that we use to train the underlying
            # meta model do not appear, we can't predict the runtime
            raise NameError(f'{forgotten_inputs} parameters missing')

        df = pd.get_dummies(df.fillna(-1))

        # adding 0 columns for columns that are not in the dataset
        dummy_inputs_to_fill = list(set(list(estimation_inputs)) - set(list((df.columns))))

        missing_inputs = list(set(list(algo_params.keys())) - set(list((params['internal_params'].keys()))))

        if self.verbose >= 1 and (len(missing_inputs) > 0):
            # if there are other params that are not in
            # the underlying meta model, let's log them
            self.logger.warning(f'''Parameters {missing_inputs} will not be accounted for''')

        for i in dummy_inputs_to_fill:
            df[i] = 0

        df = df[estimation_inputs]

        meta_X = (df[estimation_inputs]
                  ._get_numeric_data()
                  .dropna(axis=0, how='any')
                  .to_numpy())

        # scaling (for meta algo NN)
        if scaled:
            if self.verbose >= 3:
                self.logger.debug(f'''Fetching scaler: scaler_{algo_name}_estimator.pkl''')

            model_path = f'''{get_path("models")}/scaler_{algo_name}_estimator.pkl'''

            scaler = joblib.load(model_path)
            meta_X = scaler.transform(meta_X)

        return meta_X

    def _estimate_interval(self, meta_estimator,
                           X, algo_name, confidence=0.95):
        """
        estimates the prediction intervals for one data-point
        there are two ways of computing this interval.
        The first one (for meta algo RF) is using the fact
        that we have access to multiple trees (predictors)
        to compute a prediction percentile.
        The second on (for meta algo NN) is to fetch the mse
        per bin and use a t stat to compute the confidence intervals

        :param meta_estimator: the fitted random-forest meta-algo
        :param X: parameters, the same as the ones fed in the meta-algo
        :param algo_name: name of algo (not meta algo)
        :param confidence: confidence for intervals (default to 95%)
        :return: low and high values of the confidence interval
        :rtype: tuple
        """

        if type(meta_estimator).__name__ == 'RandomForestRegressor':
            predictions = [predictor.predict(X)[0]
                           for predictor in meta_estimator.estimators_]

            lower_conf = (100 - 100 * confidence) / 2
            upper_conf = 100 - (100 - 100 * confidence) / 2
            lower_bound = np.percentile(predictions, lower_conf)
            upper_bound = np.percentile(predictions, upper_conf)

        elif type(meta_estimator).__name__ == 'MLPRegressor':
            confint_path = f'''{get_path("models")}/{self.meta_algo}_{algo_name}_confint.json'''

            if self.verbose >= 3:
                self.logger.debug(f'''Fetching confint: {self.meta_algo}_{algo_name}_confint.json''')

            mse_dic = self._fetch_inputs(confint_path)
            prediction = max(meta_estimator.predict(X)[0], 0)
            bins, mse_index_list = self.bins

            if prediction < 1:
                mse_index = 0
            elif prediction >= 10 * 60:
                mse_index = 5
            else:
                for i in range(len(bins)):
                    if prediction >= bins[i][0] and prediction < bins[i][1]:
                        mse_index = i + 1

            n_obs, local_mse = mse_dic[mse_index_list[mse_index]]
            # we fetch the average mse per bin along with number of obs
            # and then use t-statistic to compute the uncertainty
            t_coef = stats.t.ppf(confidence, n_obs)
            uncertainty = t_coef * np.sqrt(2 * local_mse / n_obs)

            lower_bound = max(np.float64(0), prediction * (1 - uncertainty))
            upper_bound = max(np.float64(0), prediction * (1 + uncertainty))

        else:
            raise KeyError(f'''{type(meta_estimator).__name__} meta algo not supported''')

        return lower_bound, upper_bound

    def _estimate(self, algo, X, y=None):
        """
        estimates the model's training time given that the fit starts

        :param X: np.array of inputs to be trained
        :param y: np.array of outputs to be trained
        (set to None if unsupervised algo)
        :param algo: algo whose runtime the user wants to predict
        :return: predicted runtime,
        low and high values of the confidence interval
        :rtype: tuple
        """
        # fetching sklearn model of the end user
        param_dic = self._fetch_algo_metadata(algo)
        algo_name = param_dic['name']

        if algo_name not in config("supported_algos"):
            raise NotImplementedError(f'''{algo_name} not currently supported by this package''')

        if self.meta_algo not in config('supported_meta_algos'):
            raise KeyError(f'''meta algo {self.meta_algo} currently not supported''')

        if self.verbose >= 3:
            self.logger.debug(f'''Fetching estimator: {self.meta_algo}_{algo_name}_estimator.pkl''')

        model_path = f'''{get_path("models")}/{self.meta_algo}_{algo_name}_estimator.pkl'''

        meta_estimator = joblib.load(model_path)

        # retrieving all parameters of interest:
        df = self._fetch_params(algo, X, y)

        # Transforming the inputs:
        if self.meta_algo == 'NN':
            meta_X = self._transform_params(algo, df, scaled=True)
        else:
            meta_X = self._transform_params(algo, df)

        prediction = max(np.float64(0), meta_estimator.predict(meta_X)[0])

        # if prediction from NN is too low, let's go back to RF
        if prediction < 1 and self.meta_algo == 'NN':
            if self.verbose >= 3:
                self.logger.debug('''NN prediction too low - fetching rf meta algo instead''')

                self.logger.debug(f'''Fetching estimator: RF_{algo_name}_estimator.pkl''')

            model_path = f'{get_path("models")}/RF_{algo_name}_estimator.pkl'
            meta_estimator = joblib.load(model_path)
            meta_X = self._transform_params(algo, df)
            prediction = meta_estimator.predict(meta_X)[0]

        lower_bound, upper_bound = \
            self._estimate_interval(meta_estimator,
                                    meta_X, algo_name, self.confidence)

        cleaned_prediction = self._clean_output(round(prediction))
        cleaned_lower_bound = self._clean_output(round(lower_bound))
        cleaned_upper_bound = self._clean_output(round(upper_bound))

        if self.verbose >= 1 and prediction < 1:
            self.logger.warning('''Your model predicted training runtime is very low - no need to use this package''')

        if prediction < 1:
            cleaned_prediction = f'{prediction} seconds'
            cleaned_lower_bound = f'{lower_bound} seconds'
            cleaned_upper_bound = f'{upper_bound} seconds'

        if self.verbose >= 2:
            self.logger.info(f'''Training your {algo_name} model should take ~ {cleaned_prediction}''')

            self.logger.info(f'''The {100 * self.confidence}% prediction interval is [{cleaned_lower_bound}, {cleaned_upper_bound}]''')

        return prediction, lower_bound, upper_bound

    def time(self, algo, X, y=None):
        """
        predicts training runtime for a given training

        :param algo: algo whose runtime the user wants to predict
        :param X: np.array of inputs to be trained
        :param y: np.array of outputs to be trained (set to None is unsupervised algo)
        :return: predicted runtime, low and high values of the confidence interval
        :rtype: float
        """
        try:
            self._fit_start(algo=algo, X=X, y=y)

        except Exception as e:
            # this means that the sklearn fit has raised
            # a natural exception before we artificially
            # raised a timeouterror
            if e.__class__.__name__ != 'TimeoutError':
                self.logger.warning('sklearn throws an error for this fit (see below)')
                self.logger.warning('if the error is not relevant, try using ._estimate instead of .time')
                raise e

            else:
                if self.verbose >= 3:
                    self.logger.debug('The model would fit. Moving on')
                return self._estimate(algo, X, y)
