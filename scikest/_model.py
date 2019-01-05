"""Class used to instantiate an object for fitting the meta-algorithm"""
import os
import psutil

import numpy as np
import pandas as pd
import csv
import time
import joblib
import itertools
import importlib
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

import warnings

warnings.simplefilter("ignore")

from scikest.estimate import Estimator
from scikest._utils import get_path, config, timeout
from scikest._log import LogMixin, timeit


class Model(Estimator, LogMixin):
    # default meta-algorithm
    META_ALGO = 'RF'
    # the drop rate is used to fit the meta-algo on random parameters
    DROP_RATE = 0.9
    # the default estimated algorithm is a Random Forest from sklearn
    ALGO = 'RandomForestRegressor'

    def __init__(self, drop_rate=DROP_RATE, meta_algo=META_ALGO, algo=ALGO, verbose=0, bins=None):
        # the end user will estimate the fitting time of self.algo using the package
        super().__init__(bins)
        self.algo = algo
        self.drop_rate = drop_rate
        self.meta_algo = meta_algo
        self.verbose = verbose

    @property
    def num_cpu(self):
        return os.cpu_count()

    @property
    def memory(self):
        return psutil.virtual_memory()

    @property
    def params(self):
        """
        retrieves the estimated algorithm's parameters if the algo is supported

        :return: dictionary
        """
        if self.algo not in config("supported_algos"):
            raise KeyError(f'{self.algo} not currently supported by this package')
        return config(self.algo)

    def _add_row_to_csv(self, row_input, row_output):
        """
        writes a row into the csv results file

        :param input: row inputs
        :param output: row output
        :return:
        """
        csv_name = f'{self.algo}_result.csv'
        with open(f'{get_path(csv_name)}', 'a+') as file:
            writer = csv.writer(file)
            row = list(row_input) + [row_output]
            writer.writerows([row])

    @staticmethod
    def _generate_numbers(n, p, meta_params, num_cat=None):
        """
        generates random inputs / outputs

        :param meta_params: params from json file (equivalent to self.params)
        :param num_cat: number of categories if classification algo
        :return: X & y
        :rtype: np arrays
        """
        # generating dummy inputs / outputs in [0,1)
        X = np.random.rand(n, p)
        y = None
        if meta_params["type"] == "regression":
            y = np.random.rand(n, )
        if meta_params["type"] == "classification":
            y = np.random.randint(0, num_cat, n)
        return X, y

    @staticmethod
    def _measure_time(model, X, y, meta_params):
        """
        generates fits with the meta-algo using dummy data and tracks the training runtime

        :param X: inputs
        :param y: outputs
        :param meta_params: params from json file (equivalent to self.params)
        :return: runtime
        :rtype: float
        """
        # measuring model execution time
        start_time = time.time()
        if meta_params["type"] == "unsupervised":
            model.fit(X)
        else:
            model.fit(X, y)
        elapsed_time = time.time() - start_time
        return elapsed_time

    @staticmethod
    def _str_to_float(row):
        """
        transforms semi dummy input from csv

        :param row: row of a pandas dataframe
        :return:
        """
        try:
            return np.float(row)
        except:
            return row

    def _transform_from_csv(self, csv_name):
        """
        takes data from csv and returns inputs and outputs in right format for model_fit

        :param csv_name: name of csv from generate data
        :param rename_columns: set to True if csv columns have to be named
        :return: inputs and outputs
        """
        df = pd.read_csv(get_path(csv_name))

        meta_params = self.params
        parameters_list = list(meta_params['internal_params'].keys())
        external_parameters_list = list(meta_params['external_params'].keys())
        df.columns = meta_params['other_params'] + external_parameters_list + parameters_list + ['output']

        semi_dummy_inputs = self.params['semi_dummy_inputs']
        for col in semi_dummy_inputs:
            df[col] = df[col].apply(self._str_to_float)

        inputs = df.drop(['output'], axis=1)
        outputs = df[['output']]

        return inputs, outputs

    def _get_model(self, meta_params, params):
        """
        builds the sklearn model to be fitted 

        :param params: model params included in the estimation
        :param meta_params: params from json file (equivalent to self.params)
        :return: model
        :rtype: scikit-learn model
        """
        sub_module = importlib.import_module(meta_params['module'])
        model = getattr(sub_module, self.algo)(**params)
        return model

    @timeit
    def _permute(self, concat_dic, parameters_list, external_parameters_list, meta_params, algo_type, validation=False):
        """
        for loop over every possible param combination

        :param concat_dic: all params + all values range dictionary
        :param parameters_list: all internal parameters names
        :param external_parameters_list: all external parameters names
        :param meta_params: params from json file (equivalent to self.params)
        :param algo_type: unsupervised / supervised / classification
        :param validation: boolean, set true if data is used for validation, use only once the model has been trained
        :return: inputs, outputs
        :rtype: lists
        """
        inputs = []
        outputs = []
        estimated_outputs = []
        num_cat = None

        # in this for loop, we fit the estimated algo multiple times for random parameters and random input (and output if the estimated algo is supervised)
        # we use a drop rate to randomize the parameters that we use
        for permutation in itertools.product(*concat_dic.values()):
            # computing only for (1-self.drop_rate) % of the data
            # making sure the dataset is not empty (at least 2 data points to pass the model fit)
            random_value = np.random.uniform()
            if (random_value > self.drop_rate) | (len(inputs) < 4):
                n, p = permutation[0], permutation[1]
                if algo_type == "classification":
                    num_cat = permutation[2]
                    parameters_dic = dict(zip(parameters_list, permutation[3:]))
                else:
                    parameters_dic = dict(zip(parameters_list, permutation[2:]))
                final_params = dict(zip(external_parameters_list + parameters_list, permutation))

                try:
                    model = self._get_model(meta_params, parameters_dic)
                    # fitting the models
                    X, y = self._generate_numbers(n, p, meta_params, num_cat)
                    row_input = [self.memory.total, self.memory.available, self.num_cpu] + [i for i in permutation]
                    row_output = self._measure_time(model, X, y, meta_params)

                    outputs.append(row_output)
                    inputs.append(row_input)
                    if self.verbose >= 2:
                        self.logger.info(f'data added for {final_params} which outputs {row_output} seconds')

                    if not validation:
                        self._add_row_to_csv(row_input, row_output)
                    else:
                        row_estimated_output, _, _ = self._estimate(model, X, y)
                        estimated_outputs.append(row_estimated_output)

                except Exception as e:
                    if self.verbose >= 1:
                        self.logger.warning(f'model fit for {final_params} throws a {e.__class__.__name__}')

        return inputs, outputs, estimated_outputs

    @timeit
    def _generate_data(self, validation=False):
        """
        measures training runtimes for a set of distinct parameters
        saves results in a csv (row by row)

        :param validation: boolean, set true if data is used for validation, use only once the model has been trained
        :return: inputs, outputs
        :rtype: pd.DataFrame
        """
        if self.verbose >= 2:
            self.logger.info('Generating dummy training durations to create a training set')

        meta_params = self.params
        parameters_list = list(meta_params['internal_params'].keys())
        external_parameters_list = list(meta_params['external_params'].keys())
        concat_dic = dict(**meta_params['external_params'], **meta_params['internal_params'])
        algo_type = meta_params["type"]

        inputs, outputs, estimated_outputs = self._permute(concat_dic, parameters_list, external_parameters_list,
                                                           meta_params, algo_type, validation)

        if validation:
            estimated_outputs = pd.DataFrame(estimated_outputs, columns=['estimated_outputs'])

        inputs = pd.DataFrame(inputs, columns=meta_params['other_params'] + external_parameters_list + parameters_list)
        outputs = pd.DataFrame(outputs, columns=['output'])

        return inputs, outputs, estimated_outputs

    def _transform_data(self, inputs, outputs):
        """
        transforms the data before fitting the meta model

        :param inputs: pd.DataFrame chosen as input
        :param outputs: pd.DataFrame chosen as output
        :return: X, y
        :rtype: np arrays
        """
        # first we transform semi dummy features
        semi_dummy_inputs = self.params['semi_dummy_inputs']
        # we add columns for each semi dummy features (*number of potential dummy values)
        inputs = self._add_semi_dummy(inputs, semi_dummy_inputs)

        # we then fill artificial (and natural) NAs with -1

        data = pd.get_dummies(inputs.fillna(-1))

        if self.verbose >= 2:
            self.logger.info('Model inputs: {}'.format(list(data.columns)))

        # reshaping into arrays
        X = (data
             ._get_numeric_data()
             .dropna(axis=0, how='any')
             .as_matrix())
        y = outputs['output'].dropna(axis=0, how='any').as_matrix()

        return X, y, data.columns, inputs.columns

    def _scale_data(self, X_train, X_test, save_model):
        """
        scales the X vector in order to fit the NN meta algo

        :param X_train: pd.DataFrame chosen as input for the training set
        :param X_test: pd.DataFrame chosen as input for the test set
        :param save_model: boolean set to True if the model needs to be saved
        :return: X_train and X_test data scaled
        :rtype: pd.DataFrame
        """
        scaler = StandardScaler()
        scaler.fit(X_train)

        if save_model:
            if self.verbose >= 2:
                self.logger.info(f'Saving scaler model to scaler_{self.algo}_estimator.pkl')
            model_path = f'{get_path("models")}/scaler_{self.algo}_estimator.pkl'
            joblib.dump(scaler, model_path)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    @timeit
    def _random_search(self, inputs, outputs, iterations, save_model=False):
        """
        performs a random search on the NN meta algo to find the best params

        :param inputs: pd.DataFrame chosen as input
        :param outputs: pd.DataFrame chosen as output
        :param iterations: Number of parameter settings that are sampled
        :param save_model: boolean set to True if the model needs to be saved
        :return: best meta_algo with parameters
        :rtype: scikit learn RandomizedSearchCV object
        """
        X, y, cols, original_cols = self._transform_data(inputs, outputs)
        if self.meta_algo != 'NN':
            raise KeyError(f'meta algo {self.meta_algo} not supported for random search')

        parameter_space = config("random_search_params")
        meta_algo = MLPRegressor(max_iter=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        X_train, X_test = self._scale_data(X_train, X_test, save_model)

        meta_algo = RandomizedSearchCV(meta_algo, parameter_space, n_iter=iterations, n_jobs=2)
        meta_algo.fit(X_train, y_train)

        if self.verbose >= 2:
            self.logger.info(f'Best parameters found: {meta_algo.best_estimator_}')

        return meta_algo

    @timeit
    def model_fit(self, generate_data=True, inputs=None, outputs=None, csv_name=None, save_model=False,
                  meta_algo_params=None):
        """
        builds the actual training time estimator

        :param generate_data: bool (if set to True, calls _generate_data)
        :param inputs: pd.DataFrame chosen as input
        :param outputs: pd.DataFrame chosen as output
        :param csv_name: name if csv in case we fetch data from csv
        :param save_model: boolean set to True if the model needs to be saved
        :param meta_algo_params: params of meta algo
        :return: meta_algo
        :rtype: scikit learn model
        """
        if meta_algo_params is None:
            if self.meta_algo == 'NN':
                meta_algo_params = {'max_iter': 200, 'hidden_layer_sizes': [100, 100, 100]}
            elif self.meta_algo == 'RF':
                meta_algo_params = {'criterion': 'mse', 'max_depth': 100, 'max_features': 10}

        if generate_data:
            inputs, outputs, _ = self._generate_data()
        else:
            if csv_name is not None:
                inputs, outputs = self._transform_from_csv(csv_name=csv_name)

        if inputs is None or outputs is None:
            raise NameError('no inputs / outputs found: please enter a csv name or set generate_data to True')

        X, y, cols, original_cols = self._transform_data(inputs, outputs)

        # we decide on a meta-algorithm
        if self.meta_algo not in config('supported_meta_algos'):
            raise KeyError(f'meta algo {self.meta_algo} currently not supported')
        if self.meta_algo == 'RF':
            meta_algo = RandomForestRegressor(**meta_algo_params)
        if self.meta_algo == 'NN':
            meta_algo = MLPRegressor(**meta_algo_params)

        if self.verbose >= 2:
            self.logger.info(f'Fitting {self.meta_algo} to estimate training durations for model {self.algo}')

        # dividing into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        if self.meta_algo == 'NN':
            X_train_scaled, X_test_scaled = self._scale_data(X_train, X_test, save_model)
            meta_algo.fit(X_train_scaled, y_train)

        else:
            meta_algo.fit(X_train, y_train)

        if save_model:
            if self.verbose >= 2:
                self.logger.info(f'Saving {self.meta_algo} to {self.meta_algo}_{self.algo}_estimator.pkl')
            model_path = f'{get_path("models")}/{self.meta_algo}_{self.algo}_estimator.pkl'
            joblib.dump(meta_algo, model_path)

            json_path = f'{get_path("models")}/{self.meta_algo}_{self.algo}_estimator.json'

            with open(json_path, 'w') as outfile:
                json.dump({"dummy": list(cols), "original": list(original_cols)}, outfile)

        if self.meta_algo == 'NN':
            if self.verbose >= 2:
                self.logger.info(f'R squared on train set is {r2_score(y_train, meta_algo.predict(X_train_scaled))}')

            # MAPE is the mean absolute percentage error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
            y_pred_test = np.array([max(i, 0) for i in meta_algo.predict(X_test_scaled)])
            y_pred_train = np.array([max(i, 0) for i in meta_algo.predict(X_train_scaled)])
        else:
            if self.verbose >= 2:
                self.logger.info(f'R squared on train set is {r2_score(y_train, meta_algo.predict(X_train))}')
            y_pred_test = meta_algo.predict(X_test)
            y_pred_train = meta_algo.predict(X_train)

        mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100

        bins, mape_index_list = self.bins

        bins_values = [y_pred_test < 1] + [(y_pred_test >= i[0]) & (y_pred_test < i[1]) for i in bins] + [
            y_pred_test >= 10 * 60]

        if save_model:
            mse_tests = [mean_squared_error(y_test[bin], y_pred_test[bin]) for bin in bins_values]
            observation_tests = [y_test[bin].shape[0] for bin in bins_values]

            mse_test_dic = dict(zip(mape_index_list, zip(observation_tests, mse_tests)))

            if self.verbose >= 2:
                self.logger.info(f'Computed mse on test set (with number of observations): {mse_test_dic}')

        if self.meta_algo == 'NN':

            if save_model:
                json_conf_path = f'{get_path("models")}/{self.meta_algo}_{self.algo}_confint.json'
                self.logger.info(f'Saving confint to {self.meta_algo}_{self.algo}_confint.json')

                with open(json_conf_path, 'w') as outfile:
                    json.dump(mse_test_dic, outfile)

        if self.verbose >= 2:
            self.logger.info(f'''
            MAPE on train set is: {mape_train}
            MAPE on test set is: {mape_test}
            RMSE on train set is {np.sqrt(mean_squared_error(y_train, y_pred_train))}
            RMSE on test set is {np.sqrt(mean_squared_error(y_test, y_pred_test))} ''')

        return meta_algo

    @timeit
    def model_validate(self):
        """
        measures training runtimes and compares to actual runtimes once the model has been trained

        :return: results dataframe and error rate
        :rtype: pd.DataFrame and float
        """
        inputs, outputs, estimated_outputs = self._generate_data(validation=True)

        actual_values = outputs['output']
        estimated_values = estimated_outputs['estimated_outputs']
        avg_weighted_error = np.dot(actual_values, actual_values - estimated_values) / sum(actual_values)

        return inputs, outputs, estimated_outputs, avg_weighted_error
