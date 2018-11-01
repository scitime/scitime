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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import warnings

warnings.simplefilter("ignore")

from scikest.utils import LogMixin, get_path, config, timeit


class Trainer(LogMixin):
    # default meta-algorithm
    META_ALGO = 'RF'
    # the drop rate is used to fit the meta-algo on random parameters
    DROP_RATE = 0.9
    # the default estimated algorithm is a Random Forest from sklearn
    ALGO = 'RandomForestRegressor'

    def __init__(self, drop_rate=DROP_RATE, meta_algo=META_ALGO, algo=ALGO, verbose=0):
        # the end user will estimate the fitting time of self.algo using the package
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
            raise ValueError(f'{self.algo} not currently supported by this package')
        return config(self.algo)

    @property
    def estimation_inputs(self):
        """
        retrieves estimation inputs (made dummy)

        :return: list of inputs
        """
        return self.params['other_params'] + [i for i in self.params['external_params'].keys()] \
               + [i for i in self.params['internal_params'].keys() if i not in self.params['dummy_inputs']] \
               + [i + '_' + str(k) for i in self.params['internal_params'].keys() if i in self.params['dummy_inputs']
                  for k in self.params['internal_params'][i]]

    @staticmethod
    def _add_data_to_csv(row_input, row_output):
        """
        writes into the csv results file row by row

        :param input: row inputs
        :param output: row output
        :return:
        """
        with open(r'result.csv', 'a+') as file:
            writer = csv.writer(file)
            row = list(row_input) + [row_output]
            writer.writerows([row])

    def _generate_numbers(self, n, p, meta_params, num_cat=None):
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

    def _measure_time(self, X, y, meta_params, params, num_cat=None):
        """
        generates fits with the meta-algo using dummy data and tracks the training runtime

        :param X: inputs
        :param y: outputs
        :param params: model params included in the estimation
        :param meta_params: params from json file (equivalent to self.params)
        :param num_cat: number of categories if classification algo
        :return: runtime
        :rtype: float
        """
        # selecting a model, the estimated algo
        sub_module = importlib.import_module(meta_params['module'])
        model = getattr(sub_module, self.algo)(**params)
        # measuring model execution time
        start_time = time.time()
        if meta_params["type"] == "unsupervised":
            model.fit(X)
        else:
            model.fit(X, y)
        elapsed_time = time.time() - start_time
        return elapsed_time

    @timeit
    def _permute(self, concat_dic, parameters_list, external_parameters_list, meta_params, algo_type):
        """
        for loop over every possible param combination

        :param concat_dic: all params + all values range dictionary
        :param parameters_list: all internal parameters names
        :param external_parameters_list: all external parameters names
        :param meta_params: params from json file (equivalent to self.params)
        :param algo_type: unsupervised / supervised / classification
        :return: inputs, outputs
        :rtype: lists
        """
        inputs = []
        outputs = []
        num_cat = None
        for permutation in itertools.product(*concat_dic.values()):
            n, p = permutation[0], permutation[1]
            if algo_type == "classification":
                num_cat = permutation[2]
                parameters_dic = dict(zip(parameters_list, permutation[3:]))
            else:
                parameters_dic = dict(zip(parameters_list, permutation[2:]))

            # computing only for (1-self.drop_rate) % of the data
            random_value = np.random.uniform()
            if random_value > self.drop_rate:
                final_params = dict(zip(external_parameters_list + parameters_list, permutation))
                # handling max_features > p case
                try:
                    row_input = [self.memory.total, self.memory.available, self.num_cpu] + [i for i in permutation]
                    # fitting the models
                    X, y = self._generate_numbers(n, p, meta_params, num_cat)
                    row_output = self._measure_time(X, y, meta_params, parameters_dic, num_cat)

                    outputs.append(row_output)
                    inputs.append(row_input)
                    if self.verbose >= 2:
                        self.logger.info(f'data added for {final_params} which outputs {row_output} seconds')
                    self._add_data_to_csv(row_input, row_output)

                except Exception as e:
                    if self.verbose >= 1:
                        self.logger.warning(f'model fit for {final_params} throws a {e.__class__.__name__}')
        return inputs, outputs

    @timeit
    def _generate_data(self):
        """
        measures training runtimes for a set of distinct parameters
        saves results in a csv (row by row)

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
        # in this for loop, we fit the estimated algo multiple times for random parameters and random input (and output if the estimated algo is supervised)
        # we use a drop rate to randomize the parameters that we use
        inputs, outputs = self._permute(concat_dic, parameters_list, external_parameters_list, meta_params, algo_type)
        inputs = pd.DataFrame(inputs, columns=meta_params['other_params'] + external_parameters_list + parameters_list)
        outputs = pd.DataFrame(outputs, columns=['output'])

        return inputs, outputs

    @timeit
    def model_fit(self, generate_data=True, df=None, outputs=None):
        """
        builds the actual training time estimator

        :param generate_data: bool (if set to True, calls _generate_data)
        :param df: pd.DataFrame chosen as input
        :param outputs: pd.DataFrame chosen as output
        :return: meta_algo
        :rtype: scikit learn model
        """
        if generate_data:
            df, outputs = self._generate_data()

        data = pd.get_dummies(df)

        if self.verbose >= 2:
            self.logger.info('Model inputs: {}'.format(list(data.columns)))

        # we decide on a meta-algorithm
        if self.meta_algo not in config('supported_meta_algos'):
            raise ValueError(f'meta algo {self.meta_algo} currently not supported')
        if self.meta_algo == 'RF':
            meta_algo = RandomForestRegressor()

        if self.verbose >= 2:
            self.logger.info(f'Fitting {self.meta_algo} to estimate training durations for model {self.algo}')

        # adding 0 columns for columns that are not in the dataset, assuming it's only dummy columns
        missing_inputs = list(set(list(self.estimation_inputs)) - set(list((data.columns))))
        for i in missing_inputs:
            data[i] = 0

        # reshaping into arrays
        x = (data[self.estimation_inputs]
             ._get_numeric_data()
             .dropna(axis=0, how='any')
             .as_matrix())
        y = outputs['output'].dropna(axis=0, how='any').as_matrix()

        # dividing into train/test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
        meta_algo.fit(x_train, y_train)

        if self.verbose >= 2:
            self.logger.info(f'Saving {self.meta_algo} to {self.meta_algo}_{self.algo}_estimator.pkl')

        path = f'{get_path("models")}/{self.meta_algo}_{self.algo}_estimator.pkl'
        joblib.dump(meta_algo, path)

        if self.verbose >= 2:
            self.logger.info(f'R squared on train set is {r2_score(y_train, meta_algo.predict(x_train))}')

        # MAPE is the mean absolute percentage error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
        y_pred_test = meta_algo.predict(x_test)
        mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        y_pred_train = meta_algo.predict(x_train)
        mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        # with open('mape.txt', 'w') as f:
        # f.write(str(mape))

        if self.verbose >= 2:
            self.logger.info(f'''
            MAPE on train set is: {mape_train}
            MAPE on test set is: {mape_test}
            RMSE on train set is {np.sqrt(mean_squared_error(y_train, y_pred_train))}
            RMSE on test set is {np.sqrt(mean_squared_error(y_test, y_pred_test))} ''')

        return meta_algo
