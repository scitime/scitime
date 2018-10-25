import os
import psutil

import numpy as np
import pandas as pd
import json
import csv
import time
import joblib
import itertools

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import warnings

warnings.simplefilter("ignore")

from scikest.utils import LogMixin, get_path, config, timeit


class Trainer(LogMixin):
    ALGO_ESTIMATOR = 'RF'
    DROP_RATE = 0.9
    ALGO = 'RandomForestRegressor'

    def __init__(self, drop_rate=DROP_RATE, algo_estimator=ALGO_ESTIMATOR, algo=ALGO, verbose=True):
        self.algo = algo
        self.drop_rate = drop_rate
        self.algo_estimator = algo_estimator
        self.verbose = verbose

    @property
    def num_cpu(self):
        return os.cpu_count()

    @property
    def memory(self):
        return psutil.virtual_memory()

    @property
    def params(self):
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
        + [i + '_' + str(k) for i in self.params['internal_params'].keys() if i in self.params['dummy_inputs'] for k in self.params['internal_params'][i]]
    
    @staticmethod
    def _add_data_to_csv(row_input, row_output):
        """
        writes into a csv row by row

        :param input: row inputs
        :param output: row output
        :return:
        """
        with open(r'result.csv', 'a+') as file:
            writer = csv.writer(file)
            row = list(row_input) + [row_output]
            writer.writerows([row])

    def _measure_time(self, n, p, params, num_cat=None):
        """
        generates dummy fits and tracks the training runtime

        :param n: number of observations
        :param p: number of features
        :param params: model params included in the estimation
        :param num_cat: number of categories if classification algo
        :return: runtime
        :rtype: float
        """
        # Genrating dummy inputs / outputs
        X = np.random.rand(n, p)
        if self.params["type"] == "regression":
            y = np.random.rand(n, )
        if self.params["type"] == "classification":
            y = np.random.randint(0, num_cat, n)
        # Fitting model
        if self.algo == "RandomForestRegressor":
            model = RandomForestRegressor(**params)
        if self.algo == "SVC":
            model = SVC(**params)
        if self.algo == "KMeans":
            model = KMeans(**params)

        start_time = time.time()
        if self.params["type"] == "unsupervised":
            model.fit(X)
        else:
            model.fit(X, y)
        elapsed_time = time.time() - start_time
        return elapsed_time

    @timeit
    def _generate_data(self):
        """
        measures training runtimes for a set of distinct parameters - saves results in a csv (row by row)

        :return: inputs, outputs
        :rtype: pd.DataFrame
        """
        if self.verbose:
            self.logger.info('Generating dummy training durations to create a training set')
        inputs = []
        outputs = []
        parameters_list = list(self.params['internal_params'].keys())
        external_parameters_list = list(self.params['external_params'].keys())
        concat_dic = dict(**self.params['external_params'], **self.params['internal_params'])

        for permutation in itertools.product(*concat_dic.values()):
            n, p = permutation[0], permutation[1]
            if self.params["type"] == "classification":
                num_cat = permutation[2]
                parameters_dic = dict(zip(parameters_list, permutation[3:]))
            else:
                parameters_dic = dict(zip(parameters_list, permutation[2:]))
            # Computing only for (1-self.drop_rate) % of the data
            random_value = np.random.uniform()
            if random_value > self.drop_rate:
                final_params = dict(zip(external_parameters_list + parameters_list, permutation))
                # Handling max_features > p case
                try:
                    row_input = [self.memory.total, self.memory.available, self.num_cpu] + [i for i in permutation]
                    if self.params["type"] == "classification":
                        row_output = self._measure_time(n, p, parameters_dic, num_cat)
                    else:
                        row_output = self._measure_time(n, p, parameters_dic)
                    outputs.append(row_output)
                    inputs.append(row_input)
                    if self.verbose:
                        self.logger.info(f'data added for {final_params} which outputs {row_output} seconds')

                    self._add_data_to_csv(row_input, row_output)
                except Exception as e:
                    self.logger.warning(f'model fit for {final_params} throws an error')

        inputs = pd.DataFrame(inputs, columns=self.params['other_params'] + external_parameters_list + parameters_list)
        outputs = pd.DataFrame(outputs, columns=['output'])

        return inputs, outputs

    @timeit
    def model_fit(self, generate_data=True, df=None, outputs=None):
        """
        builds the actual training time estimator

        :param generate_data: bool (if set to True, calls _generate_data)
        :param df: pd.DataFrame chosen as input
        :param outputs: pd.DataFrame chosen as output
        :return: algo_estimator
        :rtype: pickle file
        """
        if generate_data:
            df, outputs = self._generate_data()

        data = pd.get_dummies(df)

        if self.verbose:
            self.logger.info('Model inputs: {}'.format(list(data.columns)))

        if self.algo_estimator == 'LR':
            algo_estimator = linear_model.LinearRegression()
        if self.algo_estimator == 'RF':
            algo_estimator = RandomForestRegressor()

        if self.verbose:
            self.logger.info(f'Fitting {self.algo_estimator} to estimate training durations for model {self.algo}')

        # adding 0 columns for columns that are not in the dataset, assuming it s only dummy columns
        missing_inputs = list(set(list(self.estimation_inputs)) - set(list((data.columns))))
        for i in missing_inputs:
            data[i] = 0

        # Reshaping into arrays
        x = (data[self.estimation_inputs]
             ._get_numeric_data()
             .dropna(axis=0, how='any')
             .as_matrix())
        y = outputs['output'].dropna(axis=0, how='any').as_matrix()

        # Diving into train/test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
        algo_estimator.fit(x_train, y_train)

        if self.algo_estimator == 'LR':
            if self.verbose:
                self.logger.info('Saving LR coefs in json file')
            with open('scikest/coefs/lr_coefs.json', 'w') as outfile:
                json.dump([algo_estimator.intercept_] + list(algo_estimator.coef_), outfile)
        if self.verbose:
            self.logger.info(f'Saving {self.algo_estimator} to {self.algo_estimator}_{self.algo}_estimator.pkl')

        path = f'{get_path("models")}/{self.algo_estimator}_{self.algo}_estimator.pkl'
        joblib.dump(algo_estimator, path)

        if self.verbose:
            self.logger.info(f'R squared on train set is {r2_score(y_train, algo_estimator.predict(x_train))}')

        y_pred_test = algo_estimator.predict(x_test)
        mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        y_pred_train = algo_estimator.predict(x_train)
        mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        # with open('mape.txt', 'w') as f:
        # f.write(str(mape))

        if self.verbose:
            self.logger.info(f'''
            MAPE on train set is: {mape_train}
            MAPE on test set is: {mape_test} 
            RMSE on train set is {np.sqrt(mean_squared_error(y_train, y_pred_train))} 
            RMSE on test set is {np.sqrt(mean_squared_error(y_test, y_pred_test))} ''')

        return algo_estimator