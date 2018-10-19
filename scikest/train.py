import os

import numpy as np
import pandas as pd
import json
import csv
import time
import joblib
import itertools

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import warnings

warnings.simplefilter("ignore")

from scikest.utils import LogMixin, get_path, config, timeit

class Trainer(LogMixin):
    ALGO_ESTIMATOR = 'LR'
    DROP_RATE = 0.9
    ALGO = 'RF'

    def __init__(self, drop_rate=DROP_RATE, algo_estimator=ALGO_ESTIMATOR, algo=ALGO, verbose=True):
        self.drop_rate = drop_rate
        self.algo_estimator = algo_estimator
        self.algo = algo
        self.params = config(self.algo)
        self.verbose = verbose
        self.estimation_inputs = [i for i in self.params['external_params'].keys()] + [i for i in self.params[
            'internal_params'].keys() if i not in self.params['dummy_inputs']] + [i + '_' + str(k) for i in
                                                                                  self.params['internal_params'].keys()
                                                                                  if i in self.params['dummy_inputs']
                                                                                  for k in
                                                                                  self.params['internal_params'][i]]

    @property
    def num_cpu(self):
        return os.cpu_count()

    @staticmethod
    def _add_data_to_csv(thisInput, thisOutput):
        """
        writes into a csv row by row

        :param thisInput: input
        :param thisOutput: output
        :return:
        """
        with open(r'result.csv', 'a+') as file:
            writer = csv.writer(file)
            thisRow = list(thisInput) + [thisOutput]
            writer.writerows([thisRow])

    def _measure_time(self, n, p, rf_params):
        """
        generates dummy fits and tracks the training runtime

        :param n: number of observations
        :param p: number of features
        :param rf_params: rf params included in the estimation
        :return: runtime
        :rtype: float
        """
        # Genrating dummy inputs / outputs
        X = np.random.rand(n, p)
        y = np.random.rand(n, )
        # Fitting rf
        clf = RandomForestRegressor(**rf_params)
        start_time = time.time()
        clf.fit(X, y)
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
        rf_parameters_list = list(self.params['internal_params'].keys())
        external_parameters_list = list(self.params['external_params'].keys())
        concat_dic = dict(**self.params['external_params'], **self.params['internal_params'])

        for permutation in itertools.product(*concat_dic.values()):
            n, p = permutation[0], permutation[1]
            rf_parameters_dic = dict(zip(rf_parameters_list, permutation[2:]))
            final_params = dict(zip(external_parameters_list + rf_parameters_list, permutation))

            # Computing only for (1-self.drop_rate) % of the data
            random_value = np.random.uniform()
            if random_value > self.drop_rate:
                # Handling max_features > p case
                try:
                    thisOutput = self._measure_time(n, p, rf_parameters_dic)
                    thisInput = permutation
                    outputs.append(thisOutput)
                    inputs.append(thisInput)
                    if self.verbose:
                        self.logger.info(f'data added for {final_params} which outputs {thisOutput} seconds')

                    self._add_data_to_csv(thisInput, thisOutput)
                except Exception as e:
                    self.logger.warning(f'model fit for {final_params} throws an error')

        inputs = pd.DataFrame(inputs, columns=external_parameters_list + rf_parameters_list)
        outputs = pd.DataFrame(outputs, columns=['output'])

        return inputs, outputs

    @timeit
    def model_fit(self, generate_data=True, df=None, outputs=None):
        """
        builds the actual training time estimator

        :param generate_data: bool (if set to True, calls _generate_data)
        :param df: pd.DataFrame chosen as input
        :param output: pd.DataFrame chosen as output
        :return: algo
        :rtype: pickle file
        """
        if generate_data:
            df, outputs = self._generate_data()

        data = pd.get_dummies(df)

        if self.verbose:
            self.logger.info('Model inputs: {}'.format(list(data.columns)))

        if self.algo_estimator == 'LR':
            algo = linear_model.LinearRegression()
        if self.algo_estimator == 'RF':
            algo = RandomForestRegressor()

        if self.verbose:
            self.logger.info(f'Fitting {self.algo_estimator} to estimate training durations')

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
        algo.fit(x_train, y_train)

        if self.algo_estimator == 'LR':
            if self.verbose:
                self.logger.info('Saving LR coefs in json file')
            with open('coefs/lr_coefs.json', 'w') as outfile:
                json.dump([algo.intercept_] + list(algo.coef_), outfile)
        if self.verbose:
            self.logger.info(f'Saving {self.algo_estimator} to {self.algo_estimator}_estimator.pkl')

        path = f'{get_path("models")}/{self.algo_estimator}_estimator.pkl'
        joblib.dump(algo, path)

        if self.verbose:
            self.logger.info(f'R squared on train set is {r2_score(y_train, algo.predict(x_train))}')

        y_pred_test = algo.predict(x_test)
        mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        y_pred_train = algo.predict(x_train)
        mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        # with open('mape.txt', 'w') as f:
        # f.write(str(mape))

        if self.verbose:
            self.logger.info(f'''
            MAPE on train set is: {mape_train}
            MAPE on test set is: {mape_test} 
            RMSE on train set is {np.sqrt(mean_squared_error(y_train, y_pred_train))} 
            RMSE on test set is {np.sqrt(mean_squared_error(y_test, y_pred_test))} ''')

        return algo

# TODO
# Adding n*log(n)*v (supposedly = runtime of training in big o notation)
# X_1=np.append(X,np.array(X[:,1]*X[:,0]*np.log(X[:,0])).reshape(432,1),axis=1)
