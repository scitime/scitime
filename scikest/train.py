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
    """
    This class is used to instantiate an object for fitting the meta-algorithm
    """
    ALGO_ESTIMATOR = 'RF' # This is the meta-algorithm
    DROP_RATE = 0.9 # The drop rate is used to fit the meta-algo on random parameters
    ALGO = 'RandomForestRegressor' # The default estimated algorithm is a Random Forest from sklearn

    def __init__(self, drop_rate=DROP_RATE, algo_estimator=ALGO_ESTIMATOR, algo=ALGO, verbose=True):
        self.algo = algo #The end user will estimate the fitting time of this algo using the package
        self.drop_rate = drop_rate
        self.algo_estimator = algo_estimator #This is the meta-algorithm
        self.verbose = verbose
        #@nathan when is self.params initilized?
        # it is line 51: when I define params(self) with the property decorator

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
        + [i + '_' + str(k) for i in self.params['internal_params'].keys() if i in self.params['dummy_inputs'] for k in self.params['internal_params'][i]]

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

    def _measure_time(self, n, p, params, num_cat=None):
        """
        generates fits with the meta-algo using dummy data and tracks the training runtime

        :param n: number of observations
        :param p: number of features
        :param params: model params included in the estimation
        :param num_cat: number of categories if classification algo
        :return: runtime
        :rtype: float
        """
        # Generating dummy inputs / outputs in [0,1)
        X = np.random.rand(n, p)
        if self.params["type"] == "regression":
            y = np.random.rand(n, )
        if self.params["type"] == "classification":
            y = np.random.randint(0, num_cat, n)

        # Select a model, the estimated algo
        if self.algo == "RandomForestRegressor":
            model = RandomForestRegressor(**params)
        if self.algo == "SVC":
            model = SVC(**params)
        if self.algo == "KMeans":
            model = KMeans(**params)

        # Measure model execution time
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
        measures training runtimes for a set of distinct parameters
        saves results in a csv (row by row)

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


        # In this for loop, we fit the estimated algo multiple times for random parameters and random input (and output if the estimated algo is unsupervised)
        # We use a drop rate to randomize the parameters that we use
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

                    # This is where the models are fitted
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
        # @nathan this boolean is not clear to me: please clarify the use case for false and raise exception if df and outputs are not given as arguments
        # if true you already generated some data so you dont want to call _generate_data. in that case yiu add the pregenerated data as an argument of the function (df and inputs)
        if generate_data:
            df, outputs = self._generate_data()

        data = pd.get_dummies(df)

        if self.verbose:
            self.logger.info('Model inputs: {}'.format(list(data.columns)))

        # We decide on a meta-algorithm
        if self.algo_estimator == 'LR':
            algo_estimator = linear_model.LinearRegression()
        if self.algo_estimator == 'RF':
            algo_estimator = RandomForestRegressor()

        if self.verbose:
            self.logger.info(f'Fitting {self.algo_estimator} to estimate training durations for model {self.algo}')

        #@nathan to clarify the missing inputs thing
        #missing input is when we m\are missing some columns: for instance if self.estimation_inputs contains
        #max_features_auto, max_features_20 etc, then if one of these is not in data, we need to make sure to add a correspo nding column filled with zeros.
        #this is even more relevant in estimate.py
        # adding 0 columns for columns that are not in the dataset, assuming it's only dummy columns
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

        #@nathan do we only do the LR case here? why not the RF?
        #this was when we thought LR would be a great meta algo. we wanted to save the model by saving the coefs instead of pickle file, this is why it s only relevant for LR. but not useful now
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

        #MAPE is the mean absolute percentage error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
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

        #@nathan returning the estimator actually returns the pickle file!?
        #yes, not sure that s necessary though
        return algo_estimator
