import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from sklearn import linear_model
from utils import LogMixin, add_data_to_csv, get_path, config, timeit
import warnings
import itertools
import os
import json
import pandas as pd

warnings.simplefilter("ignore")

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
    def _check_feature_condition(f, p):
        """
        makes sure the rf training doesn't break when f>p

        :param f: max feature param
        :param p: num feature param
        :return: bool
        """
        if (type(f) != int):
            return True
        else:
            if f <= p:
                return True
            else:
                return False

    def _measure_time(self, n, p, rf_params):
        """
        generates dummy fits and tracks the training runtime

        :param n: number of observations
        :param p: number of features
        :param rf_params: rf params included in the estimation
        :return: runtime
        :rtype: float
        """
        #Genrating dummy inputs / outputs
        X = np.random.rand(n, p)
        y = np.random.rand(n, )
        #Fitting rf
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
            n = permutation[0]
            p = permutation[1]
            f = permutation[7]

            rf_parameters_dic = dict(zip(rf_parameters_list, permutation[2:]))
            #Computing only for (1-self.drop_rate) % of the data
            random_value = np.random.uniform()
            if random_value > self.drop_rate:
                #Handling max_features > p case
                if self._check_feature_condition(f,p):
                    thisOutput = self._measure_time(n, p, rf_parameters_dic)
                    thisInput = permutation
                    outputs.append(thisOutput)
                    inputs.append(thisInput)
                    if self.verbose:
                        self.logger.info('data added for {p} which outputs {s} seconds'.format(p=dict(zip(external_parameters_list + rf_parameters_list, thisInput)),s=thisOutput))

                    add_data_to_csv(thisInput, thisOutput)

        inputs = pd.DataFrame(inputs, columns=external_parameters_list + rf_parameters_list)
        outputs = pd.DataFrame(outputs, columns=['output'])

        return (inputs, outputs)

    @timeit
    def model_fit(self,generate_data=True,df=None,outputs=None):
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
            algo=RandomForestRegressor()

        if self.verbose:
            self.logger.info('Fitting ' + self.algo_estimator + ' to estimate training durations')
        #adding 0 columns for columns that are not in the dataset, assuming it s only dummy columns
        missing_inputs = list(set(list(self.estimation_inputs)) - set(list((data.columns))))
        for i in missing_inputs:
            data[i]=0
        #Reshaping into arrays
        X = (data[self.estimation_inputs]
             ._get_numeric_data()
             .dropna(axis=0, how='any')
             .as_matrix())
        y = outputs['output'].dropna(axis=0, how='any').as_matrix()
        #Diving into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        algo.fit(X_train, y_train)
        if self.algo_estimator == 'LR':
            if self.verbose:
                self.logger.info('Saving LR coefs in json file')
            with open('coefs/lr_coefs.json', 'w') as outfile:
                json.dump([algo.intercept_]+list(algo.coef_), outfile)
        if self.verbose:
            self.logger.info('Saving ' + self.algo_estimator + ' to ' + self.algo_estimator + '_estimator.pkl')
        path = get_path('models')+'/'+self.algo_estimator + '_estimator.pkl'
        joblib.dump(algo, path)
        if self.verbose:
            self.logger.info('R squared on train set is {}'.format(r2_score(y_train, algo.predict(X_train))))
        y_pred_test = algo.predict(X_test)
        MAPE_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        y_pred_train = algo.predict(X_train)
        MAPE_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        #with open('MAPE.txt', 'w') as f:
            #f.write(str(MAPE))
        if self.verbose:
            self.logger.info('MAPE on train set is: {}'.format(MAPE_train))
            self.logger.info('MAPE on test set is: {}'.format(MAPE_test))
            self.logger.info('RMSE on train set is {}'.format(np.sqrt(mean_squared_error(y_train, y_pred_train))))
            self.logger.info('RMSE on test set is {}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
        return algo

# TODO
# Adding n*log(n)*v (supposedly = runtime of training in big o notation)
#X_1=np.append(X,np.array(X[:,1]*X[:,0]*np.log(X[:,0])).reshape(432,1),axis=1)
