from train import Trainer
import joblib
from utils import LogMixin, get_path, config
import os
import json
import pandas as pd


class Estimator(Trainer, LogMixin):
    ALGO_ESTIMATOR = 'LR'
    ALGO = 'RF'

    def __init__(self, algo_estimator=ALGO_ESTIMATOR, algo=ALGO, verbose=True):
        super().__init__(verbose, algo_estimator, algo)
        self.algo_estimator = algo_estimator
        self.num_cpu = os.cpu_count()
        self.algo = algo
        self.params = config(self.algo)
        self.verbose = verbose
        self.estimation_inputs = [i for i in self.params['external_params'].keys()] + [i for i in self.params[
            'internal_params'].keys() if i not in self.params['dummy_inputs']] + [i + '_' + str(k) for i in
                                                                                  self.params['internal_params'].keys()
                                                                                  if i in self.params['dummy_inputs']
                                                                                  for k in
                                                                                  self.params['internal_params'][i]]

    def estimate_duration(self, X, algo):
        """
        predicts training runtime for a given training

        :param X: np.array of inputs to be trained
        :param algo: algo used to predict runtimee
        :return: predicted runtime
        :rtype: float
        """
        if self.algo_estimator == 'LR':
            if self.verbose:
                self.logger.info('Loading LR coefs from json file')
            with open('coefs/lr_coefs.json', 'r') as f:
                coefs = json.load(f)
        else:
            if self.verbose:
                self.logger.info('Fetching estimator: ' + self.algo_estimator + '_estimator.pkl')
            path = get_path('models') + '/' + self.algo_estimator + '_estimator.pkl'
            estimator = joblib.load(path)
        # Retrieving all parameters of interest
        inputs = []
        n = X.shape[0]
        inputs.append(n)
        p = X.shape[1]
        inputs.append(p)
        params = algo.get_params()
        param_list = list(self.params['external_params'].keys()) + list(self.params['internal_params'].keys())

        for i in self.params['internal_params'].keys():
            # Handling n_jobs=-1 case
            if (i == 'n_jobs'):
                if (params[i] == -1):
                    inputs.append(self.num_cpu)
                else:
                    inputs.append(params[i])

            else:
                if i in self.params['dummy_inputs']:
                    # To make dummy
                    inputs.append(str(params[i]))
                else:
                    inputs.append(params[i])
        # Making dummy
        dic = dict(zip(param_list, [[i] for i in inputs]))
        if self.verbose:
            self.logger.info('Training your model for these params: {}'.format(dic))
        df = pd.DataFrame(dic, columns=param_list)
        df = pd.get_dummies(df)
        # adding 0 columns for columns that are not in the dataset, assuming it s only dummy columns
        missing_inputs = list(set(list(self.estimation_inputs)) - set(list((df.columns))))
        for i in missing_inputs:
            df[i] = 0

        df = df[self.estimation_inputs]
        if self.algo_estimator == 'LR':
            pred = coefs[0]
            for i in range(df.shape[1]):
                pred += df.ix[0, i] * coefs[i + 1]
        else:
            X = (df[self.estimation_inputs]
                 ._get_numeric_data()
                 .dropna(axis=0, how='any')
                 .as_matrix())
            pred = estimator.predict(X)
        if self.verbose:
            self.logger.info('Training your model should take ~ ' + str(pred[0]) + ' seconds')
        return pred