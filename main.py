import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib
from sklearn import linear_model
from utils import Logging
import warnings
import itertools
import os
import json
import pandas as pd
from sklearn.metrics import mean_squared_error
import csv

warnings.simplefilter("ignore")
log = Logging(__name__)


class RFest(object):
    '''Predicting training time of a random forest regression'''
    RAW_ESTIMATION_INPUTS = open('inputs/raw_inputs.txt').read().splitlines()
    ESTIMATION_INPUTS = open('inputs/inputs.txt').read().splitlines()
    MAX_DEPTH_RANGE = [10, 50, 100]
    INPUTS_RANGE = [5, 50, 100]
    N_ESTIMATORS_RANGE = [10, 50, 100]
    ROWS_RANGE = [100, 1000, 10000]
    ALGO_ESTIMATOR = 'LR'
    DROP_RATE = 0.9
    MAX_FEATURES = ['auto'  ,10]
    MIN_SAMPLES_SPLIT_RANGE = [2, 4, 10]
    MIN_SAMPLES_LEAF_RANGE = [1, 5, 10]
    MIN_WEIGHT_FRACTION_LEAF_RANGE = [0.1, 0.25, 0.5]
    MAX_LEAF_NODES_RANGE = [2, 4, 10]
    MIN_IMPURITY_SPLIT_RANGE = [1, 5, 10]
    MIN_IMPURITY_DECREASE_RANGE = [1, 5, 10]
    BOOTSTRAP = [True, False]
    OOB_SCORE = [False]  ##OOB SCORE CAN BE TRUE IFF BOOTSTRAP IS TRUE!
    N_JOBS_RANGE = [1, 2, 5, 8]
    DUMMY_VARIABLES=['max_features']

    # criterion
    # RANDOM_STATE
    # verbose
    # warm_start
    # class_weight

    def __init__(self, raw_estimation_inputs=RAW_ESTIMATION_INPUTS, estimation_inputs=ESTIMATION_INPUTS,
                 drop_rate=DROP_RATE, max_depth_range=MAX_DEPTH_RANGE, inputs_range=INPUTS_RANGE,
                 n_estimators_range=N_ESTIMATORS_RANGE, rows_range=ROWS_RANGE, algo_estimator=ALGO_ESTIMATOR,
                 max_features=MAX_FEATURES,
                 min_samples_split_range=MIN_SAMPLES_SPLIT_RANGE, min_samples_leaf_range=MIN_SAMPLES_LEAF_RANGE,
                 min_weight_fraction_leaf_range=MIN_WEIGHT_FRACTION_LEAF_RANGE,
                 max_leaf_nodes_range=MAX_LEAF_NODES_RANGE, min_impurity_split_range=MIN_IMPURITY_SPLIT_RANGE,
                 min_impurity_decrease_range=MIN_IMPURITY_DECREASE_RANGE, bootstrap=BOOTSTRAP, oob_score=OOB_SCORE,
                 n_jobs_range=N_JOBS_RANGE,dummy_variables=DUMMY_VARIABLES):
        '''Defining: algo paramters, algo parameters range, drop rate, number of cores'''
        self.raw_estimation_inputs = raw_estimation_inputs
        self.estimation_inputs = estimation_inputs
        self.drop_rate = drop_rate
        self.max_depth_range = max_depth_range
        self.inputs_range = inputs_range
        self.n_estimators_range = n_estimators_range
        self.rows_range = rows_range
        self.algo_estimator = algo_estimator
        self.max_features = max_features
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.min_weight_fraction_leaf_range = min_weight_fraction_leaf_range
        self.max_leaf_nodes_range = max_leaf_nodes_range
        self.min_impurity_split_range = min_impurity_split_range
        self.min_impurity_decrease_range = min_impurity_decrease_range
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs_range = n_jobs_range
        self.num_cpu = os.cpu_count()
        self.dummy_variables=dummy_variables

    def measure_time(self, n, p, rf_params):
        '''Generating fake fit and saving the training runtime'''
        #Genrating dummy inputs / outputs
        X = np.random.rand(n, p)
        y = np.random.rand(n, )
        #Fitting rf
        clf = RandomForestRegressor(**rf_params)
        start_time = time.time()
        clf.fit(X, y)
        elapsed_time = time.time() - start_time
        return elapsed_time

    def generate_data(self):
        '''Measuring training runtimes for a set of distincts parameters'''
        log.info('Generating dummy training durations to create a training set')
        inputs = []
        outputs = []
        rf_parameters_list = self.raw_estimation_inputs

        for permutation in itertools.product(
                self.rows_range,
                self.inputs_range,
                self.n_estimators_range,
                self.max_depth_range,
                self.min_samples_split_range,
                self.min_samples_leaf_range,
                self.min_weight_fraction_leaf_range,
                self.max_features,
                self.max_leaf_nodes_range,
                self.min_impurity_split_range,
                self.min_impurity_decrease_range,
                self.bootstrap,
                self.oob_score,
                self.n_jobs_range):

            n = permutation[0]
            p = permutation[1]
            f = permutation[7]

            rf_parameters_dic = dict(zip(rf_parameters_list, permutation[2:]))
            #Computing only for (1-self.drop_rate) % of the data
            if np.random.uniform() > self.drop_rate:
                #Handling max_features > p case
                if type(f)==int:
                    if f<=p:
                        thisOutput=self.measure_time(n, p, rf_parameters_dic)
                        thisInput=permutation
                        log.info('data added for {p} which outputs {s} seconds'.format(p=thisInput,s=thisOutput))
                        #outputs.append(thisOutput)
                        #inputs.append(thisInput)

                        with open(r'result.csv', 'a+') as file:
                            #columns=['num_rows'] + ['num_features'] + rf_parameters_list + ['output']
                            writer = csv.writer(file)
                            thisRow=list(thisInput)+[thisOutput]
                            writer.writerows([thisRow])      
                else:
                    thisOutput=self.measure_time(n, p, rf_parameters_dic)
                    thisInput=permutation
                    log.info('data added for {p} which outputs {s} seconds'.format(p=thisInput,s=thisOutput))
                    #outputs.append(thisOutput)
                    #inputs.append(thisInput)

                    with open(r'result.csv', 'a+') as file:
                        #columns=['num_rows'] + ['num_features'] + rf_parameters_list + ['output']
                        writer = csv.writer(file)
                        thisRow=list(thisInput)+[thisOutput]
                        writer.writerows([thisRow])                

        #inputs = pd.DataFrame(inputs, columns=['num_rows'] + ['num_features'] + rf_parameters_list)
        #outputs = pd.DataFrame(outputs, columns=['output'])

        #return (inputs, outputs)

    def model_fit(self,generate_data=True,df=None,outputs=None):
        '''Building the actual training time estimator'''
        if generate_data==True:
            df, outputs = self.generate_data()

        data = pd.get_dummies(df)

        if self.algo_estimator == 'LR':
            algo = linear_model.LinearRegression()
        log.info('Fitting ' + self.algo_estimator + ' to estimate training durations')
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
            log.info('Saving LR coefs in json file')
            with open('coefs/lr_coefs.json', 'w') as outfile:
                json.dump([algo.intercept_]+list(algo.coef_), outfile)
        log.info('Saving ' + self.algo_estimator + ' to ' + self.algo_estimator + '_estimator.pkl')
        joblib.dump(algo, self.algo_estimator + '_estimator.pkl')
        log.info('R squared on train set is {}'.format(r2_score(y_train, algo.predict(X_train))))
        y_pred_test = algo.predict(X_test)
        MAPE_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        y_pred_train = algo.predict(X_train)
        MAPE_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        #with open('MAPE.txt', 'w') as f:
            #f.write(str(MAPE))
        log.info('MAPE on train set is: {}'.format(MAPE_train))
        log.info('MAPE on test set is: {}'.format(MAPE_test))
        log.info('RMSE on train set is {}'.format(np.sqrt(mean_squared_error(y_train, y_pred_train))))
        log.info('RMSE on test set is {}'.format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
        return algo

    def estimate_duration(self, X, algo):
        '''Predicting training runtime'''
        if self.algo_estimator == 'LR':
            log.info('Loading LR coefs from json file')
            with open('coefs/lr_coefs.json', 'r') as f:
                coefs= json.load(f)
        else:
            log.info('Fetching estimator: ' + self.algo_estimator + '_estimator.pkl')
            estimator = joblib.load(self.algo_estimator + '_estimator.pkl')
        #Retrieving all parameters of interest
        inputs = []
        n = X.shape[0]
        inputs.append(n)
        p = X.shape[1]
        inputs.append(p)
        params = algo.get_params()

        for i in self.raw_estimation_inputs:
            #Handling n_jobs=-1 case
            if (i=='n_jobs'):
                if (params[i]==-1):
                    inputs.append(self.num_cpu)
                else:
                    inputs.append(params[i])
                    
            else:
                if i in self.dummy_variables:
                    #To make dummy 
                    inputs.append(str(params[i]))
                else:    
                    inputs.append(params[i])
        #Making dummy
        dic=dict(zip(['num_rows']+['num_features']+self.raw_estimation_inputs, [[i] for i in inputs]))
        df=pd.DataFrame(dic, columns=['num_rows']+['num_features']+self.raw_estimation_inputs)
        df=pd.get_dummies(df)
        missing_inputs = list(set(list(self.estimation_inputs)) - set(list((df.columns))))
        for i in missing_inputs:
            df[i]=0

        df=df[self.estimation_inputs]        
        if self.algo_estimator == 'LR':
            pred=coefs[0]
            for i in range(df.shape[1]):
                pred+=df.ix[0,i]*coefs[i+1]
        log.info('Training your model should take ~ ' + str(pred) + ' seconds')
        return pred

# TODO
# Adding n*log(n)*v (supposedly = runtime of training in big o notation)
#X_1=np.append(X,np.array(X[:,1]*X[:,0]*np.log(X[:,0])).reshape(432,1),axis=1)
