import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.metrics import r2_score
import multiprocessing
import joblib
from sklearn import linear_model
from utils import Logging
import warnings
import itertools
import os
import pandas as pd
<<<<<<< HEAD

=======
>>>>>>> 6b9cb23a78ada191eebc5457effda10b1cad477c

warnings.simplefilter("ignore")
log = Logging(__name__)


class RFest(object):
<<<<<<< HEAD
    RAW_ESTIMATION_INPUTS=['n_estimators','max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'bootstrap', 'oob_score', 'n_jobs']

    ESTIMATION_INPUTS=['num_rows','num_features','n_estimators','max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features_auto', 'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'bootstrap', 'oob_score', 'n_jobs']
    MAX_DEPTH_RANGE=[10,50,100]
    INPUTS_RANGE=[5,50,100]
    N_ESTIMATORS_RANGE=[10,50,100]
    ROWS_RANGE=[100,1000,10000]
    ALGO_ESTIMATOR='LR'
    DROP_RATE=0.9
    MAX_FEATURES='auto' #TO VERIFY
    MIN_SAMPLES_SPLIT_RANGE=[2,4,10]
    MIN_SAMPLES_LEAF_RANGE=[1,5,10]
    MIN_WEIGHT_FRACTION_LEAF_RANGE=[0.1,0.25,0.5]
    MAX_LEAF_NODES_RANGE=[2,4,10]
    MIN_IMPURITY_SPLIT_RANGE=[1,5,10]
    MIN_IMPURITY_DECREASE_RANGE=[1,5,10]
    BOOTSTRAP=[True,False]
    OOB_SCORE=[False] ##OOB SCORE CAN BE TRUE IFF BOOTSTRAP IS TRUE!
    N_JOBS_RANGE=[-1,1,2]

    #criterion
    #RANDOM_STATE
    #verbose
    #warm_start
    #class_weight

    def __init__(self,raw_estimation_inputs=RAW_ESTIMATION_INPUTS,estimation_inputs=ESTIMATION_INPUTS,drop_rate=DROP_RATE,max_depth_range=MAX_DEPTH_RANGE,inputs_range=INPUTS_RANGE,
                 n_estimators_range=N_ESTIMATORS_RANGE,rows_range=ROWS_RANGE,algo_estimator=ALGO_ESTIMATOR,max_features=MAX_FEATURES,
                 min_samples_split_range=MIN_SAMPLES_SPLIT_RANGE,min_samples_leaf_range=MIN_SAMPLES_LEAF_RANGE,min_weight_fraction_leaf_range=MIN_WEIGHT_FRACTION_LEAF_RANGE,
                 max_leaf_nodes_range=MAX_LEAF_NODES_RANGE,min_impurity_split_range=MIN_IMPURITY_SPLIT_RANGE,
                 min_impurity_decrease_range=MIN_IMPURITY_DECREASE_RANGE,bootstrap=BOOTSTRAP,oob_score=OOB_SCORE,n_jobs_range=N_JOBS_RANGE):
        self.raw_estimation_inputs = raw_estimation_inputs
        self.estimation_inputs=estimation_inputs
        self.drop_rate=drop_rate
        self.max_depth_range=max_depth_range
=======
    RAW_ESTIMATION_INPUTS = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                             'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease',
                             'min_impurity_split', 'bootstrap', 'oob_score', 'n_jobs']

    ESTIMATION_INPUTS = ['num_rows', 'num_features', 'n_estimators', 'max_depth', 'min_samples_split',
                         'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features_auto', 'max_leaf_nodes',
                         'min_impurity_decrease', 'min_impurity_split', 'bootstrap', 'oob_score', 'n_jobs']
    MAX_DEPTH_RANGE = [10, 50, 100]
    INPUTS_RANGE = [5, 50, 100]
    N_ESTIMATORS_RANGE = [10, 50, 100]
    ROWS_RANGE = [100, 1000, 10000]
    ALGO_ESTIMATOR = 'LR'
    DROP_RATE = 0.9
    MAX_FEATURES = 'auto'  # TO VERIFY
    MIN_SAMPLES_SPLIT_RANGE = [2, 4, 10]
    MIN_SAMPLES_LEAF_RANGE = [1, 5, 10]
    MIN_WEIGHT_FRACTION_LEAF_RANGE = [0.1, 0.25, 0.5]
    MAX_LEAF_NODES_RANGE = [2, 4, 10]
    MIN_IMPURITY_SPLIT_RANGE = [1, 5, 10]
    MIN_IMPURITY_DECREASE_RANGE = [1, 5, 10]
    BOOTSTRAP = [True, False]
    OOB_SCORE = [False]  ##OOB SCORE CAN BE TRUE IFF BOOTSTRAP IS TRUE!
    N_JOBS_RANGE = [-1, 1, 2]

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
                 n_jobs_range=N_JOBS_RANGE):
        self.raw_estimation_inputs = raw_estimation_inputs
        self.estimation_inputs = estimation_inputs
        self.drop_rate = drop_rate
        self.max_depth_range = max_depth_range
>>>>>>> 6b9cb23a78ada191eebc5457effda10b1cad477c
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

    def measure_time(self, n, p, rf_params):
        start_time = time.time()
        X = np.random.rand(n, p)
        y = np.random.rand(n, )
        clf = RandomForestRegressor(**rf_params)
        clf.fit(X, y)
        elapsed_time = time.time() - start_time
        return elapsed_time

    def generate_data(self):
        log.info('Generating dummy training durations to create a training set')
<<<<<<< HEAD
        inputs=[]
        outputs=[]
        rf_parameters_list=self.raw_estimation_inputs

        for permutation in itertools.product(
            self.rows_range,
            self.inputs_range,
            self.n_estimators_range,
            self.max_depth_range,
            self.min_samples_split_range,
            self.min_samples_leaf_range,
            self.min_weight_fraction_leaf_range,
            [self.max_features],
            self.max_leaf_nodes_range,
            self.min_impurity_split_range,
            self.min_impurity_decrease_range,
            self.bootstrap,
            self.oob_score,
            self.n_jobs_range):


            n=permutation[0]
            p=permutation[1]
            rf_parameters_dic = dict(zip(rf_parameters_list, permutation[2:]))
            
            if np.random.uniform()>self.drop_rate:
                outputs.append(self.measure_time(n,p,rf_parameters_dic))
                inputs.append(permutation)



        inputs=pd.DataFrame(inputs,columns=['num_rows']+['num_features']+rf_parameters_list)
        outputs=pd.DataFrame(outputs,columns=['output'])

        return (inputs,outputs)

    def model_fit(self):
        df,outputs=self.generate_data()

        data = pd.get_dummies(df)

        if self.algo_estimator=='LR':
            algo=linear_model.LinearRegression()
        log.info('Fitting '+self.algo_estimator+' to estimate training durations')
=======
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
                [self.max_features],
                self.max_leaf_nodes_range,
                self.min_impurity_split_range,
                self.min_impurity_decrease_range,
                self.bootstrap,
                self.oob_score,
                self.n_jobs_range):

            n = permutation[0]
            p = permutation[1]
            rf_parameters_dic = dict(zip(rf_parameters_list, permutation[2:]))

            if np.random.uniform() > self.drop_rate:
                outputs.append(self.measure_time(n, p, rf_parameters_dic))
                inputs.append(permutation)

        inputs = pd.DataFrame(inputs, columns=['num_rows'] + ['num_features'] + rf_parameters_list)
        outputs = pd.DataFrame(outputs, columns=['output'])

        return (inputs, outputs)

    def model_fit(self):
        df, outputs = self.generate_data()

        data = pd.get_dummies(df)

        if self.algo_estimator == 'LR':
            algo = linear_model.LinearRegression()
        log.info('Fitting ' + self.algo_estimator + ' to estimate training durations')
>>>>>>> 6b9cb23a78ada191eebc5457effda10b1cad477c

        X = (data[self.estimation_inputs]
             ._get_numeric_data()
             .dropna(axis=0, how='any')
             .as_matrix())
<<<<<<< HEAD
        y= outputs['output'].dropna(axis=0, how='any').as_matrix()
=======
        y = outputs['output'].dropna(axis=0, how='any').as_matrix()
>>>>>>> 6b9cb23a78ada191eebc5457effda10b1cad477c
        algo.fit(X, y)
        log.info('Saving ' + self.algo_estimator + ' to ' + self.algo_estimator + '_estimator.pkl')
        joblib.dump(algo, self.algo_estimator + '_estimator.pkl')
        log.info('R squared is {}'.format(r2_score(y, algo.predict(X))))
        return algo

    def estimate_duration(self, X, algo):
        log.info('Fetching estimator: ' + self.algo_estimator + '_estimator.pkl')
        estimator = joblib.load(self.algo_estimator + '_estimator.pkl')
<<<<<<< HEAD
        inputs=[]
        n=X.shape[0]
        inputs.append(n)
        p=X.shape[1]
        inputs.append(p)
        params=algo.get_params()
=======
        inputs = []
        n = X.shape[0]
        inputs.append(n)
        p = X.shape[1]
        inputs.append(p)
        params = algo.get_params()
>>>>>>> 6b9cb23a78ada191eebc5457effda10b1cad477c

        for i in self.raw_estimation_inputs:
            inputs.append(params[i])

<<<<<<< HEAD

        pred=estimator.predict(np.array([inputs]))
        log.info('Training your model should take ~ '+str(pred[0])+' seconds')
=======
        pred = estimator.predict(np.array([inputs]))
        log.info('Training your model should take ~ ' + str(pred[0]) + ' seconds')
>>>>>>> 6b9cb23a78ada191eebc5457effda10b1cad477c
        return pred

# TODO
# Adding n*log(n)*v (supposedly = runtime of training in big o notation)
#X_1=np.append(X,np.array(X[:,1]*X[:,0]*np.log(X[:,0])).reshape(432,1),axis=1)