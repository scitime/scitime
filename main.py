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


warnings.simplefilter("ignore")
log = Logging(__name__)

class RFest(object):
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

    def __init__(self,drop_rate=DROP_RATE,max_depth_range=MAX_DEPTH_RANGE,inputs_range=INPUTS_RANGE,
                 n_estimators_range=N_ESTIMATORS_RANGE,rows_range=ROWS_RANGE,algo_estimator=ALGO_ESTIMATOR,max_features=MAX_FEATURES,
                 min_samples_split_range=MIN_SAMPLES_SPLIT_RANGE,min_samples_leaf_range=MIN_SAMPLES_LEAF_RANGE,min_weight_fraction_leaf_range=MIN_WEIGHT_FRACTION_LEAF_RANGE,
                 max_leaf_nodes_range=MAX_LEAF_NODES_RANGE,min_impurity_split_range=MIN_IMPURITY_SPLIT_RANGE,
                 min_impurity_decrease_range=MIN_IMPURITY_DECREASE_RANGE,bootstrap=BOOTSTRAP,oob_score=OOB_SCORE,n_jobs_range=N_JOBS_RANGE):
        self.drop_rate=drop_rate
        self.max_depth_range=max_depth_range
        self.inputs_range = inputs_range
        self.n_estimators_range = n_estimators_range
        self.rows_range=rows_range
        self.algo_estimator=algo_estimator
        self.max_features=max_features
        self.min_samples_split_range=min_samples_split_range
        self.min_samples_leaf_range=min_samples_leaf_range
        self.min_weight_fraction_leaf_range=min_weight_fraction_leaf_range
        self.max_leaf_nodes_range=max_leaf_nodes_range
        self.min_impurity_split_range=min_impurity_split_range
        self.min_impurity_decrease_range=min_impurity_decrease_range
        self.bootstrap=bootstrap
        self.oob_score=oob_score
        self.n_jobs_range=n_jobs_range
        self.num_cpu=multiprocessing.cpu_count()


    def measure_time(self,n,p,n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1):
        start_time = time.time()
        X=np.random.rand(n,p)
        y=np.random.rand(n,)
        clf = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs)
        clf.fit(X,y)
        elapsed_time = time.time() - start_time
        return elapsed_time

    def generate_data(self):
        log.info('Generating dummy training durations to create a training set')
        inputs=[]
        outputs=[]
        for element in itertools.product(
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

            arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12,arg13=element

            if np.random.uniform()>self.drop_rate:
                outputs.append(self.measure_time(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12,arg13))
                inputs.append(np.array(element))
        return (inputs,outputs)

    def model_fit(self):
        X,y=self.generate_data()
        if self.algo_estimator=='LR':
            algo=linear_model.LinearRegression()
        log.info('Fitting '+self.algo_estimator+' to estimate training durations')
        algo.fit(X, y)
        log.info('Saving '+self.algo_estimator+' to '+self.algo_estimator + '_estimator.pkl')
        joblib.dump(algo, self.algo_estimator + '_estimator.pkl')
        log.info('R squared is {}'.format(r2_score(y, algo.predict(X))))
        return algo

    def estimate_duration(self,X,algo):
        log.info('Fetching estimator: '+self.algo_estimator + '_estimator.pkl')
        estimator = joblib.load(self.algo_estimator + '_estimator.pkl')
        n=X.shape[0]
        p=X.shape[1]
        i=algo.max_depth
        j=algo.max_features
        k=algo.n_estimators
        pred=estimator.predict(np.array([[n,p,i,j,k,-1]]))
        log.info('Training your model should take ~ '+str(pred[0])+' seconds')
        return pred

#TODO
#Adding n*log(n)*v (supposedly = runtime of training in big o notation)
#X_1=np.append(X,np.array(X[:,1]*X[:,0]*np.log(X[:,0])).reshape(432,1),axis=1)