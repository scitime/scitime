import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.metrics import r2_score
import multiprocessing
import joblib
from sklearn import linear_model
from utils import Logging
import warnings

warnings.simplefilter("ignore")
log = Logging(__name__)

class RFest(object):
    MAX_DEPTH_RANGE=[10,50,100]
    INPUTS_RANGE=[5,50,100]
    N_ESTIMATORS_RANGE=[10,50,100]
    ROWS_RANGE=[100,1000,10000]
    ALGO_ESTIMATOR='LR'
    DROP_RATE=0.9


    def __init__(self,drop_rate=DROP_RATE,max_depth_range=MAX_DEPTH_RANGE,inputs_range=INPUTS_RANGE,
                 n_estimators_range=N_ESTIMATORS_RANGE,rows_range=ROWS_RANGE,algo_estimator=ALGO_ESTIMATOR):
        self.drop_rate=drop_rate
        self.max_depth_range=max_depth_range
        self.inputs_range = inputs_range
        self.n_estimators_range = n_estimators_range
        self.rows_range=rows_range
        self.algo_estimator=algo_estimator
        self.num_cpu=multiprocessing.cpu_count()

    def measure_time(self,n=10,p=10,i=5,j=2,k=100,l=-1):
        start_time = time.time()
        X=np.random.rand(n,p)
        y=np.random.rand(n,)
        clf = RandomForestRegressor(max_depth=i, max_features=j, n_estimators=k,n_jobs=l)
        clf.fit(X,y)
        elapsed_time = time.time() - start_time
        return elapsed_time

    def generate_data(self):
        log.info('Generating dummy training durations to create a training set')
        inputs=[]
        outputs=[]
        for i in self.max_depth_range:
            for k in self.n_estimators_range:
                for p in self.inputs_range:
                    for j in range(1,p,10):
                        for n in self.rows_range:
                            if np.random.uniform()>self.drop_rate:
                                outputs.append(self.measure_time(j=j,i=i,n=n,k=k,p=p))
                                inputs.append(np.array([n,p,i,j,k,-1]))
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