import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.metrics import r2_score
import multiprocessing


def num_cpu():
#Checking number of scores
    return multiprocessing.cpu_count()


def measure_time(n=10,p=10,i=5,j=2,k=100,l=-1):
    start_time = time.time()
    X=np.random.rand(n,p)
    y=np.random.rand(n,)
    clf = RandomForestRegressor(max_depth=i, max_features=j, n_estimators=k,n_jobs=l)
    clf.fit(X,y)
    elapsed_time = time.time() - start_time
    return elapsed_time

def generate_data():
    inputs=[]
    outputs=[]
    for i in [10,50,100] :
        for k in [10,50,100]:
            for p in [5,50,100]:
                for j in range(1,p,10):
                    for n in [100,1000,10000]:
                        outputs.append(measure_time(j=j,i=i,n=n,k=k,p=p))
                        inputs.append(np.array([n,p,i,j,k,-1]))
    return (inputs,outputs)

def model_fit(algo,X,y):
    algo.fit(X, y)
    return (algo,r2_score(y, regr.predict(X)))


def model_predict(algo,X):
    n=X.shape[0]
    p=X.shape[1]
    i=algo.max_depth
    j=algo.max_features
    k=algo.n_estimators
    return algo.predict(np.array([n,p,i,j,k,-1]))


#TODO
#Adding n*log(n)*v (supposedly = runtime of training in big o notation)
#X_1=np.append(X,np.array(X[:,1]*X[:,0]*np.log(X[:,0])).reshape(432,1),axis=1)