[![Documentation Status](https://readthedocs.org/projects/scitime/badge/?version=latest)](https://scitime.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/simkimsia/UtilityBehaviors.png)](https://travis-ci.com/nathan-toubiana/scitime) [![Build status](https://ci.appveyor.com/api/projects/status/f6xp39veawdd4y43?svg=true)](https://ci.appveyor.com/project/nathan-toubiana/scitime-2l382)
 [![codecov](https://codecov.io/gh/nathan-toubiana/scitime/branch/master/graph/badge.svg?token=yWAeEV2qWc)](https://codecov.io/gh/nathan-toubiana/scitime) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
 [![PyPI version](https://badge.fury.io/py/scitime.svg)](https://badge.fury.io/py/scitime) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/scitime.svg)](https://anaconda.org/conda-forge/scitime) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/scitime.svg)](https://anaconda.org/conda-forge/scitime)
 [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# scitime
Training time estimation for scikit-learn algorithms. Method explained in this [article](https://medium.freecodecamp.org/two-hours-later-and-still-running-how-to-keep-your-sklearn-fit-under-control-cc603dc1283b?source=friends_link&sk=98e79add47516c38eeec59cf755df938)

Currently supporting:
- RandomForestRegressor
- SVC
- KMeans
- RandomForestClassifier

### Environment setup
Python version: 3.6 or higher

Package dependencies:
- scikit-learn (>=0.19.1)
- pandas (>=0.20.3)
- joblib (>=0.12.5)
- psutil (>=5.4.7)

#### Install scitime
```
❱ pip install scitime
or 
❱ conda install -c conda-forge scitime
```

### Usage

#### How to compute a runtime estimation

- Example for RandomForestRegressor

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time

from scitime import Estimator

# example for rf regressor
estimator = Estimator(meta_algo='RF', verbose=3)
rf = RandomForestRegressor()

X,y = np.random.rand(100000,10),np.random.rand(100000,1)
# run the estimation
estimation, lower_bound, upper_bound = estimator.time(rf, X, y)

# compare to the actual training time
start_time = time.time()
rf.fit(X,y)
elapsed_time = time.time() - start_time
print("elapsed time: {:.2}".format(elapsed_time))
```

- Example for KMeans

```python
from sklearn.cluster import KMeans
import numpy as np
import time

from scitime import Estimator

# example for kmeans clustering
estimator = Estimator(meta_algo='RF', verbose=3)
km = KMeans()

X = np.random.rand(100000,10)
# run the estimation
estimation, lower_bound, upper_bound = estimator.time(km, X)

# compare to the actual training time
start_time = time.time()
km.fit(X)
elapsed_time = time.time() - start_time
print("elapsed time: {:.2}".format(elapsed_time))
```

The Estimator class arguments:

- **meta_algo**: The estimator used to predict the time, either RF or NN 
- **verbose**: Controls the amount of log output (either 0, 1, 2 or 3)
- **confidence**: Confidence for intervals (defaults to 95%)

Parameters of the estimator.time function:
- **X**: np.array of inputs to be trained
- **y**: np.array of outputs to be trained (set to None for unsupervised algo)
- **algo**: algo whose runtime the user wants to predict

### --- FOR TESTERS / CONTRIBUTORS ---


#### Local Testing
Inside virtualenv (with pytest>=3.2.1):
```
(env)$ python -m pytest
```
#### How to use _data.py to generate data / fit models?
```
$ python _data.py --help

usage: _data.py [-h] [--drop_rate DROP_RATE] [--meta_algo {RF,NN}]
                [--verbose VERBOSE]
                [--algo {RandomForestRegressor,RandomForestClassifier,SVC,KMeans}]
                [--generate_data] [--fit FIT] [--save]

Gather & Persist Data of model training runtimes

optional arguments:
  -h, --help            show this help message and exit
  --drop_rate DROP_RATE
                        drop rate of number of data generated (from all param
                        combinations taken from _config.json). Default is
                        0.999
  --meta_algo {RF,NN}   meta algo used to fit the meta model (NN or RF) -
                        default is RF
  --verbose VERBOSE     verbose mode (0, 1, 2 or 3)
  --algo {RandomForestRegressor,RandomForestClassifier,SVC,KMeans}
                        algo to train data on
  --generate_data       do you want to generate & write data in a dedicated
                        csv?
  --fit FIT             do you want to fit the model? If so indicate the csv
                        name
  --save                (only used for model fit) do you want to save /
                        overwrite the meta model from this fit?
```
(_data.py uses _model.py behind the scenes)
#### How to run _model.py?

After pulling the master branch (`git pull origin master`) and setting the environment (described above),
run `ipython` and:

```python
from scitime._model import Model

# example of data generation for rf regressor
trainer = Model(drop_rate=0.99999, verbose=3, algo='RandomForestRegressor')
inputs, outputs, _ = trainer._generate_data()

# then fitting the meta model
meta_algo = trainer.model_fit(generate_data=False, inputs=inputs, outputs=outputs)
# this should not locally overwrite the pickle file located at scitime/models/{your_model}
# if you want to save the model, set the argument save_model to True
```
