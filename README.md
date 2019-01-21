# scitime
Training time estimation for sklearn algos
### Environment setup
Python version: 3.6.3
#### Virtualenv
```
❱ virtualenv env
❱ source env/bin/activate
❱ pip install "git+ssh://git@github.com/nathan-toubiana/scitime.git"
or
❱ pip install git+https://github.com/nathan-toubiana/scitime.git
```

#### How to run estimate.py?

After having a corresponding model in `scitime/models/`:

Example for RandomForestRegressor

```
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
import pandas as pd

from scitime.estimate import Estimator

# example for rf regressor
estimator = Estimator(meta_algo='RF', verbose=3)
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=200,
           max_features=10, max_leaf_nodes=10, min_impurity_decrease=10,
           min_impurity_split=10, min_samples_leaf=10,
           min_samples_split=10, min_weight_fraction_leaf=0.5,
           n_estimators=100, n_jobs=10, oob_score=False, random_state=None,
           verbose=2, warm_start=False)

X,y = np.random.rand(100000,10),np.random.rand(100000,1)
# run the estimation
estimation, lower_bound, upper_bound = estimator.time(rf, X, y)

# compare to the actual training time
start_time = time.time()
rf.fit(X,y)
elapsed_time = time.time() - start_time
print("elapsed time: {:.2}".format(elapsed_time))
print("estimated elapsed time: {:.2}. 95% confidence interval: [{:.2},{:.2}]".format(estimation, lower_bound, upper_bound))
```

Example for KMeans

```
from sklearn.cluster import KMeans
import numpy as np
import time
import pandas as pd

from scitimet.estimate import Estimator

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
print("estimated elapsed time: {:.2}. 95% confidence interval: [{:.2},{:.2}]".format(estimation, lower_bound, upper_bound))
```

### ---FOR TESTERS / CONTRIBUTORS ---


#### Testing
Inside virtualenv (with pytest>=3.2.1):
```
(env)$ python -m pytest
```
#### How to use _data.py to generate data / fit models?
```
$ python _data.py --help

usage: _data.py [-h] [--drop_rate DROP_RATE] [--meta_algo {RF,NN}]
                [--verbose VERBOSE]
                [--algo {RandomForestRegressor,SVC,KMeans}] [--generate_data]
                [--fit FIT] [--save]

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
  --algo {RandomForestRegressor,SVC,KMeans}
                        algo to train data on
  --generate_data       do you want to generate & write data in a dedicated
                        csv?
  --fit FIT             do you want to fit the model? If so indicate the csv
                        name
  --save                (only used for model fit) do you want to save /
                        overwrite the meta model from this fit?
```
#### How to run _model.py?

After pulling the master branch (`git pull origin master`) and setting the environment (described above),
run `ipython` and:

```
from scitime._model import Model

# example of data generation for rf regressor
trainer = Model(drop_rate=0.99999, verbose=3, algo='RandomForestRegressor')
inputs, outputs, _ = trainer._generate_data()

# then fitting the meta model
meta_algo = trainer.model_fit(generate_data=False, inputs=inputs, outputs=outputs)
# this should not locally overwrite the pickle file located at scitime/models/{your_model}
# if you want to save the model, set the argument save_model to True
```