# scikest
Training time estimation for sklearn algos
### Environment setup
Python version: 3.6.3
#### Virtualenv
```
❱ virtualenv env
❱ source env/bin/activate
❱ python setup.py install
```
#### How to run train.py?

After pulling the master branch (`git pull origin master`) and setting the environment (described above),
run `ipython` and:

```
from scikest.train import Trainer

# example of data generation for rf regressor
trainer = Trainer(drop_rate=0.99999, verbose=True, algo='RandomForestRegressor')
inputs, outputs = trainer._generate_data()

# then fitting the model
algo_estimator = trainer.model_fit(generate_data=False, df=inputs, outputs=outputs)
# this should overwrite the pickle file located at scikest/models/{your_model}
```
#### How to run estimate.py?

After having a corresponding model in `scikest/models/`:

```
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
import pandas as pd

from scikest.estimate import Estimator

# example for rf regressor
estimator = Estimator(algo_estimator='RF')
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=200,
           max_features=10, max_leaf_nodes=10, min_impurity_decrease=10,
           min_impurity_split=10, min_samples_leaf=10,
           min_samples_split=10, min_weight_fraction_leaf=0.5,
           n_estimators=100, n_jobs=10, oob_score=False, random_state=None,
           verbose=2, warm_start=False)

X,y = np.random.rand(100000,10),np.random.rand(100000,1)
# run the estimation
estimator.estimate_duration(X,y, rf)

# compare to the actual training time
start_time = time.time()
rf.fit(X,y)
elapsed_time = time.time() - start_time
print(elapsed_time)
```