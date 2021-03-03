Scitime Quick Start
====================================
Scitime currently supports:

* RandomForestClassifier
* SVC
* KMeans
* RandomForestRegressor

Installation
------------

To install:

.. code-block:: bash

    pip install scitime

Or:

.. code-block:: bash

    conda install -c conda-forge scitime

Usage
-------

Example of getting runtime estimation for KMeans:

.. code-block:: python

    from sklearn.cluster import KMeans
    import numpy as np
    import time

    from scitime import RuntimeEstimator

    # example for kmeans clustering
    estimator = RuntimeEstimator(meta_algo='RF', verbose=3)
    km = KMeans()

    # generating inputs for this example
    X = np.random.rand(100000,10)
    # run the estimation
    estimation, lower_bound, upper_bound = estimator.time(km, X)


Local Testing
-------
Inside virtualenv (with pytest>=3.2.1):

.. code-block:: bash

    python -m pytest


Use _data.py to generate data
-------
(for contributors)

.. code-block:: bash

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

Contribute
-------
The preferred way to contribute to scitime is to fork the main repository on GitHub, then submit a “pull request” (PR) - as done for scikit-learn contributions:

- Create an account on GitHub if you do not already have one.
- Fork the project repository: click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account.
- Clone your fork of the scitime repo from your GitHub account to your local disk:

.. code-block:: bash

    git clone git@github.com:YourLogin/scitime.git
    cd scitime
    # Install library in editable mode:
    pip install --editable .



- Create a branch to hold your development changes:

.. code-block:: bash

    git checkout -b my-feature


and start making changes. Always use a feature branch. It’s good practice to never work on the masterbranch!

- Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using git add and then git commit files:

.. code-block:: bash


    git add modified_files
    git commit


- to record your changes in Git, then push the changes to your GitHub account with:

.. code-block:: bash

    git push -u origin my-feature



- Follow GitHub instructions to create a pull request from your fork.

Some quick additional notes:
- We use appveyor and travis.ci for our tests
- We try to follow the PEP8 guidelines (using flake8, ignoring codes E501 and F401)
