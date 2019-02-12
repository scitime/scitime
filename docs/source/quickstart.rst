Scitime Quick Start
====================================

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

    from scitime import Estimator

    # example for kmeans clustering
    estimator = Estimator(meta_algo='RF', verbose=3)
    km = KMeans()

    # generating inputs for this example
    X = np.random.rand(100000,10)
    # run the estimation
    estimation, lower_bound, upper_bound = estimator.time(km, X)

