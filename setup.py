from setuptools import setup
from os import path

DIR = path.dirname(path.abspath(__file__))
INSTALL_PACKAGES = open(path.join(DIR, 'requirements.txt')).read().splitlines()

setup(
    name='scikest',
    packages=['scikest'],
    description="Training time estimator for sklearn algos",
    install_requires=INSTALL_PACKAGES,
    version='0.0.1',
    url='http://github.com/nathan-toubiana/scikest',
    author='Gabriel Lerner & Nathan Toubiana',
    author_email='toubiana.nathan@gmail.com',
    keywords=['machine-learning', 'sklearn', 'training-time'],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-sugar'
    ],
    package_data={
        # include json and pkl files
        '': ['*.json', 'models/*.pkl', 'models/*.json'],
    },
    include_package_data=True,
    python_requires='>=3'
)
