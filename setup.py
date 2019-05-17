from setuptools import setup
from os import path

DIR = path.dirname(path.abspath(__file__))
INSTALL_PACKAGES = open(path.join(DIR, 'requirements.txt')).read().splitlines()

with open(path.join(DIR, 'README.md')) as f:
    README = f.read()

# get __version__ from _version.py
ver_file = path.join('scitime', '_version.py')
with open(ver_file) as f:
    exec(f.read())

VERSION = __version__

setup(
    name='scitime',
    packages=['scitime'],
    description="Training time estimator for scikit-learn algorithms",
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=INSTALL_PACKAGES,
    version=VERSION,
    url='http://github.com/nathan-toubiana/scitime',
    author='Gabriel Lerner & Nathan Toubiana',
    author_email='toubiana.nathan@gmail.com',
    keywords=['machine-learning', 'scikit-learn', 'training-time'],
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
