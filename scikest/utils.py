import logging
import csv
import json
from os import path
import time

class LogMixin(object):
    @property
    def logger(self):
        name = '.'.join([self.__module__, self.__class__.__name__])
        FORMAT = '%(levelname)s:%(message)s'
        logging.basicConfig(format=FORMAT, level=logging.DEBUG)
        logger = logging.getLogger(name)
        return logger

def timeit(method):
    """takes method and wraps it in a timer"""
    log = LogMixin()

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        log.logger.info(f'{method.__qualname__} took {round(te - ts, 3)}s seconds')
        return result

    return timed

def get_path(file):
    """
    returns current path
    """
    return path.join(path.dirname(path.abspath(__file__)), file)

def config(key):
    """
    loads json data

    :param key: specific key to load
    :return: dictionary
    """
    return json.load(open(get_path('config.json')))[key]