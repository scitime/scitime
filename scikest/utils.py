import logging
import json
import time
from functools import wraps
import errno
import signal
import os
import warnings

warnings.simplefilter("ignore")


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
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file)


def config(key):
    """
    loads json data

    :param key: specific key to load
    :return: dictionary
    """
    return json.load(open(get_path('config.json')))[key]


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """checks if a function does not throw an instant error without actually running the entire function"""

    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator
