import os

import json

from functools import wraps
import errno
import signal

import warnings
warnings.simplefilter("ignore")


def get_path(file):
    """returns current path"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), file)


def config(key):
    """
    loads json data

    :param key: specific key to load
    :return: dictionary
    """
    return json.load(open(get_path('_config.json')))[key]


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
