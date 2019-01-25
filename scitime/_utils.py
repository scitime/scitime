import os
import json

from threading import Thread
import functools

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


def timeout(seconds_before_timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [TimeoutError('artificial timeout error')]

            def new_func():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=new_func)
            t.daemon = True
            try:
                t.start()
                t.join(seconds_before_timeout)
            except Exception as e:
                raise e
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco
