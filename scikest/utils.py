from __future__ import print_function
import os

import json

from functools import wraps
import errno
import signal
import threading
import multiprocessing


import sys
import threading
from time import sleep
try:
    import thread
except ImportError:
    import _thread as thread

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
    return json.load(open(get_path('config.json')))[key]


class TimeoutError(Exception):
    pass


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    #print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    thread.interrupt_main() # raises KeyboardInterrupt

def timeout(s):
    '''
    use as decorator to exit process if 
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer
