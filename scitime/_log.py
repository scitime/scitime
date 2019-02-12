import time
import logging

import warnings

warnings.simplefilter("ignore")


class LogMixin(object):
    @property
    def logger(self):
        name = '.'.join([self.__module__, self.__class__.__name__])
        FORMAT = '%(name)s:%(levelname)s:%(message)s'
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

        log.logger.info(f'''{method.__qualname__} took
        {round(te - ts, 3)}s seconds''')

        return result

    timed.__name__ = method.__name__
    timed.__doc__ = method.__doc__

    return timed
