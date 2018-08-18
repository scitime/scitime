import logging

def Logging(__name__):
    FORMAT = '%(asctime)-15s - %(name)s - %(message)s'
    logging.basicConfig(format=FORMAT)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    return log

