import logging
import csv

def Logging(__name__):
    """
    a function for logs
    """
    FORMAT = '%(asctime)-15s - %(name)s - %(message)s'
    logging.basicConfig(format=FORMAT)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    return log

def add_data_to_csv(thisInput,thisOutput, rf_parameters_list):
    """
    writes into a csv row by row

    :param thisInput: input
    :param thisOutput: output
    :param rf_parameters_list: algo parameter list
    :return:
    """
    with open(r'result.csv', 'a+') as file:
        columns = ['num_rows'] + ['num_features'] + rf_parameters_list + ['output']
        writer = csv.writer(file)
        thisRow = list(thisInput) + [thisOutput]
        writer.writerows([thisRow])