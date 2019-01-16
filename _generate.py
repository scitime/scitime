import argparse

from scikest._model import Model
from scikest._utils import config

SUPPORTED_META_ALGOS = config('supported_meta_algos')
SUPPORTED_ALGOS = config('supported_algos')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather & Persist Data of model training runtimes')
    parser.add_argument('--drop_rate', required=False, default=0.999,
                        help="""drop rate of number of data generated (from all param combinations taken from _config.json). Default is 0.999""")
    parser.add_argument('--meta_algo', required=False, default='RF', choices=SUPPORTED_META_ALGOS,
                        help="""meta algo used to fit the meta model (NN or RF) - default is RF""")
    parser.add_argument('--verbose', required=False, default=1,
                        help='verbose mode (0, 1, 2 or 3)')
    parser.add_argument('--algo', required=False, default='RandomForestRegressor', choices=SUPPORTED_ALGOS,
                        help='algo to train data on')
    parser.add_argument('--persist', required=False, default=False, action='store_true',
                        help='do you want to write data in a dedicated csv?')
    args = parser.parse_args()

    if args.drop_rate is None:
        drop_rate = 0.999
    else:
        drop_rate = args.drop_rate

    if args.verbose is None:
        verbose = 1
    else:
        verbose = args.verbose

    if args.meta_algo is None:
        meta_algo = 'RF'
    else:
        meta_algo = args.meta_algo

    if args.algo is None:
        algo = 'RandomForestRegressor'
    else:
        algo = args.algo

    m = Model(drop_rate=drop_rate, algo=algo, meta_algo=meta_algo, verbose=verbose)
    m._generate_data(write_csv=args.persist)
