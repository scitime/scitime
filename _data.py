import argparse
import numpy as np

from scitime._model import Model
from scitime._utils import config

SUPPORTED_META_ALGOS = config('supported_meta_algos')
SUPPORTED_ALGOS = config('supported_algos')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather & Persist Data of model training runtimes')
    parser.add_argument('--drop_rate', required=False, default=0.999,
                        help="""drop rate of number of data generated (from all param combinations taken from _config.json). Default is 0.999""")
    parser.add_argument('--meta_algo', required=False, choices=SUPPORTED_META_ALGOS,
                        help="""meta algo used to fit the meta model (NN or RF) - default is RF""")
    parser.add_argument('--verbose', required=False, default=1,
                        help='verbose mode (0, 1, 2 or 3)')
    parser.add_argument('--algo', required=False, default='RandomForestRegressor', choices=SUPPORTED_ALGOS,
                        help='algo to train data on')
    parser.add_argument('--generate_data', required=False, default=False, action='store_true',
                        help='do you want to generate & write data in a dedicated csv?')
    parser.add_argument('--fit', required=False,
                        help='do you want to fit the model? If so indicate the csv name')
    parser.add_argument('--save', required=False, default=False, action='store_true',
                        help='(only used for model fit) do you want to save / overwrite the meta model from this fit?')

    args = parser.parse_args()

    assert args.generate_data or args.fit

    if args.fit is None:
        if args.meta_algo is not None:
            raise ValueError('No need to specify the meta algo if you only want to generate data')
        if args.save:
            raise ValueError('No need to specify the --save flag if you only want to generate data')

    if args.fit is not None:
        assert args.meta_algo

    if args.drop_rate is None:
        drop_rate = 0.999
    else:
        drop_rate = np.float(args.drop_rate)

    if args.verbose is None:
        verbose = 1
    else:
        verbose = int(args.verbose)

    if args.meta_algo is None:
        meta_algo = 'RF'
    else:
        meta_algo = args.meta_algo

    if args.algo is None:
        algo = 'RandomForestRegressor'
    else:
        algo = args.algo

    m = Model(drop_rate=drop_rate, algo=algo, meta_algo=meta_algo, verbose=verbose)

    if args.generate_data:
        m._generate_data(write_csv=True)

    if args.fit is not None:
        csv_name = args.fit
        m.model_fit(generate_data=False, csv_name=csv_name, save_model=args.save)

