from typing import List, Dict
import argparse
import numpy as np

from sauce_pricer.globals import RESULTS_DIR
from sauce_pricer.utils import file_utils, plot_utils


def cli(sys_argv: List[str]):
    """Command line interface to train the models

    :param sys_argv: list of command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mse_loss', action='store_true')

    parser.add_argument('--writer_path', type=str, default=None,
                        help='path to the pickled writer. Use this option if visualizing losses.')

    parser.add_argument('--price_predictions', action='store_true')

    parser.add_argument('--target_data', type=str, default=None,
                        help="path to the pickled target data set. Use this option if visualizing predictions")

    parser.add_argument('--target_feature', type=str, default=None,
                        help="Name of the target feature. Use this option if visualizing predictions")

    parser.add_argument('--pred', type=str, default=None,
                        help="path to the pickled predictions. Use this option if visualizing predictions")

    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR,
                        help='Path to the directory where to save results')

    parser.add_argument('--output_filename', type=str,
                        help='Name of the figure')

    args = parser.parse_args(sys_argv)

    save_path = '{}/figure_{}'.format(args.output_dir, args.output_filename)

    if args.mse_loss:
        writer: Dict[str, List[float]] = file_utils.load_pickle(args.writer_path)
        print(' [visualize] Saving MSE figure at `{}`'.format(save_path))
        plot_utils.plot_train_valid_rmse_loss(writer['train_loss'], writer['valid_loss'], save_path)
        print(' [visualize] Done')
        return

    if args.price_predictions:
        dataframe = file_utils.pickle2dataframe(args.target_data)
        targets = dataframe[args.target_feature]
        predictions = file_utils.load_pickle(args.pred)

        plot_utils.scatter_plot_predictions(predictions, targets, save_path)
        return

    exit('You need to specify `--mse_loss` or `--price_predictions` in command line arguments')



