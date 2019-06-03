from typing import List
import argparse

from sauce_pricer.utils import file_utils


def cli(sys_argv: List[str]):
    """Command line interface to merge two dataset

    :param sys_argv: list of command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_dataset_1', type=str,
                        help='Path to a pickled dataset')

    parser.add_argument('path_to_dataset_2', type=str,
                        help='Path a pickled dataset')

    parser.add_argument('--out_path', type=str,
                        help='Path to save the merged dataset')

    args = parser.parse_args(sys_argv)

    dataset1 = file_utils.pickle2dataframe(args.path_to_dataset_1)
    dataset2 = file_utils.pickle2dataframe(args.path_to_dataset_2)
    merged_dataset = dataset1.append(dataset2)
    file_utils.dataframe2pickle(merged_dataset, args.out_path)
