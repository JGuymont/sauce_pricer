import os
import pickle
import pandas


def pickle2dataframe(path: str):
    """
    Load a pickle and convert it to a `pandas.DataFrame`

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(path, 'rb') as f:
        unpickled = pickle.load(f)
    dataframe = pandas.DataFrame(unpickled)
    return dataframe


def dataframe2pickle(dataframe, path):
    with open(path, 'wb') as f:
        pickle.dump(dataframe, f, pickle.HIGHEST_PROTOCOL)


def save_to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as f:
        unpickled = pickle.load(f)
    return unpickled


def get_model_path(args):
    config_file = os.path.basename(args.config)
    model_name = os.path.splitext(config_file)[0]
    model_file = '{}.pt'.format(model_name)
    model_path = os.path.join(args.model_dir, model_file)
    return model_path
