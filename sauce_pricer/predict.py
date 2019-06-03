from typing import List
import os
import argparse
import json
import torch

from sauce_pricer.globals import RESULTS_DIR
from sauce_pricer.constants import CPU, CUDA
from sauce_pricer.globals import (
    TEST_PATH,
    NUMERICAL_PREPROCESSOR_SAVE_PATH,
    CATEGORICAL_PREPROCESSOR_SAVE_PATH
)
from sauce_pricer.models.mlp_regressor import MLPRegressor
from sauce_pricer.models.autoencoder import Autoencoder
from sauce_pricer.models.imputer import Imputer
from sauce_pricer.data.preprocessing import preprocess_for_inference
from sauce_pricer.utils import file_utils
from sauce_pricer.models.utils import load_model_state


# TODO: this should probably not be here
REGISTERED_MODELS = {
    "MLPRegressor": MLPRegressor,
    "Autoencoder": Autoencoder,
    "Imputer": Imputer
}


def cli(sys_argv: List[str]):
    """Command line interface to make prediction

    :param sys_argv: list of command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_definition', type=str,
                        help='Path to json model definition')

    parser.add_argument('--model_state_path', type=str,
                        help='Path where to the trained parameters')

    parser.add_argument('--data_path', type=str, default=TEST_PATH,
                        help='path to the pickled dataframe on which prediction should be made')

    parser.add_argument('--numerical_preprocessor', type=str, default=NUMERICAL_PREPROCESSOR_SAVE_PATH,
                        help='Path of the saved numerical preprocessor')

    parser.add_argument('--categorical_preprocessor', type=str, default=CATEGORICAL_PREPROCESSOR_SAVE_PATH,
                        help='Path to the saved categorical preprocessor')

    parser.add_argument('--output_directory', type=str, default=RESULTS_DIR,
                        help='Path where to save the prediction of the experiment')

    args = parser.parse_args(sys_argv)

    # # ---------- parse config file ---------- # #
    config: dict = json.load(open(args.model_definition, 'r'))

    model_class: str = config['model_class']
    model_name: str = config['model_name']
    numerical_input_features: List[str] = config['data']['numerical_input_features']
    categorical_input_features: List[str] = config['data']['categorical_input_features']
    output_features: List[str] = config['data']['output_features']
    batch_size_test: int = config['data']['batch_size_test']

    device = torch.device(CUDA if torch.cuda.is_available() else CPU)

    # # ---------- parse model state ---------- # #
    model_state = load_model_state(args.model_state_path, device)

    model_hyperparameters: dict = model_state['hyperparameters']
    model_hyperparameters.update(config['model'])
    model_hyperparameters['device']: torch.device = device
    model_weights: dict = model_state['best_model_state_dict']

    # # ---------- initialize model ---------- # #
    model = REGISTERED_MODELS[model_class](**model_hyperparameters).to(device)
    model.load(model_weights)

    # # ---------- preprocess data for inference ---------- # #
    test_loader = preprocess_for_inference(
        args.data_path,
        numerical_input_features,
        categorical_input_features,
        output_features,
        args.numerical_preprocessor,
        args.categorical_preprocessor,
        batch_size_test=batch_size_test
    )

    # # ---------- compute and save predictions ---------- # #
    predictions = model.predict(test_loader)

    # save predictions
    data_file_name = os.path.basename(args.data_path)
    data_file_name = os.path.splitext(data_file_name)[0]  # remove extension
    model_path = '{}/predictions_{}_{}.pickle'.format(args.output_directory, model_name, data_file_name)
    print(' [predict] Saving predictions at: `{}`'.format(model_path))
    file_utils.save_to_pickle(
        predictions,
        path=model_path
    )
    print(' [predict] Done')
