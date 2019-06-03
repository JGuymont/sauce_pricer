from typing import List, Tuple
import argparse
import json
import torch

from sauce_pricer.data.preprocessing import preprocess_for_training
from sauce_pricer.models.modules.embedding import Embedding
from sauce_pricer.globals import (
    TRAIN_PATH,
    VALID_PATH,
    FULL_DATA_PATH,
    RESULTS_DIR,
    TRAINED_MODELS_DIR,
    NUMERICAL_PREPROCESSOR_SAVE_PATH,
    CATEGORICAL_PREPROCESSOR_SAVE_PATH
)
from sauce_pricer.constants import CPU, CUDA
from sauce_pricer.models.mlp_regressor import MLPRegressor
from sauce_pricer.models.utils import Writer
from sauce_pricer.models.autoencoder import Autoencoder


# TODO: this should probably not be here
REGISTERED_MODELS = {
    "MLPRegressor": MLPRegressor,
    "Autoencoder": Autoencoder
}


def cli(sys_argv: List[str]):
    """Command line interface to train the models

    :param sys_argv: list of command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=TRAIN_PATH,
                        help="Path to the pickled training data set.")

    parser.add_argument('--valid_path', type=str, default=VALID_PATH,
                        help='Path to the pickled validation data set.')

    parser.add_argument('--data_path', type=str, default=FULL_DATA_PATH,
                        help='Path to the full pickled data set. This data will be use only to list all the possible '
                             'classes of the categorical features.')

    parser.add_argument('--model_save_dir', type=str, default=TRAINED_MODELS_DIR,
                        help='Path where to save the trained parameters after training.')

    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR,
                        help='Path of the directory where to save results (e.g. the train/valid losses).')

    parser.add_argument('--numerical_preprocessor_save_path', type=str, default=NUMERICAL_PREPROCESSOR_SAVE_PATH,
                        help='Path where to save the parameters of the numerical preprocessor. The same parameters are '
                             'going to be used when making inference.')

    parser.add_argument('--categorical_preprocessor_save_path', type=str, default=CATEGORICAL_PREPROCESSOR_SAVE_PATH,
                        help='Path where to save the parameters of categorical preprocessor. The same parameters are '
                             'going to be used when making inference.')

    parser.add_argument('--model_definition', type=str,
                        help='Path to model definition. Model definition is a json file containing all the '
                             'hyperparameters of the model.')

    args = parser.parse_args(sys_argv)

    # # ---------- parse config file ---------- # #
    config: dict = json.load(open(args.model_definition, 'r'))
    model_class: str = config['model_class']
    model_name: str = config['model_name']
    batch_size_train: int = config['data']['batch_size_train']
    batch_size_valid: int = config['data']['batch_size_valid']
    numerical_input_features: List[str] = config['data']['numerical_input_features']
    categorical_input_features: List[str] = config['data']['categorical_input_features']
    output_features: List[str] = config['data']['output_features']
    scale: Tuple[int, int] = config['processing']['scale']
    apply_log: bool = config['processing']['apply_log']

    # # ---------- create data loaders ---------- # #
    train_loader, valid_loader = preprocess_for_training(
        args.train_path,
        args.valid_path,
        args.data_path,
        batch_size_train,
        batch_size_valid,
        numerical_input_features,
        categorical_input_features,
        output_features,
        scale,
        apply_log,
        args.numerical_preprocessor_save_path,
        args.categorical_preprocessor_save_path
    )

    # # ---------- hyperparameters ---------- # #
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)  # check if GPU is available
    hyperparameters = {}  # create a dictionary containing all the hyperparameters of the model
    hyperparameters.update(config['model'])
    hyperparameters.update(config['training'])
    hyperparameters['device'] = device

    # # ---------- Embedding ---------- # #

    # If categorical features are provided in config (see ./config/regressor_mix_features.json for an example of how
    # to do that), then a list of tuples [(n_category, n_embeddings), ...] need to be provided as hyperparameter.
    # (n_category, n_embeddings) means that the the categorical feature can take `n_category` different values and
    # it should be encoded using `n_embeddings`; usually `n_embeddings` is smaller then `n_category`. The list of tuples
    # [(n_category, n_embeddings), ...] is generated by Embedding.get_embedding_dimensions() (see
    # models.modules.embedding). For more details about how embeddings works, please see
    # https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html.
    if categorical_input_features:
        train_data_df = train_loader.dataset.data  # extract original dataframe from `train_loader`
        embedding_dims = Embedding.get_embedding_dimensions(train_data_df, categorical_input_features)
        hyperparameters['embedding_dims'] = embedding_dims  # add embedding_dims to hyperparameters

    # # ---------- Initialize and train model ---------- # #
    model = REGISTERED_MODELS[model_class](**hyperparameters).to(device)

    # initialize writer object: basically a dictionary to store loss information
    writer = Writer()

    # train the model
    print(' [train] Training model `{}`...'.format(model_class))
    best_model_state, best_valid_loss = model.fit(train_loader, valid_loader, writer=writer)
    print(' [train] Done')

    # Save model: by default, the weights are going to be save in `trained_models/model_state_<model_name>.pickle`
    model_path = '{}/model_state_{}.pickle'.format(args.model_save_dir, model_name)
    print(' [train] Saving trained model at: `{}`'.format(model_path))
    model.save(model_class, best_model_state, best_valid_loss, hyperparameters, model_path)
    print(' [train] Done')

    # save writer
    writer_path = '{}/writer_{}.pickle'.format(args.results_dir, model_name)
    print(' [train] Saving `writer` at: `{}`'.format(writer_path))
    writer.save(writer_path)
    print(' [train] Done')
