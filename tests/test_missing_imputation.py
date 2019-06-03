from typing import List
import unittest
import json
import torch

from sauce_pricer.constants import CUDA, CPU
from sauce_pricer.models.utils import load_model_state
from sauce_pricer.models.imputer import Imputer
from sauce_pricer.data.option_dataset import NumericalDatasetMissing
from sauce_pricer.data.preprocessing import NumericalPreprocessor
from torch.utils.data import DataLoader
from sauce_pricer.utils import file_utils


MISSING_DATA_PATH = 'tests/data/train_data_missing.pkl'
DATA_PATH = 'tests/data/train_data.pkl'
CONFIG_PATH = 'tests/config/missing_imputer.json'
NUMERICAL_PREPROCESSOR_PATH = 'tests/models/numerical_preprocessor.pickle'
AUTOENCODER_PATH = './tests/models/model_state_autoencoder.pickle'

NUMERICAL_INPUT_FEATURES = [
    "StrikePrice",
    "OPT_UNDL_PX",
    "OPT_FINANCE_RT",
    "OPT_DIV_YIELD",
    "OPT_TIME_TO_MAT",
    "Fx",
    "Volatility_PX_MID",
    "Volatility_PX_LAST",
    "Volatility_BB_BST",
    "Volatility_Ivol",
    "PX_VOLUME",
    "PX_VOLUME_1D"
]

OUTPUT_FEATURES = ["PX_LAST"]

DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)


class TestMissingImputer(unittest.TestCase):

    def test_missing_imputation(self):
        config: dict = json.load(open(CONFIG_PATH, 'r'))

        # # ---------- parse model state ---------- # #
        model_state = load_model_state(AUTOENCODER_PATH, DEVICE)

        model_hyperparameters: dict = model_state['hyperparameters']
        model_hyperparameters.update(config['model'])
        model_hyperparameters['device']: torch.device = DEVICE
        model_weights: dict = model_state['best_model_state_dict']

        numerical_preprocessor = NumericalPreprocessor()
        numerical_preprocessor.load(NUMERICAL_PREPROCESSOR_PATH)

        # # ---------- preprocess data for inference ---------- # #
        missing_data = NumericalDatasetMissing(
            path=MISSING_DATA_PATH,
            numerical_features=NUMERICAL_INPUT_FEATURES,
            output_features=OUTPUT_FEATURES,
            transform_numerical=numerical_preprocessor
        )

        missing_data.data = missing_data.data[:100]
        missing_data.raw_data = missing_data.raw_data[:100]

        missing_data_loader = DataLoader(missing_data, batch_size=1)

        # # ---------- initialize model ---------- # #
        model = Imputer(**model_hyperparameters).to(DEVICE)
        model.load(model_weights)

        model.eval()

        with torch.no_grad():

            # test prediction for a single example
            x, y, missings = next(iter(missing_data_loader))

            pred = model.forward(x.to(DEVICE), missings)

            assert 0 not in pred.cpu().numpy()

            completed_dataframe = model.predict(missing_data_loader)

        assert len(completed_dataframe[NUMERICAL_INPUT_FEATURES]) == len(completed_dataframe[NUMERICAL_INPUT_FEATURES].dropna())