import unittest

from sauce_pricer.globals import TRAIN_PATH, VALID_PATH, TEST_PATH, FULL_DATA_PATH
from sauce_pricer.data.preprocessing import get_categorical_processor
from sauce_pricer.utils import file_utils


CATEGORICAL_INPUT_FEATURES = [
    "BloombergUndlTicker",
    "OptionStyle",
    "ExerciseStyle",
    "StrikePriceCurrencyCode"
]


class TestCategoricalPreprocessing(unittest.TestCase):

    def test_label_encoder(self):
        """
        Make sure the preprocessing works for all data set
        """

        categorical_processor = get_categorical_processor(
            FULL_DATA_PATH,
            CATEGORICAL_INPUT_FEATURES
        )

        train_data_df = file_utils.pickle2dataframe(TRAIN_PATH)[CATEGORICAL_INPUT_FEATURES]
        valid_data_df = file_utils.pickle2dataframe(VALID_PATH)[CATEGORICAL_INPUT_FEATURES]
        test_data_df = file_utils.pickle2dataframe(TEST_PATH)[CATEGORICAL_INPUT_FEATURES]

        categorical_processor.transform(train_data_df)
        categorical_processor.transform(valid_data_df)
        categorical_processor.transform(test_data_df)