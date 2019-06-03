# # ---------------- paths ---------------- # #

# Data ----------------
DATA_DIR = 'data'
METADATA_PATH = '{}/metadata.json'.format(DATA_DIR)
FULL_DATA_PATH = '{}/PSPOption_RAW_Data.pkl'.format(DATA_DIR)
TRAIN_PATH = '{}/train_data.pkl'.format(DATA_DIR)
VALID_PATH = '{}/valid_data.pkl'.format(DATA_DIR)
TEST_PATH = '{}/test_data.pkl'.format(DATA_DIR)
TRAIN_PATH_MISSING = '{}/train_data_missing.pkl'.format(DATA_DIR)
TRAIN_PATH_AUGMENTED = '{}/train_data_augmented.pkl'.format(DATA_DIR)

# Models ----------------
TRAINED_MODELS_DIR = 'trained_models'
NUMERICAL_PREPROCESSOR_SAVE_PATH = '{}/numerical_preprocessor.pickle'.format(TRAINED_MODELS_DIR)
CATEGORICAL_PREPROCESSOR_SAVE_PATH = '{}/categorical_preprocessor.pickle'.format(TRAINED_MODELS_DIR)

# Results ----------------
RESULTS_DIR = 'results'

# # ---------------- Misc ---------------- # #
SEED = 1

