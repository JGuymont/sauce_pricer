from typing import NewType, List, Tuple
from torch.utils.data import DataLoader
from pandas import DataFrame

from .option_dataset import DatasetLabeled, NumericalDatasetUnlabeled, NumericalDatasetMissing
from .preprocessors import NumericalPreprocessor, CategoricalPreprocessor
from sauce_pricer.constants import SUPERVISED, UNSUPERVISED
from sauce_pricer.utils import file_utils


Dataset = NewType('Dataset', DatasetLabeled or NumericalDatasetUnlabeled or NumericalDatasetMissing)
Preprocessor = NewType('Preprocessor', NumericalPreprocessor or CategoricalPreprocessor)


def get_numerical_processor(
        data_path: str,
        features: List[str],
        scale: Tuple[int, int],
        apply_log: bool,
        save_path: str = None
) -> Preprocessor:
    """Load a training set saved as a pickle and train a NumericalPreprocessor

    :param data_path: Path to the data on which the preprocessor will be fitted
    :param features: List of numerical features to preprocess
    :param scale: See `preprocessors.NumericalPreprocessor` docstring
    :param apply_log: See `preprocessors.NumericalPreprocessor` docstring
    :param save_path: Path where to save the parameter of the preprocessor
    :return: fitted numerical preprocessor
    """
    train_data = file_utils.pickle2dataframe(data_path)
    train_data = train_data[features]
    normalizer = NumericalPreprocessor(scale, apply_log)
    normalizer.fit(train_data)
    if save_path:
        normalizer.save(save_path)
    return normalizer


def get_categorical_processor(data_path: str, features: List[str], save_path: str = None) -> Preprocessor:
    """Load a data set saved as a pickle and fit a CategoricalPreprocessor

    :param data_path: Path to the data on which the preprocessor will be fitted
    :param features: List of categorical features to preprocess
    :param save_path: Path where to save the parameter of the preprocessor
    :return: fitted categorical preprocessor
    """
    train_data: DataFrame = file_utils.pickle2dataframe(data_path)
    train_data = train_data[features].dropna()
    encoder = CategoricalPreprocessor()
    encoder.fit(train_data)
    if save_path:
        encoder.save(save_path)
    return encoder


def get_dataset(task_type: str) -> Dataset:
    """Return the right dataset depending on the features and the task

    :param task_type: `'unsupervised'` or `'supervised'`
    :return: A Dataset type class
    """
    if task_type == SUPERVISED:
        return DatasetLabeled
    elif task_type == UNSUPERVISED:
        return NumericalDatasetUnlabeled
    else:
        raise ValueError('Unsupported task type: {}'.format(task_type))


def preprocess_for_training(
        train_path: str,
        valid_path: str,
        data_path: str,
        batch_size_train: int,
        batch_size_valid: int,
        numerical_input_features: List[str] = None,
        categorical_input_features: List[str] = None,
        output_features: List[str] = None,
        scale: Tuple[int, int] = None,
        apply_log: bool = False,
        numerical_preprocessor_save_path: str = None,
        categorical_preprocessor_save_path: str = None
) -> Tuple[DataLoader, DataLoader]:
    """Return train loader and valid loader for the training of a model.

    :param train_path: Path to the pickled training data
    :param valid_path: Path to the pickled validation data
    :param data_path: Path to the complete data. The complete data is required to know how many unique labels there
        is in each categorical features.
    :param batch_size_train: Batch size for mini-batch training
    :param batch_size_valid: Batch size for validation. To large could cause memory issue.
    :param numerical_input_features: List of numerical input features
    :param categorical_input_features: List of categorical input features
    :param output_features: List of output feature
    :param apply_log: See docstring of NumericalPreprocessor
    :param scale: See docstring of NumericalPreprocessor
    :param numerical_preprocessor_save_path: Path where to save the numerical preprocessor
    :param categorical_preprocessor_save_path: Path where to save the categorical preprocessor
    :return: train and valid data loader
    """

    task_type = SUPERVISED if output_features else UNSUPERVISED

    dataset = get_dataset(task_type)

    # initialize processors
    numerical_processor = get_numerical_processor(
        train_path,
        numerical_input_features,
        scale,
        apply_log,
        numerical_preprocessor_save_path
    )

    categorical_processor = get_categorical_processor(
        data_path,
        categorical_input_features,
        categorical_preprocessor_save_path
    ) if categorical_input_features else None

    # build train and valid dataset
    train_data = dataset(
        path=train_path,
        numerical_features=numerical_input_features,
        categorical_features=categorical_input_features,
        output_features=output_features,
        transform_numerical=numerical_processor,
        transform_categorical=categorical_processor
    )

    valid_data = dataset(
        path=valid_path,
        numerical_features=numerical_input_features,
        categorical_features=categorical_input_features,
        output_features=output_features,
        transform_numerical=numerical_processor,
        transform_categorical=categorical_processor
    )

    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size_valid)

    return train_loader, valid_loader


def preprocess_for_inference(
        data_path: str,
        numerical_input_features: List[str],
        categorical_input_features: List[str] = None,
        output_features: List[str] = None,
        numerical_preprocessor_path: str = None,
        categorical_preprocessor_path: str = None,
        batch_size_test: int = 1000
) -> DataLoader:
    """Preprocess the data for inference.

    :param data_path: Path to the pickled data set
    :param numerical_input_features: list of numerical input features
    :param categorical_input_features: list of categorical input features
    :param output_features: list of output feature
    :param numerical_preprocessor_path: Path to the parameters of the numerical preprocessor.
    :param categorical_preprocessor_path: Path to the parameters of the numerical preprocessor. Should be save
        during training.
    :param batch_size_test: batch size of prediction. Too large will create memory issue
    :return: test loader
    """
    if 'missing' in data_path:
        dataset = NumericalDatasetMissing
    else:
        task_type = SUPERVISED if output_features else UNSUPERVISED
        dataset = get_dataset(task_type)

    numerical_preprocessor = NumericalPreprocessor()
    numerical_preprocessor.load(numerical_preprocessor_path)

    categorical_preprocessor = None
    if categorical_input_features and categorical_preprocessor_path:
        categorical_preprocessor = CategoricalPreprocessor()
        categorical_preprocessor.load(categorical_preprocessor_path)

    test_data = dataset(
        path=data_path,
        numerical_features=numerical_input_features,
        categorical_features=categorical_input_features,
        output_features=output_features,
        transform_numerical=numerical_preprocessor,
        transform_categorical=categorical_preprocessor
    )

    test_loader = DataLoader(test_data, batch_size=batch_size_test)

    return test_loader
