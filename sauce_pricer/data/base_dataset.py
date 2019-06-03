from abc import abstractmethod
from torch.utils.data import Dataset
import pandas as pd

from sauce_pricer.utils import file_utils


class BaseOptionDataset(Dataset):
    """
    Base abstract class for the option data

    Arguments::

        path: str
            Path to the pickle dataset.

        numerical_features: list of string
            List of the names of the numerical features

        categorical_features: list of string
            List of the names of the categorical features

        output_features: str
            Name of the output feature

        transform_numerical: callable or None, default: None
            Processing to be applied to the numerical features.

        transform_categorical: callable or None, default: None
            Processing to be applied to the categorical features.
    """
    def __init__(self, path, numerical_features=None, categorical_features=None, output_features=None, transform_numerical=None, transform_categorical=None):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.output_features = output_features
        self.transform_numerical = transform_numerical
        self.transform_categorical = transform_categorical
        self.all_features = self.get_all_features(numerical_features, categorical_features, output_features)
        self.raw_data = file_utils.pickle2dataframe(path)
        self.data = self.load_data(path, transform_numerical, transform_categorical)

    def get_all_features(self, numerical_features, categorical_features, output_features):
        all_features = []
        if categorical_features:
            all_features += categorical_features
        if numerical_features:
            all_features += numerical_features
        if output_features:
            all_features += output_features
        return all_features

    def load_data(self, path, transform_numerical, transform_categorical):
        raw_data = file_utils.pickle2dataframe(path)
        data = raw_data[self.all_features]
        if transform_numerical:
            data = transform_numerical(raw_data)
        if transform_categorical:
            data = transform_categorical(raw_data)
        return data

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def merge(self, new_data, raw=False, inverse_transform=False):
        """
        Merge `new_data` to the data

        Arguments::

            data: numpy.array
                Numpy array of dimension `n x d` where `d`
                is the number of features in the data
            raw: bool, default: False
                `False` indicated `data` is transformed (preprocessed)
                using `self.transform`, `True` indicates the data is in the original format
            inverse_transform: bool, default: False
                If `True`, the data will be untransformed after the merge.
        """
        new_data = pd.DataFrame(new_data, columns=list(self.data))
        if raw:
            return self.raw_data.copy().append(new_data)
        augmented = self.data.copy().append(new_data)
        if inverse_transform:
            if self.transform_numerical:
                augmented = self.transform_numerical.inverse_transform(augmented)
            if self.transform_categorical:
                augmented = self.transform_categorical.inverse_transform(augmented)
        return augmented
