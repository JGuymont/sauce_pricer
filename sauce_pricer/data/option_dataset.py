from typing import NewType, Union
import math
import numpy as np

from .base_dataset import BaseOptionDataset


class DatasetLabeled(BaseOptionDataset):
    """
    This data loader is used when the features are both
    categorical and numerical.

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs_num = self.data[self.numerical_features].astype(np.float32).values
        self.inputs_cat = self.data[self.categorical_features].astype(np.int64).values \
            if self.categorical_features else None
        self.targets = self.data[self.output_features].astype(np.float32).values

    def __getitem__(self, idx):
        """
        Return the data corresponding to the line `idx`:
        a tuple (X_numerical, X_categorical, log Y) is returned.
        """
        x_numerical = self.inputs_num[idx]
        x_categorical = self.inputs_cat[idx] if self.categorical_features else []
        y = self.targets[idx]
        return (x_numerical, x_categorical), y


class NumericalDatasetUnlabeled(BaseOptionDataset):
    """
    Abstract class for the unlabeled option data. i.e.
    this dataset is for unsupervised learning, Only support
    numerical features.

    Arguments:
        path: str
            Path to the pickle dataset.

        numerical_features: list of string
            List of the names of the input features

        transform_numerical: callable or None, default: None
            Processing to be applied to the numerical features.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = self.data[self.numerical_features].astype(np.float32).values

    def __getitem__(self, idx):
        """
        Return the data corresponding to the line `idx`:
        since this dataset is for unsupervised learning,
        return only the input.
        """
        return self.X[idx]
    
    def num_input_features(self):
        """
        Return the number of features
        """
        return len(self[0])


class NumericalDatasetMissing(BaseOptionDataset):
    """
    Abstract class for the unlabeled option data
    with missing values.

    Arguments:
        path: str
            Path to the pickle dataset.

        numerical_features: list of string
            List of the names of the input features

        transform_numerical: callable or None, default: None
            Processing to be applied to the numerical features.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inputs = self.data[self.numerical_features].astype(np.float32).values
        self.targets = self.data[self.output_features].astype(np.float32).values

    def __getitem__(self, idx):
        """
        Return the data corresponding to the line `idx`:
        a tuple (X, log Y) is returned.
        """
        x = self.inputs[idx]
        y = self.targets[idx]

        # check if there is missing values: `missings`
        # will contain the list of index of missing values.
        missings = [i for i, v in enumerate(x) if math.isnan(v)]

        x[missings] = 0.               # replace nans with 0s
        missings = np.array(missings)  # map missing to numpy array

        return x, y, missings


Dataset = NewType('Dataset', Union[DatasetLabeled, NumericalDatasetUnlabeled])
