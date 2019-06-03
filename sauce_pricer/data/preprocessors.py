"""
TODO:
    - improve application of log in NumericalPreprocessor: maybe test lognormality?
"""
import numpy as np
from pandas import DataFrame
from typing import List, Dict, NewType
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from sauce_pricer.utils import file_utils


EPS = 0.000001

SklearnTransformer = NewType('SklearnTransformer', MinMaxScaler or PowerTransformer)


class NumericalPreprocessor:
    """Preprocessor for numerical features

    :param scale: Tuple (min, max). If provided, the features will be scaled between the first element
        of the tuple (`min`) and the second (`max`).
    :param apply_log: If `True`, log is applied to all feature that are strictly positive (> 0). This is to facilitate
        the normalization of positive features which tend to be lognormal-ish.
    """

    def __init__(self, scale: tuple = None, apply_log: bool = False):
        self._apply_log = apply_log
        self._scale = scale
        self._transformers: Dict[str, Dict[str, SklearnTransformer]] = {}  # dictionary to store all the preprocessors
        self._registered_features: List[str] = []                          # list of features that have been transformed
        self._log_features: List[str] = []                                 # list of features on which log is applied

    def to_log(self, data: np.array) -> bool:
        return min(data) > 0 and self._apply_log

    def fit(self, dataframe: DataFrame) -> None:
        """Estimate and save the optimal parameter of PowerTransformer for each feature.
        Also store values require to scale and unscale the features if scaling needs to be apply.

        :param dataframe: dataframe containing only the features that needs to be normalize
        """
        for feature in list(dataframe):
            self._registered_features.append(feature)
            data = dataframe[feature].to_numpy().reshape(-1, 1)  # load feature into a numpy array

            self._transformers[feature] = {}                     # initialized storage for the feature transformers

            if self.to_log(data):                                # log features that should be log
                data = np.log(data)
                self._log_features.append(feature)

            power_transformer = PowerTransformer()               # log features that should be log
            power_transformer.fit(data)

            self._transformers[feature]['normalizer'] = power_transformer

            if self._scale:
                scaler = MinMaxScaler(feature_range=self._scale)
                scaler.fit(power_transformer.transform(data))
                self._transformers[feature]['scaler'] = scaler

    def _transform_feature(self, dataframe: DataFrame, feature: str) -> np.array:
        if feature not in self._registered_features:
            return dataframe[feature]

        scaler: MinMaxScaler = self._transformers[feature]['scaler'] if self._scale else None
        normalizer: PowerTransformer = self._transformers[feature]['normalizer']

        values = dataframe[feature].to_numpy().reshape(-1, 1)
        if feature in self._log_features:                            # log features if log in preprocessing
            index_to_log = [e > 0 if not np.isnan(e)                 # log only positive non NaN element
                            else False for e in values]
            index_to_log = np.array(index_to_log, dtype=bool)        # map list of index to array
            values[index_to_log] = np.log(values[index_to_log])

        values = normalizer.transform(values)                         # normalize feature
        values = scaler.transform(values) if scaler else None         # scale features id scale provided
        return values

    def _inverse_transform_feature(self, dataframe: DataFrame, feature: str) -> np.array:
        if feature not in self._registered_features:
            return dataframe[feature]

        scaler: MinMaxScaler = self._transformers[feature]['scaler'] if self._scale else None
        normalizer: PowerTransformer = self._transformers[feature]['normalizer']

        values = dataframe[feature].to_numpy().reshape(-1, 1)
        values = scaler.inverse_transform(values) if scaler else values       # unscale feature scaled in preprocessing
        values = normalizer.inverse_transform(values)                         # un-normalize features
        values = np.exp(values) if feature in self._log_features else values  # exp() if feature log in preprocessing

        return values

    def transform(self, dataframe: DataFrame) -> DataFrame:
        """Apply preprocessing to features seen when fitting.

        :param dataframe: Data to preprocess.
        :return: Preprocessed dataframe
        """
        for feature in list(dataframe):
            dataframe[feature] = self._transform_feature(dataframe, feature)
        return dataframe

    def inverse_transform(self, dataframe: DataFrame) -> DataFrame:
        """Inverse the preprocessing. Only features seen when fitting are processed.

        :param dataframe: Dataframe to process.
        :return: Dataframe with non normalized numerical features.
        """
        for feature in list(dataframe):
            dataframe[feature] = self._inverse_transform_feature(dataframe, feature)
        return dataframe

    def __call__(self, dataframe: DataFrame) -> DataFrame:
        return self.transform(dataframe)

    def save(self, path: str) -> None:
        preprocessing_parameters = {
            "scale": self._scale,
            "apply_log": self._apply_log,
            "transformers": self._transformers,
            "registered_features": self._registered_features,
            "log_features": self._log_features
        }
        file_utils.save_to_pickle(preprocessing_parameters, path)

    def load(self, path: str) -> None:
        preprocessing_parameters = file_utils.load_pickle(path)
        self._scale = preprocessing_parameters['scale']
        self._apply_log = preprocessing_parameters['apply_log']
        self._transformers = preprocessing_parameters['transformers']
        self._registered_features = preprocessing_parameters['registered_features']
        self._log_features = preprocessing_parameters['log_features']


class CategoricalPreprocessor:
    """
    Preprocessing function for categorical encoding

    Arguments
        categorical_features: list or None, default: None
            List of categorical features

    Returns:
        pandas.DataFrame: the preprocessed dataframe
    """

    def __init__(self):
        self._label_encoders: Dict[str, LabelEncoder()] = {}  # dictionary to store all the preprocessors
        self._registered_features: List[str] = []             # list of features that have been transformed

    def fit(self, dataframe: DataFrame) -> None:
        for feature in list(dataframe):
            self._registered_features.append(feature)
            data = dataframe[feature]                         # extract current feature
            label_encoder = LabelEncoder()
            label_encoder.fit(data)
            self._label_encoders[feature] = label_encoder     # store feature's label encoder

    def transform(self, dataframe: DataFrame) -> DataFrame:
        """Apply preprocessing to features seen when fitting.

        :param dataframe: Data to preprocess.
        :return: Preprocessed dataframe
        """
        transformed_df = dataframe.copy()                         # make a copy of the dataframe prevents pandas error
        for feature in self._registered_features:
            label_encoder = self._label_encoders[feature]         # get encoder for current feature
            values = dataframe[feature]
            data_classes = set(values)

            for label in data_classes:
                assert label in label_encoder.classes_, "label `{}` not seen during training for feature `{}`"\
                                                        .format(label, feature)

            n_labels_data = len(data_classes)                     # number of classes in current data
            n_labels = len(label_encoder.classes_)                # number of classes seen during training

            assert n_labels >= n_labels_data, "Number of classes for feature `{}`: {} bigger then number of classes " \
                                              "seen during training: {}".format(feature, n_labels_data, n_labels)

            labels = label_encoder.transform(values)              # this is a list of labels 1,...,n_category - 1
            transformed_df[feature] = labels                      # replace feature values with encodings

        return transformed_df

    def inverse_transform(self, dataframe: DataFrame):
        inverse_transformed_df = dataframe.copy()          # make a copy of the dataframe prevents pandas error
        for feature in self._registered_features:
            label_encoder = self._label_encoders[feature]  # get encoder for current feature
            labels = dataframe[feature]
            classes = label_encoder.transform(labels)      # map labels to original classes
            inverse_transformed_df[feature] = classes      # replace feature values with original classes
        return inverse_transformed_df

    def __call__(self, dataframe):
        return self.transform(dataframe)

    def save(self, path: str) -> None:
        preprocessing_parameters = {
            "label_encoders": self._label_encoders,
            "registered_features": self._registered_features,
        }
        file_utils.save_to_pickle(preprocessing_parameters, path)

    def load(self, path):
        preprocessing_parameters = file_utils.load_pickle(path)
        self._label_encoders = preprocessing_parameters['label_encoders']
        self._registered_features = preprocessing_parameters['registered_features']
