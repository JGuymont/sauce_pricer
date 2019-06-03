from typing import Dict, List
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sauce_pricer.models.autoencoder import Autoencoder
from sauce_pricer.data.option_dataset import Dataset
from sauce_pricer.data.preprocessors import NumericalPreprocessor, CategoricalPreprocessor


class Imputer(nn.Module):
    """
    Impute missing values
    """
    def __init__(self, tol, max_iter, **kwargs):
        super(Imputer, self).__init__()
        self.autoencoder = Autoencoder(**kwargs)
        self.TOL = tol
        self.max_iter = max_iter
        self.device = self.autoencoder.device

    def forward(self, x, missings):
        current_error = np.float('Inf')
        delta_error = np.float('Inf')
        niter = 1

        reconstruction = self.autoencoder(x)

        while delta_error < self.TOL or niter > self.max_iter:
            reconstruction = self.autoencoder(reconstruction).cpu().numpy()
            last_error = current_error
            current_error = F.mse_loss(reconstruction, x)
            delta_error = abs(last_error - current_error)
            niter += 1
        x[:, missings.long()] = reconstruction[:, missings.long()]

        return x

    def load(self, weights: Dict[str, Tensor]):
        self.autoencoder.load_state_dict(weights)

    def predict(self, data_loader: DataLoader):
        recons_list: List[np.array] = []              # list of examples where missing values replaced with predictions
        dataset: Dataset = data_loader.dataset        # original dataset
        self.eval()  # evaluation mode

        with torch.no_grad():

            print(' [Imputer] Reconstructing incomplete examples...')
            for batch_input, batch_target, batch_missing in tqdm(data_loader):

                batch_input = batch_input.to(self.device)

                batch_recon = self.forward(batch_input, batch_missing)  # predict missing values
                batch_recon = batch_recon.cpu().numpy()                 # map tensor to numpy
                batch_recon = np.append(batch_recon, batch_target)      # append inputs to targets
                recons_list.append(batch_recon)
            print(' [Imputer] Done.')

        recons_array = np.array(recons_list)  # transform the list into an array of shape (n_examples, n_features)
        print(' [Imputer] Merging missing values with original dataframe')
        completed_dataframe = self._merge(dataset, recons_array)
        print(' [Imputer] Done')

        return completed_dataframe

    def _merge(self, dataset: Dataset, new_data: np.array):
        """
        Merge `new_data` to the data

        :param dataset: Original dataset to merge predicted missing to
        :param new_data: Numpy array of dimension `n x d` where `d` is the number of features in the data
        """
        data: DataFrame = dataset.raw_data                  # original dataframe

        predicted_features: List[str] = list(dataset.all_features)  # list of features used during predictions

        new_data: DataFrame = pd.DataFrame(new_data, columns=predicted_features)

        transform_numerical: NumericalPreprocessor = dataset.transform_numerical        # Preprocessors attach to the
        transform_categorical: CategoricalPreprocessor = dataset.transform_categorical  # dataset.

        if transform_numerical:
            new_data = transform_numerical.inverse_transform(new_data)
        if transform_categorical:
            new_data = transform_categorical.inverse_transform(new_data)

        for feature in predicted_features:
            data[feature] = new_data[feature].values

        return data
