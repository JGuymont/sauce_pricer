"""
Utility functions use by the models of the module .models
"""
from typing import Dict, List
import pickle
import logging
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from option_pricer.constants import CUDA, CPU


def load_model_state(path: str, device: torch.device):
    """
    Restore the best model parameters for inference
    """
    print(' [predict] Restoring parameters located at: `{path}`'.format(path=path))
    map_location = {CUDA: CPU} if device == CPU else None
    checkpoint = torch.load(path, map_location)
    print(' [predict] Done')
    return checkpoint


def evaluate_regression_mse(model, data_loader: DataLoader, device):
    """
    Compute the mean square error loss of a regression model
    over a data loader
    """
    loss: float = 0.                           # loss that will be return at the end
    data_size: int = len(data_loader.dataset)  # number of examples in the data

    # Put model in evaluation mode with `self.eval()`:
    # some feature are only meant for training (e.g. dropout).
    model.eval()

    # `torch.no_grad()` tells Pytorch not to keep information
    # related to the gradient, We don't need to keep gradient in memory
    # since we aren't optimizing
    with torch.no_grad():

        for batch_input, batch_target in data_loader:

            batch_size = batch_target.shape[0]

            # map inputs and targets to `self.device` (GPU if available)
            batch_input = tuple(x.to(device) for x in batch_input if isinstance(x, torch.Tensor))
            batch_target = batch_target.to(device)

            # compute prediction
            batch_output = model(batch_input)

            # multiply by batch_size because loss function
            # compute the mean over the batch. We want the mean
            # over the dataset so we divide at the end
            loss += F.mse_loss(batch_target, batch_output).cpu().numpy() * batch_size

    # compute the mean loss over the dataset
    loss /= data_size

    return loss


def evaluate_reconstruction_mse(model, data_loader: DataLoader, device):
    """
    Compute the reconstruction loss of an autoencoder model
    over the all the batch of a data loader
    """
    loss: float = 0.  # loss that will be return at the end
    data_size: int = len(data_loader.dataset)  # number of examples in the data

    # Put model in evaluation mode with `self.eval()`:
    # some feature are only meant for training (e.g. dropout).
    model.eval()

    # `torch.no_grad()` tells Pytorch not to keep information
    # related to the gradient, We don't need to keep gradient in memory
    # since we aren't optimizing
    with torch.no_grad():

        for batch_input in data_loader:

            batch_size = batch_input.shape[0]

            batch_input = batch_input.to(device)
            batch_recon = model.forward(batch_input)

            loss += F.mse_loss(batch_input, batch_recon).cpu().numpy() * batch_size

    return loss / data_size


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


class Writer:
    def __init__(self):
        self._data: Dict[str, List[float]] = {}

    def __getitem__(self, item: str) -> List[float]:
        return self._data[item]

    def add(self, name: str, value: float):
        try:
            self._data[name].append(value)
        except KeyError:
            self._data[name] = [value]

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self._data, f, pickle.HIGHEST_PROTOCOL)

