from typing import Dict, Tuple
import time
import abc
from abc import abstractmethod
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from sauce_pricer.utils.time_utils import get_time
from sauce_pricer.models.utils import Writer


class BaseModel(nn.Module, abc.ABC):

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, inputs):
        pass

    def fit(
            self,
            train_loader: DataLoader,
            valid_loader: DataLoader = None,
            writer: Writer = None
    ) -> Tuple[Dict[str, Tensor], float]:
        """Train the neural network

        :param train_loader: Torch data loader containing training data
        :param valid_loader: Torch data loader containing validation data
        :param writer: Object Writer to store information about the loss (see class Writer in .models.utils)
        :return: None
        """
        start_time: float = time.time()
        best_valid_loss: float = float('Inf')
        best_model_state = self.state_dict().copy()

        for epoch in range(self.n_epochs):
            self._train_iteration(train_loader)
            train_loss = self.evaluate(self, train_loader, self.device)
            valid_loss = self.evaluate(self, valid_loader, self.device)

            if writer:
                writer.add('train_loss', train_loss)
                writer.add('valid_loss', valid_loss)

            # Early stopping: if validation set, save the model if it has improved
            if valid_loader is not None:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_model_state.update(self.state_dict().copy())

            epoch_time = get_time(start_time, time.time())
            print('  epoch: {:02} | Train mse: {:.3f} | Valid mse: {:.3f} | time: {}'.format(
                epoch + 1, train_loss,
                valid_loss,
                epoch_time))

        return best_model_state, best_valid_loss

    def save(self, model_class: str, best_parameters: dict, best_model_loss: float, hyperparameters: dict, path: str):
        """
        Save model parameters under `path`
        """
        checkpoint = {
            "model_class": model_class,
            "hyperparameters": hyperparameters,
            "best_model_state_dict": best_parameters,
            "best_model_loss": best_model_loss,
            'last_model_state_dict': self.state_dict(),
            'last_optim_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, weights: Dict[str, Tensor]):
        self.load_state_dict(weights)
