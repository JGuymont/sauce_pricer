from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from sauce_pricer.constants import CPU
from .base_model import BaseModel
from .modules.embedding import Embedding
from .utils import evaluate_regression_mse


class MLPRegressor(BaseModel):

    def __init__(self,
                 n_numerical_features: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout: float,
                 embedding_dims: List[Tuple[int, int]] = None,
                 embedding_dropout: float = 0,
                 n_epochs: int = 100,
                 lr: float = 0.001,
                 weight_decay: float = 0,
                 device: torch.device = torch.device(CPU)):
        """Multilayer perceptron regressor. Optimization is done with Adam and the
        loss that is minimize is MSE.

        :param n_numerical_features: Number of numerical input features
        :param hidden_dim: Dimension of the hidden layers
        :param output_dim: Dimension of the output
        :param dropout: Percentage of nodes to drop when applying dropout
        :param embedding_dims:
        :param embedding_dropout:
        :param n_epochs: Number of training iterations
        :param lr: Learning rate
        :param weight_decay: Weight decay parameter. Weight decay is a regularization method
        :param device: Device on which computation is done.
        """
        super(MLPRegressor, self).__init__()

        self.n_epochs = n_epochs
        self.device = device

        # initialize embedding layers if embedding_dims is not None
        if embedding_dims:
            self.embedding = Embedding(embedding_dims, embedding_dropout)
            input_dim = n_numerical_features + self.embedding.num_embeddings()
        else:
            input_dim = n_numerical_features

        # Computation of the first hidden layer. Map the
        # input to the first hidden layer:
        self.input2hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # -> Wx + b
            nn.BatchNorm1d(hidden_dim),            # -> apply bach normalization on Wx + b
            nn.ReLU(),                             # -> relu(Wx + b)
            nn.Dropout(p=dropout),                 # -> cancel some of the connection to prevent overfitting
        )

        # Computation of the second hidden layer. Map the
        # first hidden layer to the second hidden layer:
        self.hidden2hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # map the last hidden layer to the output
        self.hidden2output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

        self.optimizer = Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        self.evaluate = evaluate_regression_mse

    def forward(self, inputs: Tuple[torch.Tensor, ...]):

        if len(inputs) == 1:
            inputs = inputs[0]
        else:
            inputs, x_categorical = inputs
            x_embedding = self.embedding(x_categorical.long())
            inputs = torch.cat((inputs, x_embedding), 1)

        hidden01 = self.input2hidden(inputs)
        hidden02 = self.hidden2hidden(hidden01)
        outputs = self.hidden2output(hidden02)
        return torch.exp(outputs).view(-1, 1)  # reshape into (batch_size, 1)

    def _train_iteration(self, train_loader):
        """
        Perform an iteration of gradient descent
        """
        # Set the module in training mode; e.g. activate dropout.
        # see https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train for details
        self.train()

        for batch_input, batch_target in train_loader:

            self.optimizer.zero_grad()                     # make sure all the gradients are equal to zero.

            batch_target = batch_target.to(self.device)    # send data to GPU if available
            batch_input = tuple(x.to(self.device) for x in batch_input if isinstance(x, torch.Tensor))

            batch_output = self.forward(batch_input)       # compute prediction
            loss = F.mse_loss(batch_target, batch_output)  # compute mse loss
            loss.backward()                                # compute derivative of the loss wrt each parameter
            clip_grad_norm_(self.parameters(), max_norm=0.1)
            self.optimizer.step()                          # Parameters are updated: e.g. W = W - lr * Gradient_W

    def predict(self, data_loader: DataLoader):
        """
        Compute the mean square error loss of a regression model
        over a data loader
        """
        predictions: List = []

        # Put model in evaluation mode with `self.eval()`:
        # some feature are only meant for training (e.g. dropout).
        self.eval()

        # `torch.no_grad()` tells Pytorch not to keep information
        # related to the gradient, We don't need to keep gradient in memory
        # since we aren't optimizing
        with torch.no_grad():
            for batch_input, _ in data_loader:

                # map inputs and targets to `self.device` (GPU if available)
                batch_input = tuple(x.to(self.device) for x in batch_input if isinstance(x, torch.Tensor))

                batch_output = self.forward(batch_input)  # compute prediction

                predictions.extend(prediction.cpu().numpy() for prediction in batch_output)

        return predictions
