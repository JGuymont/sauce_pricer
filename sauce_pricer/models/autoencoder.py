#!/usr/bin/python3
"""
Class for the vanilla MLP autoencoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from .base_model import BaseModel
from .modules import linear
from .utils import evaluate_reconstruction_mse


class Autoencoder(BaseModel):
    """Vanilla MLP autoencoder. The decoder inverses all the operation applied by the encoder.

     :param encoder_config: List of layer configuration. Layer configurations are `dict` that most contain a key `type`
        and a key `param`. `type` should be one of ['linear', 'relu', 'tanh', 'dropout', '']
    """

    def __init__(self, input_dim, mask_pct, encoder_config, n_epochs, lr, weight_decay, device):
        super(BaseModel, self).__init__()
        self.encoder = linear.Encoder(config=encoder_config)
        self.decoder = linear.Decoder(input_size=input_dim, encoder=self.encoder)
        self.n_epochs = n_epochs
        self.device = device
        self.mask = nn.Dropout(mask_pct)
        self.optimizer = Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        self.evaluate = evaluate_reconstruction_mse

    def forward(self, inputs):
        """
        Forward propagation:
        encode then decode to get reconstruction
        """
        masked = self.mask(inputs)
        latent = self.encoder(masked)
        recons = self.decoder(latent)
        return torch.tanh(recons)

    def _train_iteration(self, train_loader: DataLoader):
        """
        Perform an iteration of gradient descent
        """

        # Set the module in training mode:
        # see https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train for details
        self.train()

        for batch_input in train_loader:

            self.optimizer.zero_grad()                   # make sure all the gradients are equal to zero.

            batch_input = batch_input.to(self.device)    # send inputs to GPU if available
            batch_recon = self.forward(batch_input)      # compute prediction
            loss = F.mse_loss(batch_input, batch_recon)  # compute loss
            loss.backward()                              # compute derivative of the loss wrt each parameter
            self.optimizer.step()                        # Parameters are updated: e.g. W = W - lr * Gradient_W
