import torch.nn as nn

from collections import OrderedDict, Mapping


class Encoder(nn.Module):
    """
    Vanilla multilayer perceptron
    """

    def __init__(self, config):
        super(Encoder, self).__init__()

        # Get layers and name the model
        self.layers = self.build(config)

    def build(self, config):
        encoder_network = []
        for layer in config:
            if layer['type'] == 'linear':
                encoder_network.append(nn.Linear(**layer['param']))
            elif layer['type'] == 'relu':
                encoder_network.append(nn.ReLU())
            elif layer['type'] == 'tanh':
                encoder_network.append(nn.Tanh())
            elif layer['type'] == 'dropout':
                encoder_network.append(nn.Dropout(**layer['param']))
            elif layer['type'] == 'batch_norm':
                encoder_network.append(nn.BatchNorm1d(**layer['param']))
            else:
                raise ValueError("Unsupported layer type supplied.")
        return nn.Sequential(*encoder_network)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output


class Decoder(nn.Module):
    """
    Inverts the operations of the forward MLP
    """

    def __init__(self, input_size, encoder):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.encoder = encoder

        # Inverted MLP associated with the forward MLP
        self.layers = nn.ModuleList(
            [self.inverse_module(i) for i in reversed(encoder.layers)]
        )

    def compute_output_sizes(self):
        output_sizes = [self.input_size]
        for layer in self.encoder.layers:
            output_sizes.append(output_sizes[-1])
        return output_sizes

    def inverse_module(self, layer):
        """
        Inputs:
        -------
        layer: Layer to be inversed in order to create the inverted MLP.

        Returns:
        --------
        The inverse of each operation
        """
        if isinstance(layer, nn.ReLU):
            return nn.ReLU()
        elif isinstance(layer, nn.Dropout):
            return nn.Dropout(p=layer.p)
        elif isinstance(layer, nn.BatchNorm1d):
            return nn.BatchNorm1d(layer.num_features)
        elif isinstance(layer, nn.BatchNorm1d):
            return nn.BatchNorm1d(layer.num_features)
        elif isinstance(layer, nn.Linear):
            return nn.Linear(layer.out_features, layer.in_features)
        else:
            return None

    def forward(self, x):
        output = x
        for i, layer in enumerate(self.layers):
            output = layer(output)
        return output
