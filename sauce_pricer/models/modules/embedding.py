from typing import List, Tuple
import torch
from torch import nn
import pandas


class Embedding(nn.Module):

    def __init__(self, embedding_dims, dropout):

        super(Embedding, self).__init__()

        self.embedding_dims = embedding_dims
        self.embedding_layers = self._build_embeddings()
        self.dropout = nn.Dropout(p=dropout)

    def _build_embeddings(self):
        embedding_layers = []
        for num_category, embedding_dim in self.embedding_dims:
            embedding_layer = nn.Embedding(num_embeddings=num_category, embedding_dim=embedding_dim)
            embedding_layers.append(embedding_layer)
        embedding_layers = nn.ModuleList(embedding_layers)
        return embedding_layers

    def forward(self, inputs):
        embedded_features = []

        for i, embedding_layer in enumerate(self.embedding_layers):
            feature = inputs[:, i]                           # extract ith feature
            embedded_feature = embedding_layer(feature)      # embed current feature
            embedded_features.append(embedded_feature)

        embedded_features = tuple(embedded_features)         # convert `list` to `tuple` cause torch.cat expect `tuple`)
        embedded_features = torch.cat(embedded_features, 1)  # concatenate all the categorical features in one tensor
        embedded_features = self.dropout(embedded_features)  # apply dropout

        return embedded_features

    def num_embeddings(self):
        return sum([embedding_dim for _, embedding_dim in self.embedding_dims])

    @staticmethod
    def get_embedding_dimensions(
            dataframe: pandas.DataFrame,
            categorical_features: List[str],
            max_embedding_size: int = 25
    ) -> List[Tuple[int, int]]:
        embedding_dims = []
        for feature in categorical_features:
            n_category = dataframe[feature].nunique()                        # number of unique category for feature
            embedding_size = min(max_embedding_size, (n_category + 1) // 2)  # reduce dimension by 2
            embedding_size = max(2, embedding_size)                          # make sure embedding size is at least 2
            embedding_dims.append((n_category, embedding_size))
        return embedding_dims


