import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

sys.path.insert(0, './cs224u') # hacky solution to avoid adding __init__.py in cs224u

from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

class ActivationLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_activation):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, device=device)
        self.activation = hidden_activation

    def forward(self, x):
        return self.activation(self.linear(x))


class TorchDeepNeuralClassifier(TorchShallowNeuralClassifier):
    def __init__(self,
            vocab_size,
            output_size,
            num_addends=4, 
            num_layers=1,
            embed_dim=50,
            **base_kwargs):
        """
        A dense, feed-forward network with the number of hidden layers
        set by `num_layers`.

        Parameters
        ----------
        num_layers : int
            Number of hidden layers in the network.

        **base_kwargs
            For details, see `torch_model_base.py`.

        Attributes
        ----------
        loss: nn.CrossEntropyLoss(reduction="mean")

        self.params: list
            Extends TorchModelBase.params with names for all of the
            arguments for this class to support tuning of these values
            using `sklearn.model_selection` tools.

        """
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        super().__init__(**base_kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.params += ['num_layers']
        self.input_size = num_addends * embed_dim
        self.output_size = output_size

    def build_graph(self):
        """
        Define the model's computation graph.

        Returns
        -------
        nn.Module

        """

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        # Input to hidden:
        self.layers = [
            self.embedding, 
            nn.Flatten(0),
            ActivationLayer(
                self.input_size, self.hidden_dim, self.device, self.hidden_activation)]
        # Hidden to hidden:
        for _ in range(self.num_layers-1):
            self.layers += [
                ActivationLayer(
                    self.hidden_dim, self.hidden_dim, self.device, self.hidden_activation)]
        # Hidden to output:
        self.layers.append(
            nn.Linear(self.hidden_dim, self.output_size, device=self.device))
        return nn.Sequential(*self.layers)