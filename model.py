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


class TorchDeepNeuralModel(nn.Module):
    def __init__(self,
            vocab_size,
            output_size,
            num_addends,
            device,
            hidden_activation,
            num_layers=1,
            embed_dim=40,
            hidden_dim=50):

        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_addends = num_addends
        self.hidden_dim = hidden_dim 

        self.input_size = self.num_addends * self.embed_dim
        self.output_size = output_size
        self.device = device
        self.hidden_activation = hidden_activation

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        # Input to hidden:
        self.layers = [
            #self.embedding, 
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
        self.model = nn.Sequential(*self.layers)

    def forward(self, X):
        X = self.embedding(X)
        new_x = []
        for x in X:
            new_x.append(torch.cat(tuple(x[i] for i in range(self.num_addends))))
        new_x = torch.stack(new_x)
        output = self.model(new_x)
        return output


class TorchDeepNeuralClassifier(TorchShallowNeuralClassifier):
    def __init__(self,
            vocab_size,
            output_size,
            num_addends,
            num_layers=1,
            embed_dim=40,
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
        self.num_addends = num_addends
        super().__init__(**base_kwargs)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.params += ['num_layers']
        self.output_size = output_size
      
    def build_dataset(self, X, y=None):
        X = np.array(X)
        self.input_dim = X.shape[1]
        X = torch.LongTensor(X)
        if y is None:
            dataset = torch.utils.data.TensorDataset(X)
        else:
            self.classes_ = sorted(set(y))
            self.n_classes_ = len(self.classes_)
            class2index = dict(zip(self.classes_, range(self.n_classes_)))
            y = [class2index[label] for label in y]
            y = torch.tensor(y)
            dataset = torch.utils.data.TensorDataset(X, y)
        return dataset


    def build_graph(self):
        """
        Define the model's computation graph.

        Returns
        -------
        nn.Module

        """
        return TorchDeepNeuralModel(self.vocab_size, self.output_size,
                        self.num_addends, self.device, self.hidden_activation, 
                        self.num_layers, self.embed_dim, self.hidden_dim)