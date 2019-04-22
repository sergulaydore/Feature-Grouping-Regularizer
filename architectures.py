import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


def get_layer_units(n_inputs=5, n_outputs=3,
                    hidden_layer_sizes=10):
    """
    Computer the number of units at each layer
    :param hidden_layer_sizes: hidden layer
    :param n_inputs: number of features
    :param n_outputs: number of outputs
    :return: combined list of units including input,
     output and hidden layers.
    """
    if hidden_layer_sizes:
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)
        layer_units = [n_inputs] + hidden_layer_sizes + \
                      [n_outputs]
    else:
        layer_units = [n_inputs, n_outputs]

    return layer_units


class FeatureGrouping(nn.Module):
    def __init__(self, n_cluster, in_features, out_features, bias=True):
        super(FeatureGrouping, self).__init__()
        self.weight_orig = Parameter(torch.Tensor(out_features,
                                                  in_features),
                                     requires_grad=True
                                     )
        self.weight_reduced = Parameter(torch.Tensor(out_features,
                                                     n_cluster),
                                        requires_grad=True
                                        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_features),
                                  requires_grad=True
                                  )
        else:
            self.register_parameter('bias', None)
        self.n_cluster = n_cluster
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_orig.size(1))
        self.weight_orig.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight_reduced.data.uniform_(-stdv, stdv)

    def forward(self, x, cluster, training):
        """
        Apply feature grouping during training:
            - reduce the dimension of input
            - apply weights in reduced dimenstion
        Apply a standard linear transformation at prediction:
            - apply weights in original dimension
        bias term is not affected
        """
        # print("Training is", training)
        if training is True:
            reduced = torch.from_numpy(cluster.transform(x.data.numpy()))
            reduced_var = autograd.Variable(reduced.data, requires_grad=True)
            self.weight_reduced.data = torch.from_numpy(cluster.transform(self.weight_orig.data.numpy()))
            output = F.linear(reduced_var.float(), weight=self.weight_reduced.float(), bias=self.bias.float())
            return output
        else:
            return F.linear(x, weight=self.weight_orig, bias=self.bias)


class Net(nn.Module):
    def __init__(self,  n_inputs=5, n_outputs=3, n_cluster=None,
                 hidden_layer_sizes=None, activation=None, dropout=0,
                 ):
        """
        Initialize the network
        :param hidden_layer_sizes: list of hidden layers
        :param n_input: dimension of the input
        :param n_output: dimension of the output
        :param activation: activation function
        """
        super(Net, self).__init__()
        self.activation = activation
        self.n_cluster = n_cluster  # initial cluster just to get shapes
        self.layers = nn.ModuleList()
        self.dropout = dropout

        if n_cluster is not None:
            if not hasattr(hidden_layer_sizes, "__iter__") and hidden_layer_sizes is not None:
                hidden_layer_sizes = [hidden_layer_sizes]
            if hasattr(hidden_layer_sizes, "__iter__"):
                hidden_layer_sizes = list(hidden_layer_sizes)
            out_features = hidden_layer_sizes[0] if hidden_layer_sizes is not None else n_outputs
            feature_grouping = FeatureGrouping(n_cluster, n_inputs, out_features)
            if hidden_layer_sizes:
                rest_hidden_layer_sizes = hidden_layer_sizes[1:] if len(hidden_layer_sizes) > 1 else None
                layer_units = get_layer_units(hidden_layer_sizes[0], n_outputs,
                                              rest_hidden_layer_sizes)
            else:
                layer_units = None
            self.layers.append(feature_grouping)
        else:
            layer_units = get_layer_units(n_inputs, n_outputs,
                                          hidden_layer_sizes)
        if layer_units:
            for k in range(len(layer_units) - 1):
                self.layers.append(nn.Linear(layer_units[k], layer_units[k + 1]))

    def forward(self, x, cluster, training):
        for idx, f in enumerate(self.layers[:-1]):
            if hasattr(f, 'n_cluster'):
                x = f(x, cluster, training)
            else:
                x = F.dropout(x, p=self.dropout, training=training)
                x = f(x)
            if self.activation and self.activation != 'linear':
                x = getattr(F, self.activation)(x)
        # apply last layer
        if len(self.layers) >= 1:
            f = self.layers[-1]
            if hasattr(f, 'n_cluster'):
                x = f(x, cluster, training)
            else:
                x = F.dropout(x, p=self.dropout, training=training)
                x = f(x)
        x = F.log_softmax(x, dim=1)
        return x

    
class LeNet(nn.Module):
    def __init__(self, n_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 50)
        self.fc2 = nn.Linear(50, n_classes)
        self.n_cluster = None

    def forward(self, x, cluster, training):
        # cluster is None!
        if len(x.size()) != 4:
            shape = x.shape
            n_features = shape[-1]
            n_dim = int(n_features ** 0.5)
            x = x.view(-1, 1, n_dim, n_dim)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x), training=training), 2))
        shape = x.shape
        x = x.view(-1, shape[1] * shape[2] * shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
