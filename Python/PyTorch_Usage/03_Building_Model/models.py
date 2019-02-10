import torch
from torch import nn


def TwoLayerNet(in_features=1, hidden_features=20, out_features=1):
    hidden = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
    activation = nn.ReLU()
    output = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
    
    net = nn.Sequential(hidden, activation, output)
    
    return net


class TwoLinearLayerNet(nn.Module):
    
    def __init__(self):
        super(TwoLinearLayerNet, self).__init__()
        pass
    
    def forward(self, *input):
        pass