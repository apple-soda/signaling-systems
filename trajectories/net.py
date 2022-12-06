import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
Collection of LSTM-based models to classify ligand identity from time-series trajectory data
'''
class TSC_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, linear_hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear1 = nn.Linear(num_layers * hidden_dim, linear_hidden_dim)
        self.linear2 = nn.Linear(linear_hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        h = h.permute(1, 0, 2) # [b, h, n]
        h = h.flatten(start_dim=1)
        out = self.sigmoid(self.linear2(self.relu(self.linear1(h))))
        return out
        