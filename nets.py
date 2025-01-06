import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MockNetwork(nn.Module):
    def __init__(self):
        super(MockNetwork, self).__init__()
        self.p = nn.Parameter(torch.zeros((1)))

    def forward(self, x):
        x = self.p * x
        return x
