"""
A model for 1 channel images.
"""

import torch
import torch.nn.functional as F
from torch import nn
import Resources.helpers as helpers


class LeNet5D(nn.Module):
    def __init__(self):
        super(LeNet5D, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 50, 5)
        self.fc1 = nn.Linear(50 * 5 * 5, 572)
        self.fc2 = nn.Linear(572, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 50 * 5 * 5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)