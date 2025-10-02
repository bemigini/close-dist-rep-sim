"""

Medium CNN 

Based on pytorch CIFAR-10 tutorial:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


"""


import torch

import torch.nn as nn


class MedCNN(nn.Module):
    """ A medium CNN """
    def __init__(self, 
                 input_channels:int, final_dim:int, 
                 num_nodes: int,
                 nonlinearity: str = 'relu'):
        super().__init__()

        self.nonlinearity = nonlinearity
        self.input_channels = input_channels
        self.final_dim = final_dim

        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, num_nodes)
        self.fc3 = nn.Linear(num_nodes, num_nodes)
        self.fc4 = nn.Linear(num_nodes, int(num_nodes/2))
        self.fc5 = nn.Linear(int(num_nodes/2), final_dim)

        if self.nonlinearity == 'relu':
            self.activation = nn.ReLU()
        elif self.nonlinearity == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f'Nonlinearity not implemented: {self.nonlinearity}')
        

    def forward(self, x):
        """ The forward pass """
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x
