import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any

class SimpleNN(nn.Module):
    """
    Defines a simple feed-forward neural network architecture.

    Uses Linear layers, BatchNorm1d for stabilization, ReLU activation,
    and Dropout for regularization.
    """

    def __init__(self, input_size: int, output_size: int, hidden_layers: list[int] = [128, 128, 64], dropout_rate: float = 0.2):
        """
        Initializes the neural network layers.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output values to predict.
            hidden_layers (list[int]): A list of integers, where each integer is the number of neurons in a hidden layer.
            dropout_rate (float): The dropout rate to apply.
        """
        super(SimpleNN, self).__init__()
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Input/Output sizes must be positive integers.")
        self.input_size = input_size

        self.layers = nn.ModuleList()
        
        # Input layer
        in_features = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(in_features, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
            
        # Output layer
        self.layers.append(nn.Linear(in_features, output_size))

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
   
        for layer in self.layers:
            x = layer(x)
        return x




