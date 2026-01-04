"""
LSTM (Flow and Snow Hydrological LSTM) Model Definition.

This module contains the PyTorch model definition for LSTM.
"""

import torch
import torch.nn as nn
import logging

class LSTMModel(nn.Module):
    """
    LSTM-based model for hydrological predictions.
    
    Predicts streamflow and optionally snow water equivalent (SWE)
    from meteorological forcing data.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout_rate: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in the LSTM layers.
            num_layers (int): Number of LSTM layers.
            output_size (int): Number of output variables.
            dropout_rate (float): Dropout probability.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

        self.init_weights()

    def init_weights(self):
        """Initialize model weights using orthogonal and Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = self.ln(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return out