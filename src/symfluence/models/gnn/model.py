
"""
GNN (Graph Neural Network) Model Definition.

This module contains the PyTorch model definition for a DAG-based
Spatio-Temporal GNN for hydrological routing.
"""

import torch
import torch.nn as nn
import logging

class DirectedGraphLayer(nn.Module):
    """
    Layer to propagate information downstream through the river network DAG.
    """
    def __init__(self, input_size: int, output_size: int, adjacency_matrix: torch.Tensor):
        super(DirectedGraphLayer, self).__init__()
        self.adj = adjacency_matrix # Sparse tensor (Nodes, Nodes), A_ij = 1 if j -> i
        # Weights for transforming upstream inputs before aggregation
        self.weight = nn.Linear(input_size, output_size, bias=False)
        # Weights for the node's self-state
        self.self_weight = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Nodes, Features)
        Returns:
            out: Tensor of shape (Batch, Nodes, Output_Features)
        """
        # x shape: (B, N, F)
        
        # 1. Transform inputs from neighbors (upstream)
        # We want to aggregate upstream: h_i = Self(x_i) + Agg(Transform(x_j)) for j->i
        
        # Apply weight matrix to all nodes: (B, N, OutF)
        x_trans = self.weight(x) 
        
        # Propagate: Out = Adj @ x_trans
        # Adj is (N, N), x_trans is (B, N, OutF).
        # We need to broadcast matrix multiplication.
        # Since Adj is sparse and constant, we can use torch.matmul or sparse mm
        
        # For efficiency with Batches, we usually permute
        # (B, N, F) -> (F, B, N) or similar, but let's try standard broadcast
        
        # If x_trans is (B, N, F), we want result (B, N, F)
        # result[b, i, f] = sum_j ( A[i, j] * x_trans[b, j, f] )
        
        # Torch sparse matmul (spmm) usually supports (S, D) @ (D, Dense)
        # We can reshape x_trans to (N, B*F) for the multiplication
        B, N, F = x_trans.shape
        x_reshaped = x_trans.permute(1, 0, 2).reshape(N, B * F)
        
        # Aggregation from upstream
        aggregated = torch.sparse.mm(self.adj, x_reshaped) # (N, B*F)
        
        # Reshape back
        aggregated = aggregated.reshape(N, B, F).permute(1, 0, 2) # (B, N, F)
        
        # 2. Add self-contribution
        self_contribution = self.self_weight(x)
        
        out = self_contribution + aggregated
        return self.activation(out)


class GNNModel(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for Hydrology.
    
    Structure:
    1. Input Layer
    2. Shared LSTM (Time Processing) applied to each Node independenty.
    3. Directed Graph Layer (Spatial Processing/Routing) applied to the LSTM outputs.
    4. Readout Layer to predict Streamflow.
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 gnn_output_size: int,
                 adjacency_matrix: torch.Tensor,
                 dropout_rate: float = 0.2):
        """
        Args:
            input_size: Number of forcing features per node.
            hidden_size: Hidden size of the LSTM.
            gnn_output_size: Size of the node embedding after GNN layer.
            adjacency_matrix: Sparse tensor representing the DAG (Rows=DS, Cols=US).
        """
        super(GNNModel, self).__init__()
        
        # Temporal Feature Extraction (Shared weights across all nodes)
        # LSTM dropout is only applied when num_layers > 1, so avoid warnings.
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.0)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(hidden_size)
        
        # Spatial Routing / Graph Layer
        # We can stack multiple GNN layers if we want deeper routing, 
        # but for a simple DAG representation, one might suffice if we assume 
        # linear routing or just 1-hop aggregation per step. 
        # However, water travels far.
        # Ideally, we sort nodes topologically and accumulate.
        # But for a "GNN" approach, we typically use fixed layers.
        # Let's use 1 GNN layer to represent "mixing" then a readout.
        self.gnn = DirectedGraphLayer(hidden_size, gnn_output_size, adjacency_matrix)
        
        # Final Prediction
        self.fc = nn.Linear(gnn_output_size, 1) # Predict Q (scalar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (Batch, Time, Nodes, Features)
            
        Returns:
            out: Streamflow prediction (Batch, Nodes, 1)
        """
        B, T, N, F = x.shape
        
        # 1. Temporal Processing (LSTM)
        # Flatten Batch and Nodes to process time series independently
        x_reshaped = x.view(B * N, T, F)
        
        # LSTM output: (B*N, T, H)
        # We take the last time step hidden state
        _, (h_n, _) = self.lstm(x_reshaped)
        # h_n shape: (NumLayers, B*N, H) -> Take last layer: (B*N, H)
        h_last = h_n[-1]
        
        h_last = self.ln(h_last)
        h_last = self.dropout(h_last)
        
        # Reshape back to (B, N, H) for Graph processing
        h_nodes = h_last.view(B, N, -1)
        
        # 2. Spatial Processing (GNN)
        # Propagate info from US to DS
        h_routed = self.gnn(h_nodes)
        
        # 3. Readout
        out = self.fc(h_routed) # (B, N, 1)
        
        return out
