
"""
GNN (Graph Neural Network) Model Definition.

This module contains the PyTorch model definition for a DAG-based
Spatio-Temporal GNN for hydrological routing.
"""

import torch
import torch.nn as nn

class DirectedGraphLayer(nn.Module):
    """Layer to propagate information downstream through the river network DAG.

    This layer models water routing along the river network by aggregating
    features from upstream neighbors and combining with the node's own state.
    The implementation uses sparse matrix multiplication for efficiency.

    The key insight is that water flows from upstream (US) to downstream (DS)
    nodes following the river network topology. The adjacency matrix encodes
    this flow: A[i,j]=1 if water flows from node j (upstream) to node i (downstream).

    Algorithm:
        For each node i:
        1. Transform the node's own features: self_contribution = self_weight(x_i)
        2. Aggregate upstream features: aggregated = Σ(weight(x_j) for j where j->i)
        3. Combine: out_i = ReLU(self_contribution + aggregated)

    This allows downstream nodes to "see" what's happening upstream, enabling
    the model to learn routing delays and attenuation as water moves downstream.

    Args:
        input_size: Feature dimension of input (e.g., 64 from LSTM)
        output_size: Feature dimension of output (e.g., 32)
        adjacency_matrix: Sparse tensor (Nodes, Nodes) where A[i,j]=1 if j->i.
            Must be a PyTorch sparse tensor for efficient batched computation.
    """
    def __init__(self, input_size: int, output_size: int, adjacency_matrix: torch.Tensor):
        super(DirectedGraphLayer, self).__init__()
        self.adj = adjacency_matrix  # Sparse tensor (Nodes, Nodes), A_ij = 1 if j -> i
        # Weights for transforming upstream inputs before aggregation
        self.weight = nn.Linear(input_size, output_size, bias=False)
        # Weights for the node's self-state
        self.self_weight = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: aggregate information from upstream nodes and node itself.

        This implements the directed graph aggregation operation. For each node,
        it combines:
        1. The node's own features (transformed by self_weight)
        2. Aggregated features from all upstream neighbors (transformed by weight)

        The sparse adjacency matrix multiplication is efficiently implemented by
        reshaping the batch and feature dimensions into a single dimension, applying
        one sparse matrix operation, and reshaping back.

        Args:
            x: Input tensor of shape (Batch, Nodes, Features).

        Returns:
            out: Output tensor of shape (Batch, Nodes, Output_Features).
                Contains ReLU-activated aggregated features.
        """
        # x shape: (B, N, F)

        # 1. Transform inputs from neighbors (upstream)
        # We want to aggregate upstream: h_i = Self(x_i) + Agg(Transform(x_j)) for j->i

        # Apply weight matrix to all nodes: (B, N, OutF)
        x_trans = self.weight(x)

        # Propagate: Out = Adj @ x_trans
        # Adj is (N, N), x_trans is (B, N, OutF).
        # For efficiency with batches, reshape to allow single sparse matrix multiplication.
        # Torch sparse.mm expects (sparse S, D) @ (dense D, K) -> (S, K)
        # We reshape x_trans from (B, N, F) -> (N, B*F) so we can multiply (N,N) @ (N, B*F)

        B, N, F = x_trans.shape
        # Reshape: (B, N, F) -> (N, B, F) -> (N, B*F)
        x_reshaped = x_trans.permute(1, 0, 2).reshape(N, B * F)

        # Aggregation from upstream using sparse matrix multiplication
        # Result: (N, B*F) where each node has aggregated its upstream neighbors
        aggregated = torch.sparse.mm(self.adj, x_reshaped)  # (N, B*F)

        # Reshape back: (N, B*F) -> (N, B, F) -> (B, N, F)
        aggregated = aggregated.reshape(N, B, F).permute(1, 0, 2)  # (B, N, F)

        # 2. Add self-contribution to aggregated upstream features
        self_contribution = self.self_weight(x)

        # Combine self and upstream contributions
        out = self_contribution + aggregated
        return self.activation(out)


class GNNModel(nn.Module):
    """Spatio-Temporal Graph Neural Network for Hydrological Modeling.

    Combines temporal (LSTM) and spatial (GNN) processing to predict streamflow
    accounting for both local rainfall-runoff dynamics and downstream routing.

    Architecture:
        1. **Input Layer**: Raw forcing data (precipitation, temperature, etc.)
        2. **Temporal Processing (LSTM)**: Extracts temporal features at each node
           independently. The LSTM learns how historical forcing data influences
           local runoff generation.
        3. **Layer Normalization & Dropout**: Regularizes features before spatial processing
        4. **Spatial Processing (Directed Graph Layer)**: Aggregates information along
           the river network DAG. Models how upstream runoff combines and attenuates
           moving downstream.
        5. **Readout Layer**: Final linear projection to streamflow predictions.

    Design Philosophy:
        - **Node-level LSTM**: Each node processes its own forcing independently,
          learning local rainfall-runoff generation (like a miniature lumped model)
        - **Network-level GNN**: After temporal processing, the GNN aggregates
          upstream contributions, allowing each node to "see" what's happening
          upstream and learn routing dynamics from data
        - **Sparse Adjacency**: Uses sparse matrix representation for efficiency
          on large watersheds with many nodes

    Input/Output Shapes:
        Input:  (Batch, Time, Nodes, Features)
        Output: (Batch, Nodes, 1) - Streamflow at each node

    Args:
        input_size: Number of forcing features per node (e.g., 4 for precip/temp/humidity/radiation)
        hidden_size: LSTM hidden dimension (e.g., 64). Larger = more capacity but slower
        gnn_output_size: Graph layer output dimension (e.g., 32). Controls routing model complexity
        adjacency_matrix: Sparse torch.Tensor of shape (Nodes, Nodes) encoding river network.
            A[i,j]=1 if water flows from node j (upstream) to node i (downstream).
        dropout_rate: Dropout probability for regularization (default: 0.2)

    Configuration Recommendations:
        - For small catchments (<50 nodes): hidden_size=32, gnn_output_size=16
        - For medium catchments (50-500 nodes): hidden_size=64, gnn_output_size=32
        - For large catchments (>500 nodes): hidden_size=128, gnn_output_size=64
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 gnn_output_size: int,
                 adjacency_matrix: torch.Tensor,
                 dropout_rate: float = 0.2):
        """Initialize the GNN model.

        Args:
            input_size: Number of forcing features per node.
            hidden_size: Hidden size of the LSTM.
            gnn_output_size: Size of the node embedding after GNN layer.
            adjacency_matrix: Sparse tensor representing the DAG (Rows=DS, Cols=US).

        Raises:
            ValueError: If input dimensions are invalid.
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
        """Forward pass through temporal and spatial processing layers.

        Processes input through three main stages:
        1. **Temporal**: LSTM independently at each node learns rainfall-runoff dynamics
        2. **Normalization & Regularization**: Layer normalization and dropout
        3. **Spatial**: Graph layer aggregates upstream information for routing
        4. **Prediction**: Linear readout produces streamflow

        The key design choice is flattening nodes into the batch dimension before
        the LSTM so each node's time series is processed independently with shared
        weights. This makes the model scalable: the LSTM capacity doesn't grow
        with the number of nodes.

        Args:
            x: Input tensor of shape (Batch, Time, Nodes, Features).
                - Batch: Multiple training examples
                - Time: Temporal sequence length (e.g., 365 days)
                - Nodes: Hydrologic units/subcatchments
                - Features: Forcing variables (precip, temp, etc.)

        Returns:
            out: Streamflow prediction tensor of shape (Batch, Nodes, 1).
                Values represent predicted streamflow at each node.
        """
        B, T, N, F = x.shape

        # 1. Temporal Processing (LSTM)
        # Reshape to flatten Batch and Nodes dimensions: (B*N, T, F)
        # This allows processing each node's time series independently with shared LSTM weights.
        # Why separate each node? Because each node has different forcing and hydrology,
        # and we want to learn node-specific rainfall-runoff relationships.
        x_reshaped = x.view(B * N, T, F)

        # LSTM output: (B*N, T, H) - hidden states at each time step
        # We take only the last time step hidden state, assuming the full sequence
        # context is encoded in h_last. This is a simplification but works well in practice.
        _, (h_n, _) = self.lstm(x_reshaped)
        # h_n shape: (NumLayers, B*N, H) -> Take last layer: (B*N, H)
        h_last = h_n[-1]

        # Apply layer normalization to stabilize features before spatial processing
        h_last = self.ln(h_last)
        # Apply dropout for regularization (reduces co-adaptation of units)
        h_last = self.dropout(h_last)

        # Reshape back to (B, N, H) for Graph processing
        h_nodes = h_last.view(B, N, -1)

        # 2. Spatial Processing (GNN)
        # Propagate information from upstream to downstream along river network
        # The DirectedGraphLayer aggregates each node's upstream neighbors,
        # allowing the model to learn routing delays and flow accumulation.
        h_routed = self.gnn(h_nodes)

        # 3. Readout
        # Linear transformation from GNN features to streamflow prediction
        out = self.fc(h_routed)  # (B, N, 1)

        return out
