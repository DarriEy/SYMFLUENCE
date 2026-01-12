"""Graph Neural Network (GNN) Hydrological Model.

This module implements a Spatio-Temporal Graph Neural Network for hydrological
prediction and water routing. The model combines:

1. **Temporal Processing**: LSTM processes time series forcing data (precipitation,
   temperature, etc.) independently at each node to extract temporal features.

2. **Spatial Processing**: A directed graph layer propagates information along the
   river network DAG (directed acyclic graph), modeling how water flows from upstream
   to downstream nodes.

3. **Prediction**: Final readout layer produces streamflow predictions at each node.

Design Rationale:
    The GNN approach addresses limitations of lumped and purely distributed models:
    - Unlike lumped models, GNNs capture spatial heterogeneity and routing dynamics
    - Unlike purely distributed models, GNNs learn routing patterns from data rather
      than requiring physical equations
    - The adjacency matrix encodes the river network topology, ensuring predictions
      respect the actual watershed structure

Graph Structure:
    - Nodes: Hydrologic units (HRUs) or subcatchments
    - Edges: Directed edges from upstream (US) to downstream (DS) nodes
    - Adjacency Matrix: Sparse N×N matrix where A[i,j]=1 if water flows from j to i

Key Components:
    GNNRunner: Main orchestrator for model workflow (data, training, simulation)
    GNNModel: PyTorch model architecture (LSTM + GNN layers)
    GNNPreprocessor: Data loading, normalization, tensor preparation
    GNNPostprocessor: Output formatting and result saving

Configuration Parameters:
    GNN_HIDDEN_SIZE: LSTM hidden dimension (default: 64)
    GNN_OUTPUT_SIZE: GNN layer output dimension (default: 32)
    GNN_DROPOUT: Dropout rate for regularization (default: 0.2)
    GNN_EPOCHS: Training epochs (default: 100)
    GNN_BATCH_SIZE: Batch size for training (default: 16)
    GNN_LEARNING_RATE: Adam optimizer learning rate (default: 0.005)
    GNN_USE_SNOW: Include snow cover as input feature (default: False)
    GNN_LOAD: Load pre-trained model instead of training (default: False)

Typical Workflow:
    1. Initialize GNNRunner with config
    2. Load forcing data (precipitation, temperature) and observations
    3. Construct river network adjacency matrix from delineation
    4. Train model on historical data (or load pre-trained)
    5. Simulate forward period to generate streamflow predictions

Limitations and Considerations:
    - Requires spatially distributed domain (not suitable for lumped mode)
    - Graph structure must be consistent between training and simulation
    - Model learns routing implicitly; interpret results with domain knowledge
    - For operational forecasting, retrain regularly with new observations
"""

from .runner import GNNRunner
from .preprocessor import GNNPreprocessor
from .postprocessor import GNNPostprocessor

__all__ = ['GNNRunner', 'GNNPreprocessor', 'GNNPostprocessor']


# Register config adapter with ModelRegistry
from symfluence.models.registry import ModelRegistry
from .config import GNNConfigAdapter
ModelRegistry.register_config_adapter('GNN')(GNNConfigAdapter)

# Register result extractor with ModelRegistry
from .extractor import GNNResultExtractor
ModelRegistry.register_result_extractor('GNN')(GNNResultExtractor)

# Register preprocessor with ModelRegistry
ModelRegistry.register_preprocessor('GNN')(GNNPreprocessor)

# Register runner with ModelRegistry
ModelRegistry.register_runner('GNN')(GNNRunner)

# Register postprocessor with ModelRegistry
ModelRegistry.register_postprocessor('GNN')(GNNPostprocessor)
