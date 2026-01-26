=========================================
GNN Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The GNN (Graph Neural Network) model in SYMFLUENCE is a cutting-edge data-driven approach that leverages graph-based deep learning to predict streamflow across river networks. Unlike traditional LSTMs that treat each basin independently, GNNs explicitly model the spatial connectivity and upstream-downstream relationships within river networks.

**Key Capabilities:**

- Spatial graph-based streamflow prediction
- Explicit river network topology modeling
- Multi-site simultaneous prediction
- Information propagation along river network
- Transfer learning across basins in same network
- Scalable to large river networks
- Regional-scale prediction

**Typical Applications:**

- River network-wide streamflow prediction
- Multi-site calibration and prediction
- Ungauged basin prediction (network-informed)
- Regional hydrological modeling
- Data assimilation in river networks
- Ensemble forecasting across networks
- Climate change impact on river systems

**Spatial Scales:** River network (100s to 1000s of connected basins)

**Temporal Resolution:** Daily to hourly

GNN Architecture for Hydrology
==============================

Graph Neural Networks Fundamentals
----------------------------------

**What is a Graph?**

In hydrology, a graph represents a river network:

.. code-block:: text

   Nodes:  Basins/catchments
   Edges:  River connections (upstream → downstream)

   Example River Network:

   Basin A ──→ Basin B ──→ Basin D (outlet)
               ↗
   Basin C ───┘

**Graph Structure:**

- **Nodes (V):** Each basin with attributes (area, elevation, land cover, etc.)
- **Edges (E):** Directed connections representing flow direction
- **Adjacency Matrix (A):** Defines connectivity

GNN Message Passing
-------------------

GNNs learn by passing information along graph edges:

**Process:**

1. **Node features:** Each basin has forcing data + static attributes
2. **Message passing:** Information flows from upstream to downstream
3. **Aggregation:** Each node receives messages from upstream neighbors
4. **Update:** Node updates its hidden state based on messages
5. **Prediction:** Final layer predicts streamflow

**Hydrological Intuition:**

- Upstream precipitation affects downstream flow (captured via message passing)
- Basin characteristics influence how water is routed (learned node embeddings)
- River network topology constrains predictions (graph structure)

GNN vs LSTM
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - LSTM
     - GNN
   * - Spatial structure
     - Independent basins
     - Explicit network topology
   * - Training
     - One basin at a time
     - Entire network simultaneously
   * - Ungauged basins
     - Requires transfer learning
     - Network-informed interpolation
   * - Computational cost
     - Low (per basin)
     - Higher (entire network)
   * - Data requirements
     - Moderate (per basin)
     - High (network + all basins)
   * - Interpretability
     - Limited
     - Slightly better (graph structure)

Network Architecture
--------------------

SYMFLUENCE GNN configuration:

.. code-block:: text

   Input Layer (per node):
   ├─ Forcing time series (precipitation, temperature, ...)
   ├─ Static attributes (area, elevation, land cover, ...)
   └─ Lagged streamflow (if available)

   ↓

   GNN Layers (message passing):
   ├─ Layer 1: Graph Convolution + ReLU
   ├─ Layer 2: Graph Convolution + ReLU
   └─ Layer N: Graph Convolution + ReLU

   ↓

   Output Layer (per node):
   └─ Streamflow prediction

Configuration in SYMFLUENCE
===========================

Model Selection
---------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: GNN

Key Configuration Parameters
----------------------------

Network Architecture
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - GNN_HIDDEN_SIZE
     - 128
     - Hidden layer dimension (64, 128, 256)
   * - GNN_NUM_LAYERS
     - 3
     - Number of graph convolution layers (2-5)
   * - GNN_DROPOUT
     - 0.2
     - Dropout rate for regularization
   * - GNN_L2_REGULARIZATION
     - 1e-6
     - L2 weight decay

Training Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - GNN_EPOCHS
     - 300
     - Training epochs
   * - GNN_BATCH_SIZE
     - 64
     - Mini-batch size
   * - GNN_LEARNING_RATE
     - 0.001
     - Adam optimizer learning rate
   * - GNN_LEARNING_PATIENCE
     - 30
     - Early stopping patience

Model Options
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - GNN_LOAD
     - false
     - Load pre-trained model
   * - GNN_PARAMS_TO_CALIBRATE
     - null
     - Hyperparameters to tune
   * - GNN_PARAMETER_BOUNDS
     - null
     - Bounds for hyperparameter tuning

Input Data Requirements
=======================

Network Structure
-----------------

**River network topology file:**

**File:** ``<domain>_river_network.json``

.. code-block:: json

   {
     "nodes": [
       {
         "id": "basin_001",
         "area_km2": 250.5,
         "elevation_m": 1450,
         "latitude": 51.2,
         "longitude": -115.3,
         "land_cover": {"forest": 0.6, "grass": 0.3, "urban": 0.1},
         "soil_type": "loam"
       },
       {
         "id": "basin_002",
         "area_km2": 180.3,
         "elevation_m": 1200,
         "latitude": 51.1,
         "longitude": -115.1,
         "land_cover": {"forest": 0.4, "grass": 0.5, "urban": 0.1},
         "soil_type": "clay"
       }
     ],
     "edges": [
       {"source": "basin_001", "target": "basin_002"},
       {"source": "basin_002", "target": "basin_003"}
     ]
   }

**Adjacency matrix (alternative format):**

**File:** ``<domain>_adjacency.csv``

.. code-block:: text

   ,basin_001,basin_002,basin_003
   basin_001,0,1,0
   basin_002,0,0,1
   basin_003,0,0,0

Training Data
-------------

**Forcing data (per basin):**

**File:** ``forcing_<basin_id>.csv``

.. code-block:: text

   Date,Precip_mm,Temp_C,Rad_W/m2,Humidity_pct
   2015-01-01,5.2,2.3,120.5,65
   2015-01-02,0.0,3.1,135.2,58
   ...

**Streamflow observations (per gauged basin):**

**File:** ``streamflow_<basin_id>.csv``

.. code-block:: text

   Date,Flow_m3s
   2015-01-01,45.3
   2015-01-02,42.1
   ...

**Note:** Not all basins need observations (GNN can predict for ungauged basins within network)

Static Attributes
-----------------

**File:** ``basin_attributes.csv``

.. code-block:: text

   basin_id,area_km2,elev_mean_m,slope_deg,forest_frac,soil_clay_frac
   basin_001,250.5,1450,8.5,0.60,0.25
   basin_002,180.3,1200,5.2,0.40,0.35
   basin_003,420.1,980,3.1,0.30,0.40
   ...

Output Specifications
=====================

During Training
---------------

**Training logs:**

.. code-block:: text

   Epoch 1/300:  Train Loss: 0.621  Val Loss: 0.745  Avg NSE: 0.42
   Epoch 2/300:  Train Loss: 0.489  Val Loss: 0.602  Avg NSE: 0.55
   ...
   Epoch 95/300: Train Loss: 0.112  Val Loss: 0.156  Avg NSE: 0.81
   Early stopping at epoch 125

**Model checkpoint:**

**File:** ``<project_dir>/models/GNN/best_model.pt``

After Training
--------------

**Network-wide predictions:**

**File:** ``<network>_GNN_predictions.csv``

.. code-block:: text

   Date,basin_001_obs,basin_001_pred,basin_002_obs,basin_002_pred,basin_003_obs,basin_003_pred
   2015-01-01,45.3,43.8,32.1,31.5,78.4,76.9
   2015-01-02,42.1,41.2,29.8,29.1,72.8,71.5
   ...

**Per-basin performance:**

**File:** ``GNN_basin_metrics.csv``

.. code-block:: text

   basin_id,NSE,KGE,RMSE_m3s,MAE_m3s,Bias_pct,has_observations
   basin_001,0.82,0.78,5.2,3.8,-2.1,true
   basin_002,0.76,0.73,3.8,2.9,1.5,true
   basin_003,0.68,0.65,8.1,6.2,-4.3,false  # Ungauged - validated with proxy
   ...

Model-Specific Workflows
========================

Basic GNN Workflow
------------------

River network with multiple gauged basins:

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: river_network
   HYDROLOGICAL_MODEL: GNN

   # Define network domain
   DOMAIN_DEFINITION_METHOD: river_network
   RIVER_NETWORK_FILE: ./network_topology.json

   # Or use basin delineation (auto-creates network)
   DOMAIN_DEFINITION_METHOD: merit_basins
   MERIT_BASIN_IDS: [10234, 10235, 10236, 10237, 10238]  # Connected basins

   # Forcing
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2020

   # GNN configuration
   GNN_HIDDEN_SIZE: 128
   GNN_NUM_LAYERS: 3
   GNN_EPOCHS: 300

   # Data split
   CALIBRATION_PERIOD: [2010, 2017]  # Training + validation
   VALIDATION_PERIOD: [2018, 2020]   # Testing

Run:

.. code-block:: bash

   symfluence workflow run --config config.yaml

   # Training uses all gauged basins simultaneously
   # Predicts for ungauged basins within network

Large River Network Application
-------------------------------

For major river systems (e.g., Mississippi, Amazon):

.. code-block:: yaml

   # config.yaml
   DOMAIN_DEFINITION_METHOD: merit_basins
   MERIT_BASIN_IDS: [...]  # 500+ connected basins

   HYDROLOGICAL_MODEL: GNN

   # Larger network for big system
   GNN_HIDDEN_SIZE: 256
   GNN_NUM_LAYERS: 4

   # More epochs for complex network
   GNN_EPOCHS: 500

   # GPU essential
   USE_GPU: true

   # Batch training
   GNN_BATCH_SIZE: 128

Ungauged Basin Prediction
-------------------------

Predict streamflow at ungauged locations:

.. code-block:: yaml

   # config.yaml
   # Define network including ungauged basins
   MERIT_BASIN_IDS: [101, 102, 103, 104, 105]  # 5 basins

   # Observations available for basins 101, 103, 105
   # Basins 102, 104 are ungauged

   HYDROLOGICAL_MODEL: GNN

   # GNN will use network structure to inform predictions
   # at ungauged basins 102 and 104

Transfer Learning Across Networks
---------------------------------

Train on one network, apply to another:

.. code-block:: yaml

   # Step 1: Train on data-rich network
   # config_source_network.yaml
   DOMAIN_NAME: colorado_river
   MERIT_BASIN_IDS: [...]  # Well-gauged network

   HYDROLOGICAL_MODEL: GNN
   GNN_EPOCHS: 300

   # Save model
   GNN_SAVE_MODEL: true

.. code-block:: yaml

   # Step 2: Apply to data-sparse network
   # config_target_network.yaml
   DOMAIN_NAME: snake_river
   MERIT_BASIN_IDS: [...]  # Sparse gauge network

   HYDROLOGICAL_MODEL: GNN

   # Load pre-trained model
   GNN_LOAD: true
   GNN_PRETRAINED_MODEL: ../colorado_river/models/GNN/best_model.pt

   # Fine-tune with available data
   GNN_EPOCHS: 50

Hyperparameter Tuning
=====================

Key Hyperparameters
-------------------

**1. Hidden Size**

.. code-block:: yaml

   GNN_HIDDEN_SIZE: [64, 128, 256, 512]

- Larger = more capacity, more parameters
- 128-256 typical for medium networks (50-100 basins)
- 512+ for very large networks (500+ basins)

**2. Number of Layers**

.. code-block:: yaml

   GNN_NUM_LAYERS: [2, 3, 4, 5]

- More layers = information propagates farther in network
- 2-3 layers: local information (immediate neighbors)
- 4-5 layers: basin receives info from basins several hops away
- Diminishing returns after 4-5 layers

**3. Dropout**

.. code-block:: yaml

   GNN_DROPOUT: [0.0, 0.1, 0.2, 0.3, 0.5]

- Higher = more regularization
- 0.2-0.3 typical
- Increase if overfitting

**4. Learning Rate**

.. code-block:: yaml

   GNN_LEARNING_RATE: [0.0001, 0.0005, 0.001, 0.005]

- 0.001 is good starting point
- Lower if training unstable
- Higher for faster convergence

Automated Tuning
----------------

.. code-block:: yaml

   # Use optimization framework
   OPTIMIZATION_ALGORITHM: RandomSearch

   # Define search space
   GNN_HIDDEN_SIZE: [128, 256]
   GNN_NUM_LAYERS: [3, 4]
   GNN_DROPOUT: [0.2, 0.3]

   # Optimize network-averaged metric
   OPTIMIZATION_METRIC: KGE_network_avg
   OPTIMIZATION_MAX_ITERATIONS: 15

Known Limitations
=================

1. **Network Data Required:**

   - Needs river network topology
   - All basins in network need forcing data
   - Can't apply to isolated single basins easily

2. **Computational Cost:**

   - Training entire network is expensive
   - GPU strongly recommended
   - Scales poorly to 1000+ basin networks without optimization

3. **Data Hungry:**

   - Needs many gauged basins for training (ideally 20+)
   - Fewer gauges = degraded performance
   - Small networks (<10 basins) may not benefit over LSTM

4. **Black Box:**

   - Even less interpretable than LSTM
   - Hard to diagnose why predictions fail
   - Graph structure helps slightly but still opaque

5. **Extrapolation Issues:**

   - Same issues as LSTM for climate change
   - Cannot extrapolate outside training distribution
   - Ungauged basins still challenging if very different from gauged

6. **Network Topology Sensitivity:**

   - Incorrect network structure = poor performance
   - Errors in basin connectivity propagate
   - Need accurate DEM and flow routing

Troubleshooting
===============

Common Issues
-------------

**Error: "PyTorch Geometric not found"**

.. code-block:: bash

   # Install PyTorch Geometric (GNN library)
   pip install torch-geometric

   # Or with conda:
   conda install pyg -c pyg

**Error: "River network file missing"**

.. code-block:: yaml

   # Provide network topology
   RIVER_NETWORK_FILE: ./network_topology.json

   # Or use MERIT basins (auto-creates network)
   DOMAIN_DEFINITION_METHOD: merit_basins
   MERIT_BASIN_IDS: [...]

**Error: "Graph connectivity error"**

Check network topology:

.. code-block:: python

   import json
   with open('network_topology.json') as f:
       net = json.load(f)

   # Verify all edges reference valid nodes
   node_ids = {n['id'] for n in net['nodes']}
   for edge in net['edges']:
       assert edge['source'] in node_ids
       assert edge['target'] in node_ids

**Poor performance on ungauged basins**

1. **Increase number of gauged basins in training**
2. **Add more static attributes** (land cover, soil, topography)
3. **Use more GNN layers** (information propagates farther)
4. **Ensure network topology is correct**

**Overfitting (train >> val performance)**

.. code-block:: yaml

   # Increase regularization
   GNN_DROPOUT: 0.4
   GNN_L2_REGULARIZATION: 1e-5

   # Reduce capacity
   GNN_HIDDEN_SIZE: 64
   GNN_NUM_LAYERS: 2

**Slow training**

.. code-block:: yaml

   # Use GPU (essential for GNN)
   USE_GPU: true

   # Larger batches
   GNN_BATCH_SIZE: 128

   # Fewer layers
   GNN_NUM_LAYERS: 2

**NaN predictions**

1. Check for missing forcing data across network
2. Verify normalization didn't produce NaNs
3. Reduce learning rate
4. Check network topology for cycles or disconnected components

Performance Tips
================

Improving Accuracy
------------------

1. **More gauged basins:** 20+ gauges >> 5 gauges
2. **Rich static attributes:** Land cover, soil, geology, climate indices
3. **Accurate network topology:** Verify with DEM-derived flow directions
4. **Deeper networks:** 4-5 layers for large networks
5. **Multi-task learning:** Predict flow + other variables (e.g., snow)

Speeding Up Training
--------------------

1. **Use GPU** (10-100x faster)
2. **Smaller networks** (reduce hidden size, layers)
3. **Graph sampling** (train on subgraphs for very large networks)
4. **Early stopping** (patience = 20-30)
5. **Larger batches** (if memory allows)

Deployment
----------

After training, GNN is fast for inference:

.. code-block:: python

   # Predict entire network in milliseconds
   # Ideal for operational forecasting

Comparing with Other Models
===========================

**GNN vs LSTM:**

- Use GNN if: Many connected basins, network structure important
- Use LSTM if: Single basin or independent basins

**GNN vs Physics-Based:**

- GNN advantages: Fast, learns complex patterns, no calibration
- Physics advantages: Interpretable, extrapolates better, works with less data

**Recommendation:**

.. code-block:: yaml

   # Multi-model ensemble
   HYDROLOGICAL_MODEL: [SUMMA, LSTM, GNN]

   # GNN excels at spatial patterns
   # LSTM at temporal patterns
   # SUMMA at physical processes

Additional Resources
====================

**Graph Neural Networks for Hydrology:**

- Kratzert et al. (2023): "Graph Neural Networks for rainfall-runoff modeling"
- Nearing et al. (2023): "Graph-based learning for river networks"
- Shen et al. (2023): "Differentiable graph network for streamflow prediction"

**PyTorch Geometric:**

- Documentation: https://pytorch-geometric.readthedocs.io
- Tutorials: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

**Graph Theory for River Networks:**

- Rinaldo et al. (2006): "River networks as ecological corridors"
- Rodríguez-Iturbe & Rinaldo (1997): "Fractal River Basins"

**SYMFLUENCE-specific:**

- :doc:`../configuration`: GNN parameter reference
- :doc:`model_lstm`: Comparison with LSTM
- :doc:`model_summa`: Comparison with physics-based models
- :doc:`../troubleshooting`: General troubleshooting

**Datasets with River Networks:**

- MERIT-Basins: Global river basin network
- NHDPlus: US river network
- HydroSHEDS: Global drainage network

**Example Notebooks:**

.. code-block:: bash

   # GNN examples
   symfluence examples list | grep GNN

**Advanced GNN Architectures:**

- Graph Attention Networks (GAT)
- Graph Convolutional Networks (GCN)
- Message Passing Neural Networks (MPNN)
- Spatial-Temporal Graph Networks (STGNN)

**Future Directions:**

- Physics-informed GNNs (combining data-driven + physics)
- Hybrid models (GNN + process-based routing)
- Uncertainty quantification with GNN ensembles
