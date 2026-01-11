=========================================
LSTM Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The LSTM (Long Short-Term Memory) model in SYMFLUENCE is a data-driven deep learning approach for streamflow prediction. Built on PyTorch, this implementation uses recurrent neural networks to learn complex rainfall-runoff relationships directly from data, offering an alternative to physics-based models.

**Key Capabilities:**

- Data-driven streamflow prediction
- Long-term dependency learning (catchment memory)
- Optional snow modeling mode
- Attention mechanism support
- Transfer learning across basins
- Uncertainty quantification
- Fast execution after training
- Minimal physical parameterization

**Typical Applications:**

- Data-rich basins with limited process understanding
- Rapid assessment and benchmarking
- Ensemble member for multi-model approaches
- Operational forecasting (after training)
- Ungauged basins (via regionalization/transfer learning)
- Data assimilation experiments

**Spatial Scales:** Primarily lumped (basin-scale) or semi-distributed

**Temporal Resolution:** Daily to hourly

Model Architecture
==================

LSTM Fundamentals
----------------

**Standard LSTM Cell:**

LSTMs excel at capturing temporal dependencies through gates:

- **Forget gate:** What information to discard
- **Input gate:** What new information to store
- **Output gate:** What to output
- **Cell state:** Long-term memory

**For Hydrology:**

LSTMs learn relationships like:

- Snowmelt timing and lag
- Soil moisture memory
- Baseflow recession
- Precipitation-runoff response

**Advantages over traditional RNNs:**

- Captures long-term dependencies (e.g., multi-month snow storage)
- Avoids vanishing gradient problem
- Handles irregular forcing patterns

Network Structure
----------------

SYMFLUENCE LSTM configuration:

.. code-block:: text

   Input Layer:
   ├─ Precipitation
   ├─ Temperature
   ├─ Radiation (optional)
   ├─ Humidity (optional)
   ├─ Other forcing variables
   └─ Static catchment attributes

   ↓
   LSTM Layers (stacked):
   ├─ Layer 1 (hidden_size neurons)
   ├─ Layer 2
   └─ Layer N

   ↓
   Attention Layer (optional):
   └─ Weighted temporal attention

   ↓
   Dense Layer:
   └─ Output: Streamflow (+ Snow if LSTM_USE_SNOW=true)

Key Hyperparameters
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Hyperparameter
     - Default
     - Description
   * - LSTM_HIDDEN_SIZE
     - 128
     - Number of neurons per LSTM layer
   * - LSTM_NUM_LAYERS
     - 3
     - Number of stacked LSTM layers
   * - LSTM_LOOKBACK
     - 700
     - Input sequence length (days)
   * - LSTM_DROPOUT
     - 0.2
     - Dropout rate for regularization
   * - LSTM_USE_ATTENTION
     - true
     - Enable attention mechanism

Configuration in SYMFLUENCE
===========================

Model Selection
--------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: LSTM

Key Configuration Parameters
----------------------------

Network Architecture
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - LSTM_HIDDEN_SIZE
     - 128
     - Hidden layer size (64, 128, 256)
   * - LSTM_NUM_LAYERS
     - 3
     - Stacked LSTM layers (1-5)
   * - LSTM_DROPOUT
     - 0.2
     - Dropout probability (0-0.5)
   * - LSTM_USE_ATTENTION
     - true
     - Attention mechanism
   * - LSTM_USE_SNOW
     - false
     - Multi-task: flow + snow

Training Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - LSTM_EPOCHS
     - 300
     - Training epochs
   * - LSTM_BATCH_SIZE
     - 64
     - Mini-batch size
   * - LSTM_LEARNING_RATE
     - 0.001
     - Adam optimizer learning rate
   * - LSTM_LEARNING_PATIENCE
     - 30
     - Early stopping patience
   * - LSTM_L2_REGULARIZATION
     - 1e-6
     - L2 weight decay

Data Configuration
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - LSTM_LOOKBACK
     - 700
     - Input sequence length (days)
   * - LSTM_LOAD
     - false
     - Load pre-trained model
   * - LSTM_TRAIN_THROUGH_ROUTING
     - false
     - Train with routing for distributed models

Input Data Requirements
=======================

Training Data
------------

LSTMs require substantial training data:

**Minimum recommended:**

- 5+ years of daily data
- 10+ years preferred for robust training
- Continuous observations (minimal gaps)

**Input features:**

.. code-block:: yaml

   # Meteorological forcing (required)
   - Precipitation [mm/day]
   - Temperature [°C]

   # Additional forcing (recommended)
   - Shortwave radiation [W/m²]
   - Longwave radiation [W/m²]
   - Humidity [%]
   - Wind speed [m/s]

   # Static attributes (optional but helpful)
   - Basin area [km²]
   - Mean elevation [m]
   - Mean slope [degrees]
   - Land cover fractions
   - Soil properties

**Target variable:**

- Streamflow observations [m³/s or mm/day]

**Optional (if LSTM_USE_SNOW=true):**

- Snow water equivalent observations [mm]

Data Split
---------

Standard split for training/validation/testing:

.. code-block:: python

   # Example: 10 years total
   Training:    Years 1-6   (60%)
   Validation:  Years 7-8   (20%)
   Testing:     Years 9-10  (20%)

**Recommendations:**

- Use contiguous periods (avoid temporal leakage)
- Ensure all splits cover different climate conditions
- Holdout testing period should be truly unseen

Output Specifications
====================

During Training
--------------

**Logs:**

.. code-block:: text

   Epoch 1/300:  Train Loss: 0.543  Val Loss: 0.621  NSE: 0.45
   Epoch 2/300:  Train Loss: 0.412  Val Loss: 0.529  NSE: 0.58
   ...
   Epoch 85/300: Train Loss: 0.102  Val Loss: 0.145  NSE: 0.82
   Early stopping triggered at epoch 115

**Model checkpoints:**

- Best model saved based on validation performance
- Stored as PyTorch ``.pt`` file

**File:** ``<project_dir>/models/LSTM/best_model.pt``

After Training
-------------

**Predictions (CSV):**

**File:** ``<basin>_LSTM_predictions.csv``

.. code-block:: text

   Date,Observed_m3s,Predicted_m3s,Residual
   2015-01-01,45.3,43.8,1.5
   2015-01-02,42.1,41.2,0.9
   2015-01-03,55.8,54.3,1.5
   ...

**Performance metrics:**

- NSE (Nash-Sutcliffe Efficiency)
- KGE (Kling-Gupta Efficiency)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Bias

**If LSTM_USE_SNOW=true:**

.. code-block:: text

   Date,Q_obs,Q_pred,SWE_obs,SWE_pred
   2015-01-01,45.3,43.8,150.2,145.3
   ...

Model-Specific Workflows
========================

Basic LSTM Workflow
------------------

Simple basin-scale LSTM:

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: my_basin
   HYDROLOGICAL_MODEL: LSTM

   # Domain (lumped)
   DOMAIN_DEFINITION_METHOD: polygon
   CATCHMENT_SHP_PATH: ./basin.shp

   # Forcing
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2020  # 10 years

   # LSTM configuration
   LSTM_HIDDEN_SIZE: 128
   LSTM_NUM_LAYERS: 3
   LSTM_EPOCHS: 300
   LSTM_LOOKBACK: 365  # 1 year memory

   # Data split
   CALIBRATION_PERIOD: [2010, 2017]  # 8 years training+val
   VALIDATION_PERIOD: [2018, 2020]   # 3 years testing

Run:

.. code-block:: bash

   symfluence workflow run --config config.yaml

   # Training takes ~30 min to several hours (GPU recommended)

LSTM with Snow Modeling
-----------------------

Multi-task learning for flow and snow:

.. code-block:: yaml

   # config.yaml
   HYDROLOGICAL_MODEL: LSTM

   # Enable snow prediction
   LSTM_USE_SNOW: true

   # Requires snow observations for training
   # Place SWE data in: <project_dir>/observations/snow/

   # Adjust network for added complexity
   LSTM_HIDDEN_SIZE: 256  # Larger network
   LSTM_NUM_LAYERS: 4
   LSTM_EPOCHS: 500

Transfer Learning Workflow
--------------------------

Train on data-rich basin, transfer to ungauged basin:

.. code-block:: yaml

   # Step 1: Train on donor basin
   # config_donor.yaml
   DOMAIN_NAME: donor_basin
   HYDROLOGICAL_MODEL: LSTM

   # (standard config...)

   # Train and save model
   LSTM_SAVE_MODEL: true  # Saves to models/LSTM/donor_basin.pt

.. code-block:: yaml

   # Step 2: Apply to ungauged basin
   # config_ungauged.yaml
   DOMAIN_NAME: ungauged_basin
   HYDROLOGICAL_MODEL: LSTM

   # Load pre-trained model
   LSTM_LOAD: true
   LSTM_PRETRAINED_MODEL: ../donor_basin/models/LSTM/donor_basin.pt

   # Fine-tune (if any observations available)
   LSTM_EPOCHS: 50  # Short fine-tuning

Distributed LSTM
---------------

LSTM on distributed domain:

.. code-block:: yaml

   # Semi-distributed LSTM
   DOMAIN_DEFINITION_METHOD: delineate
   POUR_POINT_COORDS: [-115.0, 51.0]

   HYDROLOGICAL_MODEL: LSTM

   # Train LSTM per subcatchment
   LSTM_SPATIAL_MODE: distributed

   # Optional: Train through routing
   LSTM_TRAIN_THROUGH_ROUTING: true
   ROUTING_MODEL: mizuRoute

Hyperparameter Tuning
=====================

Key Hyperparameters
------------------

**1. Hidden Size**

.. code-block:: yaml

   LSTM_HIDDEN_SIZE: [64, 128, 256, 512]

- Larger = more capacity, more parameters
- 128-256 typical for single basin
- Use 64 for simple basins, 512 for complex multi-variate

**2. Number of Layers**

.. code-block:: yaml

   LSTM_NUM_LAYERS: [1, 2, 3, 4]

- More layers = deeper temporal abstractions
- Diminishing returns after 3-4 layers
- Increases training time significantly

**3. Lookback Window**

.. code-block:: yaml

   LSTM_LOOKBACK: [365, 540, 730]

- Longer = captures longer memory (e.g., snowmelt lag)
- 365 (1 year): standard
- 730 (2 years): for systems with multi-year memory
- Increases memory usage

**4. Dropout**

.. code-block:: yaml

   LSTM_DROPOUT: [0.0, 0.1, 0.2, 0.3, 0.5]

- Higher = more regularization (prevents overfitting)
- 0.2-0.3 typical
- Use 0.4-0.5 if overfitting evident

**5. Learning Rate**

.. code-block:: yaml

   LSTM_LEARNING_RATE: [0.0001, 0.0005, 0.001, 0.005]

- Default 0.001 works well usually
- Lower if training unstable
- Higher for faster convergence (but risk instability)

Hyperparameter Optimization
---------------------------

Use SYMFLUENCE's optimization framework:

.. code-block:: yaml

   # Treat hyperparameters as "parameters" to optimize
   OPTIMIZATION_ALGORITHM: RandomSearch  # Or GridSearch

   # Define search space
   LSTM_HIDDEN_SIZE: [64, 128, 256]
   LSTM_NUM_LAYERS: [2, 3, 4]
   LSTM_DROPOUT: [0.1, 0.2, 0.3]

   # Evaluate on validation set
   OPTIMIZATION_METRIC: KGE
   OPTIMIZATION_MAX_ITERATIONS: 20  # Try 20 combinations

Known Limitations
================

1. **Data Hungry:**

   - Requires substantial training data (5-10+ years)
   - Performance degrades with sparse observations
   - Difficult to apply in data-scarce regions

2. **Black Box:**

   - Limited physical interpretability
   - Hard to diagnose failure modes
   - Cannot extrapolate to untested conditions

3. **Extrapolation Issues:**

   - Poor performance outside training distribution
   - Climate change applications questionable
   - Struggles with unprecedented events

4. **Computational Cost (Training):**

   - Training can take hours to days
   - GPU recommended for practical use
   - Hyperparameter tuning multiplies cost

5. **Overfitting Risk:**

   - Can memorize training data without generalizing
   - Requires careful regularization (dropout, L2)
   - Validation set essential

6. **Ungauged Basins:**

   - Transfer learning less robust than process-based models
   - Requires similar donor basins
   - Physical similarity metrics needed

Troubleshooting
==============

Common Issues
-------------

**Error: "PyTorch not found"**

.. code-block:: bash

   # Install PyTorch
   pip install torch torchvision

   # For GPU support (recommended):
   # Visit: https://pytorch.org/get-started/locally/
   # Select your system and install appropriate version

**Error: "CUDA not available" (GPU)**

.. code-block:: bash

   # Check if GPU is detected
   python
   >>> import torch
   >>> print(torch.cuda.is_available())
   False  # GPU not available

   # Solution: Install CUDA-enabled PyTorch or use CPU

**Poor training performance (high loss)**

1. **Check data quality:**

   .. code-block:: python

      import pandas as pd
      df = pd.read_csv('basin_forcing.csv')
      print(df.isnull().sum())  # Check for NaNs
      print(df.describe())       # Check for outliers

2. **Normalize inputs:**

   - SYMFLUENCE auto-normalizes, but verify
   - Precipitation, temp should be standardized

3. **Adjust learning rate:**

   .. code-block:: yaml

      LSTM_LEARNING_RATE: 0.0001  # Lower if unstable

4. **Increase training data**

**Overfitting (train NSE >> val NSE)**

.. code-block:: yaml

   # Increase regularization
   LSTM_DROPOUT: 0.4           # From 0.2
   LSTM_L2_REGULARIZATION: 1e-5  # From 1e-6

   # Reduce model capacity
   LSTM_HIDDEN_SIZE: 64        # From 128
   LSTM_NUM_LAYERS: 2          # From 3

   # More training data
   CALIBRATION_PERIOD: [2005, 2017]  # Extend period

**Underfitting (both train and val NSE low)**

.. code-block:: yaml

   # Increase model capacity
   LSTM_HIDDEN_SIZE: 256       # From 128
   LSTM_NUM_LAYERS: 4          # From 3

   # Longer lookback
   LSTM_LOOKBACK: 730          # From 365

   # Train longer
   LSTM_EPOCHS: 500            # From 300

**NaN predictions**

1. Check for missing forcing data
2. Verify normalization didn't produce NaNs
3. Reduce learning rate (training instability)
4. Check for extreme values in forcing

**Out of memory error**

.. code-block:: yaml

   # Reduce batch size
   LSTM_BATCH_SIZE: 32         # From 64

   # Reduce lookback
   LSTM_LOOKBACK: 365          # From 730

   # Reduce hidden size
   LSTM_HIDDEN_SIZE: 64        # From 128

**Slow training**

.. code-block:: bash

   # Use GPU
   # Check GPU usage:
   nvidia-smi

   # If GPU underutilized, increase batch size
   LSTM_BATCH_SIZE: 128

Performance Tips
===============

Improving Accuracy
------------------

1. **More data:**

   - 10+ years > 5 years
   - Include diverse climate years

2. **Better features:**

   - Add radiation, humidity (not just precip/temp)
   - Include static catchment attributes
   - Derived features (e.g., temperature indices)

3. **Attention mechanism:**

   .. code-block:: yaml

      LSTM_USE_ATTENTION: true

4. **Multi-task learning:**

   .. code-block:: yaml

      LSTM_USE_SNOW: true  # If SWE data available

5. **Ensemble:**

   - Train 5-10 models with different random seeds
   - Average predictions

Speeding Up Training
-------------------

1. **Use GPU:**

   - 10-50x faster than CPU
   - Essential for hyperparameter tuning

2. **Smaller architecture:**

   .. code-block:: yaml

      LSTM_HIDDEN_SIZE: 64
      LSTM_NUM_LAYERS: 2

3. **Early stopping:**

   .. code-block:: yaml

      LSTM_LEARNING_PATIENCE: 20  # Stop if no improvement in 20 epochs

4. **Larger batches (if memory allows):**

   .. code-block:: yaml

      LSTM_BATCH_SIZE: 128

Deployment
---------

After training, LSTM is very fast:

.. code-block:: python

   # Inference (prediction) takes milliseconds per timestep
   # Ideal for operational forecasting

   # Export model for production:
   model_path = 'models/LSTM/best_model.pt'
   # Load and run in real-time forecasting system

Comparing with Physics-Based Models
===================================

**Advantages of LSTM:**

- No manual parameter calibration
- Fast execution (after training)
- Can capture complex nonlinear relationships
- Good for basins with unclear processes

**Advantages of Physics-Based (SUMMA, FUSE, GR):**

- Physically interpretable
- Better extrapolation to unseen conditions
- More robust to climate change
- Work with less data (1-3 years sometimes sufficient)

**Recommendation:**

Use LSTM alongside physics-based models:

.. code-block:: yaml

   # Multi-model ensemble
   HYDROLOGICAL_MODEL: [SUMMA, FUSE, GR, LSTM]

   # Compare performance
   # Use best performer or weighted ensemble

Additional Resources
===================

**Deep Learning for Hydrology:**

- Kratzert et al. (2018): "Rainfall-Runoff modeling using LSTMs"
  https://doi.org/10.5194/hess-22-6005-2018

- Kratzert et al. (2019): "Towards learning universal hydrological representations"
  https://doi.org/10.5194/hess-23-5089-2019

- Feng et al. (2020): "Differentiable, Learnable, Regionalized Process-Based Models"
  https://doi.org/10.1029/2020WR027710

**PyTorch:**

- Official tutorials: https://pytorch.org/tutorials/
- LSTM tutorial: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

**SYMFLUENCE-specific:**

- :doc:`../configuration`: LSTM parameter reference
- :doc:`../calibration`: Training strategies
- :doc:`model_summa`: Comparison with physics-based models
- :doc:`../troubleshooting`: General troubleshooting

**Datasets:**

- CAMELS: https://ral.ucar.edu/solutions/products/camels
- CAMELS-US, CAMELS-GB, CAMELS-AUS, CAMELS-BR, etc.
- LamaH: https://doi.org/10.5194/essd-13-4529-2021

**Example Notebooks:**

.. code-block:: bash

   # LSTM examples
   symfluence examples list | grep LSTM

**Neuralhydrology Package (Alternative Implementation):**

- https://github.com/neuralhydrology/neuralhydrology
- Kratzert et al.'s research package
- More features, LSTM variants (EA-LSTM, MC-LSTM)
