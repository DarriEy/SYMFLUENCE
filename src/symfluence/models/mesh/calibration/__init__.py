"""
MESH Model Calibration Module.

Provides calibration infrastructure for the MESH (Modélisation Environmentale
Communautaire - Surface Hydrology) land surface and hydrology model.

Components:
    optimizer: MESH-specific calibration optimizer with Ostrich integration
    parameter_manager: Manages CLASS/SVS parameters per GRU in MESH_parameters.ini
    worker: Executes MESH model runs with isolated working directories

The calibration system supports:
- CLASS (Canadian Land Surface Scheme) parameters
- SVS (Soil, Vegetation, and Snow) scheme parameters when enabled
- GRU-specific parameter calibration
- Parallel calibration with process-isolated directories
- MESH 1.5 and 2.0 compatibility
"""
