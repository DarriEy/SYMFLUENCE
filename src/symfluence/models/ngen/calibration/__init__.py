"""
NGEN Model Calibration Module.

Provides calibration infrastructure for the NGEN (Next Generation) modular
hydrological modeling framework, supporting BMI-compliant model calibration.

Components:
    optimizer: NGEN-specific calibration optimizer for multi-model configurations
    parameter_manager: Manages parameters across CFE, PET, Noah-OWP modules
    targets: Defines calibration targets with flexible spatial aggregation
    worker: Executes NGEN model runs via the ngen executable

The calibration system supports:
- CFE (Conceptual Functional Equivalent) parameter calibration
- PET module parameter calibration
- Noah-OWP land surface parameters
- Multi-catchment parallel calibration
- Parameter transfer between catchments
"""
