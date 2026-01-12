"""
T-Route Model Calibration Module.

Provides calibration infrastructure for the T-Route (NOAA OWP Routing) model,
supporting channel and reservoir routing parameter optimization.

Components:
    optimizer: T-Route-specific calibration optimizer for routing parameters

The calibration system supports:
- Muskingum-Cunge routing parameters
- Diffusive wave routing parameters
- Reservoir routing parameters
- Channel Manning's roughness optimization
- Cross-section geometry calibration
"""
