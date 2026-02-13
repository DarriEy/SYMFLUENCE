"""
Snow-17 BMI-like Interface.

BMI-inspired (Basic Model Interface) wrapper for Snow-17 model coupling.
Any rainfall-runoff model can use this to get snow-adjusted precipitation.

Usage:
    snow = Snow17BMI(params={'SCF': 1.1, 'MFMAX': 1.5}, latitude=51.17)
    snow.initialize()
    for t in range(n_timesteps):
        rain_plus_melt = snow.update(precip[t], temp[t], day_of_year[t])
        # Feed rain_plus_melt to any rainfall-runoff model
    snow.finalize()
"""

from typing import Dict, List, Optional

import numpy as np

from .parameters import (
    Snow17State,
    Snow17Params,
    SNOW17_DEFAULTS,
    DEFAULT_ADC,
    params_dict_to_namedtuple,
    create_initial_state,
)
from .model import snow17_step, snow17_simulate_numpy


class Snow17BMI:
    """BMI-inspired interface for Snow-17 model.

    Provides initialize/update/finalize lifecycle for imperative coupling
    with any rainfall-runoff model.
    """

    def __init__(
        self,
        params: Optional[Dict[str, float]] = None,
        latitude: float = 45.0,
        si: float = 100.0,
        dt: float = 1.0,
    ):
        """Create Snow17BMI instance.

        Args:
            params: Parameter dictionary (uses defaults if None)
            latitude: Catchment latitude (degrees, positive=North)
            si: SWE threshold for 100% areal coverage (mm)
            dt: Timestep in days
        """
        self._params_dict = params if params is not None else SNOW17_DEFAULTS.copy()
        self._latitude = latitude
        self._si = si
        self._dt = dt
        self._params: Optional[Snow17Params] = None
        self._state: Optional[Snow17State] = None
        self._adc = DEFAULT_ADC.copy()
        self._initialized = False
        self._last_outflow = 0.0

    def initialize(
        self,
        params: Optional[Dict[str, float]] = None,
        latitude: Optional[float] = None,
        si: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> None:
        """Initialize model state and parameters.

        Args:
            params: Override parameter dictionary
            latitude: Override latitude
            si: Override SWE threshold
            dt: Override timestep
        """
        if params is not None:
            self._params_dict = params
        if latitude is not None:
            self._latitude = latitude
        if si is not None:
            self._si = si
        if dt is not None:
            self._dt = dt

        self._params = params_dict_to_namedtuple(self._params_dict, use_jax=False)
        self._state = create_initial_state(use_jax=False)
        self._initialized = True
        self._last_outflow = 0.0

    def update(self, precip: float, temp: float, day_of_year: int) -> float:
        """Execute one timestep and return rain_plus_melt.

        Args:
            precip: Total precipitation (mm/dt)
            temp: Air temperature (C)
            day_of_year: Julian day (1-366)

        Returns:
            Rain plus snowmelt (mm/dt)
        """
        if not self._initialized:
            self.initialize()

        self._state, outflow = snow17_step(
            np.float64(precip), np.float64(temp), self._dt,
            self._state, self._params, np.float64(day_of_year),
            self._latitude, self._si, self._adc, xp=np,
        )
        self._last_outflow = float(outflow)
        return self._last_outflow

    def update_batch(
        self,
        precip: np.ndarray,
        temp: np.ndarray,
        day_of_year: np.ndarray,
    ) -> np.ndarray:
        """Execute multiple timesteps and return rain_plus_melt array.

        Args:
            precip: Precipitation array (mm/dt)
            temp: Temperature array (C)
            day_of_year: Day of year array (1-366)

        Returns:
            Rain plus snowmelt array (mm/dt)
        """
        if not self._initialized:
            self.initialize()

        rain_plus_melt, self._state = snow17_simulate_numpy(
            precip, temp, day_of_year,
            self._params, self._state,
            self._latitude, self._si, self._dt, self._adc,
        )
        if len(rain_plus_melt) > 0:
            self._last_outflow = float(rain_plus_melt[-1])
        return rain_plus_melt

    def get_state(self) -> Snow17State:
        """Get current model state."""
        if self._state is None:
            return create_initial_state(use_jax=False)
        return self._state

    def set_state(self, state: Snow17State) -> None:
        """Set model state."""
        self._state = state

    def get_output_var_names(self) -> List[str]:
        """Get output variable names."""
        return ['rain_plus_melt', 'swe', 'w_i', 'w_q', 'deficit']

    def get_value(self, name: str) -> float:
        """Get current value of an output variable.

        Args:
            name: Variable name

        Returns:
            Current value
        """
        if self._state is None:
            return 0.0
        if name == 'rain_plus_melt':
            return self._last_outflow
        elif name == 'swe':
            return float(self._state.w_i + self._state.w_q)
        elif name == 'w_i':
            return float(self._state.w_i)
        elif name == 'w_q':
            return float(self._state.w_q)
        elif name == 'deficit':
            return float(self._state.deficit)
        raise ValueError(f"Unknown variable: {name}")

    def finalize(self) -> None:
        """Finalize model (reset state)."""
        self._state = None
        self._initialized = False
        self._last_outflow = 0.0
