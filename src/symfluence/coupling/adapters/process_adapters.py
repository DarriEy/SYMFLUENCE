"""Process-based adapters wrapping SYMFLUENCE external-executable models.

Each adapter wraps a SYMFLUENCE model's runner as a dCoupler ProcessComponent,
marshalling data between PyTorch tensors and the file formats expected by each
external model.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from dcoupler.core.component import (
    FluxDirection,
    FluxSpec,
)
from dcoupler.wrappers.process import ProcessComponent

logger = logging.getLogger(__name__)


class SUMMAProcessComponent(ProcessComponent):
    """Wraps SUMMA executable as a dCoupler ProcessComponent with BMI lifecycle.

    Reuses SYMFLUENCE's SUMMARunner for actual execution.
    """

    def __init__(self, name: str = "summa", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("forcing", "mixed", FluxDirection.INPUT, "hru", 3600,
                     ("time", "hru", "var")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("runoff", "kg/m2/s", FluxDirection.OUTPUT, "hru", 3600,
                     ("time", "hru"), conserved_quantity="water_mass"),
            FluxSpec("soil_drainage", "kg/m2/s", FluxDirection.OUTPUT, "hru", 3600,
                     ("time", "hru"), conserved_quantity="water_mass"),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        try:
            from symfluence.models.summa.runner import SUMMARunner
            self._runner = SUMMARunner(config, logger)
        except ImportError:
            logger.warning("SUMMARunner not available; execute() will fail")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        # SUMMA reads forcing from NetCDF; inputs are pre-written by the preprocessor
        pass

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("SUMMARunner not initialized. Call bmi_initialize first.")
        try:
            success = self._runner.run_summa()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"SUMMA execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            import xarray as xr
            output_dir = Path(self._model_config.get('EXPERIMENT_OUTPUT_SUMMA', work_dir))
            output_files = sorted(output_dir.glob("*_output_*.nc"))
            if not output_files:
                raise FileNotFoundError(f"No SUMMA output files in {output_dir}")

            ds = xr.open_mfdataset(output_files, data_vars='minimal', coords='minimal', compat='override')
            runoff = torch.tensor(
                ds["scalarTotalRunoff"].values.astype(np.float32),
                dtype=torch.float32,
            )
            drainage = torch.tensor(
                ds["scalarSoilDrainage"].values.astype(np.float32),
                dtype=torch.float32,
            )
            ds.close()
            return {"runoff": runoff, "soil_drainage": drainage}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read SUMMA outputs: {e}")
            return {
                "runoff": torch.zeros(1),
                "soil_drainage": torch.zeros(1),
            }


class MizuRouteProcessComponent(ProcessComponent):
    """Wraps mizuRoute as ProcessComponent with time-fix logic.

    Reuses SYMFLUENCE's MizuRouteRunner for actual execution.
    """

    def __init__(self, name: str = "mizuroute", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("lateral_inflow", "m3/s", FluxDirection.INPUT, "reach", 3600,
                     ("time", "reach")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("discharge", "m3/s", FluxDirection.OUTPUT, "reach", 3600,
                     ("time", "reach")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        try:
            from symfluence.models.mizuroute.runner import MizuRouteRunner
            self._runner = MizuRouteRunner(config, logger)
        except ImportError:
            logger.warning("MizuRouteRunner not available")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        pass  # mizuRoute reads SUMMA output directly

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("MizuRouteRunner not initialized")
        try:
            self._runner.fix_time_precision()
            self._runner.sync_control_file_dimensions()
            success = self._runner.run_mizuroute()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"mizuRoute execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            import xarray as xr
            output_dir = Path(self._model_config.get(
                'EXPERIMENT_OUTPUT_MIZUROUTE', work_dir
            ))
            output_files = sorted(output_dir.glob("*.nc"))
            if not output_files:
                raise FileNotFoundError(f"No mizuRoute output in {output_dir}")

            ds = xr.open_mfdataset(output_files, data_vars='minimal', coords='minimal', compat='override')
            discharge = torch.tensor(
                ds["IRFroutedRunoff"].values.astype(np.float32),
                dtype=torch.float32,
            )
            ds.close()
            return {"discharge": discharge}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read mizuRoute outputs: {e}")
            return {"discharge": torch.zeros(1)}


class ParFlowProcessComponent(ProcessComponent):
    """Wraps ParFlow as ProcessComponent.

    Reuses SUMMAToParFlowCoupler unit conversion logic.
    Unit conversion: kg/m2/s -> m/hr (factor 3.6)
    """

    KG_M2_S_TO_M_HR = 3.6

    def __init__(self, name: str = "parflow", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("recharge", "m/hr", FluxDirection.INPUT, "grid", 3600,
                     ("time", "cell")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("baseflow", "m3/hr", FluxDirection.OUTPUT, "grid", 3600,
                     ("time", "cell")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        try:
            from symfluence.models.parflow.runner import ParFlowRunner
            self._runner = ParFlowRunner(config, logger)
        except ImportError:
            logger.warning("ParFlowRunner not available")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        if "recharge" in inputs:
            recharge = inputs["recharge"].detach().numpy()
            output_path = work_dir / "recharge.csv"
            with open(output_path, "w") as f:
                f.write("sim_hour,recharge_m_hr\n")
                for i, val in enumerate(recharge):
                    f.write(f"{i},{float(val)}\n")

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("ParFlowRunner not initialized")
        try:
            success = self._runner.run_parflow()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"ParFlow execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            from symfluence.models.parflow.extractor import ParFlowResultExtractor

            output_dir = Path(self._model_config.get('PARFLOW_OUTPUT_DIR', work_dir))
            extractor = ParFlowResultExtractor()
            kwargs = {
                'start_date': self._model_config.get('SIMULATION_START', '2000-01-01'),
                'timestep_hours': float(self._model_config.get('TIMESTEP_HOURS', 1.0)),
            }

            overland = extractor.extract_variable(
                output_dir, 'overland_flow', **kwargs
            )
            subsurface = extractor.extract_variable(
                output_dir, 'subsurface_drainage', **kwargs
            )
            # Combine into total baseflow (m3/hr for FluxSpec units)
            combined = overland * 3600 + subsurface  # overland m3/s→m3/hr; subsurface already m3/hr
            return {"baseflow": torch.tensor(
                combined.values.astype(np.float32), dtype=torch.float32
            )}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read ParFlow outputs: {e}")
            return {"baseflow": torch.zeros(1)}


class MODFLOWProcessComponent(ProcessComponent):
    """Wraps MODFLOW 6 as ProcessComponent.

    Reuses SUMMAToMODFLOWCoupler TAS6 format.
    Unit conversion: kg/m2/s -> m/d (factor 86.4)
    """

    KG_M2_S_TO_M_D = 86.4

    def __init__(self, name: str = "modflow", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("recharge", "m/d", FluxDirection.INPUT, "grid", 86400,
                     ("time", "cell")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("drain_discharge", "m3/d", FluxDirection.OUTPUT, "grid", 86400,
                     ("time", "cell")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        try:
            from symfluence.models.modflow.runner import MODFLOWRunner
            self._runner = MODFLOWRunner(config, logger)
        except ImportError:
            logger.warning("MODFLOWRunner not available")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        if "recharge" in inputs:
            recharge = inputs["recharge"].detach().numpy()
            output_path = work_dir / "recharge_tas6.txt"
            with open(output_path, "w") as f:
                f.write("BEGIN TIMEARRAYDATA\n")
                for i, val in enumerate(recharge):
                    f.write(f"  {float(i):.1f}  {float(val):.8e}\n")
                f.write("END TIMEARRAYDATA\n")

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("MODFLOWRunner not initialized")
        try:
            success = self._runner.run_modflow()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"MODFLOW execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            from symfluence.models.modflow.extractor import MODFLOWResultExtractor

            output_dir = Path(self._model_config.get('MODFLOW_OUTPUT_DIR', work_dir))
            extractor = MODFLOWResultExtractor()
            kwargs = {
                'start_date': self._model_config.get('SIMULATION_START', '2000-01-01'),
                'stress_period_length': float(
                    self._model_config.get('MODFLOW_STRESS_PERIOD_DAYS', 1.0)
                ),
            }

            drain = extractor.extract_variable(
                output_dir, 'drain_discharge', **kwargs
            )
            # drain is in m3/d (matching FluxSpec units)
            return {"drain_discharge": torch.tensor(
                drain.values.astype(np.float32), dtype=torch.float32
            )}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read MODFLOW outputs: {e}")
            return {"drain_discharge": torch.zeros(1)}


class MESHProcessComponent(ProcessComponent):
    """Wraps MESH executable as a dCoupler ProcessComponent."""

    def __init__(self, name: str = "mesh", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("forcing", "mixed", FluxDirection.INPUT, "subbasin", 3600,
                     ("time", "subbasin", "var")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("discharge", "m3/s", FluxDirection.OUTPUT, "subbasin", 86400,
                     ("time",)),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        try:
            from symfluence.models.mesh.runner import MESHRunner
            self._runner = MESHRunner(config, logger)
        except ImportError:
            logger.warning("MESHRunner not available")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        pass  # MESH reads forcing files created by preprocessor

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("MESHRunner not initialized")
        try:
            success = self._runner.run_mesh()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"MESH execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            from symfluence.models.mesh.extractor import MESHResultExtractor

            output_dir = Path(self._model_config.get('EXPERIMENT_OUTPUT_MESH', work_dir))
            extractor = MESHResultExtractor('MESH')
            kwargs = {
                'start_date': self._model_config.get('SIMULATION_START', '2001-01-01'),
                'aggregate': 'daily',
            }

            # Try Basin_average_water_balance.csv first (preferred for lumped)
            basin_wb = output_dir / 'Basin_average_water_balance.csv'
            if basin_wb.exists():
                discharge = extractor.extract_variable(
                    basin_wb, 'streamflow', **kwargs
                )
            else:
                # Fall back to MESH_output_streamflow.csv or GRU_water_balance
                discharge = extractor.extract_variable(
                    output_dir, 'streamflow', **kwargs
                )
            return {"discharge": torch.tensor(
                discharge.values.astype(np.float32), dtype=torch.float32
            )}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read MESH outputs: {e}")
            return {"discharge": torch.zeros(1)}


class TRouteProcessComponent(ProcessComponent):
    """Wraps t-route (NOAA OWP) as ProcessComponent for channel routing.

    Reuses SYMFLUENCE's TRouteRunner for actual execution.
    """

    def __init__(self, name: str = "troute", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("lateral_inflow", "m3/s", FluxDirection.INPUT, "reach", 3600,
                     ("time", "reach")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("discharge", "m3/s", FluxDirection.OUTPUT, "reach", 3600,
                     ("time", "reach")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        try:
            from symfluence.models.troute.runner import TRouteRunner
            self._runner = TRouteRunner(config, logger)
        except ImportError:
            logger.warning("TRouteRunner not available")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        pass  # t-route reads upstream model output directly

    def execute(self, work_dir: Path) -> int:
        if self._runner is None:
            raise RuntimeError("TRouteRunner not initialized")
        try:
            success = self._runner.run_troute()
            return 0 if success else 1
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"t-route execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            import xarray as xr
            output_dir = Path(self._model_config.get(
                'EXPERIMENT_OUTPUT_TROUTE', work_dir
            ))
            # Check for both built-in output and nwm_routing output
            output_files = (
                sorted(output_dir.glob("troute_output.nc"))
                + sorted(output_dir.glob("*flowveldepth*.nc"))
                + sorted(output_dir.glob("nex-troute-out.nc"))
            )
            if not output_files:
                raise FileNotFoundError(f"No t-route output in {output_dir}")

            ds = xr.open_dataset(output_files[0])
            for var in ('flow', 'streamflow', 'discharge', 'q_lateral'):
                if var in ds:
                    discharge = torch.tensor(
                        ds[var].values.astype(np.float32),
                        dtype=torch.float32,
                    )
                    ds.close()
                    return {"discharge": discharge}
            ds.close()
            return {"discharge": torch.zeros(1)}
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read t-route outputs: {e}")
            return {"discharge": torch.zeros(1)}


class CLMProcessComponent(ProcessComponent):
    """Wraps CLM/CTSM executable as a dCoupler ProcessComponent."""

    def __init__(self, name: str = "clm", config: Optional[dict] = None, **kwargs):
        super().__init__(name, **kwargs)
        self._model_config = config or {}
        self._runner = None

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("forcing", "mixed", FluxDirection.INPUT, "grid", 3600,
                     ("time", "grid", "var")),
        ]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [
            FluxSpec("runoff", "mm/s", FluxDirection.OUTPUT, "grid", 3600,
                     ("time", "grid"), conserved_quantity="water_mass"),
            FluxSpec("evapotranspiration", "mm/s", FluxDirection.OUTPUT, "grid", 3600,
                     ("time", "grid")),
        ]

    def bmi_initialize(self, config: dict) -> None:
        self._model_config = config
        try:
            from symfluence.models.clm.runner import CLMRunner
            self._runner = CLMRunner(config, logger)
        except ImportError:
            logger.warning("CLMRunner not available; execute() will use raw subprocess")
        self._state = self.get_initial_state()

    def write_inputs(self, inputs: Dict[str, torch.Tensor], work_dir: Path) -> None:
        pass  # CLM reads forcing created by preprocessor

    def execute(self, work_dir: Path) -> int:
        if self._runner is not None:
            try:
                result = self._runner.run()
                return 0 if result else 1
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                logger.error(f"CLM execution failed: {e}")
                return 1

        # Fallback to raw subprocess if CLMRunner unavailable
        cesm_exe = self._model_config.get('CLM_CESM_EXE')
        if cesm_exe is None:
            raise RuntimeError("CLM_CESM_EXE not configured and CLMRunner unavailable")
        try:
            result = subprocess.run(
                [str(cesm_exe)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
            )
            return result.returncode
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"CLM execution failed: {e}")
            return 1

    def read_outputs(self, work_dir: Path) -> Dict[str, torch.Tensor]:
        try:
            from symfluence.models.clm.extractor import CLMResultExtractor

            output_dir = Path(self._model_config.get('EXPERIMENT_OUTPUT_CLM', work_dir))
            extractor = CLMResultExtractor()
            kwargs = {
                'catchment_area_km2': self._model_config.get('CATCHMENT_AREA_KM2'),
            }

            runoff = extractor.extract_variable(
                output_dir, 'streamflow', **kwargs
            )
            et = extractor.extract_variable(
                output_dir, 'evapotranspiration', **kwargs
            )
            return {
                "runoff": torch.tensor(
                    runoff.values.astype(np.float32), dtype=torch.float32
                ),
                "evapotranspiration": torch.tensor(
                    et.values.astype(np.float32), dtype=torch.float32
                ),
            }
        except Exception as e:  # noqa: BLE001 — must-not-raise contract
            logger.error(f"Failed to read CLM outputs: {e}")
            return {
                "runoff": torch.zeros(1),
                "evapotranspiration": torch.zeros(1),
            }
