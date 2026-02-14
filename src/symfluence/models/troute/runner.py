"""
TRoute Model Runner.

Manages the execution of the t-route routing model.
Supports two modes:
1. Native nwm_routing subprocess (requires compiled troute Cython extensions)
2. Built-in pure-Python Muskingum-Cunge routing (no external dependencies)

The built-in mode is used as fallback when troute is not installed.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import xarray as xr

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelRunner
from symfluence.models.execution import ModelExecutor
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler


def _check_troute_available() -> bool:
    """Check if the compiled troute Cython package is importable."""
    try:
        import troute  # noqa: F401
        return True
    except ImportError:
        return False


@ModelRegistry.register_runner('TROUTE', method_name='run_troute')
class TRouteRunner(BaseModelRunner, ModelExecutor):  # type: ignore[misc]
    """
    A standalone runner for the t-route model.

    If the compiled troute package is available, delegates to nwm_routing.
    Otherwise, uses a built-in pure-Python Muskingum-Cunge routing kernel
    that reads the same topology file and produces compatible output.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _get_model_name(self) -> str:
        return "TRoute"

    def _should_create_output_dir(self) -> bool:
        return False

    def run_troute(self):
        """
        Prepares runoff data and executes t-route routing.
        """
        self.logger.info("--- Starting t-route Run ---")

        # 1. Prepare runoff file
        runoff_filepath = self._prepare_runoff_file()

        # 2. Set up paths
        settings_path = self.project_dir / 'settings' / 'troute'
        topology_file = settings_path / self.config_dict.get(
            'SETTINGS_TROUTE_TOPOLOGY', 'troute_topology.nc'
        )
        troute_out_path = self.get_experiment_output_dir()
        troute_out_path.mkdir(parents=True, exist_ok=True)

        # 3. Choose execution mode
        if _check_troute_available():
            self.logger.info("Using compiled troute (nwm_routing) for routing.")
            self._run_nwm_routing(settings_path, troute_out_path)
        else:
            self.logger.info(
                "Compiled troute not available. Using built-in Muskingum-Cunge routing."
            )
            self._run_builtin_muskingum_cunge(
                runoff_filepath, topology_file, troute_out_path
            )

        self.logger.info("--- t-route Run Finished ---")
        return troute_out_path

    # ------------------------------------------------------------------
    # Runoff preparation
    # ------------------------------------------------------------------

    def _prepare_runoff_file(self) -> Path:
        """
        Loads the hydrological model output and renames the runoff variable
        to 'q_lateral' as required by t-route.

        Returns:
            Path to the (possibly renamed) runoff file.
        """
        self.logger.info("Preparing runoff file for t-route...")

        source_model = self.config_dict.get('TROUTE_FROM_MODEL', 'SUMMA').upper()
        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        runoff_filepath = (
            self.project_dir
            / f"simulations/{experiment_id}/{source_model}/{experiment_id}_timestep.nc"
        )

        self.verify_required_files(runoff_filepath, context="t-route runoff preparation")

        original_var_config = self.config_dict.get(
            'SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff'
        )
        if original_var_config in ('default', None, ''):
            original_var = 'averageRoutedRunoff'
        else:
            original_var = original_var_config

        self.logger.debug(f"Checking for variable '{original_var}' in {runoff_filepath}")

        with xr.open_dataset(runoff_filepath) as ds:
            if original_var in ds.data_vars:
                self.logger.info(f"Found '{original_var}', renaming to 'q_lateral'.")
                ds_mem = ds.rename({original_var: 'q_lateral'}).load()
            elif 'q_lateral' in ds.data_vars:
                self.logger.info("Runoff variable already named 'q_lateral'.")
                return runoff_filepath
            else:
                self.logger.error(
                    f"Expected runoff variable '{original_var}' not found."
                )
                raise ValueError(f"Runoff variable not found in {runoff_filepath}")

        ds_mem.to_netcdf(runoff_filepath, mode='w', format='NETCDF4')
        ds_mem.close()
        self.logger.info("Runoff variable successfully renamed.")
        return runoff_filepath

    # ------------------------------------------------------------------
    # Mode 1: nwm_routing subprocess
    # ------------------------------------------------------------------

    def _run_nwm_routing(self, settings_path: Path, troute_out_path: Path):
        """Execute routing via nwm_routing subprocess."""
        config_file = self.config_dict.get(
            'SETTINGS_TROUTE_CONFIG_FILE', 'troute_config.yml'
        )
        config_filepath = settings_path / config_file
        log_path = self.get_log_path()
        log_file_path = log_path / "troute_run.log"

        command = [sys.executable, "-m", "nwm_routing", str(config_filepath)]
        self.logger.info(f'Executing t-route command: {" ".join(command)}')

        with symfluence_error_handler(
            "t-route model execution", self.logger, error_type=ModelExecutionError
        ):
            self.execute_model_subprocess(
                command,
                log_file_path,
                success_message=(
                    f"t-route run completed. Log: {log_file_path}"
                ),
            )

    # ------------------------------------------------------------------
    # Mode 2: Built-in pure-Python Muskingum-Cunge
    # ------------------------------------------------------------------

    def _run_builtin_muskingum_cunge(
        self,
        runoff_filepath: Path,
        topology_filepath: Path,
        output_dir: Path,
    ):
        """
        Pure-Python Muskingum-Cunge channel routing.

        Reads:
            - Topology NetCDF: segment connectivity, lengths, slopes, Manning's n
            - Runoff NetCDF: lateral inflows per HRU per timestep

        Writes:
            - troute_output.nc: routed discharge (flow), velocity, depth per segment
        """
        start = time.time()
        dt = float(self.config_dict.get('SETTINGS_TROUTE_DT_SECONDS', 3600))

        # --- Load topology ---
        topo = xr.open_dataset(topology_filepath)
        seg_ids = topo['comid'].values          # segment IDs
        to_node = topo['to_node'].values        # downstream segment ID
        lengths = topo['length'].values         # segment length (m)
        slopes = topo['slope'].values           # segment slope (m/m)
        mannings = topo['n'].values             # Manning's roughness
        hru_to_seg = topo['link_id_hru'].values # HRU → segment mapping
        hru_areas = topo['hru_area_m2'].values  # HRU areas (m²)

        n_seg = len(seg_ids)
        seg_id_to_idx = {int(sid): i for i, sid in enumerate(seg_ids)}

        # Sanitize: clamp slopes away from zero
        slopes = np.maximum(np.abs(slopes), 1e-5)
        lengths = np.maximum(lengths, 1.0)
        mannings = np.where(mannings > 0, mannings, 0.035)

        # Build downstream connectivity: idx → downstream idx (or -1 for outlet)
        downstream = np.full(n_seg, -1, dtype=int)
        for i, tn in enumerate(to_node):
            ds_idx = seg_id_to_idx.get(int(tn), -1)
            if ds_idx != i:  # avoid self-loops
                downstream[i] = ds_idx

        # Compute topological order (upstream → downstream)
        topo_order = self._topological_sort(downstream, n_seg)

        self.logger.info(
            f"Topology loaded: {n_seg} segments, "
            f"dt={dt}s, outlet segments: "
            f"{sum(1 for d in downstream if d == -1)}"
        )

        # --- Load lateral inflows ---
        ds_runoff = xr.open_dataset(runoff_filepath)
        q_lateral_raw = ds_runoff['q_lateral']  # (time, hru)

        # Determine spatial dim
        spatial_dim = None
        for dim in ['hru', 'gru', 'gruId', 'feature_id']:
            if dim in q_lateral_raw.dims:
                spatial_dim = dim
                break
        if spatial_dim is None:
            non_time_dims = [d for d in q_lateral_raw.dims if d != 'time']
            spatial_dim = non_time_dims[0] if non_time_dims else None

        q_lat_vals = q_lateral_raw.values  # (ntime, nhru)
        time_vals = ds_runoff['time'].values
        n_time = len(time_vals)

        self.logger.info(f"Lateral inflows loaded: {n_time} timesteps, {q_lat_vals.shape[-1]} HRUs")

        # Map HRU inflows to segment inflows
        # q_lateral is in m/s (depth rate per unit area) → convert to m³/s
        n_hru = q_lat_vals.shape[-1] if q_lat_vals.ndim > 1 else 1
        if q_lat_vals.ndim == 1:
            q_lat_vals = q_lat_vals.reshape(-1, 1)

        # Build HRU → segment inflow mapping
        q_seg = np.zeros((n_time, n_seg), dtype=np.float64)
        for h in range(min(n_hru, len(hru_to_seg))):
            seg_for_hru = int(hru_to_seg[h])
            seg_idx = seg_id_to_idx.get(seg_for_hru, -1)
            if seg_idx >= 0 and h < len(hru_areas):
                # Convert depth rate (m/s) to volume rate (m³/s)
                q_seg[:, seg_idx] += q_lat_vals[:, h] * hru_areas[h]

        # --- Muskingum-Cunge routing ---
        bw = np.full(n_seg, 10.0)  # default channel width (m)

        # Initialize flow arrays
        q_out = np.zeros((n_time, n_seg), dtype=np.float64)
        v_out = np.zeros((n_time, n_seg), dtype=np.float64)
        d_out = np.zeros((n_time, n_seg), dtype=np.float64)

        # Track reach inflows (lateral + upstream) separately from outflows
        q_in_prev = np.maximum(q_seg[0, :], 1e-6)
        q_out[0, :] = q_in_prev.copy()

        for t in range(1, n_time):
            q_in_t = q_seg[t, :].copy()  # starts as lateral only

            # Accumulate upstream routed outflows in topological order
            for seg_idx in topo_order:
                # Total reach inflow = this segment's lateral + upstream outlets
                q_reach_in = q_in_t[seg_idx]

                # Muskingum-Cunge: Q_out(t) = C1*Q_in(t) + C2*Q_in(t-1) + C3*Q_out(t-1)
                q_in_curr = q_reach_in
                q_in_last = q_in_prev[seg_idx]
                q_out_last = q_out[t - 1, seg_idx]

                q_ref = max(0.5 * (q_in_curr + q_out_last), 1e-6)

                n_val = mannings[seg_idx]
                s0 = slopes[seg_idx]
                w = bw[seg_idx]
                depth = (q_ref * n_val / (w * np.sqrt(s0))) ** 0.6
                depth = max(depth, 0.01)

                velocity = q_ref / (w * depth) if w * depth > 0 else 0.01
                celerity = max((5.0 / 3.0) * velocity, 0.01)

                dx = lengths[seg_idx]
                K = max(dx / celerity, dt * 0.01)

                denom = 2.0 * celerity * s0 * dx
                X = 0.5 * (1.0 - q_ref / denom) if denom > 0 else 0.0
                X = np.clip(X, 0.0, 0.5)

                denom2 = 2.0 * K * (1.0 - X) + dt
                if denom2 > 0:
                    C1 = (dt - 2.0 * K * X) / denom2
                    C2 = (dt + 2.0 * K * X) / denom2
                    C3 = (2.0 * K * (1.0 - X) - dt) / denom2
                else:
                    C1 = C2 = 0.5
                    C3 = 0.0

                # Clamp and normalize for stability
                C1 = max(C1, 0.0)
                C2 = max(C2, 0.0)
                C3 = max(C3, 0.0)
                c_sum = C1 + C2 + C3
                if c_sum > 0:
                    C1 /= c_sum
                    C2 /= c_sum
                    C3 /= c_sum

                q_routed = max(C1 * q_in_curr + C2 * q_in_last + C3 * q_out_last, 0.0)

                q_out[t, seg_idx] = q_routed
                v_out[t, seg_idx] = velocity
                d_out[t, seg_idx] = depth

                # Record this timestep's inflow for next iteration
                q_in_prev[seg_idx] = q_reach_in

                # Pass routed outflow to downstream segment's inflow
                ds_idx = downstream[seg_idx]
                if ds_idx >= 0:
                    q_in_t[ds_idx] += q_routed

        elapsed = time.time() - start
        self.logger.info(f"Muskingum-Cunge routing completed in {elapsed:.1f}s")

        # --- Write output ---
        ds_out = xr.Dataset(
            {
                'flow': (['time', 'feature_id'], q_out),
                'velocity': (['time', 'feature_id'], v_out),
                'depth': (['time', 'feature_id'], d_out),
            },
            coords={
                'time': time_vals,
                'feature_id': seg_ids,
            },
            attrs={
                'title': 't-route Muskingum-Cunge routing output',
                'routing_method': 'muskingum_cunge',
                'dt_seconds': dt,
                'mannings_n_source': 'topology',
                'created_by': 'SYMFLUENCE built-in t-route runner',
            },
        )
        ds_out['flow'].attrs = {'units': 'm3/s', 'long_name': 'Routed discharge'}
        ds_out['velocity'].attrs = {'units': 'm/s', 'long_name': 'Flow velocity'}
        ds_out['depth'].attrs = {'units': 'm', 'long_name': 'Flow depth'}

        out_file = output_dir / 'troute_output.nc'
        ds_out.to_netcdf(out_file, format='NETCDF4')
        ds_out.close()

        # Summary statistics at outlet
        outlet_indices = [i for i in range(n_seg) if downstream[i] == -1]
        if outlet_indices:
            outlet_q = q_out[:, outlet_indices[0]]
            self.logger.info(
                f"Outlet segment {seg_ids[outlet_indices[0]]}: "
                f"mean Q = {np.mean(outlet_q):.3f} m³/s, "
                f"max Q = {np.max(outlet_q):.3f} m³/s"
            )

        self.logger.info(f"Output written to {out_file}")

        topo.close()
        ds_runoff.close()

    @staticmethod
    def _topological_sort(downstream: np.ndarray, n: int) -> list:
        """
        Return segment indices in topological order (headwaters first).
        """
        in_degree = np.zeros(n, dtype=int)
        for i in range(n):
            ds = downstream[i]
            if ds >= 0:
                in_degree[ds] += 1

        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            ds = downstream[node]
            if ds >= 0:
                in_degree[ds] -= 1
                if in_degree[ds] == 0:
                    queue.append(ds)

        # Any remaining nodes (cycles) appended at end
        if len(order) < n:
            remaining = set(range(n)) - set(order)
            order.extend(remaining)

        return order
