"""DCouplerWorker: BaseWorker implementation that delegates to dCoupler CouplingGraph.

Bridges dCoupler's PyTorch-based training loop with SYMFLUENCE's BaseWorker
calibration interface (DDS, Ostrich, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from symfluence.coupling.bmi_registry import BMIRegistry
from symfluence.coupling.graph_builder import CouplingGraphBuilder
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.optimization.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('DCOUPLER')
class DCouplerWorker(BaseWorker):
    """BaseWorker implementation that delegates to a dCoupler CouplingGraph.

    This worker can operate in two modes:

    1. **Standard mode** (process-based models): Applies parameters to files,
       executes external models through the graph, reads outputs.
    2. **Gradient mode** (JAX models): Uses PyTorch autograd through the graph
       for gradient-based optimization.

    The worker constructs the CouplingGraph once on initialization and reuses
    it for all evaluations.
    """

    def __init__(self, config: dict, logger_instance: Optional[logging.Logger] = None):
        super().__init__(config, logger_instance or logger)
        self._registry = BMIRegistry()
        self._builder = CouplingGraphBuilder(registry=self._registry)
        self._graph = None
        self._external_inputs = None
        self._n_timesteps = None
        self._dt = None

    @property
    def graph(self):
        """Lazy-initialize the coupling graph from config."""
        if self._graph is None:
            self._graph = self._builder.build(self.config)
        return self._graph

    def supports_native_gradients(self) -> bool:
        """True if all components in the graph are differentiable."""
        from dcoupler.core.component import GradientMethod
        return all(
            comp.gradient_method != GradientMethod.NONE
            for comp in self.graph.components.values()
        )

    def set_external_inputs(
        self,
        external_inputs: Dict[str, Dict[str, torch.Tensor]],
        n_timesteps: int,
        dt: float,
    ) -> None:
        """Set the forcing data for graph forward passes.

        Call this before evaluate() when using the graph in calibration loops.

        Args:
            external_inputs: Dict of {component_name: {flux_name: tensor}}
            n_timesteps: Number of timesteps to simulate
            dt: Timestep size in seconds
        """
        self._external_inputs = external_inputs
        self._n_timesteps = n_timesteps
        self._dt = dt

    def apply_parameters(
        self, params: Dict[str, float], settings_dir: Path, **kwargs
    ) -> bool:
        """Apply parameters to graph components.

        For JAX/autograd components, sets the raw nn.Parameters.
        For process components, writes parameter files to settings_dir.
        """
        try:
            for comp_name, comp in self.graph.components.items():
                comp_params = {
                    k: v for k, v in params.items()
                    if any(s.name == k for s in comp.parameters)
                }
                if not comp_params:
                    continue

                # For JAX components: set raw parameters via sigmoid inverse
                from dcoupler.wrappers.jax import JAXComponent
                if isinstance(comp, JAXComponent):
                    for i, spec in enumerate(comp.parameters):
                        if spec.name in comp_params:
                            val = comp_params[spec.name]
                            lo, hi = spec.lower_bound, spec.upper_bound
                            # Inverse sigmoid to get raw value
                            normalized = (val - lo) / (hi - lo)
                            normalized = max(min(normalized, 0.999), 0.001)
                            raw = np.log(normalized / (1 - normalized))
                            comp._raw_params[i].data = torch.tensor(
                                raw, dtype=torch.float32
                            )

                # For process components: delegate to model-specific parameter writing
                from dcoupler.wrappers.process import ProcessComponent
                if isinstance(comp, ProcessComponent):
                    self._apply_process_params(comp, comp_params, settings_dir)

            return True
        except Exception as e:
            self.logger.error(f"Failed to apply parameters: {e}")
            return False

    def _apply_process_params(
        self, comp, params: Dict[str, float], settings_dir: str
    ) -> None:
        """Write parameters to files for process-based components.

        Process models receive their parameters through model-specific
        preprocessors that write config/parameter files before execution.
        This method logs which parameters are being passed through but
        does not duplicate the preprocessor's work.
        """
        if params:
            self.logger.debug(
                f"Process component '{comp.name}' received params "
                f"{list(params.keys())}. Parameter application for process "
                "models is handled by model-specific preprocessors."
            )

    def run_model(
        self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs
    ) -> bool:
        """Execute the coupling graph forward pass."""
        try:
            # For process-only graphs, external_inputs aren't needed
            # (ProcessComponents handle their own I/O from disk)
            external_inputs: Dict[str, Any] = self._external_inputs
            n_timesteps = self._n_timesteps
            dt = self._dt

            if external_inputs is None:
                external_inputs = {}
                n_timesteps = n_timesteps or 1
                dt = dt or 86400.0

            outputs = self.graph.forward(
                external_inputs=external_inputs,
                n_timesteps=n_timesteps,
                dt=dt,
            )

            # Store outputs for metric calculation
            self._last_outputs = outputs

            # Save outputs to disk
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            for comp_name, comp_outputs in outputs.items():
                for flux_name, tensor in comp_outputs.items():
                    save_path = output_path / f"{comp_name}_{flux_name}.pt"
                    torch.save(tensor, save_path)

            return True
        except Exception as e:
            self.logger.error(f"Graph forward pass failed: {e}")
            return False

    def calculate_metrics(
        self, output_dir: Path, config: Dict[str, Any], **kwargs
    ) -> Dict[str, float]:
        """Calculate calibration metrics from graph outputs.

        Uses the stored outputs from the last run_model() call.
        Falls back to reading saved tensors from output_dir.
        """
        try:
            outputs = getattr(self, "_last_outputs", None)
            if outputs is None:
                # Try loading from disk
                output_path = Path(output_dir)
                outputs = {}
                for pt_file in output_path.glob("*.pt"):
                    parts = pt_file.stem.split("_", 1)
                    if len(parts) == 2:
                        comp_name, flux_name = parts
                        outputs.setdefault(comp_name, {})[flux_name] = torch.load(
                            pt_file, weights_only=True
                        )

            # Extract simulated discharge
            sim = self._extract_simulation(outputs, config)
            if sim is None:
                return {"KGE": -999.0}

            # Load observations
            obs = self._load_observations(config)
            if obs is None:
                return {"KGE": -999.0}

            # Apply warmup period
            warmup = int(config.get('WARMUP_DAYS', 0))
            if warmup > 0:
                sim = sim[warmup:]
                obs = obs[warmup:]

            # Calculate metrics
            metrics = self._compute_metrics(sim, obs, config)
            return metrics

        except Exception as e:
            self.logger.error(f"Metric calculation failed: {e}")
            return {"KGE": -999.0}

    def _extract_simulation(
        self, outputs: Dict, config: dict
    ) -> Optional[torch.Tensor]:
        """Extract the primary simulation variable from graph outputs."""
        # Look for discharge in routing, then runoff in land
        for comp_name in ("routing", "land"):
            if comp_name in outputs:
                for var in ("discharge", "runoff"):
                    if var in outputs[comp_name]:
                        return outputs[comp_name][var]
        return None

    def _load_observations(self, config: dict) -> Optional[torch.Tensor]:
        """Load observation data for metric calculation."""
        obs_file = config.get("OBSERVATIONS_FILE")
        if obs_file is None:
            return None
        try:
            import pandas as pd
            df = pd.read_csv(obs_file)
            return torch.tensor(df.iloc[:, 1].values, dtype=torch.float32)
        except Exception as e:
            self.logger.error(f"Failed to load observations: {e}")
            return None

    def _compute_metrics(
        self,
        sim: torch.Tensor,
        obs: torch.Tensor,
        config: dict,
    ) -> Dict[str, float]:
        """Compute KGE and other metrics."""
        # Align lengths
        min_len = min(len(sim), len(obs))
        sim = sim[:min_len].detach().cpu()
        obs = obs[:min_len]

        # Remove NaN
        valid = ~(torch.isnan(sim) | torch.isnan(obs))
        sim = sim[valid]
        obs = obs[valid]

        if len(sim) < 10:
            return {"KGE": -999.0}

        # KGE
        sim_mean = sim.mean()
        obs_mean = obs.mean()
        sim_std = sim.std()
        obs_std = obs.std()

        if obs_std < 1e-10:
            return {"KGE": -999.0}

        r = torch.corrcoef(torch.stack([sim, obs]))[0, 1]
        alpha = sim_std / obs_std
        beta = sim_mean / obs_mean

        kge = 1.0 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # NSE
        ss_res = torch.sum((sim - obs) ** 2)
        ss_tot = torch.sum((obs - obs_mean) ** 2)
        nse = 1.0 - ss_res / (ss_tot + 1e-10)

        return {
            "KGE": float(kge.item()),
            "NSE": float(nse.item()),
            "r": float(r.item()),
            "alpha": float(alpha.item()),
            "beta": float(beta.item()),
        }

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = "kge",
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """Evaluate with gradient computation for differentiable models.

        Only works when all graph components support autograd.

        Returns:
            Tuple of (score, gradients_dict). Gradients is None if unavailable.
        """
        if not self.supports_native_gradients():
            raise RuntimeError(
                "Not all components support native gradients. "
                "Use evaluate() for non-differentiable components."
            )

        # Apply parameters
        self.apply_parameters(params, Path())

        # Forward pass with grad tracking
        for p in self.graph.get_all_parameters():
            p.requires_grad_(True)

        outputs = self.graph.forward(
            external_inputs=self._external_inputs,
            n_timesteps=self._n_timesteps,
            dt=self._dt,
        )

        # Compute differentiable loss
        from dcoupler.losses.hydrological import kge_loss, nse_loss
        sim = self._extract_simulation(outputs, self.config)
        obs = self._load_observations(self.config)

        loss_fn = kge_loss if metric == "kge" else nse_loss
        loss = loss_fn(sim, obs)

        # Backward pass
        loss.backward()

        # Collect gradients
        gradients = {}
        for comp_name, comp in self.graph.components.items():
            for spec, param in zip(comp.parameters, comp.get_torch_parameters()):
                if param.grad is not None:
                    gradients[f"{comp_name}.{spec.name}"] = param.grad.detach().numpy()

        return float(loss.item()), gradients if gradients else None
