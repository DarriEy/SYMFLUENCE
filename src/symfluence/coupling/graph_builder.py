"""Config-driven CouplingGraph construction for SYMFLUENCE.

Reads SYMFLUENCE configuration and constructs a dCoupler CouplingGraph
with appropriate components, connections, spatial remappers, and unit
conversions.
"""

from __future__ import annotations

import logging
from typing import Optional, Union


from dcoupler.core.graph import CouplingGraph
from dcoupler.core.connection import SpatialRemapper

from symfluence.coupling.bmi_registry import BMIRegistry

logger = logging.getLogger(__name__)


def _get_cfg(config, typed_accessor, default=None):
    """Get config value from typed config or fall back to dict access.

    Supports both typed SymfluenceConfig and plain dict configs.
    """
    if not isinstance(config, dict):
        try:
            value = typed_accessor()
            if value is not None:
                return value
        except (AttributeError, KeyError, TypeError):
            pass
    return default

# Unit conversion factors for common coupling interfaces
UNIT_CONVERSIONS = {
    # SUMMA soil drainage (kg/m2/s) -> ParFlow recharge (m/hr)
    ("SUMMA", "PARFLOW"): 3.6,
    # SUMMA soil drainage (kg/m2/s) -> MODFLOW recharge (m/d)
    ("SUMMA", "MODFLOW"): 86.4,
    # SUMMA runoff (kg/m2/s) -> mizuRoute lateral inflow (m3/s)
    # Actual conversion depends on HRU areas; placeholder 1.0 (remapper handles it)
    ("SUMMA", "MIZUROUTE"): 1.0,
    # SUMMA runoff (kg/m2/s) -> t-route lateral inflow (m3/s)
    ("SUMMA", "TROUTE"): 1.0,
    # Snow-17 rain_plus_melt (mm/dt) -> XAJ precip (mm/dt)
    ("SNOW17", "XAJ"): 1.0,
    ("SNOW17", "XINANJIANG"): 1.0,
    # Snow-17 rain_plus_melt (mm/dt) -> SAC-SMA precip (mm/dt)
    ("SNOW17", "SACSMA"): 1.0,
    ("SNOW17", "SAC-SMA"): 1.0,
}


class CouplingGraphBuilder:
    """Constructs a dCoupler CouplingGraph from SYMFLUENCE configuration.

    The builder reads config keys like HYDROLOGICAL_MODEL, ROUTING_MODEL,
    GROUNDWATER_MODEL, and SNOW_MODULE to determine which components to
    create and how to connect them.

    Usage::

        builder = CouplingGraphBuilder()
        graph = builder.build(config_dict)
        outputs = graph.forward(external_inputs, n_timesteps=365, dt=86400)
    """

    def __init__(self, registry: Optional[BMIRegistry] = None):
        self._registry = registry or BMIRegistry()

    def build(self, config: Union[dict, object]) -> CouplingGraph:
        """Build a CouplingGraph from SYMFLUENCE config.

        Accepts either a typed SymfluenceConfig or a plain dict.

        Config keys used:
            HYDROLOGICAL_MODEL: Primary land surface model (SUMMA, MESH, CLM, XAJ, SACSMA)
            ROUTING_MODEL: Optional routing model (MIZUROUTE)
            GROUNDWATER_MODEL: Optional GW model (PARFLOW, MODFLOW)
            SNOW_MODULE: Optional snow module (SNOW17) for coupled JAX models
            COUPLING_MODE: 'sequential' (default), 'per_timestep', 'dcoupler'
            CONSERVATION_MODE: Optional 'check' or 'enforce'

        Args:
            config: Typed SymfluenceConfig or plain dict with config keys.

        Returns:
            Configured CouplingGraph with all components and connections.
        """
        # Build a dict view for component constructors that still expect dict
        config_dict = config if isinstance(config, dict) else (
            config.to_dict() if hasattr(config, 'to_dict') else {}
        )

        conservation_mode = _get_cfg(
            config, lambda: config.model.conservation_mode, default=None
        ) or config_dict.get("CONSERVATION_MODE")
        graph = CouplingGraph(conservation_mode=conservation_mode)

        model_name = _get_cfg(
            config, lambda: config.model.hydrological_model,
            default=config_dict.get("HYDROLOGICAL_MODEL", "")
        )
        model_name = (model_name if isinstance(model_name, str) else str(model_name or "")).upper()

        routing_name = _get_cfg(
            config, lambda: config.model.routing_model,
            default=config_dict.get("ROUTING_MODEL", "")
        )
        routing_name = (routing_name or "").upper()

        gw_name = _get_cfg(
            config, lambda: config.model.groundwater_model,
            default=config_dict.get("GROUNDWATER_MODEL", "")
        )
        gw_name = (gw_name or "").upper()

        snow_module = _get_cfg(
            config, lambda: config.model.snow_module,
            default=config_dict.get("SNOW_MODULE", "")
        )
        snow_module = (snow_module or "").upper()

        if not model_name:
            raise ValueError("HYDROLOGICAL_MODEL must be specified in config")

        # Create primary land surface component
        land_cls = self._registry.get(model_name)
        land = land_cls(name="land", config=config_dict)
        graph.add_component(land)
        logger.info(f"Added land component: {model_name}")

        # Snow module coupling (for JAX models)
        if snow_module and snow_module in ("SNOW17",):
            snow_cls = self._registry.get(snow_module)
            snow = snow_cls(name="snow", config=config_dict)
            graph.add_component(snow)

            unit_conv = UNIT_CONVERSIONS.get((snow_module, model_name), 1.0)
            graph.connect(
                "snow", "rain_plus_melt",
                "land", "precip",
                unit_conversion=unit_conv if unit_conv != 1.0 else None,
            )
            logger.info(f"Connected snow ({snow_module}) -> land ({model_name})")

        # Groundwater coupling
        if gw_name:
            gw_cls = self._registry.get(gw_name)
            gw = gw_cls(name="groundwater", config=config_dict)
            graph.add_component(gw)

            unit_conv = self._get_unit_conversion(model_name, gw_name)
            source_flux = "soil_drainage" if model_name == "SUMMA" else "runoff"
            # Provide identity remapper for hru->grid spatial mismatch
            spatial_remap = self._build_identity_remapper_if_needed(
                graph.components["land"], source_flux,
                gw, "recharge",
            )
            graph.connect(
                "land", source_flux,
                "groundwater", "recharge",
                unit_conversion=unit_conv,
                spatial_remap=spatial_remap,
            )
            logger.info(f"Connected land -> groundwater ({gw_name}), conv={unit_conv}")

        # Routing coupling
        if routing_name:
            route_cls = self._registry.get(routing_name)
            route = route_cls(name="routing", config=config_dict)
            graph.add_component(route)

            remapper = self._build_remapper(config)
            unit_conv = self._get_unit_conversion(model_name, routing_name)
            # Need spatial remap for hru->reach mismatch
            if remapper is None:
                remapper = self._build_identity_remapper_if_needed(
                    graph.components["land"], "runoff",
                    route, "lateral_inflow",
                )
            graph.connect(
                "land", "runoff",
                "routing", "lateral_inflow",
                spatial_remap=remapper,
                unit_conversion=unit_conv,
            )
            logger.info(f"Connected land -> routing ({routing_name})")

        # Validate
        warnings = graph.validate()
        for w in warnings:
            logger.warning(f"Graph validation: {w}")

        return graph

    def _build_identity_remapper_if_needed(
        self, source_comp, source_flux_name: str, target_comp, target_flux_name: str,
    ) -> Optional[SpatialRemapper]:
        """Return an identity SpatialRemapper if spatial types differ, else None."""
        src_spec = next(
            (f for f in source_comp.output_fluxes if f.name == source_flux_name), None
        )
        tgt_spec = next(
            (f for f in target_comp.input_fluxes if f.name == target_flux_name), None
        )
        if src_spec and tgt_spec and src_spec.spatial_type != tgt_spec.spatial_type:
            # Use a 1x1 identity remapper as placeholder
            return SpatialRemapper.identity(1)
        return None

    def _get_unit_conversion(self, source_model: str, target_model: str) -> Optional[float]:
        """Look up unit conversion factor between two models."""
        key = (source_model.upper(), target_model.upper())
        conv = UNIT_CONVERSIONS.get(key)
        if conv is None:
            logger.warning(
                f"No unit conversion defined for {source_model} -> {target_model}; "
                f"assuming no conversion needed"
            )
        return conv

    def _build_remapper(self, config: Union[dict, object]) -> Optional[SpatialRemapper]:
        """Build spatial remapper from config if mapping data is available.

        Looks for HRU-to-reach mapping in the topology file.
        """
        if isinstance(config, dict):
            topology_file = config.get("TOPOLOGY_FILE")
        else:
            topology_file = _get_cfg(config, lambda: config.model.topology_file, default=None)
        if topology_file is None:
            return None

        try:
            import xarray as xr
            ds = xr.open_dataset(topology_file)
            source_ids = ds.get("hruId", ds.get("hru_id"))
            target_ids = ds.get("segId", ds.get("seg_id"))
            hru_to_seg = ds.get("hruToSegId", ds.get("hru_to_seg_id"))

            if source_ids is None or target_ids is None or hru_to_seg is None:
                logger.warning("Topology file missing required variables")
                ds.close()
                return None

            areas = ds.get("hruArea", ds.get("hru_area"))
            area_vals = areas.values if areas is not None else None

            remapper = SpatialRemapper.from_mapping_table(
                source_ids=source_ids.values,
                target_ids=target_ids.values,
                hru_to_seg=hru_to_seg.values,
                areas=area_vals,
            )
            ds.close()
            return remapper
        except Exception as e:
            logger.warning(f"Failed to build spatial remapper: {e}")
            return None
