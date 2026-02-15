"""
Variable ID mapper for FEWS <-> SYMFLUENCE translation.

Supports three tiers of mapping:
1. Explicit entries (inline config or YAML file)
2. Auto-detect via VariableStandardizer
3. Pass-through (names match)
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import xarray as xr

from .config import FEWSConfig, IDMapEntry
from .exceptions import IDMappingError

logger = logging.getLogger(__name__)

# Common FEWS parameterId -> SYMFLUENCE standard name mappings
_AUTO_MAP: Dict[str, str] = {
    "P.obs": "pptrate",
    "P.forecast": "pptrate",
    "P": "pptrate",
    "T.obs": "airtemp",
    "T.forecast": "airtemp",
    "T": "airtemp",
    "RH": "relhum",
    "SW": "SWRadAtm",
    "LW": "LWRadAtm",
    "WS": "windspd",
    "SP": "airpres",
    "Q.obs": "discharge",
    "Q.sim": "discharge",
    "H.obs": "water_level",
    "H.sim": "water_level",
    "ET": "pet",
    "E.act": "aet",
}


class IDMapper:
    """Bidirectional variable name/unit translator between FEWS and SYMFLUENCE.

    Args:
        config: FEWSConfig with inline id_map and id_map_file
    """

    def __init__(self, config: FEWSConfig) -> None:
        self._fews_to_sym: Dict[str, IDMapEntry] = {}
        self._sym_to_fews: Dict[str, IDMapEntry] = {}

        # Tier 1: explicit inline entries
        for entry in config.id_map:
            self._fews_to_sym[entry.fews_id] = entry
            self._sym_to_fews[entry.symfluence_id] = entry

        # Tier 1b: YAML file entries
        if config.id_map_file:
            self._load_yaml(Path(config.id_map_file))

        # Tier 2: auto-detect common forcings
        if config.auto_id_map:
            self._add_auto_map()

    def _load_yaml(self, path: Path) -> None:
        """Load ID mappings from a YAML file."""
        if not path.is_file():
            logger.warning("ID map file not found: %s", path)
            return

        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            raise IDMappingError(f"Failed to load ID map YAML {path}: {exc}") from exc

        if not isinstance(data, list):
            raise IDMappingError(f"ID map YAML must be a list of mappings, got {type(data).__name__}")

        for item in data:
            entry = IDMapEntry(**item)
            if entry.fews_id not in self._fews_to_sym:
                self._fews_to_sym[entry.fews_id] = entry
            if entry.symfluence_id not in self._sym_to_fews:
                self._sym_to_fews[entry.symfluence_id] = entry

    def _add_auto_map(self) -> None:
        """Add auto-detected common variable mappings (Tier 2)."""
        for fews_id, sym_id in _AUTO_MAP.items():
            if fews_id not in self._fews_to_sym:
                entry = IDMapEntry(fews_id=fews_id, symfluence_id=sym_id)
                self._fews_to_sym[fews_id] = entry
            if sym_id not in self._sym_to_fews:
                entry = IDMapEntry(fews_id=fews_id, symfluence_id=sym_id)
                self._sym_to_fews[sym_id] = entry

    def fews_to_symfluence(self, fews_id: str) -> Tuple[str, float, float]:
        """Map a FEWS variable ID to SYMFLUENCE name with conversion.

        Args:
            fews_id: FEWS parameterId

        Returns:
            Tuple of (symfluence_name, conversion_factor, conversion_offset)
        """
        if fews_id in self._fews_to_sym:
            entry = self._fews_to_sym[fews_id]
            return entry.symfluence_id, entry.conversion_factor, entry.conversion_offset
        # Tier 3: pass-through
        return fews_id, 1.0, 0.0

    def symfluence_to_fews(self, sym_id: str) -> Tuple[str, float, float]:
        """Map a SYMFLUENCE variable name to FEWS parameterId with reverse conversion.

        Args:
            sym_id: SYMFLUENCE variable name

        Returns:
            Tuple of (fews_id, inverse_factor, inverse_offset)
        """
        if sym_id in self._sym_to_fews:
            entry = self._sym_to_fews[sym_id]
            factor = entry.conversion_factor
            offset = entry.conversion_offset
            # Reverse conversion: fews = (sym - offset) / factor
            if factor == 0:
                raise IDMappingError(f"Conversion factor is zero for {sym_id}")
            inv_factor = 1.0 / factor
            inv_offset = -offset / factor
            return entry.fews_id, inv_factor, inv_offset
        # Tier 3: pass-through
        return sym_id, 1.0, 0.0

    def rename_dataset_fews_to_sym(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename dataset variables from FEWS IDs to SYMFLUENCE names and apply conversion.

        Args:
            ds: Dataset with FEWS variable names

        Returns:
            Dataset with SYMFLUENCE variable names and converted values
        """
        rename_map: Dict[str, str] = {}
        for var in list(ds.data_vars):
            sym_name, factor, offset = self.fews_to_symfluence(str(var))
            if sym_name != var:
                rename_map[str(var)] = sym_name
            if factor != 1.0 or offset != 0.0:
                ds[var] = ds[var] * factor + offset

        if rename_map:
            ds = ds.rename(rename_map)
        return ds

    def rename_dataset_sym_to_fews(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename dataset variables from SYMFLUENCE names to FEWS IDs and apply reverse conversion.

        Args:
            ds: Dataset with SYMFLUENCE variable names

        Returns:
            Dataset with FEWS variable names and reverse-converted values
        """
        rename_map: Dict[str, str] = {}
        for var in list(ds.data_vars):
            fews_name, inv_factor, inv_offset = self.symfluence_to_fews(str(var))
            if fews_name != var:
                rename_map[str(var)] = fews_name
            if inv_factor != 1.0 or inv_offset != 0.0:
                ds[var] = ds[var] * inv_factor + inv_offset

        if rename_map:
            ds = ds.rename(rename_map)
        return ds
