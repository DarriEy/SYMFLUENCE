"""
dRoute Model Preprocessor.

Handles spatial preprocessing and network topology generation for the dRoute routing model.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any
import xarray as xr

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelPreProcessor
from symfluence.geospatial.geometry_utils import GeospatialUtilsMixin

try:
    import droute
    HAS_DROUTE = True
except ImportError:
    HAS_DROUTE = False


@ModelRegistry.register_preprocessor('DROUTE')
class DRoutePreProcessor(BaseModelPreProcessor, GeospatialUtilsMixin):
    """
    Preprocessor for the dRoute differentiable routing model.

    Handles spatial preprocessing and network topology generation for dRoute,
    including conversion of river network shapefiles to dRoute's Network object
    format and pickle serialization for fast loading during model execution.

    dRoute is a differentiable routing model that can be integrated with
    neural network training for end-to-end hydrological modeling.

    Attributes:
        setup_dir (Path): Directory for dRoute configuration files.
        project_dir (Path): Root project directory.
        domain_name (str): Name of the modeling domain.
    """

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "dRoute"

    def __init__(self, config: Dict[str, Any], logger: Any):
        # Initialize base class (handles standard paths and directories)
        super().__init__(config, logger)
        
        self.logger.debug(f"DRoutePreProcessor initialized. Default setup_dir: {self.setup_dir}")
        
        # Ensure setup directory exists
        if not self.setup_dir.exists():
            self.logger.info(f"Creating dRoute setup directory: {self.setup_dir}")
            self.setup_dir.mkdir(parents=True, exist_ok=True)

    def run_preprocessing(self):
        """Run the complete dRoute preprocessing workflow."""
        self.logger.info("Starting dRoute spatial preprocessing")
        
        if not HAS_DROUTE:
            self.logger.error("dRoute not found. Please install it to use dRoute routing.")
            return

        # 1. Create or ensure topology.nc exists (can reuse mizuRoute topology if already created)
        # We'll use the one created by mizuRoute preprocessor if available, 
        # or create a new one if it doesn't exist.
        topology_path = self.setup_dir / self.config_dict.get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
        
        if not topology_path.exists():
            self.logger.info("Creating network topology file for dRoute")
            self._create_topology_file(topology_path)
        else:
            self.logger.info(f"Using existing topology file: {topology_path}")

        # 2. Convert topology.nc to dRoute Network object and pickle it for fast loading
        self._create_droute_network_pickle(topology_path)

        self.logger.info("dRoute spatial preprocessing completed")

    def _create_topology_file(self, output_path: Path):
        """
        Create a mizuRoute-compatible topology.nc file.
        This leverages existing logic from MizuRoutePreProcessor if possible,
        but for simplicity here we'll implement a basic version.
        """
        from symfluence.models.mizuroute import MizuRoutePreProcessor
        mizu_pre = MizuRoutePreProcessor(self.config_dict, self.logger)
        # Override setup_dir to point to dRoute setup dir
        mizu_pre.setup_dir = self.setup_dir
        mizu_pre.create_network_topology_file()

    def _create_droute_network_pickle(self, topology_path: Path):
        """
        Load topology.nc and create a droute.Network object, then save as pickle.
        """
        self.logger.info(f"Building dRoute Network from {topology_path}")
        
        try:
            ds = xr.open_dataset(topology_path)
            
            # Extract data
            seg_ids = ds['segId'].values
            down_seg_ids = ds['downSegId'].values
            slopes = ds['slope'].values
            lengths = ds['length'].values
            
            # Use Manning's n from config or default
            default_n = self.config_dict.get('DROUTE_DEFAULT_MANNING_N', 0.035)
            if 'mann_n' in ds:
                mann_n = ds['mann_n'].values
            else:
                mann_n = np.full(len(seg_ids), default_n)
            
            hru_ids = ds['hruId'].values
            hru_to_seg = ds['hruToSegId'].values
            hru_areas = ds['area'].values  # m²
            
            ds.close()
            
            n_segs = len(seg_ids)
            seg_id_to_idx = {int(seg_id): i for i, seg_id in enumerate(seg_ids)}
            
            # Build upstream connectivity map
            upstream_map: dict[int, list[int]] = {i: [] for i in range(n_segs)}
            for i, down_id in enumerate(down_seg_ids):
                down_id_int = int(down_id)
                if down_id_int in seg_id_to_idx:
                    down_idx = seg_id_to_idx[down_id_int]
                    upstream_map[down_idx].append(i)
            
            # Build network
            network = droute.Network()
            
            for i in range(n_segs):
                reach = droute.Reach()
                reach.id = i
                reach.length = float(lengths[i])
                reach.slope = max(float(slopes[i]), 0.0001)
                reach.manning_n = float(mann_n[i])
                
                # Default geometry (power law) - can be made configurable
                reach.geometry.width_coef = self.config_dict.get('DROUTE_WIDTH_COEF', 7.2)
                reach.geometry.width_exp = self.config_dict.get('DROUTE_WIDTH_EXP', 0.5)
                reach.geometry.depth_coef = self.config_dict.get('DROUTE_DEPTH_COEF', 0.27)
                reach.geometry.depth_exp = self.config_dict.get('DROUTE_DEPTH_EXP', 0.3)
                
                reach.upstream_junction_id = i
                down_id = int(down_seg_ids[i])
                if down_id in seg_id_to_idx:
                    reach.downstream_junction_id = seg_id_to_idx[down_id]
                else:
                    reach.downstream_junction_id = -1  # Outlet
                
                network.add_reach(reach)
            
            for i in range(n_segs):
                junc = droute.Junction()
                junc.id = i
                junc.upstream_reach_ids = upstream_map[i]
                junc.downstream_reach_ids = [i]
                network.add_junction(junc)
            
            network.build_topology()
            
            # Create HRU ID to segment index mapping
            hru_to_seg_idx = {}
            for i, hru_id in enumerate(hru_ids):
                seg_id = int(hru_to_seg[i])
                if seg_id in seg_id_to_idx:
                    hru_to_seg_idx[int(hru_id)] = seg_id_to_idx[seg_id]
            
            # Create area array indexed by reach index
            seg_areas = np.zeros(n_segs)
            for i, hru_id in enumerate(hru_ids):
                seg_id = int(hru_to_seg[i])
                if seg_id in seg_id_to_idx:
                    seg_idx = seg_id_to_idx[seg_id]
                    seg_areas[seg_idx] = hru_areas[i]

            # Find outlet index
            outlet_idx = 0
            for i, down_id in enumerate(down_seg_ids):
                if int(down_id) not in seg_id_to_idx:
                    outlet_idx = i
                    break

            # Metadata for the runner
            network_data = {
                'network': network,
                'seg_areas': seg_areas,
                'outlet_idx': outlet_idx,
                'hru_to_seg_idx': hru_to_seg_idx,
                'hru_ids': hru_ids,
                'seg_ids': seg_ids
            }
            
            pickle_path = self.setup_dir / 'dRoute_network.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(network_data, f)
                
            self.logger.info(f"dRoute Network pickled to {pickle_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating dRoute network pickle: {e}")
            raise
