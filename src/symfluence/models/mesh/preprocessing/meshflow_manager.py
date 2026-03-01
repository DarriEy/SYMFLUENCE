# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MESH Meshflow Manager

Handles meshflow execution for MESH preprocessing.
Meshflow is the single required pathway - no fallbacks.
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict


def _patch_meshflow_network_bug():
    """
    Patch a bug in meshflow's network.py extract_rank_next function.

    The bug is in meshflow/utility/network.py at line ~129 where:
        next_var[k] = r
    should be:
        next_var[k] = r[0]

    The issue is that np.where() returns an array, even with a single match,
    so assigning it directly to a scalar array element causes:
        ValueError: setting an array element with a sequence.

    This patch monkey-patches the extract_rank_next function with a corrected
    version that properly extracts the scalar value from the np.where result.

    This fix should be removed once meshflow releases a corrected version.
    See: https://github.com/CH-Earth/meshflow (issue to be filed)
    """
    try:
        import numpy as np
        from meshflow.utility import network as meshflow_network

        def _patched_extract_rank_next(seg, ds_seg, outlet_value=-9999):
            """
            Patched version of extract_rank_next that fixes the array assignment bug.

            This is a corrected copy of meshflow.utility.network.extract_rank_next.
            """
            from meshflow.utility.network import _adjust_ids

            # extracting numpy array out of input iterables
            seg_arr = np.array(seg)
            ds_seg_arr = np.array(ds_seg)

            # re-order ids to match MESH's requirements
            seg_id, to_segment = _adjust_ids(seg_arr, ds_seg_arr)

            # Count the number of outlets
            outlets = np.where(to_segment == outlet_value)[0]

            # Search over to extract the subbasins drain into each outlet
            rank_var_id_domain = np.array([]).astype(int)
            outlet_number = np.array([]).astype(int)

            for k in range(len(outlets)):
                # initial step
                seg_id_target = seg_id[outlets[k]]
                # set the rank_var of the outlet
                rank_var_id = outlets[k]

                # find upstream seg_ids draining into the chosen outlet
                while (np.size(seg_id_target) >= 1):
                    if (np.size(seg_id_target) == 1):
                        r = np.where(to_segment == seg_id_target)[0]
                    else:
                        r = np.where(to_segment == seg_id_target[0])[0]
                    # updated the target seg_id
                    seg_id_target = np.append(seg_id_target, seg_id[r])
                    # remove the first searched target
                    seg_id_target = np.delete(seg_id_target, 0, 0)
                    if (len(seg_id_target) == 0):
                        break
                    # update the rank_var_id
                    rank_var_id = np.append(rank_var_id, r)
                rank_var_id = np.flip(rank_var_id)
                if (np.size(rank_var_id) > 1):
                    outlet_number = np.append(
                        outlet_number,
                        (k) * np.ones((len(rank_var_id), 1)).astype(int)
                    )
                else:
                    outlet_number = np.append(outlet_number, (k))
                rank_var_id_domain = np.append(rank_var_id_domain, rank_var_id)
                rank_var_id = []

            # reorder seg_id and to_segment
            seg_id = seg_id[rank_var_id_domain]
            to_segment = to_segment[rank_var_id_domain]

            # rearrange outlets to be consistent with MESH outlet structure
            na = len(rank_var_id_domain)
            fid1 = np.where(to_segment != outlet_value)[0]
            fid2 = np.where(to_segment == outlet_value)[0]
            fid = np.append(fid1, fid2)

            rank_var_id_domain = rank_var_id_domain[fid]
            seg_id = seg_id[fid]
            to_segment = to_segment[fid]
            outlet_number = outlet_number[fid]

            # construct rank_var and next_var variables
            next_var = np.zeros(na).astype(np.int32)

            for k in range(na):
                if (to_segment[k] != outlet_value):
                    r = np.where(to_segment[k] == seg_id)[0] + 1
                    # BUG FIX: Extract scalar from array (original bug: next_var[k] = r)
                    next_var[k] = r[0] if len(r) > 0 else 0
                else:
                    next_var[k] = 0

            # Construct rank_var from 1:na
            rank_var = np.arange(1, na + 1).astype(np.int32)

            return rank_var, next_var, seg_id, to_segment

        # Apply the patch
        meshflow_network.extract_rank_next = _patched_extract_rank_next

        # Log that patch was applied
        logger = logging.getLogger(__name__)
        logger.debug("Applied runtime patch for meshflow network.py extract_rank_next bug")

    except Exception as e:  # noqa: BLE001 — model execution resilience
        # If patching fails, log warning but don't prevent import
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to apply meshflow network.py patch: {e}")


# Import meshflow and apply patch
try:
    from meshflow.core import MESHWorkflow
    MESHFLOW_AVAILABLE = True
    _meshflow_import_error = None
    # Apply runtime patch for meshflow bug
    _patch_meshflow_network_bug()
except Exception as e:  # noqa: BLE001 — model execution resilience
    MESHFLOW_AVAILABLE = False
    MESHWorkflow = None
    _meshflow_import_error = str(e)
    # Use debug level since this is an optional dependency most users don't need
    logging.getLogger(__name__).debug(f"meshflow import failed; MESH preprocessing disabled: {e}")


class MESHFlowManager:
    """
    Manages meshflow execution for MESH preprocessing.

    Meshflow is the required preprocessing pathway. If meshflow fails,
    preprocessing fails - there are no fallback strategies.
    """

    def __init__(
        self,
        forcing_dir: Path,
        config: Dict[str, Any],
        logger: logging.Logger = None
    ):
        """
        Initialize meshflow manager.

        Args:
            forcing_dir: Directory for MESH files
            config: Meshflow configuration dictionary
            logger: Optional logger instance
        """
        self.forcing_dir = forcing_dir
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def is_available() -> bool:
        """Check if meshflow is available."""
        return MESHFLOW_AVAILABLE

    def run(self, prepare_forcing_callback=None, postprocess_callback=None) -> None:
        """
        Run meshflow to generate MESH input files.

        Args:
            prepare_forcing_callback: Callback for direct forcing preparation
            postprocess_callback: Callback for post-processing output

        Raises:
            ModelExecutionError: If meshflow is not available or fails.
        """
        if not MESHFLOW_AVAILABLE:
            from symfluence.core.exceptions import ModelExecutionError
            detail = f" ({_meshflow_import_error})" if _meshflow_import_error else ""
            raise ModelExecutionError(
                f"meshflow is not available{detail}. Install with:\n"
                "  pip install git+https://github.com/kasra-keshavarz/hydrant.git\n"
                "  pip install git+https://github.com/CH-Earth/meshflow.git@main\n"
                "Note: hydrant must be installed BEFORE meshflow to avoid pulling "
                "the wrong 'hydrant' package from PyPI."
            )

        self._check_required_files()
        self._clean_output_files()

        try:
            import meshflow
            self.logger.info(f"Using meshflow version: {getattr(meshflow, '__version__', 'unknown')}")

            self.logger.info("Initializing MESHWorkflow with config")
            workflow = MESHWorkflow(**self.config)

            self.logger.info("Running meshflow workflow")
            workflow.run(save_path=str(self.forcing_dir))
            workflow.save(output_dir=str(self.forcing_dir))
            self.logger.info("Meshflow workflow completed successfully")

            # Call prepare_forcing_callback to generate/fix forcing data
            # This is needed because meshflow's CDO-based forcing prep has issues
            # with frequency inference on multi-file datasets
            if prepare_forcing_callback:
                self.logger.info("Running forcing preparation callback")
                prepare_forcing_callback()

            # Post-process
            if postprocess_callback:
                postprocess_callback()

            self.logger.info("Meshflow preprocessing completed successfully")

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Meshflow preprocessing failed: {e}")
            self.logger.debug(traceback.format_exc())
            from symfluence.core.exceptions import ModelExecutionError
            raise ModelExecutionError(f"Meshflow preprocessing failed: {e}") from e

    def _check_required_files(self) -> None:
        """Check that required input files exist."""
        from symfluence.core.exceptions import ConfigurationError

        required_files = [self.config.get('riv'), self.config.get('cat')]
        missing_files = [f for f in required_files if f and not Path(f).exists()]

        if missing_files:
            raise ConfigurationError(
                f"MESH preprocessing requires these files: {missing_files}. "
                "Run geospatial preprocessing first."
            )

    def _clean_output_files(self) -> None:
        """Clean existing output files."""
        output_files = [
            self.forcing_dir / "MESH_forcing.nc",
            self.forcing_dir / "MESH_drainage_database.nc",
        ]
        for f in output_files:
            if f.exists():
                f.unlink()
