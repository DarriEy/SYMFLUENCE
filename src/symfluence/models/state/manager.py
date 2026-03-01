# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
State manager.

Orchestrates state operations across models and external systems
(e.g. FEWS). Handles serialization for transport, ensemble state
management, and FEWS import/export bridging.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .exceptions import StateError
from .types import ModelState, StateMetadata

logger = logging.getLogger(__name__)


class StateManager:
    """Orchestrates state save/restore across models and external systems."""

    # ------------------------------------------------------------------
    # Serialization for WorkerTask / WorkerResult transport
    # ------------------------------------------------------------------

    @staticmethod
    def serialize(state: ModelState) -> Dict[str, Any]:
        """Serialize a ModelState to a dictionary for transport.

        Args:
            state: The ModelState to serialize.

        Returns:
            A JSON-compatible dictionary.
        """
        return state.to_dict()

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> ModelState:
        """Deserialize a dictionary back into a ModelState.

        Args:
            data: Dictionary produced by serialize().

        Returns:
            Reconstructed ModelState.
        """
        return ModelState.from_dict(data)

    # ------------------------------------------------------------------
    # FEWS bridge
    # ------------------------------------------------------------------

    @staticmethod
    def import_from_fews(
        fews_state_dir: Path,
        model_runner: Any,
    ) -> Optional[ModelState]:
        """Import state from FEWS input directory into a model runner.

        Delegates to the model's own ``save_state`` / ``load_state``
        after copying files from the FEWS directory into the model's
        state directory.

        Args:
            fews_state_dir: FEWS state input directory.
            model_runner: A runner implementing StateCapableMixin.

        Returns:
            The loaded ModelState, or None if nothing to import.
        """
        fews_state_dir = Path(fews_state_dir)
        if not fews_state_dir.is_dir():
            logger.debug("FEWS state dir does not exist: %s", fews_state_dir)
            return None

        pattern = model_runner.get_state_file_pattern()
        state_files = sorted(fews_state_dir.glob(pattern))
        if not state_files:
            logger.debug("No state files matching '%s' in %s", pattern, fews_state_dir)
            return None

        # Copy files into model's state directory
        model_state_dir = model_runner.get_state_directory()
        if model_state_dir is None:
            model_state_dir = fews_state_dir
        else:
            model_state_dir = Path(model_state_dir)
            model_state_dir.mkdir(parents=True, exist_ok=True)
            for src in state_files:
                dst = model_state_dir / src.name
                shutil.copy2(str(src), str(dst))
                logger.info("Copied FEWS state: %s -> %s", src, dst)

        # Build a ModelState from the copied files
        model_name = model_runner._get_model_name() if hasattr(model_runner, '_get_model_name') else 'unknown'
        metadata = StateMetadata(
            model_name=model_name,
            timestamp='',
            format=model_runner.get_state_format(),
            variables=model_runner.get_state_variables(),
        )
        state = ModelState(
            metadata=metadata,
            files=[model_state_dir / f.name for f in state_files],
        )

        model_runner.load_state(state)
        logger.info("Imported FEWS state into %s", model_name)
        return state

    @staticmethod
    def export_to_fews(
        model_runner: Any,
        fews_state_dir: Path,
        timestamp: str = '',
    ) -> Optional[ModelState]:
        """Export model state to FEWS output directory.

        Args:
            model_runner: A runner implementing StateCapableMixin.
            fews_state_dir: FEWS state output directory.
            timestamp: Timestamp label for the state.

        Returns:
            The exported ModelState, or None on failure.
        """
        fews_state_dir = Path(fews_state_dir)
        fews_state_dir.mkdir(parents=True, exist_ok=True)

        # Let the model save its state to a temporary staging area
        staging = fews_state_dir / '.staging'
        staging.mkdir(parents=True, exist_ok=True)

        try:
            state = model_runner.save_state(staging, timestamp)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error("Failed to save model state for FEWS export: %s", e)
            return None

        # Copy state files to FEWS output
        for src in state.files:
            dst = fews_state_dir / src.name
            shutil.copy2(str(src), str(dst))
            logger.info("Exported state to FEWS: %s -> %s", src, dst)

        # Clean up staging
        shutil.rmtree(staging, ignore_errors=True)

        logger.info("Exported state to FEWS dir: %s", fews_state_dir)
        return state

    # ------------------------------------------------------------------
    # Ensemble state management
    # ------------------------------------------------------------------

    @staticmethod
    def save_ensemble(
        states: List[ModelState],
        archive_dir: Path,
    ) -> None:
        """Persist N member states to archive_dir/member_000/, member_001/, etc.

        Args:
            states: List of ModelState objects, one per ensemble member.
            archive_dir: Root directory for the ensemble archive.
        """
        archive_dir = Path(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)

        for i, state in enumerate(states):
            member_dir = archive_dir / f"member_{i:03d}"
            member_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for src in state.files:
                dst = member_dir / src.name
                shutil.copy2(str(src), str(dst))

            # Save arrays as .npz
            if state.arrays:
                np.savez(member_dir / 'state_arrays.npz', **state.arrays)

            logger.debug("Saved ensemble member %d to %s", i, member_dir)

        logger.info("Saved %d ensemble members to %s", len(states), archive_dir)

    @staticmethod
    def load_ensemble(
        archive_dir: Path,
        model_runner: Any,
        n_members: int,
    ) -> List[ModelState]:
        """Restore ensemble states from archive.

        Args:
            archive_dir: Root directory of the ensemble archive.
            model_runner: Runner for metadata context.
            n_members: Number of members to load.

        Returns:
            List of restored ModelState objects.
        """
        archive_dir = Path(archive_dir)
        states: List[ModelState] = []

        model_name = model_runner._get_model_name() if hasattr(model_runner, '_get_model_name') else 'unknown'
        state_format = model_runner.get_state_format()
        state_variables = model_runner.get_state_variables()

        for i in range(n_members):
            member_dir = archive_dir / f"member_{i:03d}"
            if not member_dir.is_dir():
                raise StateError(f"Ensemble member directory missing: {member_dir}")

            # Load files
            pattern = model_runner.get_state_file_pattern()
            files = sorted(member_dir.glob(pattern))

            # Load arrays
            arrays: Dict[str, np.ndarray] = {}
            npz_file = member_dir / 'state_arrays.npz'
            if npz_file.exists():
                with np.load(npz_file) as data:
                    arrays = {k: data[k] for k in data.files}

            metadata = StateMetadata(
                model_name=model_name,
                timestamp='',
                format=state_format,
                variables=state_variables,
                ensemble_member=i,
            )
            states.append(ModelState(metadata=metadata, files=files, arrays=arrays))

        logger.info("Loaded %d ensemble members from %s", len(states), archive_dir)
        return states

    @staticmethod
    def create_ensemble(
        base_state: ModelState,
        n_members: int,
        perturbation_fn: Callable[[ModelState, int], ModelState],
    ) -> List[ModelState]:
        """Create N perturbed copies of a base state.

        Args:
            base_state: The unperturbed base state.
            n_members: Number of ensemble members to create.
            perturbation_fn: Function (base_state, member_idx) -> perturbed ModelState.

        Returns:
            List of N perturbed ModelState objects.
        """
        states = []
        for i in range(n_members):
            perturbed = perturbation_fn(base_state, i)
            states.append(perturbed)

        logger.info("Created ensemble of %d members", n_members)
        return states
