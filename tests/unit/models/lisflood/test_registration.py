# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for LISFLOOD model component registration."""
from __future__ import annotations

import pytest

from symfluence.core.registries import R


class TestModelRegistryRegistration:
    """Tests for LISFLOOD components in ModelRegistry."""

    def test_runner_is_registered(self):
        # Trigger registration by importing the module
        import symfluence.models.lisflood  # noqa: F401

        runners = R.runners
        assert "LISFLOOD" in runners

    def test_preprocessor_is_registered(self):
        import symfluence.models.lisflood  # noqa: F401

        preprocessors = R.preprocessors
        assert "LISFLOOD" in preprocessors

    def test_postprocessor_is_registered(self):
        import symfluence.models.lisflood  # noqa: F401

        postprocessors = R.postprocessors
        assert "LISFLOOD" in postprocessors

    def test_config_adapter_is_registered(self):
        import symfluence.models.lisflood  # noqa: F401
        from symfluence.core.registries import Registries as R

        assert "LISFLOOD" in R.config_adapters

    def test_result_extractor_is_importable(self):
        from symfluence.models.lisflood.extractor import LisfloodResultExtractor

        assert LisfloodResultExtractor is not None

    def test_result_extractor_registered_via_manifest(self):
        import symfluence.models.lisflood  # noqa: F401
        from symfluence.core.registries import Registries as R

        assert "LISFLOOD" in R.result_extractors


class TestModelManifest:
    """Tests for LISFLOOD model_manifest registration."""

    def test_manifest_created_lisflood_entry(self):
        import symfluence.models.lisflood  # noqa: F401
        from symfluence.core.registries import Registries as R

        # model_manifest registers into R.preprocessors, R.runners, etc.
        assert "LISFLOOD" in R.preprocessors
        assert "LISFLOOD" in R.runners
        assert "LISFLOOD" in R.result_extractors
        assert "LISFLOOD" in R.config_adapters


class TestBuildInstructions:
    """Tests for LISFLOOD build instructions registration."""

    def test_build_instructions_registered(self):
        # Trigger registration by importing the build_instructions module
        import symfluence.models.lisflood.build_instructions  # noqa: F401
        from symfluence.cli.services import BuildInstructionsRegistry

        instructions = BuildInstructionsRegistry.get_instructions("lisflood")
        assert instructions is not None

    def test_build_instructions_has_description(self):
        import symfluence.models.lisflood.build_instructions  # noqa: F401
        from symfluence.cli.services import BuildInstructionsRegistry

        instructions = BuildInstructionsRegistry.get_instructions("lisflood")
        assert "description" in instructions
        assert "LISFLOOD" in instructions["description"]

    def test_build_instructions_has_install_dir(self):
        import symfluence.models.lisflood.build_instructions  # noqa: F401
        from symfluence.cli.services import BuildInstructionsRegistry

        instructions = BuildInstructionsRegistry.get_instructions("lisflood")
        assert instructions["install_dir"] == "lisflood"

    def test_build_instructions_is_optional(self):
        import symfluence.models.lisflood.build_instructions  # noqa: F401
        from symfluence.cli.services import BuildInstructionsRegistry

        instructions = BuildInstructionsRegistry.get_instructions("lisflood")
        assert instructions.get("optional") is True


class TestCalibrationImports:
    """Tests for LISFLOOD calibration component imports."""

    def test_optimizer_importable(self):
        from symfluence.models.lisflood.calibration.optimizer import LisfloodModelOptimizer

        assert LisfloodModelOptimizer is not None

    def test_parameter_manager_importable(self):
        from symfluence.models.lisflood.calibration.parameter_manager import LisfloodParameterManager

        assert LisfloodParameterManager is not None

    def test_worker_importable(self):
        from symfluence.models.lisflood.calibration.worker import LisfloodWorker

        assert LisfloodWorker is not None

    def test_calibration_package_exports(self):
        from symfluence.models.lisflood.calibration import (
            LisfloodModelOptimizer,
            LisfloodParameterManager,
            LisfloodWorker,
        )

        assert LisfloodModelOptimizer is not None
        assert LisfloodParameterManager is not None
        assert LisfloodWorker is not None


class TestOptimizerRegistry:
    """Tests for LISFLOOD in OptimizerRegistry."""

    def test_optimizer_registry_has_lisflood(self):
        import symfluence.models.lisflood  # noqa: F401

        optimizer_cls = R.optimizers.get("LISFLOOD")
        assert optimizer_cls is not None

    def test_worker_registry_has_lisflood(self):
        import symfluence.models.lisflood  # noqa: F401

        worker_cls = R.workers.get("LISFLOOD")
        assert worker_cls is not None

    def test_parameter_manager_registry_has_lisflood(self):
        import symfluence.models.lisflood  # noqa: F401

        pm_cls = R.parameter_managers.get("LISFLOOD")
        assert pm_cls is not None
