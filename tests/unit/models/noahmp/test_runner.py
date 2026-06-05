from __future__ import annotations

from symfluence.core.registries import R

"""Tests for Noah-MP runner."""


class TestNoahMPRunner:
    """Tests for Noah-MP runner."""

    def test_runner_can_be_imported(self):
        from symfluence.models.noahmp.runner import NoahMPRunner
        assert NoahMPRunner is not None

    def test_runner_registered_with_registry(self):
        import symfluence.models.noahmp  # noqa: F401
        assert 'NOAHMP' in R.runners

    def test_model_name(self):
        from symfluence.models.noahmp.runner import NoahMPRunner
        assert NoahMPRunner.MODEL_NAME == "NOAHMP"

    def test_expected_outputs(self):
        from symfluence.models.noahmp.runner import NoahMPRunner
        runner = NoahMPRunner.__new__(NoahMPRunner)
        outputs = runner._get_expected_outputs()
        assert 'output.nc' in outputs
