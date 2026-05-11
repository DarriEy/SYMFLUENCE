"""Tests for CWatM runner."""


class TestCWatMRunner:
    """Tests for CWatM runner."""

    def test_runner_can_be_imported(self):
        from symfluence.models.cwatm.runner import CWatMRunner
        assert CWatMRunner is not None

    def test_runner_registered_with_registry(self):
        import symfluence.models.cwatm  # noqa: F401
        from symfluence.models.registry import ModelRegistry
        assert 'CWATM' in ModelRegistry._runners

    def test_model_name(self):
        from symfluence.models.cwatm.runner import CWatMRunner
        assert CWatMRunner.MODEL_NAME == "CWATM"

    def test_expected_outputs(self):
        from symfluence.models.cwatm.runner import CWatMRunner
        runner = CWatMRunner.__new__(CWatMRunner)
        outputs = runner._get_expected_outputs()
        assert 'discharge_daily.nc' in outputs
