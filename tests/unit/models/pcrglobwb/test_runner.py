"""Tests for PCR-GLOBWB runner."""


class TestPCRGLOBWBRunner:
    """Tests for PCR-GLOBWB runner."""

    def test_runner_can_be_imported(self):
        from symfluence.models.pcrglobwb.runner import PCRGLOBWBRunner
        assert PCRGLOBWBRunner is not None

    def test_runner_registered_with_registry(self):
        import symfluence.models.pcrglobwb  # noqa: F401
        from symfluence.models.registry import ModelRegistry
        assert 'PCRGLOBWB' in ModelRegistry._runners

    def test_model_name(self):
        from symfluence.models.pcrglobwb.runner import PCRGLOBWBRunner
        assert PCRGLOBWBRunner.MODEL_NAME == "PCRGLOBWB"

    def test_expected_outputs(self):
        from symfluence.models.pcrglobwb.runner import PCRGLOBWBRunner
        runner = PCRGLOBWBRunner.__new__(PCRGLOBWBRunner)
        outputs = runner._get_expected_outputs()
        assert 'discharge_dailyTot_output.nc' in outputs
