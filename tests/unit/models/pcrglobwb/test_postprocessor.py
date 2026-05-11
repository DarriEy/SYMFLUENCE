"""Tests for PCR-GLOBWB postprocessor."""


class TestPCRGLOBWBPostProcessor:
    """Tests for PCR-GLOBWB postprocessor."""

    def test_postprocessor_can_be_imported(self):
        from symfluence.models.pcrglobwb.postprocessor import PCRGLOBWBPostProcessor
        assert PCRGLOBWBPostProcessor is not None

    def test_postprocessor_registered_with_registry(self):
        import symfluence.models.pcrglobwb  # noqa: F401
        from symfluence.models.registry import ModelRegistry
        assert 'PCRGLOBWB' in ModelRegistry._postprocessors

    def test_model_name(self):
        from symfluence.models.pcrglobwb.postprocessor import PCRGLOBWBPostProcessor
        assert PCRGLOBWBPostProcessor.model_name == "PCRGLOBWB"

    def test_streamflow_unit(self):
        from symfluence.models.pcrglobwb.postprocessor import PCRGLOBWBPostProcessor
        assert PCRGLOBWBPostProcessor.streamflow_unit == "cms"

    def test_streamflow_variable(self):
        from symfluence.models.pcrglobwb.postprocessor import PCRGLOBWBPostProcessor
        assert PCRGLOBWBPostProcessor.streamflow_variable == "discharge"

    def test_output_file_pattern(self):
        from symfluence.models.pcrglobwb.postprocessor import PCRGLOBWBPostProcessor
        assert 'discharge' in PCRGLOBWBPostProcessor.output_file_pattern
        assert 'output.nc' in PCRGLOBWBPostProcessor.output_file_pattern
