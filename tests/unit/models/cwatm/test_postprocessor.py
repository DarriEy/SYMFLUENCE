from __future__ import annotations

from symfluence.core.registries import R

"""Tests for CWatM postprocessor."""


class TestCWatMPostProcessor:
    """Tests for CWatM postprocessor."""

    def test_postprocessor_can_be_imported(self):
        from symfluence.models.cwatm.postprocessor import CWatMPostProcessor
        assert CWatMPostProcessor is not None

    def test_postprocessor_registered_with_registry(self):
        import symfluence.models.cwatm  # noqa: F401
        assert 'CWATM' in R.postprocessors

    def test_model_name(self):
        from symfluence.models.cwatm.postprocessor import CWatMPostProcessor
        assert CWatMPostProcessor.model_name == "CWATM"

    def test_streamflow_unit(self):
        from symfluence.models.cwatm.postprocessor import CWatMPostProcessor
        assert CWatMPostProcessor.streamflow_unit == "cms"

    def test_streamflow_variable(self):
        from symfluence.models.cwatm.postprocessor import CWatMPostProcessor
        assert CWatMPostProcessor.streamflow_variable == "discharge"
