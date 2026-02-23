"""GUI component widgets for the SYMFLUENCE Panel application."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis_panel import AnalysisPanel
    from .attributes_panel import AttributesPanel
    from .calibration_panel import CalibrationPanel
    from .command_panel import CommandPanel
    from .config_editor import ConfigEditor
    from .data_panel import DataPanel
    from .discretization_panel import DiscretizationPanel
    from .domain_browser import DomainBrowser
    from .domain_panel import DomainPanel
    from .forcings_panel import ForcingsPanel
    from .gauge_setup_panel import GaugeSetupPanel
    from .iterative_run_panel import IterativeRunPanel
    from .log_viewer import LogViewer
    from .map_view import MapView
    from .model_panel import ModelPanel
    from .results_viewer import ResultsViewer
    from .workflow_runner import WorkflowRunner

_LAZY_IMPORTS = {
    'MapView': '.map_view',
    'CommandPanel': '.command_panel',
    'AttributesPanel': '.attributes_panel',
    'DomainPanel': '.domain_panel',
    'DiscretizationPanel': '.discretization_panel',
    'DataPanel': '.data_panel',
    'ForcingsPanel': '.forcings_panel',
    'ModelPanel': '.model_panel',
    'CalibrationPanel': '.calibration_panel',
    'AnalysisPanel': '.analysis_panel',
    'ConfigEditor': '.config_editor',
    'WorkflowRunner': '.workflow_runner',
    'LogViewer': '.log_viewer',
    'ResultsViewer': '.results_viewer',
    'IterativeRunPanel': '.iterative_run_panel',
    'GaugeSetupPanel': '.gauge_setup_panel',
    'DomainBrowser': '.domain_browser',
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'MapView',
    'CommandPanel',
    'AttributesPanel',
    'DomainPanel',
    'DiscretizationPanel',
    'DataPanel',
    'ForcingsPanel',
    'ModelPanel',
    'CalibrationPanel',
    'AnalysisPanel',
    'ConfigEditor',
    'WorkflowRunner',
    'LogViewer',
    'ResultsViewer',
    'IterativeRunPanel',
    'GaugeSetupPanel',
    'DomainBrowser',
]
