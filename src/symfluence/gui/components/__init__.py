"""
GUI component widgets for the SYMFLUENCE Panel application.
"""

from .map_view import MapView
from .command_panel import CommandPanel
from .attributes_panel import AttributesPanel
from .domain_panel import DomainPanel
from .discretization_panel import DiscretizationPanel
from .data_panel import DataPanel
from .forcings_panel import ForcingsPanel
from .model_panel import ModelPanel
from .calibration_panel import CalibrationPanel
from .analysis_panel import AnalysisPanel
from .config_editor import ConfigEditor
from .workflow_runner import WorkflowRunner
from .log_viewer import LogViewer
from .results_viewer import ResultsViewer
from .iterative_run_panel import IterativeRunPanel
from .gauge_setup_panel import GaugeSetupPanel
from .domain_browser import DomainBrowser

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
