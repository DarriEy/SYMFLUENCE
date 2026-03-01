# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""TUI widget modules."""

from .config_tree import ConfigTreeWidget
from .domain_list import DomainListWidget
from .log_panel import LogPanel
from .metrics_table import MetricsTable
from .run_summary_table import RunSummaryTable
from .slurm_table import SlurmJobTable
from .sparkline import SparklineWidget
from .step_progress import StepProgressWidget

__all__ = [
    'DomainListWidget',
    'RunSummaryTable',
    'StepProgressWidget',
    'MetricsTable',
    'LogPanel',
    'SparklineWidget',
    'ConfigTreeWidget',
    'SlurmJobTable',
]
