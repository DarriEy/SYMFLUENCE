# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base panel class for model comparison visualizations.

Provides a common interface and shared utilities for all panel types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BasePanel(ABC):
    """Abstract base class for visualization panels.

    Panels are reusable visualization components that can be composed
    into larger multi-panel figures. Each panel is responsible for
    rendering a specific type of visualization onto a matplotlib axis.

    Attributes:
        plot_config: Configuration object with plot styling settings
        logger: Logger instance for diagnostic messages

    Usage:
        class MyPanel(BasePanel):
            def render(self, ax, data):
                ax.plot(data['x'], data['y'])
                self._apply_styling(ax, title='My Plot')
    """

    # Default color palette for models
    MODEL_COLORS = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
    ]

    def __init__(
        self,
        plot_config: Any,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the panel.

        Args:
            plot_config: Configuration object with plot styling settings
            logger: Logger instance for diagnostic messages
        """
        self.plot_config = plot_config
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def render(self, ax: Any, data: Dict[str, Any]) -> None:
        """Render the panel onto the given axis.

        Args:
            ax: Matplotlib axis to render onto
            data: Dictionary containing data needed for rendering.
                  The exact structure depends on the panel type.

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement render()")

    def _apply_styling(
        self,
        ax: Any,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend: bool = False,
        legend_loc: str = 'best',
        grid: bool = True
    ) -> None:
        """Apply standard styling to an axis.

        Args:
            ax: Matplotlib axis
            title: Optional title for the plot
            xlabel: Optional x-axis label
            ylabel: Optional y-axis label
            legend: Whether to show legend
            legend_loc: Legend location
            grid: Whether to show grid
        """
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if legend:
            ax.legend(loc=legend_loc, framealpha=0.9)
        if grid:
            ax.grid(True, alpha=0.3)

    def _get_color(self, index: int) -> str:
        """Get color for a model at the given index.

        Args:
            index: Model index (0-based)

        Returns:
            Color string for the model
        """
        return self.MODEL_COLORS[index % len(self.MODEL_COLORS)]

    def _extract_model_name(self, column_name: str) -> str:
        """Extract clean model name from column name.

        Args:
            column_name: Full column name (e.g., 'SUMMA_discharge_cms')

        Returns:
            Clean model name (e.g., 'SUMMA')
        """
        return column_name.replace('_discharge_cms', '').replace('_discharge', '')

    def _get_valid_data(
        self,
        obs: np.ndarray,
        sim: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get aligned data with NaN values removed.

        Args:
            obs: Observation array
            sim: Simulation array

        Returns:
            Tuple of (clean_obs, clean_sim) with NaN values removed
        """
        valid_mask = ~(np.isnan(obs) | np.isnan(sim))
        return obs[valid_mask], sim[valid_mask]

    def _validate_data(self, data: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that required data keys are present.

        Args:
            data: Data dictionary
            required_keys: List of required keys

        Returns:
            True if all required keys present, False otherwise
        """
        missing = [k for k in required_keys if k not in data]
        if missing:
            self.logger.warning(f"Missing required data keys: {missing}")
            return False
        return True
