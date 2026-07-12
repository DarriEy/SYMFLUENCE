#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Bow at Banff Figures - Shared Style Configuration
Design: Clean minimal, AGU journal, colorblind-safe (Okabe-Ito palette)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

# =============================================================================
# COLOR PALETTE (Okabe-Ito - CVD-safe)
# =============================================================================

COLORS = {
    # Primary data colors
    'default': '#0072B2',      # Blue - uncalibrated/default model
    'calibrated': '#009E73',   # Bluish green - calibrated model
    'obs': '#E69F00',          # Orange - observations
    'obs_secondary': '#CC79A7', # Reddish purple - secondary observations

    # Additional data colors (if needed)
    'accent1': '#F0E442',      # Yellow
    'accent2': '#56B4E9',      # Sky blue
    'accent3': '#D55E00',      # Vermillion (use sparingly)

    # Structural colors
    'text': '#333333',         # Dark gray for text
    'axis': '#333333',         # Dark gray for axes
    'grid': '#E5E5E5',         # Light gray for gridlines
    'grid_subtle': '#F0F0F0',  # Very light gray
    'shading': '#F8F8F8',      # Barely-there gray for period shading
    'shading_alt': '#FAFAFA',  # Alternative shading

    # Table colors
    'table_header': '#F5F5F5',
    'table_positive': '#E8F5E9',  # Light green for improvements
    'table_negative': '#FFF3E0',  # Light orange for degradation
    'table_neutral': '#FAFAFA',

    # Map colors
    'catchment_fill': '#E3F2FD',   # Very light blue
    'catchment_edge': '#0072B2',   # Blue
    'river': '#64B5F6',            # Light blue
    'terrain_water': '#B3E5FC',
}

# Color aliases for backward compatibility
COLORS['uncal'] = COLORS['default']
COLORS['cal'] = COLORS['calibrated']

# =============================================================================
# TYPOGRAPHY
# =============================================================================

FONT = {
    'family': 'sans-serif',
    'sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'size': {
        'title': 14,
        'panel_label': 13,
        'axis_label': 12,
        'tick': 10,
        'legend': 10,
        'annotation': 10,
        'table': 11,
    }
}

# =============================================================================
# LINE STYLES
# =============================================================================

LINES = {
    'obs': {'linewidth': 1.2, 'linestyle': '-', 'alpha': 0.9},
    'default': {'linewidth': 1.8, 'linestyle': '--', 'alpha': 0.85},
    'calibrated': {'linewidth': 1.8, 'linestyle': '-', 'alpha': 0.95},
    'reference': {'linewidth': 1.0, 'linestyle': ':', 'color': '#888888'},
    'grid': {'linewidth': 0.5, 'linestyle': '-', 'alpha': 0.5},
}

# =============================================================================
# MARKERS
# =============================================================================

MARKERS = {
    'obs': {'marker': 'o', 's': 35, 'alpha': 0.8, 'edgecolors': 'white', 'linewidths': 0.5},
    'obs_large': {'marker': 'o', 's': 50, 'alpha': 0.8, 'edgecolors': 'white', 'linewidths': 0.8},
    'scatter': {'s': 30, 'alpha': 0.6, 'edgecolors': 'none'},
    'gauge': {'marker': '^', 's': 120, 'edgecolors': 'white', 'linewidths': 1.5, 'zorder': 10},
    'station': {'marker': 'o', 's': 50, 'edgecolors': 'white', 'linewidths': 1.0, 'zorder': 9},
}

# =============================================================================
# FIGURE DIMENSIONS (AGU specifications)
# =============================================================================

FIGURE = {
    'single_col': (3.3, 3.3),      # inches (8.4 cm)
    'double_col': (6.85, 5.0),     # inches (17.4 cm)
    'full_page': (7.5, 9.0),       # inches (19 cm)
    'dpi': 300,
    'dpi_preview': 150,
}

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_style():
    """Apply the publication style to matplotlib."""
    plt.rcParams.update({
        # Font
        'font.family': FONT['family'],
        'font.sans-serif': FONT['sans-serif'],
        'font.size': FONT['size']['axis_label'],

        # Axes
        'axes.labelsize': FONT['size']['axis_label'],
        'axes.titlesize': FONT['size']['title'],
        'axes.titleweight': 'bold',
        'axes.labelcolor': COLORS['text'],
        'axes.edgecolor': COLORS['axis'],
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': cycler('color', [
            COLORS['default'], COLORS['calibrated'], COLORS['obs'],
            COLORS['obs_secondary'], COLORS['accent1'], COLORS['accent2']
        ]),

        # Ticks
        'xtick.labelsize': FONT['size']['tick'],
        'ytick.labelsize': FONT['size']['tick'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,

        # Legend
        'legend.fontsize': FONT['size']['legend'],
        'legend.frameon': False,
        'legend.borderpad': 0.4,
        'legend.labelspacing': 0.4,
        'legend.handletextpad': 0.5,

        # Grid
        'axes.grid': False,
        'grid.color': COLORS['grid'],
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,

        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.dpi': FIGURE['dpi_preview'],
        'savefig.dpi': FIGURE['dpi'],
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Other
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'patch.edgecolor': COLORS['axis'],
    })


def despine(ax, left=True, bottom=True):
    """Remove top and right spines, optionally keep left and bottom."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def add_panel_label(ax, label, loc='upper left', offset=(0.02, 0.98)):
    """Add panel label (a), (b), etc. to axis."""
    ax.text(offset[0], offset[1], f'({label})', transform=ax.transAxes,
            fontsize=FONT['size']['panel_label'], fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='none', alpha=0.8))


def add_grid(ax, which='major', axis='both'):
    """Add subtle grid lines."""
    ax.grid(True, which=which, axis=axis, color=COLORS['grid'],
            linewidth=LINES['grid']['linewidth'], alpha=LINES['grid']['alpha'])
    ax.set_axisbelow(True)


def format_axis_dates(ax, interval_years=2):
    """Format x-axis for date display."""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(interval_years))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


def create_minimal_table(ax, data, col_widths=None, row_colors=None):
    """Create a minimal, clean table."""
    ax.axis('off')

    n_cols = len(data[0])
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    table = ax.table(
        cellText=data[1:],
        colLabels=data[0],
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(FONT['size']['table'])
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(n_cols):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['table_header'])
        cell.set_text_props(fontweight='bold')
        cell.set_edgecolor(COLORS['grid'])
        cell.set_linewidth(0.5)

    # Style data cells
    for i in range(1, len(data)):
        for j in range(n_cols):
            cell = table[(i, j)]
            cell.set_edgecolor(COLORS['grid'])
            cell.set_linewidth(0.5)

            # Apply row colors if specified
            if row_colors and i-1 < len(row_colors):
                if row_colors[i-1] == 'positive':
                    cell.set_facecolor(COLORS['table_positive'])
                elif row_colors[i-1] == 'negative':
                    cell.set_facecolor(COLORS['table_negative'])

    return table


def color_table_changes(table, change_col, data):
    """Color the change column based on positive/negative values."""
    for i in range(1, len(data)):
        cell = table[(i, change_col)]
        try:
            val_text = data[i][change_col]
            val = float(val_text)
            if val > 0.005:  # Positive improvement
                cell.set_facecolor(COLORS['table_positive'])
            elif val < -0.005:  # Negative (degradation)
                cell.set_facecolor(COLORS['table_negative'])
        except (ValueError, IndexError):
            pass


# =============================================================================
# PLOT HELPERS
# =============================================================================

def plot_timeseries_comparison(ax, times, default_data, cal_data, obs_times=None,
                                obs_data=None, obs_label='Observations',
                                default_label='Default', cal_label='Calibrated',
                                default_metric=None, cal_metric=None, metric_name='r'):
    """Plot a standardized time series comparison."""

    # Build labels with metrics if provided
    def_lbl = default_label
    cal_lbl = cal_label
    if default_metric is not None:
        def_lbl = f'{default_label} ({metric_name}={default_metric:.2f})'
    if cal_metric is not None:
        cal_lbl = f'{cal_label} ({metric_name}={cal_metric:.2f})'

    # Plot in order: default (back), calibrated (middle), obs (front)
    ax.plot(times, default_data, color=COLORS['default'],
            label=def_lbl, **LINES['default'])
    ax.plot(times, cal_data, color=COLORS['calibrated'],
            label=cal_lbl, **LINES['calibrated'])

    if obs_times is not None and obs_data is not None:
        ax.scatter(obs_times, obs_data, color=COLORS['obs'],
                   label=obs_label, **MARKERS['obs'])

    despine(ax)
    return ax


def plot_scatter_comparison(ax, obs, default, calibrated,
                            default_label='Default', cal_label='Calibrated',
                            xlabel='Observed', ylabel='Simulated',
                            equal_axes=True):
    """Plot a standardized scatter comparison."""

    ax.scatter(obs, default, color=COLORS['default'],
               label=default_label, **MARKERS['scatter'])
    ax.scatter(obs, calibrated, color=COLORS['calibrated'],
               label=cal_label, **MARKERS['scatter'])

    # 1:1 line
    if equal_axes:
        all_data = list(obs) + list(default) + list(calibrated)
        valid_data = [x for x in all_data if not (x != x)]  # Remove NaN
        if valid_data:
            min_val = min(valid_data) * 0.9
            max_val = max(valid_data) * 1.1
            ax.plot([min_val, max_val], [min_val, max_val],
                    color=COLORS['text'], **LINES['reference'], label='1:1')
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    despine(ax)
    add_grid(ax, axis='both')

    return ax


# =============================================================================
# Initialize on import
# =============================================================================

setup_style()
