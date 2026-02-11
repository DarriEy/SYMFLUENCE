"""
Unicode sparkline widget for score convergence display.
"""

from typing import List

from textual.widgets import Static

from ..constants import SPARKLINE_CHARS


class SparklineWidget(Static):
    """Single-line sparkline using Unicode block characters."""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._values: List[float] = []

    def set_values(self, values: List[float], width: int = 60) -> None:
        """Render a sparkline from a list of float values.

        Args:
            values: Numeric values to plot.
            width: Maximum character width for the sparkline.
        """
        self._values = list(values)
        if not self._values:
            self.update("")
            return

        # Downsample if too many points
        if len(self._values) > width:
            step = len(self._values) / width
            sampled = []
            for i in range(width):
                idx = int(i * step)
                sampled.append(self._values[idx])
            display = sampled
        else:
            display = self._values

        vmin = min(display)
        vmax = max(display)
        span = vmax - vmin if vmax != vmin else 1.0
        n_chars = len(SPARKLINE_CHARS) - 1

        chars = []
        for v in display:
            idx = int((v - vmin) / span * n_chars)
            idx = max(0, min(n_chars, idx))
            chars.append(SPARKLINE_CHARS[idx])

        self.update("".join(chars))
