"""
Common imports and base configuration for config models.

This module provides shared imports and the base ConfigDict used
across all configuration model classes.
"""

from pydantic import ConfigDict

# Standard ConfigDict for all config models
FROZEN_CONFIG = ConfigDict(extra='allow', populate_by_name=True, frozen=True)
