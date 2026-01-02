"""
Common imports and base configuration for config models.

This module provides shared imports and the base ConfigDict used
across all configuration model classes.
"""

from typing import List, Optional, Any, Dict, Union, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from functools import cached_property
import pandas as pd
import warnings

# Standard ConfigDict for all config models
FROZEN_CONFIG = ConfigDict(extra='allow', populate_by_name=True, frozen=True)
