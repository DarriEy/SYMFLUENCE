"""
Validation mixin for SYMFLUENCE modules.

Provides standard validation operations.
"""

from pathlib import Path
from typing import List, Union


class ValidationMixin:
    """
    Mixin providing standard validation operations.

    Requires self.config_dict to be available (e.g., from ConfigMixin).
    """

    def validate_config(self, required_keys: List[str], operation: str) -> None:
        """Validate configuration keys."""
        from ..validation import validate_config_keys
        config_dict = getattr(self, 'config_dict', {})
        validate_config_keys(config_dict, required_keys, operation)

    def validate_file(self, file_path: Union[str, Path], description: str = "file") -> Path:
        """Validate that a file exists."""
        from ..validation import validate_file_exists
        return validate_file_exists(file_path, description)

    def validate_dir(self, dir_path: Union[str, Path], description: str = "directory") -> Path:
        """Validate that a directory exists."""
        from ..validation import validate_directory_exists
        return validate_directory_exists(dir_path, description)
