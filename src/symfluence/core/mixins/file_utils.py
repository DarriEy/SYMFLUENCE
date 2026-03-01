# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
File utilities mixin for SYMFLUENCE modules.

Provides standard file and directory operations.
"""

from pathlib import Path
from typing import List, Optional, Union


class FileUtilsMixin:
    """
    Mixin providing standard file and directory operations.

    Requires self.logger to be available (e.g., from LoggingMixin).
    """

    def ensure_dir(
        self,
        path: Union[str, Path],
        parents: bool = True,
        exist_ok: bool = True
    ) -> Path:
        """Ensure a directory exists."""
        from ..file_utils import ensure_dir
        logger = getattr(self, 'logger', None)
        return ensure_dir(path, logger=logger, parents=parents, exist_ok=exist_ok)

    def copy_file(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        preserve_metadata: bool = True
    ) -> Path:
        """Copy a file."""
        from ..file_utils import copy_file
        logger = getattr(self, 'logger', None)
        return copy_file(src, dst, logger=logger, preserve_metadata=preserve_metadata)

    def copy_tree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        dirs_exist_ok: bool = True,
        ignore_patterns: Optional[List[str]] = None
    ) -> Path:
        """Copy a directory tree."""
        from ..file_utils import copy_tree
        logger = getattr(self, 'logger', None)
        return copy_tree(
            src, dst,
            logger=logger,
            dirs_exist_ok=dirs_exist_ok,
            ignore_patterns=ignore_patterns
        )

    def safe_delete(self, path: Union[str, Path], ignore_errors: bool = True) -> bool:
        """Safely delete a file or directory."""
        from ..file_utils import safe_delete
        logger = getattr(self, 'logger', None)
        return safe_delete(path, logger=logger, ignore_errors=ignore_errors)
