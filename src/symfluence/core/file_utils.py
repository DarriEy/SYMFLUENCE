"""
File and directory utilities for SYMFLUENCE.

Provides standardized file operations with consistent logging and error handling
to eliminate boilerplate code across the codebase.
"""

import shutil
import logging
from pathlib import Path
from typing import Union, List, Optional
from symfluence.core.exceptions import FileOperationError


def ensure_dir(
    path: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    parents: bool = True,
    exist_ok: bool = True
) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory
        logger: Optional logger for info messages
        parents: If True, create parent directories
        exist_ok: If True, don't raise error if directory exists

    Returns:
        Path object for the directory

    Raises:
        FileOperationError: If directory creation fails
    """
    dir_path = Path(path)
    try:
        if dir_path.exists():
            if not dir_path.is_dir():
                raise FileOperationError(
                    f"Failed to create directory {dir_path}: Path exists but is not a directory"
                )
        else:
            dir_path.mkdir(parents=parents, exist_ok=exist_ok)
            if logger:
                logger.info(f"Created directory: {dir_path}")
        return dir_path
    except FileOperationError:
        raise
    except Exception as e:
        raise FileOperationError(
            f"Failed to create directory {dir_path}: {e}"
        ) from e


def copy_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    preserve_metadata: bool = True
) -> Path:
    """
    Copy a file from source to destination with error handling.

    Args:
        src: Source file path
        dst: Destination file or directory path
        logger: Optional logger for info messages
        preserve_metadata: If True, use copy2 to preserve metadata

    Returns:
        Path object for the destination file

    Raises:
        FileOperationError: If copy fails
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    try:
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
            
        # Ensure destination directory exists
        if dst_path.suffix: # Likely a file path
            ensure_dir(dst_path.parent, logger=logger)
        else: # Likely a directory path
            ensure_dir(dst_path, logger=logger)
            
        copy_func = shutil.copy2 if preserve_metadata else shutil.copy
        result = Path(copy_func(str(src_path), str(dst_path)))
        
        if logger:
            logger.debug(f"Copied {src_path.name} to {result}")
        return result
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to copy {src_path} to {dst_path}: {e}"
        ) from e


def copy_tree(
    src: Union[str, Path],
    dst: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    dirs_exist_ok: bool = True,
    ignore_patterns: Optional[List[str]] = None
) -> Path:
    """
    Copy an entire directory tree.

    Args:
        src: Source directory
        dst: Destination directory
        logger: Optional logger for info messages
        dirs_exist_ok: If True, don't raise error if destination exists
        ignore_patterns: List of glob patterns to ignore

    Returns:
        Path object for the destination directory

    Raises:
        FileOperationError: If copy fails
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    try:
        if not src_path.exists():
            raise FileNotFoundError(f"Source directory not found: {src_path}")
            
        ignore = shutil.ignore_patterns(*ignore_patterns) if ignore_patterns else None
        
        # shutil.copytree handles the actual copy
        result = Path(shutil.copytree(
            str(src_path), 
            str(dst_path), 
            dirs_exist_ok=dirs_exist_ok,
            ignore=ignore
        ))
        
        if logger:
            logger.info(f"Copied directory tree from {src_path} to {dst_path}")
        return result
        
    except Exception as e:
        raise FileOperationError(
            f"Failed to copy directory tree {src_path} to {dst_path}: {e}"
        ) from e


def safe_delete(
    path: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    ignore_errors: bool = True
) -> bool:
    """
    Safely delete a file or directory.

    Args:
        path: Path to delete
        logger: Optional logger for info messages
        ignore_errors: If True, don't raise on failure

    Returns:
        True if deleted or didn't exist, False on failure (if ignore_errors=True)
    """
    del_path = Path(path)
    if not del_path.exists():
        return True
        
    try:
        if del_path.is_file():
            del_path.unlink()
        else:
            shutil.rmtree(del_path)
            
        if logger:
            logger.info(f"Deleted: {del_path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"Failed to delete {del_path}: {e}")
        if ignore_errors:
            return False
        raise FileOperationError(f"Failed to delete {del_path}: {e}") from e
