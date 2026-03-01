"""Tests for symfluence.core.file_utils module."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from symfluence.core.exceptions import FileOperationError
from symfluence.core.file_utils import copy_file, copy_tree, ensure_dir, safe_delete

# =============================================================================
# ensure_dir
# =============================================================================

class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_creates_new_directory(self, tmp_path):
        target = tmp_path / "new_dir"
        result = ensure_dir(target)
        assert result == target
        assert target.is_dir()

    def test_creates_nested_parents(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target, parents=True)
        assert result == target
        assert target.is_dir()

    def test_returns_existing_directory(self, tmp_path):
        result = ensure_dir(tmp_path)
        assert result == tmp_path

    def test_accepts_string_path(self, tmp_path):
        target = str(tmp_path / "str_dir")
        result = ensure_dir(target)
        assert Path(target).is_dir()

    def test_logs_on_creation(self, tmp_path):
        logger = MagicMock()
        target = tmp_path / "logged_dir"
        ensure_dir(target, logger=logger)
        logger.info.assert_called_once()

    def test_no_log_when_exists(self, tmp_path):
        logger = MagicMock()
        ensure_dir(tmp_path, logger=logger)
        logger.info.assert_not_called()

    def test_raises_when_path_is_file(self, tmp_path):
        file_path = tmp_path / "afile.txt"
        file_path.write_text("data")
        with pytest.raises(FileOperationError, match="not a directory"):
            ensure_dir(file_path)

    def test_raises_on_permission_error(self, tmp_path, monkeypatch):
        target = tmp_path / "no_perms"
        monkeypatch.setattr(Path, "exists", lambda self: False)
        monkeypatch.setattr(Path, "mkdir", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("denied")))
        with pytest.raises(FileOperationError):
            ensure_dir(target)


# =============================================================================
# copy_file
# =============================================================================

class TestCopyFile:
    """Tests for copy_file function."""

    def test_copies_file_successfully(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"
        result = copy_file(src, dst)
        assert result.read_text() == "hello"

    def test_copies_to_directory(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst_dir = tmp_path / "output"
        dst_dir.mkdir()
        result = copy_file(src, dst_dir)
        assert result.exists()

    def test_creates_destination_directory(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("data")
        dst = tmp_path / "nested" / "dest.txt"
        copy_file(src, dst)
        assert dst.read_text() == "data"

    def test_raises_on_missing_source(self, tmp_path):
        with pytest.raises(FileOperationError, match="Failed to copy"):
            copy_file(tmp_path / "nope.txt", tmp_path / "dst.txt")

    def test_preserve_metadata_flag(self, tmp_path):
        src = tmp_path / "src.txt"
        src.write_text("data")
        dst = tmp_path / "dst_no_meta.txt"
        copy_file(src, dst, preserve_metadata=False)
        assert dst.read_text() == "data"

    def test_logs_on_copy(self, tmp_path):
        logger = MagicMock()
        src = tmp_path / "src.txt"
        src.write_text("x")
        copy_file(src, tmp_path / "dst.txt", logger=logger)
        logger.debug.assert_called()


# =============================================================================
# copy_tree
# =============================================================================

class TestCopyTree:
    """Tests for copy_tree function."""

    def test_copies_directory_tree(self, tmp_path):
        src = tmp_path / "src_tree"
        src.mkdir()
        (src / "a.txt").write_text("a")
        (src / "sub").mkdir()
        (src / "sub" / "b.txt").write_text("b")
        dst = tmp_path / "dst_tree"
        result = copy_tree(src, dst)
        assert (dst / "a.txt").read_text() == "a"
        assert (dst / "sub" / "b.txt").read_text() == "b"
        assert result == dst

    def test_raises_on_missing_source(self, tmp_path):
        with pytest.raises(FileOperationError, match="Failed to copy"):
            copy_tree(tmp_path / "missing", tmp_path / "dst")

    def test_ignore_patterns(self, tmp_path):
        src = tmp_path / "src_tree"
        src.mkdir()
        (src / "keep.txt").write_text("keep")
        (src / "skip.pyc").write_text("skip")
        dst = tmp_path / "dst_tree"
        copy_tree(src, dst, ignore_patterns=["*.pyc"])
        assert (dst / "keep.txt").exists()
        assert not (dst / "skip.pyc").exists()

    def test_logs_on_copy(self, tmp_path):
        logger = MagicMock()
        src = tmp_path / "src_tree"
        src.mkdir()
        (src / "f.txt").write_text("x")
        copy_tree(src, tmp_path / "dst_tree", logger=logger)
        logger.info.assert_called()


# =============================================================================
# safe_delete
# =============================================================================

class TestSafeDelete:
    """Tests for safe_delete function."""

    def test_deletes_file(self, tmp_path):
        f = tmp_path / "todel.txt"
        f.write_text("x")
        assert safe_delete(f) is True
        assert not f.exists()

    def test_deletes_directory(self, tmp_path):
        d = tmp_path / "todel_dir"
        d.mkdir()
        (d / "inner.txt").write_text("x")
        assert safe_delete(d) is True
        assert not d.exists()

    def test_nonexistent_returns_true(self, tmp_path):
        assert safe_delete(tmp_path / "nope") is True

    def test_ignore_errors_returns_false(self, tmp_path, monkeypatch):
        f = tmp_path / "locked"
        f.write_text("x")
        monkeypatch.setattr(Path, "unlink", lambda self: (_ for _ in ()).throw(PermissionError("denied")))
        result = safe_delete(f, ignore_errors=True)
        assert result is False

    def test_raises_when_ignore_errors_false(self, tmp_path, monkeypatch):
        f = tmp_path / "locked"
        f.write_text("x")
        monkeypatch.setattr(Path, "unlink", lambda self: (_ for _ in ()).throw(PermissionError("denied")))
        with pytest.raises(FileOperationError):
            safe_delete(f, ignore_errors=False)

    def test_logs_on_delete(self, tmp_path):
        logger = MagicMock()
        f = tmp_path / "todel.txt"
        f.write_text("x")
        safe_delete(f, logger=logger)
        logger.info.assert_called()
