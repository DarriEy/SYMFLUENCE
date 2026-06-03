# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""safe_zip_extract / safe_tar_extract reject path traversal (CVE-2007-4559 family)."""

from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from symfluence.core.archive_extraction import (
    ArchiveExtractionError,
    safe_tar_extract,
    safe_zip_extract,
)


def _add_tar_file(tf: tarfile.TarFile, name: str, data: bytes = b"x") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


# ---- zip ----------------------------------------------------------------


def test_zip_benign_extracts(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sub/ok.txt", b"hi")
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(archive) as zf:
        safe_zip_extract(zf, dest)
    assert (dest / "sub" / "ok.txt").read_bytes() == b"hi"


@pytest.mark.parametrize("evil", ["../escape.txt", "a/../../escape.txt"])
def test_zip_traversal_rejected(tmp_path, evil):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(evil, b"x")
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ArchiveExtractionError):
            safe_zip_extract(zf, dest)
    assert not (tmp_path / "escape.txt").exists()


# ---- tar ----------------------------------------------------------------


def test_tar_benign_extracts(tmp_path):
    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tf:
        _add_tar_file(tf, "sub/ok.txt", b"hi")
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(archive) as tf:
        safe_tar_extract(tf, dest)
    assert (dest / "sub" / "ok.txt").read_bytes() == b"hi"


def test_tar_traversal_rejected(tmp_path):
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        _add_tar_file(tf, "../escape.txt")
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(archive) as tf:
        with pytest.raises(ArchiveExtractionError):
            safe_tar_extract(tf, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_tar_symlink_escape_rejected(tmp_path):
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        tf.addfile(info)
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(archive) as tf:
        with pytest.raises(ArchiveExtractionError):
            safe_tar_extract(tf, dest)
