# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Safe archive extraction guarding against path traversal (CVE-2007-4559 family).

``zipfile.ZipFile.extractall`` and ``tarfile.TarFile.extractall`` do not, by
default, stop an archive member from writing outside the destination directory
via an absolute path, a ``..`` traversal, or (for tar) a symlink whose target
escapes the destination. These helpers validate every member's resolved
destination *before* extracting, then extract members individually, and replace
the bare ``extractall`` calls across the data handlers.

The tar ``filter='data'`` argument (Python 3.11.4+/3.12+) is used as additional
defense in depth where available, but correctness does not depend on it — the
containment checks here are version-independent.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Union

__all__ = ["ArchiveExtractionError", "safe_zip_extract", "safe_tar_extract"]


class ArchiveExtractionError(Exception):
    """Raised when an archive member would extract outside the destination."""


def _resolved_within(dest: Path, name: str) -> bool:
    """True if *name*, resolved under *dest*, stays inside *dest*."""
    if Path(name).is_absolute():
        return False
    try:
        (dest / name).resolve().relative_to(dest.resolve())
        return True
    except ValueError:
        return False


def safe_zip_extract(
    zf: zipfile.ZipFile,
    dest: Union[str, Path],
    members: Optional[Iterable[str]] = None,
) -> None:
    """Extract *zf* into *dest*, rejecting any member that escapes *dest*."""
    dest = Path(dest)
    names = list(members) if members is not None else zf.namelist()
    for name in names:
        if not _resolved_within(dest, name):
            raise ArchiveExtractionError(
                f"Refusing to extract unsafe zip member {name!r}: escapes {dest}"
            )
    for name in names:
        zf.extract(name, dest)


def safe_tar_extract(
    tar: tarfile.TarFile,
    dest: Union[str, Path],
    members: Optional[Iterable[tarfile.TarInfo]] = None,
) -> None:
    """Extract *tar* into *dest*, rejecting members or links that escape *dest*."""
    dest = Path(dest)
    infos = list(members) if members is not None else tar.getmembers()
    for info in infos:
        if not _resolved_within(dest, info.name):
            raise ArchiveExtractionError(
                f"Refusing to extract unsafe tar member {info.name!r}: escapes {dest}"
            )
        if info.issym() or info.islnk():
            # A link's target must also stay within dest (resolved relative to the
            # link's own directory).
            link_dir = (dest / info.name).parent
            if Path(info.linkname).is_absolute():
                raise ArchiveExtractionError(
                    f"Refusing absolute link target {info.linkname!r} for {info.name!r}"
                )
            try:
                (link_dir / info.linkname).resolve().relative_to(dest.resolve())
            except ValueError:
                raise ArchiveExtractionError(
                    f"Refusing link {info.name!r} -> {info.linkname!r}: escapes {dest}"
                ) from None
    for info in infos:
        if sys.version_info >= (3, 12):
            tar.extract(info, dest, filter="data")
        else:
            tar.extract(info, dest)
