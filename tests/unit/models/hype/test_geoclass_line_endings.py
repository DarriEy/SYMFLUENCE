# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GeoClass.txt must not carry doubled carriage returns.

Opening the file in text mode without ``newline=''`` corrupts it on Windows:
pandas writes ``os.linesep`` ('\\r\\n') and Python's text layer translates the
'\\n' again, so every data row ends '\\r\\r\\n' while the header written by
``f.write`` ends '\\r\\n'.

HYPE parses only the header, reports ``nclass=1``, and drops the rest with
"WARNING: slc larger than nclass skipped". On the Bow-at-Banff calibration
catchment that silently removed ~85% of the area from runoff generation:
simulated volume fell to 14-19% of observed while correlation stayed ~0.6-0.84,
so every parameter set scored about zero and the calibration converged to a
degenerate corner (KGE 0.054 on Windows vs 0.741 on macOS). Nothing errored.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _write_geoclass(path, *, fixed: bool):
    """Reproduce the writer both ways: fixed=False is the original code."""
    frame = pd.DataFrame({"slc": [1, 2, 3, 4], "lulc": [5, 6, 7, 8]})
    header = "!          SLC\tLULC\tSoil layer depth 3 \n"
    if fixed:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(header)
            frame.to_csv(fh, sep="\t", index=False, header=False,
                         lineterminator="\n")
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header)
            frame.to_csv(fh, sep="\t", index=False, header=False)
    return path.read_bytes()


def test_written_geoclass_has_no_doubled_carriage_return(tmp_path):
    """The defect, stated directly."""
    raw = _write_geoclass(tmp_path / "GeoClass.txt", fixed=True)
    assert b"\r\r" not in raw


def test_every_class_row_survives_a_line_split(tmp_path):
    """What HYPE actually does: split into lines and count classes."""
    raw = _write_geoclass(tmp_path / "GeoClass.txt", fixed=True)
    rows = [ln for ln in raw.split(b"\n") if ln.strip() and not ln.startswith(b"!")]
    assert len(rows) == 4, "all four land classes must be parseable"
    for row in rows:
        assert not row.endswith(b"\r\r")


def test_output_is_byte_identical_across_platforms(tmp_path):
    """Pinning the terminator keeps Windows output equal to Linux/macOS.

    A reproduction campaign compares results across platforms; an input file
    that differs by line ending is a difference waiting to be misattributed.
    """
    raw = _write_geoclass(tmp_path / "GeoClass.txt", fixed=True)
    assert b"\r" not in raw, "expected pure LF regardless of host platform"


@pytest.mark.skipif(pd.__version__ < "1.5", reason="lineterminator kwarg name")
def test_the_unfixed_writer_is_actually_broken_on_windows(tmp_path):
    """Guard against the fix being reverted as cosmetic.

    On POSIX os.linesep is '\\n' and the original code happens to be fine,
    which is exactly why this went unnoticed until the Windows port.
    """
    import os

    raw = _write_geoclass(tmp_path / "bad.txt", fixed=False)
    if os.linesep == "\r\n":
        assert b"\r\r\n" in raw, "expected the original code to double the CR here"
    else:
        assert b"\r" not in raw
