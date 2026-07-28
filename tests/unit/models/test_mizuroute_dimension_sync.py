# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The mizuRoute control file must name the dimension the runoff file uses.

Getting this wrong does not crash. mizuRoute takes the space dimension's length
from ``<dname_hruid>`` and the IDs from ``<vname_hruid>``, but never checks that
``<vname_qsim>`` is actually dimensioned on it. A SUMMA output file carries BOTH
``hru`` and ``gru`` dimensions, so a wrong name still resolves to a real
dimension of a plausible length; unmatched IDs then fall through to a
zero-initialised array, and the ID-consistency check that would catch it sits
behind mizuRoute's ``debug`` flag, which is false by default. The result is zero
streamflow, no warning, exit code 0.

Two things therefore have to hold, and both are pinned here: the declaration
must be right in the first place, and the runtime sync must be able to correct
it in either direction.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pytest
import xarray as xr

from symfluence.core.modeling.config_schema import parallel_calibration_config
from symfluence.core.modeling.utilities.runoff_loader import get_model_config
from symfluence.models.mizuroute.runner import MizuRouteRunner

pytestmark = [pytest.mark.unit]


def _write_runoff(path, dim: str, var: str) -> None:
    """A runoff file carrying BOTH dimensions, as SUMMA's output does."""
    other = "hru" if dim == "gru" else "gru"
    xr.Dataset(
        {
            "averageRoutedRunoff": (("time", dim), np.zeros((2, 1))),
            var: ((dim,), np.array([1])),
        },
        coords={"time": [0, 1], dim: [1], other: [1]},
    ).to_netcdf(path)


def _control(dim: str, var: str) -> str:
    # <vname_qsim> is present in every real control file and is what tells the
    # sync which variable mizuRoute will read — and therefore which dimension
    # is authoritative. Omitting it here would exercise only the fallback.
    return (
        "<vname_qsim>            averageRoutedRunoff    ! Variable name for runoff \n"
        f"<dname_hruid>           {dim}     ! Dimension name for HM_HRU ID \n"
        f"<vname_hruid>           {var}   ! Variable name for HM_HRU ID \n"
    )


def _values(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"<(\w+)>\s+(\S+)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _runner() -> MizuRouteRunner:
    runner = MizuRouteRunner.__new__(MizuRouteRunner)
    runner.logger = logging.getLogger("test.mizuroute.sync")
    return runner


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

def test_summa_declares_the_dimension_summa_actually_writes():
    """SUMMA's runoff is gru-dimensioned, in every configuration.

    ``averageRoutedRunoff`` is registered in SUMMA's ``bvar_meta`` and every
    bvar is defined with ``needGRU``, so it is ``(time, gru)`` regardless of
    spatial mode, HRU:GRU ratio or output settings. The declaration said
    ``hru``/``hruId`` and was corrected; it only ever appeared to work because
    the runtime sync rewrote it before mizuRoute ran.
    """
    runoff = get_model_config("SUMMA")
    assert (runoff.hru_dim, runoff.hru_var) == ("gru", "gruId")


def test_the_two_summa_paths_agree():
    """Serial and parallel calibration must name the same dimension.

    They disagreed: the parallel declaration said gru/gruId (correct) while the
    serial one said hru/hruId, upgraded to gru only when a count-based
    heuristic saw n_hrus > n_grus — which is false at 1 HRU per GRU, the
    common case.
    """
    runoff = get_model_config("SUMMA")
    parallel = parallel_calibration_config("SUMMA")
    assert (runoff.hru_dim, runoff.hru_var) == (parallel.hru_dim, parallel.hru_var)


# ---------------------------------------------------------------------------
# The runtime sync
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "control_dim,control_var,file_dim,file_var",
    [
        ("hru", "hruId", "gru", "gruId"),  # promote
        ("gru", "gruId", "hru", "hruId"),  # demote — used to be impossible
    ],
)
def test_sync_corrects_the_control_file_in_both_directions(
    tmp_path, control_dim, control_var, file_dim, file_var
):
    """The demote direction was unreachable.

    The guard was ``if dname not in line`` — a substring test against a line
    whose own tag is ``<dname_hruid>``, which contains 'hru'. So for
    ``dname='hru'`` it was always False and the dimension could never be
    rewritten to hru, while the variable could (its tag is lowercase 'hruid',
    the value is 'hruId'). That left the control file internally inconsistent —
    dimension gru, variable hruId — while logging that it had applied hru.
    """
    control = tmp_path / "mizuroute.control"
    runoff = tmp_path / "runoff.nc"
    control.write_text(_control(control_dim, control_var), encoding="utf-8")
    _write_runoff(runoff, file_dim, file_var)

    _runner().sync_control_file_dimensions(control, runoff)

    values = _values(control.read_text(encoding="utf-8"))
    assert values["dname_hruid"] == file_dim
    assert values["vname_hruid"] == file_var


def test_sync_leaves_a_correct_control_file_alone(tmp_path):
    control = tmp_path / "mizuroute.control"
    runoff = tmp_path / "runoff.nc"
    original = _control("gru", "gruId")
    control.write_text(original, encoding="utf-8")
    _write_runoff(runoff, "gru", "gruId")

    _runner().sync_control_file_dimensions(control, runoff)

    assert control.read_text(encoding="utf-8") == original


def test_sync_never_leaves_dimension_and_variable_disagreeing(tmp_path):
    """The failure mode that matters: a half-corrected file.

    A gru dimension with an hruId variable is worse than either consistent
    pairing, because mizuRoute reads the length from one and the IDs from the
    other and reports nothing.
    """
    for control_dim, control_var in [("hru", "hruId"), ("gru", "gruId"),
                                     ("gru", "hruId"), ("hru", "gruId")]:
        control = tmp_path / f"c_{control_dim}_{control_var}.control"
        runoff = tmp_path / f"r_{control_dim}_{control_var}.nc"
        control.write_text(_control(control_dim, control_var), encoding="utf-8")
        _write_runoff(runoff, "gru", "gruId")

        _runner().sync_control_file_dimensions(control, runoff)

        values = _values(control.read_text(encoding="utf-8"))
        assert (values["dname_hruid"], values["vname_hruid"]) == ("gru", "gruId"), (
            f"starting from {control_dim}/{control_var} the sync produced "
            f"{values['dname_hruid']}/{values['vname_hruid']}"
        )
