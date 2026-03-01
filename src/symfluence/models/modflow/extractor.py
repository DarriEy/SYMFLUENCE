# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
MODFLOW 6 Result Extractor

Extracts results from MODFLOW 6 output files:
- Head values from binary .hds files
- Drain discharge from budget .bud files
- Listing file (.lst) for water balance summaries

MODFLOW 6 binary file format:
    Each record: header (text + kstp/kper/pertim/totim/ncol/nrow/nlay) + float64 array
"""

import logging
import struct
from pathlib import Path
from typing import Dict, List

import pandas as pd

from symfluence.models.base.base_extractor import ModelResultExtractor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_result_extractor("MODFLOW")
class MODFLOWResultExtractor(ModelResultExtractor):
    """
    Extracts results from MODFLOW 6 output files.

    Supports extraction of:
    - Groundwater head from binary .hds files
    - Drain discharge from binary .bud files
    - Budget summaries from listing file
    """

    def __init__(self, model_name: str = 'MODFLOW'):
        super().__init__(model_name)
        self.logger = logger

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for locating MODFLOW outputs."""
        return {
            'head': ['*.hds'],
            'budget': ['*.bud'],
            'listing': ['mfsim.lst', '*.lst'],
        }

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs,
    ) -> pd.Series:
        """Extract a variable time series from MODFLOW output.

        Args:
            output_file: Path to output file or directory
            variable_type: Type ('head', 'drain_discharge', 'budget')
            **kwargs: Additional args (stress_period_length for time index)

        Returns:
            Time series of extracted variable
        """
        output_path = Path(output_file)

        if variable_type == 'head':
            return self._extract_heads(output_path, **kwargs)
        elif variable_type in ('drain_discharge', 'baseflow'):
            return self._extract_drain_discharge(output_path, **kwargs)
        else:
            raise ValueError(f"Unknown variable type: {variable_type}")

    def _extract_heads(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract head time series from .hds binary file.

        MODFLOW 6 binary head file format per record:
            - KSTP (int32): time step number
            - KPER (int32): stress period number
            - PERTIM (float64): time in period
            - TOTIM (float64): total simulation time
            - TEXT (16 bytes): record identifier
            - NCOL (int32), NROW (int32), NLAY (int32)
            - HEAD array (float64, size = ncol * nrow * nlay)
        """
        if output_path.is_dir():
            hds_files = list(output_path.glob("*.hds"))
            if not hds_files:
                raise ValueError(f"No .hds files found in {output_path}")
            hds_file = hds_files[0]
        else:
            hds_file = output_path

        times = []
        heads = []

        with open(hds_file, 'rb') as f:
            while True:
                try:
                    header = self._read_binary_header(f)
                    if header is None:
                        break
                    kstp, kper, pertim, totim, text, ncol, nrow, nlay = header
                    n_values = ncol * nrow * nlay
                    data = struct.unpack(f'{n_values}d', f.read(n_values * 8))
                    times.append(totim)
                    # For lumped model (1×1×1), head is a single value
                    heads.append(data[0])
                except (struct.error, ValueError):
                    break

        sp_length = kwargs.get('stress_period_length', 1.0)
        start_date = kwargs.get('start_date', '2000-01-01')
        date_index = pd.date_range(
            start=start_date,
            periods=len(heads),
            freq=f'{int(sp_length)}D',
        )

        return pd.Series(heads, index=date_index, name='head_m')

    def _extract_drain_discharge(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract drain discharge from budget .bud binary file.

        The budget file contains flow terms for each package.
        We look for DRN (drain) records.
        """
        if output_path.is_dir():
            bud_files = list(output_path.glob("*.bud"))
            if not bud_files:
                raise ValueError(f"No .bud files found in {output_path}")
            bud_file = bud_files[0]
        else:
            bud_file = output_path

        times = []
        drain_flows = []

        with open(bud_file, 'rb') as f:
            while True:
                try:
                    header = self._read_budget_header(f)
                    if header is None:
                        break
                    kstp, kper, text, ndim1, ndim2, ndim3, imeth, delt, pertim, totim = header

                    text_clean = text.strip()
                    n_values = ndim1 * ndim2

                    if imeth == 0 or imeth == 1:
                        # Full 3D array
                        data = struct.unpack(f'{n_values}d', f.read(n_values * 8))
                    elif imeth == 6:
                        data = self._read_imeth6_data(f, text_clean)
                    else:
                        # Skip unknown imeth
                        self.logger.debug(f"Skipping budget record with imeth={imeth}")
                        if n_values > 0:
                            f.read(n_values * 8)
                        continue

                    if 'DRN' in text_clean:
                        times.append(totim)
                        total_drain = sum(data) if isinstance(data, list) else sum(data)
                        # Drain flows are negative (outflow), take absolute value
                        drain_flows.append(abs(total_drain))

                except (struct.error, ValueError, EOFError):
                    break

        if not drain_flows:
            self.logger.warning("No drain discharge records found in budget file")
            return pd.Series(dtype=float, name='drain_m3d')

        sp_length = kwargs.get('stress_period_length', 1.0)
        start_date = kwargs.get('start_date', '2000-01-01')
        date_index = pd.date_range(
            start=start_date,
            periods=len(drain_flows),
            freq=f'{int(sp_length)}D',
        )

        return pd.Series(drain_flows, index=date_index, name='drain_m3d')

    def _read_imeth6_data(self, f, text_clean: str) -> list:
        """Read compact-list (imeth=6) budget data from a MODFLOW 6 .bud file.

        MODFLOW 6 writes ``naux + 1`` as the naux field (where naux is
        the actual number of auxiliary variables). It then writes only
        ``naux`` aux-name records. Each list entry contains ``naux + 1``
        double values (flow followed by auxiliary values).
        """
        # Read 4 text IDs (model/package names)
        f.read(16)  # txt1id1
        f.read(16)  # txt2id1
        f.read(16)  # txt1id2
        f.read(16)  # txt2id2

        # ndat = actual naux + 1 (includes flow column)
        ndat = struct.unpack('i', f.read(4))[0]
        naux = max(ndat - 1, 0)

        for _ in range(naux):
            f.read(16)  # aux name

        nlist = struct.unpack('i', f.read(4))[0]

        data = []
        for _ in range(nlist):
            f.read(4)  # node1
            f.read(4)  # node2
            # Read ndat double values: first is flow, rest are auxiliary
            vals = struct.unpack(f'{ndat}d', f.read(ndat * 8))
            data.append(vals[0])  # flow value

        return data

    def _read_binary_header(self, f):
        """Read a MODFLOW 6 binary head file record header.

        Returns:
            Tuple of (kstp, kper, pertim, totim, text, ncol, nrow, nlay) or None at EOF
        """
        raw = f.read(4)
        if len(raw) < 4:
            return None
        kstp = struct.unpack('i', raw)[0]
        kper = struct.unpack('i', f.read(4))[0]
        pertim = struct.unpack('d', f.read(8))[0]
        totim = struct.unpack('d', f.read(8))[0]
        text = f.read(16).decode('ascii').strip()
        ncol = struct.unpack('i', f.read(4))[0]
        nrow = struct.unpack('i', f.read(4))[0]
        nlay = struct.unpack('i', f.read(4))[0]
        return kstp, kper, pertim, totim, text, ncol, nrow, nlay

    def _read_budget_header(self, f):
        """Read a MODFLOW 6 budget file record header.

        Returns:
            Tuple of (kstp, kper, text, ndim1, ndim2, ndim3, imeth, delt, pertim, totim)
            or None at EOF
        """
        raw = f.read(4)
        if len(raw) < 4:
            return None
        kstp = struct.unpack('i', raw)[0]
        kper = struct.unpack('i', f.read(4))[0]
        text = f.read(16).decode('ascii').strip()
        ndim1 = struct.unpack('i', f.read(4))[0]
        ndim2 = struct.unpack('i', f.read(4))[0]
        ndim3 = struct.unpack('i', f.read(4))[0]
        imeth = struct.unpack('i', f.read(4))[0]
        delt = struct.unpack('d', f.read(8))[0]
        pertim = struct.unpack('d', f.read(8))[0]
        totim = struct.unpack('d', f.read(8))[0]
        return kstp, kper, text, ndim1, ndim2, ndim3, imeth, delt, pertim, totim
