# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional-dependency behavior for SALib-backed sensitivity methods."""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from symfluence.evaluation import sensitivity_analysis
from symfluence.evaluation.sensitivity_analysis import SensitivityAnalyzer


@pytest.mark.parametrize("method_name", ["perform_sobol_analysis", "perform_rbd_fast_analysis"])
def test_salib_methods_explain_optional_extra(monkeypatch, method_name):
    monkeypatch.setattr(sensitivity_analysis, "rbd_fast", None)
    monkeypatch.setattr(sensitivity_analysis, "sobol", None)
    monkeypatch.setattr(sensitivity_analysis, "sobol_sample", None)

    analyzer = SensitivityAnalyzer.__new__(SensitivityAnalyzer)
    analyzer.logger = logging.getLogger(__name__)
    samples = pd.DataFrame({"parameter": [0.0, 1.0], "RMSE": [1.0, 0.5]})

    with pytest.raises(ImportError, match=r"symfluence\[sensitivity\]"):
        getattr(analyzer, method_name)(samples, metric="RMSE")
