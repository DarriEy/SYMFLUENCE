"""Tests for Noah-MP calibration modules."""
import textwrap

import pytest


class TestNoahMPOptimizerRegistration:
    def test_import(self):
        from symfluence.models.noahmp.calibration.optimizer import NoahMPModelOptimizer
        assert NoahMPModelOptimizer is not None

    def test_registered(self):
        import symfluence.models.noahmp.calibration.optimizer  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry
        assert OptimizerRegistry.get_optimizer('NOAHMP') is not None

    def test_model_name(self):
        from symfluence.models.noahmp.calibration.optimizer import NoahMPModelOptimizer
        o = NoahMPModelOptimizer.__new__(NoahMPModelOptimizer)
        assert o._get_model_name() == 'NOAHMP'


class TestNoahMPWorkerRegistration:
    def test_import(self):
        from symfluence.models.noahmp.calibration.worker import NoahMPWorker
        assert NoahMPWorker is not None

    def test_registered(self):
        import symfluence.models.noahmp.calibration.worker  # noqa: F401
        from symfluence.optimization.registry import OptimizerRegistry
        assert OptimizerRegistry.get_worker('NOAHMP') is not None

    def test_namelist_params(self):
        from symfluence.models.noahmp.calibration.worker import NoahMPWorker
        assert 'rain_snow_thresh' in NoahMPWorker.NAMELIST_PARAMS
        assert 'ZREF' in NoahMPWorker.NAMELIST_PARAMS

    def test_soilparm_columns(self):
        from symfluence.models.noahmp.calibration.worker import NoahMPWorker
        assert NoahMPWorker.SOILPARM_COLUMNS['bexp'] == 1
        assert NoahMPWorker.SOILPARM_COLUMNS['dksat'] == 7


class TestNoahMPWorkerApplyParameters:
    def test_apply_namelist(self, tmp_path):
        import logging

        from symfluence.models.noahmp.calibration.worker import NoahMPWorker
        d = tmp_path / 'NOAHMP'; d.mkdir()
        (d / 'namelist.input').write_text("&forcing\n ZREF = 2.0\n rain_snow_thresh = 1.0\n/\n")
        w = NoahMPWorker(config={}, logger=logging.getLogger('t'))
        assert w.apply_parameters({'rain_snow_thresh': -1.5, 'ZREF': 8.0}, d)
        t = (d / 'namelist.input').read_text()
        assert '-1.500000' in t and '8.000000' in t

    def test_apply_soilparm(self, tmp_path):
        import logging

        from symfluence.models.noahmp.calibration.worker import NoahMPWorker
        d = tmp_path / 'NOAHMP'; d.mkdir(); (d / 'parameters').mkdir()
        (d / 'namelist.input').write_text("&forcing\n ZREF = 2.0\n/\n&structure\n isltyp = 1\n/\n")
        (d / 'parameters' / 'SOILPARM.TBL').write_text(
            "Soil Parameters\nSTAS\n19,1 'BB'\n"
            "1, 2.79, 0.010, -0.472, 0.339, 0.192, 0.069, 4.66E-5, 2.65E-5, 0.010, 0.92, 'SAND'\n"
            "2, 4.26, 0.028, -1.044, 0.421, 0.283, 0.036, 1.41E-5, 5.14E-6, 0.028, 0.82, 'LOAMY'\n")
        w = NoahMPWorker(config={}, logger=logging.getLogger('t'))
        assert w.apply_parameters({'bexp': 3.5, 'smcmax': 0.45}, d)
        t = (d / 'parameters' / 'SOILPARM.TBL').read_text()
        assert '3.500' in t and '0.450' in t and '4.26' in t


class TestNoahMPParameterBounds:
    def test_loaded(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_noahmp_bounds
        assert len(get_noahmp_bounds()) > 0

    def test_expected_params(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_noahmp_bounds
        b = get_noahmp_bounds()
        for p in ['dksat', 'bexp', 'smcmax', 'slope', 'refkdt']:
            assert p in b

    def test_min_lt_max(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_noahmp_bounds
        for n, b in get_noahmp_bounds().items():
            assert b['min'] < b['max'], n

    def test_dksat_log(self):
        from symfluence.optimization.core.parameter_bounds_registry import get_noahmp_bounds
        assert get_noahmp_bounds()['dksat'].get('transform') == 'log'
