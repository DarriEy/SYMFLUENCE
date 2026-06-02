"""Tests for dRoute optimizer."""

import pytest

from symfluence.core.registries import R


class TestDRouteOptimizerRegistration:
    """Tests for dRoute optimizer registration."""

    def test_optimizer_can_be_imported(self):
        from droute.calibration.optimizer import DRouteModelOptimizer
        assert DRouteModelOptimizer is not None

    def test_optimizer_registered(self):
        assert 'DROUTE' in R.optimizers

    def test_optimizer_is_correct_class(self):
        from droute.calibration.optimizer import DRouteModelOptimizer

        assert R.optimizers.get('DROUTE') == DRouteModelOptimizer


class TestDRouteWorkerRegistration:
    """Tests for dRoute worker registration."""

    def test_worker_registered(self):
        assert 'DROUTE' in R.workers

    def test_worker_is_correct_class(self):
        from droute.calibration.worker import DRouteWorker

        assert R.workers.get('DROUTE') == DRouteWorker


class TestDRouteGradientSupport:
    """Tests for dRoute gradient support delegation."""

    def test_gradient_support_returns_bool(self):
        from droute.calibration.worker import DRouteWorker
        worker = DRouteWorker()
        result = worker.supports_native_gradients()
        assert isinstance(result, bool)

    def test_gradient_support_without_droute(self):
        """Without droute installed, gradients should not be available."""
        from droute.calibration.worker import HAS_DROUTE, DRouteWorker
        worker = DRouteWorker()
        if not HAS_DROUTE:
            assert worker.supports_native_gradients() is False
