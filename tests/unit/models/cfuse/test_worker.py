"""Tests for cFUSE calibration worker."""
from __future__ import annotations

import pytest

from symfluence.core.registries import R


class TestCFUSEWorkerRegistration:
    """Tests for cFUSE worker registration."""

    def test_worker_can_be_imported(self):
        from cfuse.calibration.worker import CFUSEWorker
        assert CFUSEWorker is not None

    def test_worker_registered(self):
        assert 'CFUSE' in R.workers

    def test_worker_is_correct_class(self):
        from cfuse.calibration.worker import CFUSEWorker

        assert R.workers.get('CFUSE') == CFUSEWorker


class TestCFUSEWorkerProperties:
    """Tests for cFUSE worker properties."""

    def test_gradient_support_returns_bool(self):
        from cfuse.calibration.worker import CFUSEWorker
        worker = CFUSEWorker()
        result = worker.supports_native_gradients()
        assert isinstance(result, bool)

    def test_worker_has_penalty_score(self):
        from cfuse.calibration.worker import CFUSEWorker
        worker = CFUSEWorker()
        assert hasattr(worker, 'penalty_score')

    def test_worker_has_evaluate_worker_function(self):
        """Worker should have a static function for process pool."""
        from cfuse.calibration.worker import CFUSEWorker
        assert hasattr(CFUSEWorker, 'evaluate_worker_function')
        assert callable(CFUSEWorker.evaluate_worker_function)
