# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for the SYMFLUENCE logging protocol.

Covers the shared utilities in ``symfluence.core.logging_utils`` and the
``LoggingManager`` setup in ``symfluence.project.logging_manager``: idempotent
setup, the ``symfluence_{domain}_{experiment}_{ts}.log`` filename, the
canonical file format with short logger names, single-record step headers,
completion messages, and symfluence-rooted logger naming.
"""
from __future__ import annotations

import logging
import re

import pytest

import symfluence.project.logging_manager as lm
from symfluence.core.logging_utils import (
    FILE_FORMAT,
    FILE_FORMAT_DEBUG,
    THIRD_PARTY_LOGGER_LEVELS,
    ShortNameFilter,
    get_worker_logger,
    log_once,
    reset_log_once,
    silence_third_party,
)
from symfluence.core.mixins.logging import LoggingMixin, _class_logger_name
from symfluence.project.logging_manager import LoggingManager, get_logger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_logging_state():
    """Isolate the process-wide logging state around every test."""
    sym = logging.getLogger('symfluence')
    root = logging.getLogger()
    saved = (sym.handlers[:], sym.level, sym.propagate,
             root.handlers[:], root.level,
             lm._active_setup_key, lm._active_log_file)

    sym.handlers = []
    lm._active_setup_key = None
    lm._active_log_file = None
    reset_log_once()

    yield

    for handler in sym.handlers:
        if handler not in saved[0]:
            handler.close()
    sym.handlers, sym.level, sym.propagate = saved[0], saved[1], saved[2]
    root.handlers, root.level = saved[3], saved[4]
    lm._active_setup_key, lm._active_log_file = saved[5], saved[6]
    # Worker loggers configured by get_worker_logger keep handlers/propagate
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith('symfluence.worker.'):
            worker = logging.getLogger(name)
            worker.handlers = []
            worker.propagate = True
    reset_log_once()


def _make_config(tmp_path, experiment_id='expA', domain='testdom'):
    config = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'DOMAIN_NAME': domain,
    }
    if experiment_id is not None:
        config['EXPERIMENT_ID'] = experiment_id
    return config


def _file_handlers(logger=None):
    logger = logger or logging.getLogger('symfluence')
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


class _RecordCollector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _attach_collector():
    """Attach a record collector to the (already configured) 'symfluence' logger."""
    handler = _RecordCollector()
    logging.getLogger('symfluence').addHandler(handler)
    return handler


# ---------------------------------------------------------------------------
# setup_logging: idempotence and filenames
# ---------------------------------------------------------------------------


def test_setup_is_idempotent_for_same_domain_and_experiment(tmp_path):
    manager1 = LoggingManager(_make_config(tmp_path))
    first_file = manager1.log_file

    manager2 = LoggingManager(_make_config(tmp_path))

    assert len(_file_handlers()) == 1, "re-setup must not add a second file handler"
    assert manager2.log_file == first_file, "re-setup must reuse the same log file"


def test_setup_reconfigures_for_different_experiment(tmp_path):
    manager1 = LoggingManager(_make_config(tmp_path, experiment_id='expA'))
    manager2 = LoggingManager(_make_config(tmp_path, experiment_id='expB'))

    assert len(_file_handlers()) == 1
    assert manager1.log_file != manager2.log_file
    assert '_expB_' in manager2.log_file.name


def test_log_filename_contains_domain_and_experiment(tmp_path):
    manager = LoggingManager(_make_config(tmp_path, experiment_id='run_1'))
    assert re.fullmatch(
        r'symfluence_testdom_run_1_\d{8}_\d{6}\.log', manager.log_file.name
    )


def test_log_filename_sanitizes_experiment_id(tmp_path):
    manager = LoggingManager(_make_config(tmp_path, experiment_id='run 1/a:b'))
    assert re.fullmatch(
        r'symfluence_testdom_run-1-a-b_\d{8}_\d{6}\.log', manager.log_file.name
    )


def test_log_filename_omits_experiment_segment_when_unset(tmp_path, monkeypatch):
    # Typed configs always default an experiment id ('run_1'); the unset path
    # only occurs for plain-dict configs with an empty id, so keep the config
    # a dict here by bypassing coercion.
    monkeypatch.setattr(
        'symfluence.core.config.coercion.coerce_config',
        lambda config, warn=True, strict=None: config,
    )
    manager = LoggingManager(_make_config(tmp_path, experiment_id=''))
    assert manager.experiment_id == ''
    assert re.fullmatch(
        r'symfluence_testdom_\d{8}_\d{6}\.log', manager.log_file.name
    )


def test_get_log_file_path_finds_new_and_legacy_names(tmp_path):
    manager = LoggingManager(_make_config(tmp_path))
    assert manager.get_log_file_path('general') == manager.log_file

    # Legacy files remain discoverable
    legacy = manager.log_dir / 'symfluence_general_testdom_99990101_000000.log'
    legacy.write_text('old\n', encoding='utf-8')
    found = manager.get_log_file_path('general')
    assert found in (legacy, manager.log_file)
    legacy.unlink()


# ---------------------------------------------------------------------------
# File format
# ---------------------------------------------------------------------------


def test_file_format_strips_symfluence_prefix(tmp_path):
    manager = LoggingManager(_make_config(tmp_path))
    get_logger('data.acquisition').info('fetching forcings')

    for handler in _file_handlers():
        handler.flush()
    content = manager.log_file.read_text(encoding='utf-8')
    assert re.search(
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\s+\[data\.acquisition\] '
        r'fetching forcings$',
        content,
        flags=re.MULTILINE,
    )
    # The fixed 'symfluence.' prefix must not appear in the bracketed name
    assert '[symfluence.data.acquisition]' not in content


def test_short_name_filter_sname_values():
    filt = ShortNameFilter()
    for name, expected in [
        ('symfluence', 'symfluence'),
        ('symfluence.data.foo', 'data.foo'),
        ('rasterio', 'rasterio'),
    ]:
        record = logging.LogRecord(name, logging.INFO, __file__, 1, 'm', (), None)
        assert filt.filter(record) is True
        assert record.sname == expected


def test_file_format_constants_shape():
    assert '%(sname)s' in FILE_FORMAT
    assert '%(sname)s' in FILE_FORMAT_DEBUG
    assert '%(module)s.%(funcName)s' in FILE_FORMAT_DEBUG


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_has_no_basicconfig_side_effect():
    root = logging.getLogger()
    before = root.handlers[:]

    logger = get_logger('models.summa')

    assert logger.name == 'symfluence.models.summa'
    assert root.handlers == before, "get_logger must not call logging.basicConfig"
    sym = logging.getLogger('symfluence')
    assert any(isinstance(h, logging.NullHandler) for h in sym.handlers)


def test_get_logger_root_and_prefix_tolerance():
    assert get_logger().name == 'symfluence'
    assert get_logger('symfluence').name == 'symfluence'
    assert get_logger('symfluence.data.x').name == 'symfluence.data.x'


# ---------------------------------------------------------------------------
# log_once
# ---------------------------------------------------------------------------


def test_log_once_first_then_debug(caplog):
    logger = logging.getLogger('symfluence.test.log_once')
    logger.propagate = True
    with caplog.at_level(logging.DEBUG, logger='symfluence.test.log_once'):
        assert log_once(logger, logging.WARNING, 'k1', 'first') is True
        assert log_once(logger, logging.WARNING, 'k1', 'again') is False

    levels = [r.levelno for r in caplog.records]
    assert levels == [logging.WARNING, logging.DEBUG]

    reset_log_once()
    with caplog.at_level(logging.DEBUG, logger='symfluence.test.log_once'):
        assert log_once(logger, logging.WARNING, 'k1', 'after reset') is True


# ---------------------------------------------------------------------------
# Third-party suppression
# ---------------------------------------------------------------------------


def test_silence_third_party_applies_table():
    saved = {name: logging.getLogger(name).level
             for name in THIRD_PARTY_LOGGER_LEVELS}
    try:
        silence_third_party()
        for name, level in THIRD_PARTY_LOGGER_LEVELS.items():
            assert logging.getLogger(name).level == level == logging.WARNING
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


# ---------------------------------------------------------------------------
# Step headers and completion messages
# ---------------------------------------------------------------------------


def test_step_header_is_single_record_and_aligned(tmp_path):
    manager = LoggingManager(_make_config(tmp_path))
    collector = _attach_collector()

    manager.log_step_header(2, 16, 'Domain Definition', 'Delineate the basin')
    manager.log_step_header(12, 16, 'Calibration', 'Run DDS optimization')

    assert len(collector.records) == 2, "each header must be one log record"
    for record in collector.records:
        lines = record.getMessage().strip('\n').split('\n')
        assert lines[0].startswith('┌') and lines[-1].startswith('└')
        assert len({len(line) for line in lines}) == 1, "box lines must align"


def test_log_completion_success_and_failure(tmp_path):
    manager = LoggingManager(_make_config(tmp_path))
    collector = _attach_collector()

    manager.log_completion(success=True, message='step done', duration=1.5)
    manager.log_completion(success=False, message='step blew up')

    ok, bad = collector.records
    assert '✓ Completed: step done' in ok.getMessage()
    assert ok.levelno == logging.INFO
    assert '✗ Failed: step blew up' in bad.getMessage()
    assert bad.levelno == logging.ERROR


# ---------------------------------------------------------------------------
# LoggingMixin naming
# ---------------------------------------------------------------------------


def test_logging_mixin_logger_rooted_at_symfluence():
    class Widget(LoggingMixin):
        pass

    Widget.__module__ = 'symfluence.data.widgets'
    assert Widget().logger.name == 'symfluence.data.widgets.Widget'


def test_logging_mixin_wraps_foreign_modules_under_symfluence():
    class Foreign(LoggingMixin):
        pass

    Foreign.__module__ = 'someplugin.module'
    name = Foreign().logger.name
    assert name == 'symfluence.someplugin.module.Foreign'
    assert name.startswith('symfluence.')


def test_class_logger_name_no_double_prefix():
    class Thing:
        pass

    Thing.__module__ = 'symfluence'
    assert _class_logger_name(Thing) == 'symfluence.Thing'
    Thing.__module__ = 'symfluence.core.utils'
    assert _class_logger_name(Thing) == 'symfluence.core.utils.Thing'


# ---------------------------------------------------------------------------
# get_worker_logger
# ---------------------------------------------------------------------------


def test_get_worker_logger_names():
    assert get_worker_logger(3).name == 'symfluence.worker.P03'
    assert get_worker_logger(3, individual_id=7).name == 'symfluence.worker.P03.I007'


def test_get_worker_logger_attaches_handler_only_when_unconfigured():
    # symfluence root has no handlers here (spawned-subprocess situation)
    worker = get_worker_logger(1, individual_id=2)
    assert len(worker.handlers) == 1
    fmt = worker.handlers[0].formatter._fmt
    assert fmt == '[P01-I002] %(levelname)s: %(message)s'

    # Idempotent: a second call must not stack handlers
    get_worker_logger(1, individual_id=2)
    assert len(worker.handlers) == 1


def test_get_worker_logger_no_handler_when_root_configured():
    sym = logging.getLogger('symfluence')
    stream = logging.StreamHandler()
    sym.addHandler(stream)
    try:
        worker = get_worker_logger(9)
        assert worker.handlers == []
        assert worker.propagate is True
    finally:
        sym.removeHandler(stream)
