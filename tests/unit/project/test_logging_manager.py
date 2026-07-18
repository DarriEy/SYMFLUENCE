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
from symfluence.project.logging_manager import (
    CountingHandler,
    LoggingManager,
    get_logger,
)

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
             lm._active_setup_key, lm._active_log_file,
             lm._active_counting_handler)

    sym.handlers = []
    lm._active_setup_key = None
    lm._active_log_file = None
    lm._active_counting_handler = None
    reset_log_once()

    yield

    for handler in sym.handlers:
        if handler not in saved[0]:
            handler.close()
    sym.handlers, sym.level, sym.propagate = saved[0], saved[1], saved[2]
    root.handlers, root.level = saved[3], saved[4]
    lm._active_setup_key, lm._active_log_file = saved[5], saved[6]
    lm._active_counting_handler = saved[7]
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
# CountingHandler and log counts
# ---------------------------------------------------------------------------


def _emit(handler, level, message):
    handler.emit(logging.LogRecord('symfluence.t', level, __file__, 1,
                                   message, (), None))


def test_counting_handler_counts_levels():
    handler = CountingHandler()
    _emit(handler, logging.DEBUG, 'ignored')
    _emit(handler, logging.INFO, 'ignored')
    _emit(handler, logging.WARNING, 'warn 1')
    _emit(handler, logging.WARNING, 'warn 2')
    _emit(handler, logging.ERROR, 'err 1')
    _emit(handler, logging.CRITICAL, 'crit 1')

    assert handler.warning_count == 2
    assert handler.error_count == 2  # ERROR + CRITICAL
    assert handler.recent_errors == ['err 1', 'crit 1']


def test_counting_handler_recent_errors_unique_and_capped():
    handler = CountingHandler(max_recent_errors=3)
    for i in range(5):
        _emit(handler, logging.ERROR, f'error {i}')
    # Repeat an old message: moves to most-recent, no duplicate entry
    _emit(handler, logging.ERROR, 'error 3')

    assert handler.error_count == 6
    assert handler.recent_errors == ['error 2', 'error 4', 'error 3']
    assert len(handler.recent_errors) <= 3


def test_counting_handler_reset():
    handler = CountingHandler()
    _emit(handler, logging.WARNING, 'w')
    _emit(handler, logging.ERROR, 'e')
    handler.reset()
    assert handler.warning_count == 0
    assert handler.error_count == 0
    assert handler.recent_errors == []


def test_setup_logging_attaches_counting_handler_and_counts(tmp_path):
    manager = LoggingManager(_make_config(tmp_path))

    logger = get_logger('data.acquisition')
    logger.warning('low snow fraction')
    logger.error('download failed')
    logger.error('download failed')  # duplicate message, counted twice

    assert manager.log_counts == {'warnings': 1, 'errors': 2}
    assert manager.recent_errors == ['download failed']


def test_log_counts_shared_across_idempotent_re_setup(tmp_path):
    manager1 = LoggingManager(_make_config(tmp_path))
    get_logger('x').error('boom')

    manager2 = LoggingManager(_make_config(tmp_path))
    assert manager2.log_counts['errors'] == 1, \
        "re-setup must reuse the existing counting handler"
    assert manager1.log_counts == manager2.log_counts


# ---------------------------------------------------------------------------
# Run summary: real totals and fixed step schema
# ---------------------------------------------------------------------------


def test_run_summary_reports_real_error_and_warning_totals(tmp_path):
    import json

    manager = LoggingManager(_make_config(tmp_path))
    logger = get_logger('models.summa')
    for _ in range(3):
        logger.error('layerThickness out of bounds')
    logger.warning('CFL condition violated')
    logger.warning('CFL condition violated')

    summary_file = manager.create_run_summary(
        steps=[], execution_time=1.0, status='failed'
    )
    data = json.loads(summary_file.read_text(encoding='utf-8'))

    assert data['total_errors'] == 3, "totals must come from counted log records"
    assert data['total_warnings'] == 2
    assert data['recent_errors'] == ['layerThickness out of bounds']


def test_run_summary_fixed_step_schema_and_pointers(tmp_path):
    import json

    manager = LoggingManager(_make_config(tmp_path))
    steps = [
        {'name': 'setup_project', 'cli_name': 'setup_project',
         'description': 'Setup', 'status': 'completed', 'duration_s': 1.234567},
        {'name': 'define_domain', 'cli_name': 'define_domain',
         'description': 'Define', 'status': 'skipped', 'duration_s': 0.0},
        {'name': 'run_models', 'cli_name': 'run_model',
         'description': 'Run', 'status': 'failed', 'duration_s': 0.0,
         'error': 'executable not found'},
    ]

    summary_file = manager.create_run_summary(
        steps=steps, execution_time=12.3456, status='partial',
        errors=[{'step': 'run_model', 'error': 'executable not found'}],
    )
    data = json.loads(summary_file.read_text(encoding='utf-8'))

    assert data['schema_version'] == 2
    assert data['status'] == 'partial'
    assert data['total_steps'] == 3
    assert data['steps_completed'] == 1
    assert data['steps_skipped'] == 1
    assert data['steps_failed'] == 1

    for entry in data['steps']:
        assert set(entry) >= {'name', 'cli_name', 'description',
                              'status', 'duration_s'}
        assert entry['status'] in ('completed', 'skipped', 'failed')
    assert data['steps'][0]['duration_s'] == 1.235  # rounded, not re-measured

    # Pointers instead of duplicated manifest content
    assert data['log_file'] == str(manager.log_file)
    assert data['run_manifest'] == str(manager.log_dir / 'run_manifest.json')
    assert data['workflow_errors'] == [
        {'step': 'run_model', 'error': 'executable not found'}
    ]


def test_run_summary_normalizes_legacy_step_entries(tmp_path):
    import json

    manager = LoggingManager(_make_config(tmp_path))
    steps = [
        {'cli': 'setup_project', 'fn': 'setup_project', 'success': True,
         'duration': 2.5},
        {'cli': 'run_model', 'fn': 'run_models', 'success': False,
         'error': 'boom'},
        'plain_step_name',
    ]

    summary_file = manager.create_run_summary(steps=steps, execution_time=3.0)
    data = json.loads(summary_file.read_text(encoding='utf-8'))

    statuses = [s['status'] for s in data['steps']]
    assert statuses == ['completed', 'failed', 'completed']
    assert data['steps'][0]['duration_s'] == 2.5
    assert data['steps'][1]['error'] == 'boom'
    assert data['steps'][2]['name'] == 'plain_step_name'


# ---------------------------------------------------------------------------
# Quiet mode: console handler level
# ---------------------------------------------------------------------------


def _console_handlers(logger=None):
    logger = logger or logging.getLogger('symfluence')
    return [h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)]


def test_quiet_mode_raises_console_handler_to_warning(tmp_path):
    manager = LoggingManager(_make_config(tmp_path), quiet_mode=True)
    assert manager.quiet_mode is True

    handlers = _console_handlers()
    assert handlers, "console handler must still be attached in quiet mode"
    assert all(h.level == logging.WARNING for h in handlers)

    # File handler still captures everything
    assert all(h.level == logging.DEBUG for h in _file_handlers())


def test_quiet_mode_default_console_level_is_info(tmp_path):
    LoggingManager(_make_config(tmp_path))
    assert all(h.level == logging.INFO for h in _console_handlers())


def test_debug_mode_wins_over_quiet(tmp_path):
    manager = LoggingManager(_make_config(tmp_path), debug_mode=True,
                             quiet_mode=True)
    assert manager.debug_mode is True
    assert all(h.level == logging.DEBUG for h in _console_handlers())


def test_quiet_toggle_applies_on_idempotent_re_setup(tmp_path):
    LoggingManager(_make_config(tmp_path))
    assert all(h.level == logging.INFO for h in _console_handlers())

    LoggingManager(_make_config(tmp_path), quiet_mode=True)
    assert len(_file_handlers()) == 1, "re-setup must not add handlers"
    assert all(h.level == logging.WARNING for h in _console_handlers())


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
