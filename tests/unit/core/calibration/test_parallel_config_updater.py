# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Parallel-calibration config rewriting: model-owned, value-for-value.

``ConfigurationUpdater`` used to carry the last hardcoded per-model table in
``core``: an ``if model_upper == 'SUMMA' / 'FUSE' / 'GR' / 'HYPE'`` chain
choosing the mizuRoute control-file settings for a process, plus two 'HYPE'
special cases in the file-manager rewriter. Those values now live on each
model's registered ``ModelConfigSchema.parallel_calibration``.

This module pins the *generated control-file settings*, per model, line for
line — not merely that the code runs. The values asserted are the ones core
produced before the move, including the ones known to disagree with what the
model actually writes (GR's ``hru``/``hruId`` and ``3600``; HYPE's
``*_timestep.nc``). Those are recorded as-is and tracked separately: an
extraction pass must not change a calibration artifact mid-campaign, and a
value that quietly "improves" here is exactly the kind of regression that is
expensive to attribute.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from symfluence.core.calibration.mixins.parallel import config_updater as cu_module
from symfluence.core.calibration.mixins.parallel.config_updater import ConfigurationUpdater
from symfluence.core.modeling import config_schema as cs
from symfluence.core.modeling.config_schema import (
    DEFAULT_PARALLEL_CALIBRATION,
    ExecutionConfig,
    InputConfig,
    InstallationConfig,
    ModelConfigSchema,
    OutputConfig,
    ParallelCalibrationConfig,
    parallel_calibration_config,
    register_model_schema,
)

pytestmark = [pytest.mark.unit]


CONTROL_TEMPLATE = """! mizuRoute control file
<ancil_dir>             /old/ancil/    ! ancillary
<input_dir>             /old/input/    ! input
<output_dir>            /old/output/   ! output
<case_name>             oldcase        ! case
<fname_qsim>            old.nc         ! netCDF name
<vname_qsim>            oldvar         ! Variable name
<dt_qsim>               999            ! Time interval
<dname_hruid>           oldhru         ! Dimension name
<vname_hruid>           oldhruId       ! Variable name
<sim_start>             2010-01-01 06:00    ! start
<sim_end>               2011-12-31 06:00    ! end
<other_key>             untouched      ! passthrough
"""

FILE_MANAGER_TEMPLATE = """controlVersion       'SUMMA_FILE_MANAGER_V3.0.0'
settingsPath         '/old/settings/'
outputPath           '/old/output/'
outFilePrefix        'oldprefix'
forcingFreq          'month'
! a comment line
resultdir            noQuotesHere
"""

HYPE_INFO_TEMPLATE = """!! HYPE info file
bdate\t2000-01-01
edate\t2001-12-31
resultdir\t/old/results/
instate\tn
"""

PROC_ID = 3
EXPERIMENT_ID = 'run_1'
BASE_CONFIG = {'DOMAIN_NAME': 'testdom', 'EXPERIMENT_ID': EXPERIMENT_ID}


def _parallel_dirs(root: Path, model: str) -> dict:
    proc = root / f'process_{PROC_ID}'
    dirs = {
        'root': proc,
        'sim_dir': proc / 'simulations' / EXPERIMENT_ID / model,
        'settings_dir': proc / 'settings' / model,
        'output_dir': proc / 'output',
    }
    for key in ('sim_dir', 'settings_dir', 'output_dir'):
        dirs[key].mkdir(parents=True, exist_ok=True)
    (proc / 'settings' / 'mizuRoute').mkdir(parents=True, exist_ok=True)
    return {PROC_ID: dirs}


def _write_control(tmp_path: Path, model: str, config: dict) -> list:
    """Run the real updater over a control-file template; return its lines."""
    dirs = _parallel_dirs(tmp_path, model)
    control = dirs[PROC_ID]['settings_dir'].parent / 'mizuRoute' / 'mizuroute.control'
    control.write_text(CONTROL_TEMPLATE, encoding='utf-8')

    ConfigurationUpdater(dict(config)).update_mizuroute_controls(
        dirs, model, EXPERIMENT_ID)
    return control.read_text(encoding='utf-8').splitlines()


def _write_settings(tmp_path: Path, model: str, template: str, name: str) -> list:
    dirs = _parallel_dirs(tmp_path, model)
    (dirs[PROC_ID]['settings_dir'] / name).write_text(template, encoding='utf-8')

    ConfigurationUpdater(dict(BASE_CONFIG)).update_file_managers(
        dirs, model, EXPERIMENT_ID, name)
    return (dirs[PROC_ID]['settings_dir'] / name).read_text(
        encoding='utf-8').splitlines()


# ---------------------------------------------------------------------------
# The control-file settings each model produces, line for line
# ---------------------------------------------------------------------------

# proc_id is 3 and experiment_id 'run_1', so a proc-prefixed name reads
# 'proc_03_run_1_timestep.nc'. The <sim_start>/<sim_end> dates come from the
# template; only the time-of-day is model-declared.
_EXPECTED_CONTROL = {
    # SUMMA writes proc_{NN}_{exp}_timestep.nc because ConfigurationUpdater
    # sets its fileManager outFilePrefix to 'proc_{NN}_{exp}'.
    'SUMMA': [
        '<fname_qsim>            proc_03_run_1_timestep.nc    ! netCDF name for SUMMA runoff',
        '<vname_qsim>            averageRoutedRunoff    ! Variable name for SUMMA runoff',
        '<dt_qsim>               3600    ! Time interval of input runoff in seconds',
        '<dname_hruid>           gru     ! Dimension name for HM_HRU ID',
        '<vname_hruid>           gruId   ! Variable name for HM_HRU ID',
        '<sim_start>             2010-01-01 00:00    ! Time of simulation start',
        '<sim_end>               2011-12-31 00:00    ! Time of simulation end',
    ],
    # FUSE's parallel artifact is NOT its runoff declaration's
    # '{domain_name}_{experiment_id}_runs_def.nc': FuseToMizurouteConverter
    # rewrites that into 'proc_{NN}_{exp}_timestep.nc' before mizuRoute runs.
    'FUSE': [
        '<fname_qsim>            proc_03_run_1_timestep.nc    ! netCDF name for FUSE runoff',
        '<vname_qsim>            q_routed    ! Variable name for FUSE runoff',
        '<dt_qsim>               86400    ! Time interval of input runoff in seconds',
        '<dname_hruid>           gru     ! Dimension name for HM_HRU ID',
        '<vname_hruid>           gruId   ! Variable name for HM_HRU ID',
        '<sim_start>             2010-01-01 00:00    ! Time of simulation start',
        '<sim_end>               2011-12-31 00:00    ! Time of simulation end',
    ],
    # GR keeps its runoff-declaration filename with NO proc_ prefix — correct,
    # because GRRunner writes it into the per-process sim dir, which is what
    # <input_dir> points at. The hru/hruId, 3600 and 01:00/23:00 below are the
    # preserved-as-is values that disagree with GR's daily (time, gru) output.
    'GR': [
        '<fname_qsim>            testdom_run_1_runs_def.nc    ! netCDF name for GR runoff',
        '<vname_qsim>            q_routed    ! Variable name for GR runoff',
        '<dt_qsim>               3600    ! Time interval of input runoff in seconds',
        '<dname_hruid>           hru     ! Dimension name for HM_HRU ID',
        '<vname_hruid>           hruId   ! Variable name for HM_HRU ID',
        '<sim_start>             2010-01-01 01:00    ! Time of simulation start',
        '<sim_end>               2011-12-31 23:00    ! Time of simulation end',
    ],
    # HYPE: preserved verbatim although nothing in the HYPE package writes a
    # '*_timestep.nc'. See the schema comment — reachable only via a hand-placed
    # settings/mizuRoute, and removing it is a behaviour change of its own.
    'HYPE': [
        '<fname_qsim>            proc_03_run_1_timestep.nc    ! netCDF name for HYPE runoff',
        '<vname_qsim>            q_routed    ! Variable name for HYPE runoff',
        '<dt_qsim>               86400    ! Time interval of input runoff in seconds',
        '<dname_hruid>           hru     ! Dimension name for HM_HRU ID',
        '<vname_hruid>           hruId   ! Variable name for HM_HRU ID',
        '<sim_start>             2010-01-01 00:00    ! Time of simulation start',
        '<sim_end>               2011-12-31 00:00    ! Time of simulation end',
    ],
    # A model with no declaration: the old else-branch, unchanged.
    'NGEN': [
        '<fname_qsim>            proc_03_run_1_timestep.nc    ! netCDF name for NGEN runoff',
        '<vname_qsim>            q_routed    ! Variable name for NGEN runoff',
        '<dt_qsim>               3600    ! Time interval of input runoff in seconds',
        '<dname_hruid>           hru     ! Dimension name for HM_HRU ID',
        '<vname_hruid>           hruId   ! Variable name for HM_HRU ID',
        '<sim_start>             2010-01-01 01:00    ! Time of simulation start',
        '<sim_end>               2011-12-31 23:00    ! Time of simulation end',
    ],
}


@pytest.mark.parametrize('model', sorted(_EXPECTED_CONTROL))
def test_generated_control_file_settings(tmp_path, model):
    """Every model-dependent control-file line, exactly as core produced it."""
    lines = _write_control(tmp_path / model, model, BASE_CONFIG)
    for expected in _EXPECTED_CONTROL[model]:
        directive = expected.split()[0]
        actual = [ln for ln in lines if ln.startswith(directive)]
        assert actual == [expected], f'{model} {directive}'


@pytest.mark.parametrize('model', sorted(_EXPECTED_CONTROL))
def test_model_independent_lines_are_untouched_by_the_move(tmp_path, model):
    """Paths, case name and unrecognised directives stay core's business."""
    lines = _write_control(tmp_path / model, model, BASE_CONFIG)
    assert any(ln.startswith('<case_name>             proc_03_run_1') for ln in lines)
    assert '<other_key>             untouched      ! passthrough' in lines


def test_unregistered_model_keeps_the_historical_else_branch(tmp_path):
    """A name core never had a branch for must not start raising or changing."""
    lines = _write_control(tmp_path, 'NOT_A_MODEL', BASE_CONFIG)
    assert ('<fname_qsim>            proc_03_run_1_timestep.nc    '
            '! netCDF name for NOT_A_MODEL runoff') in lines
    assert ('<vname_qsim>            q_routed    '
            '! Variable name for NOT_A_MODEL runoff') in lines
    assert parallel_calibration_config('NOT_A_MODEL') is DEFAULT_PARALLEL_CALIBRATION


@pytest.mark.parametrize('model', ['summa', 'FuSe', '  gr  '])
def test_declaration_lookup_is_case_and_whitespace_insensitive(model):
    canonical = parallel_calibration_config(model.strip().upper())
    assert parallel_calibration_config(model) is canonical
    assert canonical is not DEFAULT_PARALLEL_CALIBRATION


# ---------------------------------------------------------------------------
# Which config overrides each model honours
# ---------------------------------------------------------------------------

# SETTINGS_MIZU_ROUTING_VAR was read on the FUSE/GR/HYPE branches only. SUMMA's
# averageRoutedRunoff and the else-branch's q_routed were literals.
@pytest.mark.parametrize('model,expected', [
    ('SUMMA', 'averageRoutedRunoff'),
    ('FUSE', 'custom_var'),
    ('GR', 'custom_var'),
    ('HYPE', 'custom_var'),
    ('NGEN', 'q_routed'),
])
def test_routing_var_override_reaches_only_declaring_models(tmp_path, model, expected):
    config = dict(BASE_CONFIG, SETTINGS_MIZU_ROUTING_VAR='custom_var')
    lines = _write_control(tmp_path / model, model, config)
    assert (f'<vname_qsim>            {expected}    '
            f'! Variable name for {model} runoff') in lines


@pytest.mark.parametrize('sentinel', ['default', ''])
@pytest.mark.parametrize('model', ['FUSE', 'GR', 'HYPE'])
def test_routing_var_sentinels_fall_back_to_the_declared_default(
        tmp_path, model, sentinel):
    config = dict(BASE_CONFIG, SETTINGS_MIZU_ROUTING_VAR=sentinel)
    lines = _write_control(tmp_path / f'{model}{sentinel}', model, config)
    assert (f'<vname_qsim>            q_routed    '
            f'! Variable name for {model} runoff') in lines


# FUSE and HYPE pinned dt_qsim to 86400 regardless of config; everyone else
# took SETTINGS_MIZU_ROUTING_DT.
@pytest.mark.parametrize('model,expected', [
    ('SUMMA', '7200'),
    ('FUSE', '86400'),
    ('GR', '7200'),
    ('HYPE', '86400'),
    ('NGEN', '7200'),
])
def test_routing_dt_override_is_ignored_where_the_cadence_is_pinned(
        tmp_path, model, expected):
    config = dict(BASE_CONFIG, SETTINGS_MIZU_ROUTING_DT='7200')
    lines = _write_control(tmp_path / model, model, config)
    assert (f'<dt_qsim>               {expected}    '
            '! Time interval of input runoff in seconds') in lines


@pytest.mark.parametrize('sentinel', ['default', ''])
def test_routing_dt_sentinels_fall_back_to_3600(tmp_path, sentinel):
    config = dict(BASE_CONFIG, SETTINGS_MIZU_ROUTING_DT=sentinel)
    lines = _write_control(tmp_path / f'summa{sentinel}', 'SUMMA', config)
    assert ('<dt_qsim>               3600    '
            '! Time interval of input runoff in seconds') in lines


# ---------------------------------------------------------------------------
# Settings-file dialect (the two 'HYPE' branches that were not about routing)
# ---------------------------------------------------------------------------

def test_hype_info_file_dialect_is_declared_not_hardcoded(tmp_path):
    """HYPE's info.txt is bare tab-separated key/value, and 'resultdir' is its
    output directive. Both were 'HYPE' string tests in core."""
    lines = _write_settings(tmp_path, 'HYPE', HYPE_INFO_TEMPLATE, 'info.txt')

    result = [ln for ln in lines if ln.startswith('resultdir')]
    assert len(result) == 1
    directive, value = result[0].split('\t')
    assert directive == 'resultdir'
    assert value.endswith(f'process_{PROC_ID}/output/')
    # Everything else in an unquoted file survives untouched.
    assert 'bdate\t2000-01-01' in lines
    assert 'instate\tn' in lines
    assert '!! HYPE info file' in lines


def test_quoted_dialect_models_pass_unquoted_lines_through(tmp_path):
    """A model whose file manager quotes values keeps the old early-continue.

    The template's ``resultdir noQuotesHere`` line is the guard: for a
    quoted-dialect model it must be preserved verbatim, never treated as an
    output directive.
    """
    lines = _write_settings(tmp_path, 'SUMMA', FILE_MANAGER_TEMPLATE,
                            'fileManager.txt')

    assert 'resultdir            noQuotesHere' in lines
    assert "outFilePrefix        'proc_03_run_1'" in lines
    assert any(ln.startswith('settingsPath') and ln.endswith("/settings/SUMMA/'")
               for ln in lines)
    assert any(ln.startswith('outputPath') and ln.endswith("/run_1/SUMMA/'")
               for ln in lines)
    assert "forcingFreq          'month'" in lines
    assert '! a comment line' in lines


# ---------------------------------------------------------------------------
# The seam: registration is what makes core see a model
# ---------------------------------------------------------------------------

@pytest.fixture
def _clean_registry():
    saved = dict(cs.REGISTERED_SCHEMAS)
    yield
    cs.REGISTERED_SCHEMAS.clear()
    cs.REGISTERED_SCHEMAS.update(saved)


def _ext_schema(**overrides) -> ModelConfigSchema:
    kwargs = dict(
        model_name='ExtModel',
        installation=InstallationConfig('EXT_INSTALL_PATH', 'installs/ext'),
        execution=ExecutionConfig(),
        input=InputConfig('FORCING_EXT_PATH', 'forcing/EXT_input'),
        output=OutputConfig('EXPERIMENT_OUTPUT_EXT', 'simulations/{experiment_id}/EXT'),
    )
    kwargs.update(overrides)
    return ModelConfigSchema(**kwargs)


def test_external_package_drives_the_control_file_by_registering(
        tmp_path, _clean_registry):
    """An out-of-tree model becomes parallel-calibratable with no core edit."""
    register_model_schema('extmodel', _ext_schema(
        parallel_calibration=ParallelCalibrationConfig(
            fname_pattern='{domain_name}_proc{proc_id:03d}_{experiment_id}_ext.nc',
            runoff_var='q_ext',
            runoff_var_from_config=False,
            dt_qsim='900',
            sim_start_time='12:00',
            sim_end_time='12:00',
            hru_dim='subbasin',
            hru_var='subbasinId',
        ),
    ))

    lines = _write_control(tmp_path, 'EXTMODEL', BASE_CONFIG)

    assert ('<fname_qsim>            testdom_proc003_run_1_ext.nc    '
            '! netCDF name for EXTMODEL runoff') in lines
    assert ('<vname_qsim>            q_ext    '
            '! Variable name for EXTMODEL runoff') in lines
    assert ('<dt_qsim>               900    '
            '! Time interval of input runoff in seconds') in lines
    assert '<dname_hruid>           subbasin     ! Dimension name for HM_HRU ID' in lines
    assert '<vname_hruid>           subbasinId   ! Variable name for HM_HRU ID' in lines
    assert '<sim_start>             2010-01-01 12:00    ! Time of simulation start' in lines
    assert '<sim_end>               2011-12-31 12:00    ! Time of simulation end' in lines


def test_external_package_can_declare_an_unquoted_settings_dialect(
        tmp_path, _clean_registry):
    register_model_schema('extmodel', _ext_schema(
        parallel_calibration=ParallelCalibrationConfig(
            settings_values_quoted=False,
            output_dir_directive='outdir',
        ),
    ))

    lines = _write_settings(tmp_path, 'EXTMODEL', 'outdir\t/old/\nkeep\tme\n',
                            'ext_settings.txt')

    assert any(ln.startswith('outdir\t') and ln.endswith('/output/') for ln in lines)
    assert 'keep\tme' in lines


def test_a_registered_schema_without_the_section_keeps_the_defaults(_clean_registry):
    """Absence is a declaration: registering a schema changes nothing by itself."""
    register_model_schema('extmodel', _ext_schema())
    assert parallel_calibration_config('EXTMODEL') is DEFAULT_PARALLEL_CALIBRATION


def test_the_default_is_the_historical_else_branch():
    """Stated once, so a default change cannot slip past as a refactor."""
    assert DEFAULT_PARALLEL_CALIBRATION == ParallelCalibrationConfig(
        fname_pattern='proc_{proc_id:02d}_{experiment_id}_timestep.nc',
        runoff_var='q_routed',
        runoff_var_from_config=False,
        dt_qsim=None,
        sim_start_time='01:00',
        sim_end_time='23:00',
        hru_dim='hru',
        hru_var='hruId',
        settings_values_quoted=True,
        output_dir_directive=None,
    )


# ---------------------------------------------------------------------------
# Core must hold no model-name knowledge in this module
# ---------------------------------------------------------------------------

def test_config_updater_branches_on_no_model_name():
    """The point of the move: no ``model_name == 'SUMMA'`` may come back.

    An AST scan of comparisons (and ``in``-tests against literal tuples), so
    docstrings and comments that merely *mention* a model stay legal while a
    literal model name reaching a branch condition fails.
    """
    source = Path(cu_module.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)

    known = {name.upper() for name in cs.REGISTERED_SCHEMAS}
    assert 'SUMMA' in known and 'HYPE' in known, 'schemas must be registered'

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        literals = []
        for operand in operands:
            if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                literals.append(operand.value)
            elif isinstance(operand, (ast.Tuple, ast.List, ast.Set)):
                literals.extend(
                    elt.value for elt in operand.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                )
        offenders.extend(
            (node.lineno, value) for value in literals if value.upper() in known
        )

    assert not offenders, (
        f'{cu_module.__file__} compares against model-name literals {offenders} — '
        'declare the value on the model\'s ModelConfigSchema.parallel_calibration '
        'instead'
    )
