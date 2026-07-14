# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Concurrency contract for the domain-shared, model-agnostic forcing.

Several workflows (different models, different experiments) routinely run
against a single domain at the same time. The basin-averaged forcing and the
model-ready store built from it are shared by all of them: they are produced
once by the model-agnostic stages and are read — never rewritten — by per-model
preprocessing.

These tests pin the invariants that make that safe:

* an existing shared forcing output is never unlinked ahead of a rebuild
  (readers must never see a gap),
* a rebuild leaves already-enriched shared files byte-identical,
* a rebuild never prunes a store link whose source is present (it may have been
  published by another process after our snapshot was taken),
* concurrent store builds plus concurrent readers never lose or truncate a file.
"""
from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path

import pytest

from symfluence.data.model_ready.forcings_builder import ForcingsStoreBuilder
from symfluence.data.preprocessing.resampling.file_processor import FileProcessor

netCDF4 = pytest.importorskip('netCDF4')

pytestmark = [pytest.mark.unit]

TEMP_K = 280.0
# Long enough that the file clears FileValidator.MIN_FILE_SIZE (100 kB): these
# fixtures stand in for real remapped forcing, and a file under that floor is
# treated as metadata-only and queued for reprocessing.
N_TIME = 20000


def _write_forcing(path: Path) -> None:
    """Write a minimal, complete basin-averaged forcing file."""
    import numpy as np

    ds = netCDF4.Dataset(str(path), 'w', format='NETCDF4_CLASSIC')
    ds.createDimension('time', N_TIME)
    ds.createDimension('hru', 1)
    t = ds.createVariable('time', 'f8', ('time',))
    t.units = 'hours since 2000-01-01 00:00:00'
    t[:] = np.arange(N_TIME, dtype='f8')
    v = ds.createVariable('air_temperature', 'f4', ('time', 'hru'))
    v[:] = np.full((N_TIME, 1), TEMP_K, dtype='f4')
    p = ds.createVariable('surface_air_pressure', 'f4', ('time', 'hru'))
    p[:] = np.full((N_TIME, 1), 90000.0, dtype='f4')
    ds.close()


def _assert_complete(path: Path) -> None:
    """A reader must always see a complete file: full time axis, sane values."""
    with netCDF4.Dataset(str(path), 'r') as ds:
        assert ds.dimensions['time'].size == N_TIME, f"truncated time axis in {path.name}"
        temps = ds.variables['air_temperature'][:]
        assert temps.shape == (N_TIME, 1)
        assert (temps == pytest.approx(TEMP_K)).all(), f"corrupt values in {path.name}"
        pres = ds.variables['surface_air_pressure'][:]
        # The exact failure seen in the field: a mid-write read yields 0.0,
        # which the SUMMA forcing validator rejects as out of range.
        assert (pres > 50000.0).all(), f"mid-write read of {path.name}"


def _source_dir(project_dir: Path) -> Path:
    return project_dir / 'forcing' / 'basin_averaged_data'


def _builder(project_dir: Path) -> ForcingsStoreBuilder:
    return ForcingsStoreBuilder(
        project_dir=project_dir,
        domain_name='bow',
        forcing_dataset='RDRS',
        strategy='symlink',
    )


def _build_worker(project_dir: str, rounds: int) -> None:
    """One workflow's repeated model-ready store builds (separate process)."""
    from symfluence.data.model_ready.forcings_builder import ForcingsStoreBuilder

    for _ in range(rounds):
        ForcingsStoreBuilder(
            project_dir=Path(project_dir),
            domain_name='bow',
            forcing_dataset='RDRS',
            strategy='symlink',
        ).build()


def _read_worker(project_dir: str, names: list, rounds: int) -> None:
    """Another workflow's per-model preprocessing reading the shared store.

    Any missing/truncated/corrupt read raises, so a non-zero exit code is the
    signal the test asserts on.
    """
    target = Path(project_dir) / 'data' / 'model_ready' / 'forcings'
    for _ in range(rounds):
        for name in names:
            _assert_complete(target / name)


class TestSharedForcingIsNotUnlinked:
    """FileProcessor must not delete the shared remapped forcing to rebuild it."""

    def test_forced_rerun_keeps_existing_output_until_republished(self, tmp_path):
        src_dir = tmp_path / 'forcing' / 'raw'
        out_dir = _source_dir(tmp_path)
        src_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)

        source = src_dir / 'bow_RDRS_2015.nc'
        _write_forcing(source)

        config = {
            'DOMAIN_NAME': 'bow',
            'FORCING_DATASET': 'RDRS',
            'SYMFLUENCE_DATA_DIR': str(tmp_path.parent),
            'FORCE_RUN_ALL_STEPS': True,
        }
        processor = FileProcessor(config, out_dir, logging.getLogger('test'))

        output = processor.determine_output_filename(source)
        _write_forcing(output)

        remaining = processor.filter_processed_files([source])

        assert remaining == [source], "forced rerun must queue the file for reprocessing"
        assert output.exists(), (
            "the shared remapped forcing was unlinked ahead of the rebuild — "
            "concurrent readers on this domain get FileNotFoundError"
        )
        _assert_complete(output)

    def test_unforced_rerun_skips_current_output(self, tmp_path):
        src_dir = tmp_path / 'forcing' / 'raw'
        out_dir = _source_dir(tmp_path)
        src_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)

        source = src_dir / 'bow_RDRS_2015.nc'
        _write_forcing(source)

        config = {
            'DOMAIN_NAME': 'bow',
            'FORCING_DATASET': 'RDRS',
            'SYMFLUENCE_DATA_DIR': str(tmp_path.parent),
            'FORCE_RUN_ALL_STEPS': False,
        }
        processor = FileProcessor(config, out_dir, logging.getLogger('test'))
        _write_forcing(processor.determine_output_filename(source))

        assert processor.filter_processed_files([source]) == []


class TestStoreBuildIsImmutable:
    """A rebuild must not touch shared forcing files it has already enriched."""

    def test_rebuild_leaves_enriched_source_untouched(self, tmp_path):
        src = _source_dir(tmp_path)
        src.mkdir(parents=True)
        forcing = src / 'bow_RDRS_remapped_2015.nc'
        _write_forcing(forcing)

        _builder(tmp_path).build()
        before = forcing.stat()

        # A second model's workflow rebuilds the store on the same domain.
        _builder(tmp_path).build()
        after = forcing.stat()

        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns), (
            "rebuild rewrote the shared basin-averaged forcing in place; "
            "another model reading it would see a half-updated NetCDF"
        )
        _assert_complete(forcing)

    def test_rebuild_keeps_link_published_by_another_process(self, tmp_path):
        src = _source_dir(tmp_path)
        src.mkdir(parents=True)
        first = src / 'bow_RDRS_remapped_2015.nc'
        _write_forcing(first)

        builder = _builder(tmp_path)
        builder.build()
        snapshot = [first]  # this process's view of the source dir

        # Another process publishes a second remapped file and links it.
        second = src / 'bow_RDRS_remapped_2016.nc'
        _write_forcing(second)
        _builder(tmp_path).build()

        link = tmp_path / 'data' / 'model_ready' / 'forcings' / second.name
        assert link.exists()

        # Our stale snapshot must not prune the link the other process just made.
        builder._create_links(snapshot)

        assert link.exists(), (
            "prune deleted a store link whose source exists — a concurrent "
            "workflow loses the forcing file it is about to read"
        )
        _assert_complete(link)

    def test_prune_removes_links_with_no_source(self, tmp_path):
        src = _source_dir(tmp_path)
        src.mkdir(parents=True)
        forcing = src / 'bow_RDRS_remapped_2015.nc'
        _write_forcing(forcing)

        builder = _builder(tmp_path)
        builder.build()

        target = tmp_path / 'data' / 'model_ready' / 'forcings'
        orphan = target / 'bow_RDRS_remapped_1999.nc'
        orphan.symlink_to(src / 'gone.nc')

        builder._create_links([forcing])

        assert not orphan.exists() and not orphan.is_symlink(), (
            "a link with no source file must still be pruned"
        )


class TestConcurrentModelsOnOneDomain:
    """Two models preprocessing on one domain: builds and reads must not collide.

    Modelled with processes, not threads: the field failure is two ``symfluence
    workflow run`` invocations (different models, one domain), and HDF5 is not
    thread-safe, so threads would test the wrong thing.
    """

    def test_concurrent_builds_and_reads_never_lose_a_file(self, tmp_path):
        src = _source_dir(tmp_path)
        src.mkdir(parents=True)
        forcings = [src / f'bow_RDRS_remapped_{year}.nc' for year in (2015, 2016, 2017)]
        for f in forcings:
            _write_forcing(f)

        target = tmp_path / 'data' / 'model_ready' / 'forcings'
        _builder(tmp_path).build()  # each workflow builds the store before preprocessing

        ctx = multiprocessing.get_context('spawn')
        procs = [
            ctx.Process(target=_build_worker, args=(str(tmp_path), 6)),
            ctx.Process(target=_build_worker, args=(str(tmp_path), 6)),
            ctx.Process(target=_read_worker, args=(str(tmp_path), [f.name for f in forcings], 20)),
            ctx.Process(target=_read_worker, args=(str(tmp_path), [f.name for f in forcings], 20)),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)

        failures = [
            f"worker {i} exited with {p.exitcode}"
            for i, p in enumerate(procs)
            if p.exitcode != 0
        ]
        for p in procs:
            if p.is_alive():
                p.terminate()

        assert not failures, (
            "concurrent models on one domain corrupted the shared forcing:\n"
            + "\n".join(failures)
        )

        for f in forcings:
            assert f.exists()
            _assert_complete(f)
            assert (target / f.name).exists()
