# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression guards for the Windows binary-release packaging.

Two defects were found by running the paper reproduction on native Windows, and
both are invisible to a normal test run because they only manifest in the
release pipeline:

1. MSYS2's stock netCDF links the AWS S3 SDK. Its ``DLL_PROCESS_DETACH`` hook
   calls ``Aws::ShutdownAPI``, which waits forever on AWS CRT worker threads
   that ``RtlExitUserProcess`` has already terminated, so every model process
   became an unkillable zombie holding its output NetCDF handles.
2. The Windows dependency bundler enumerated images with an ``*.exe`` glob, but
   staging deliberately drops the ``.exe`` suffix from tool names. Every Fortran
   model was therefore skipped, ``libnetcdff-7.dll`` was never bundled, and HYPE
   aborted with 0xC0000135 before ``main()`` — printing nothing at all, which
   calibration recorded as a -9999 score rather than a failure.

These assertions pin the shape of the fixes so neither can silently regress.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release-binaries.yml"

STAGE_SCRIPT = SCRIPTS / "stage_release_artifacts.sh"
CLOSURE_SCRIPT = SCRIPTS / "check_windows_dll_closure.sh"
NETCDF_SCRIPT = SCRIPTS / "build_netcdf_no_s3_mingw.sh"


@pytest.fixture(scope="module")
def stage_text() -> str:
    return STAGE_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def closure_text() -> str:
    return CLOSURE_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def netcdf_text() -> str:
    return NETCDF_SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_packaging_helper_scripts_exist() -> None:
    for script in (STAGE_SCRIPT, CLOSURE_SCRIPT, NETCDF_SCRIPT):
        assert script.is_file(), f"missing release script: {script}"


def test_windows_bundler_enumerates_pe_images_by_content(stage_text: str) -> None:
    """The bundler must not filter staged images by filename extension.

    ``stage_binary`` stores Windows executables under their bare tool names
    (bin/hype, bin/summa, ...) so the runners and npm shims can find them, so an
    extension glob sees almost nothing.
    """
    # Comments quote the old glob to explain the bug; only code matters here.
    code = "\n".join(
        line for line in stage_text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "ls bin/*.exe" not in code, (
        "Windows DLL bundling must not enumerate staged images with an *.exe "
        "glob — staged tool names have no extension, so Fortran models "
        "(and therefore libnetcdff-7.dll) would be skipped."
    )
    assert "PE32" in stage_text, (
        "Windows DLL bundling should detect PE images by file content."
    )


def test_staging_gates_on_the_windows_import_closure(stage_text: str) -> None:
    assert "check_windows_dll_closure.sh" in stage_text
    # The gate has to be fatal: a package with an unsatisfied import closure
    # fails silently at runtime, so it must never be produced.
    gate = stage_text.split("check_windows_dll_closure.sh")[-1]
    assert "exit 1" in gate, "the import-closure check must fail the staging step"


def test_closure_check_reads_real_import_tables(closure_text: str) -> None:
    assert "objdump" in closure_text
    assert "DLL Name:" in closure_text, (
        "the closure check must derive dependencies from each image's import "
        "table rather than a hand-maintained per-tool DLL list"
    )
    assert "0xC0000135" in closure_text, "explain the failure mode being guarded"


def test_netcdf_build_disables_s3_and_keeps_byterange(netcdf_text: str) -> None:
    assert "-DNETCDF_ENABLE_S3=OFF" in netcdf_text
    assert "-DNETCDF_ENABLE_S3_AWS=OFF" in netcdf_text
    # Byte-range reads are curl-only and were never implicated; dropping them
    # would remove functionality users may depend on.
    assert "-DNETCDF_ENABLE_BYTERANGE=ON" in netcdf_text
    assert "-DNETCDF_ENABLE_DAP=ON" in netcdf_text


def test_netcdf_build_proves_the_result_is_a_drop_in(netcdf_text: str) -> None:
    """The rebuilt DLL replaces a packaged one; assert the checks that prove it."""
    assert "NC_s3sdk" in netcdf_text, "export-table diff must be allow-listed"
    assert "libnetcdff-7.dll" in netcdf_text, (
        "must verify the Fortran binding's imported symbols survive the rebuild"
    )
    assert "S3 Support:" in netcdf_text, "must assert on libnetcdf.settings"


def test_windows_release_job_rebuilds_netcdf(workflow_text: str) -> None:
    assert "build_netcdf_no_s3_mingw.sh" in workflow_text, (
        "the Windows release job must produce an S3-free netCDF rather than "
        "consuming the stock MSYS2 package"
    )


def test_release_workflow_verifies_the_shipped_package(workflow_text: str) -> None:
    # Once for the extracted tarball, once for what npm actually installs.
    assert workflow_text.count("check_windows_dll_closure.sh") >= 2, (
        "verify the import closure of both the built tarball and the published "
        "npm package"
    )
    assert "DLL Name: libaws" in workflow_text, (
        "the shipped libnetcdf.dll must be checked for AWS SDK imports"
    )
