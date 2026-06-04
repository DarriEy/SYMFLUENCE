# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for BaseService data-directory resolution (issue #156, gap G7).

Covers the cwd-inference + writability-validation behaviour added so that
``symfluence binary install`` lands in a deterministic, writable location for
ad-hoc installs, while preserving the explicit env/config/repo-sibling paths.
"""

from pathlib import Path

import pytest

from symfluence.cli.services.base import BaseService


@pytest.fixture
def service():
    return BaseService()


@pytest.fixture(autouse=True)
def _clear_data_env(monkeypatch):
    """Ensure tests start from a clean env so resolution is deterministic."""
    monkeypatch.delenv("SYMFLUENCE_DATA_DIR", raising=False)
    monkeypatch.delenv("SYMFLUENCE_DATA", raising=False)


class TestResolveDataDir:
    def test_env_var_takes_priority(self, service, monkeypatch, tmp_path):
        monkeypatch.setenv("SYMFLUENCE_DATA_DIR", str(tmp_path / "env_data"))
        path, reason = service._resolve_data_dir({})
        assert path == tmp_path / "env_data"
        assert "environment" in reason.lower()

    def test_config_value_used_when_no_env(self, service, tmp_path):
        cfg = {"SYMFLUENCE_DATA_DIR": str(tmp_path / "cfg_data")}
        path, reason = service._resolve_data_dir(cfg)
        assert path == tmp_path / "cfg_data"
        assert "config" in reason.lower()

    def test_default_sentinel_is_ignored(self, service, monkeypatch, tmp_path):
        """'default' is a sentinel, not a real path — fall through to inference."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(BaseService, "_running_from_repo", staticmethod(lambda: False))
        path, _ = service._resolve_data_dir({"SYMFLUENCE_DATA_DIR": "default"})
        assert path == tmp_path / "SYMFLUENCE_data"

    def test_existing_workspace_at_cwd_is_reused(self, service, monkeypatch, tmp_path):
        workspace = tmp_path / "SYMFLUENCE_data"
        (workspace / "installs").mkdir(parents=True)  # marker: real workspace
        monkeypatch.chdir(tmp_path)
        path, reason = service._resolve_data_dir({})
        assert path == workspace
        assert "existing workspace" in reason.lower()

    def test_existing_workspace_above_cwd_is_reused(self, service, monkeypatch, tmp_path):
        workspace = tmp_path / "SYMFLUENCE_data"
        (workspace / "domain_test").mkdir(parents=True)  # marker: real workspace
        nested = tmp_path / "project" / "sub"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        path, _ = service._resolve_data_dir({})
        assert path == workspace

    def test_empty_named_dir_is_not_adopted(self, service, monkeypatch, tmp_path):
        """A SYMFLUENCE_data dir with no workspace marker is NOT reused."""
        (tmp_path / "SYMFLUENCE_data").mkdir()  # no markers
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(BaseService, "_running_from_repo", staticmethod(lambda: False))
        path, reason = service._resolve_data_dir({})
        # Falls through to the ad-hoc cwd path, which happens to be the same dir
        # name — but the reason proves it was created fresh, not "adopted".
        assert path == tmp_path / "SYMFLUENCE_data"
        assert "current working directory" in reason.lower()

    def test_adhoc_install_infers_cwd(self, service, monkeypatch, tmp_path):
        """No env/config/repo/workspace -> <cwd>/SYMFLUENCE_data, not cwd.parent."""
        project = tmp_path / "myproject"
        project.mkdir()
        monkeypatch.chdir(project)
        monkeypatch.setattr(BaseService, "_running_from_repo", staticmethod(lambda: False))
        path, reason = service._resolve_data_dir({})
        assert path == project / "SYMFLUENCE_data"
        assert "current working directory" in reason.lower()

    def test_repo_sibling_used_when_running_from_repo(self, service, monkeypatch, tmp_path):
        repo = tmp_path / "SYMFLUENCE"
        repo.mkdir()
        monkeypatch.chdir(repo)
        monkeypatch.setattr(BaseService, "_running_from_repo", staticmethod(lambda: True))
        monkeypatch.setattr(
            "symfluence.core.config.factories._resolve_default_data_dir",
            lambda *a, **k: str(tmp_path / "SYMFLUENCE_data"),
        )
        path, reason = service._resolve_data_dir({})
        assert path == tmp_path / "SYMFLUENCE_data"
        assert "repository" in reason.lower()

    def test_non_writable_repo_sibling_falls_back_to_cwd(self, service, monkeypatch, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        monkeypatch.chdir(project)
        monkeypatch.setattr(BaseService, "_running_from_repo", staticmethod(lambda: True))
        monkeypatch.setattr(
            "symfluence.core.config.factories._resolve_default_data_dir",
            lambda *a, **k: "/nonexistent-root/SYMFLUENCE_data",
        )
        # /nonexistent-root has no existing writable ancestor -> not writable.
        path, reason = service._resolve_data_dir({})
        assert path == project / "SYMFLUENCE_data"
        assert "current working directory" in reason.lower()


class TestHelpers:
    def test_is_writable_dir_for_existing_dir(self, tmp_path):
        assert BaseService._is_writable_dir(tmp_path) is True

    def test_is_writable_dir_for_creatable_child(self, tmp_path):
        assert BaseService._is_writable_dir(tmp_path / "new" / "deep") is True

    def test_is_writable_dir_for_unrootable_path(self):
        assert BaseService._is_writable_dir(Path("/this-root-should-not-exist-xyz/x")) is False

    def test_find_existing_data_dir_none_when_absent(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert BaseService._find_existing_data_dir() is None

    def test_find_existing_data_dir_respects_depth_limit(self, monkeypatch, tmp_path):
        ws = tmp_path / "SYMFLUENCE_data"
        (ws / "installs").mkdir(parents=True)  # a real workspace, but far away
        deep = tmp_path.joinpath(*[f"l{i}" for i in range(8)])
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)
        # The workspace is more than max_levels above cwd -> not found.
        assert BaseService._find_existing_data_dir(max_levels=3) is None

    def test_looks_like_workspace_requires_marker(self, tmp_path):
        plain = tmp_path / "SYMFLUENCE_data"
        plain.mkdir()
        assert BaseService._looks_like_workspace(plain) is False
        (plain / "installs").mkdir()
        assert BaseService._looks_like_workspace(plain) is True

    def test_looks_like_workspace_accepts_domain_marker(self, tmp_path):
        ws = tmp_path / "SYMFLUENCE_data"
        (ws / "domain_bow").mkdir(parents=True)
        assert BaseService._looks_like_workspace(ws) is True

    def test_get_data_dir_returns_path_only(self, service, monkeypatch, tmp_path):
        monkeypatch.setenv("SYMFLUENCE_DATA_DIR", str(tmp_path / "d"))
        assert service._get_data_dir({}) == tmp_path / "d"
