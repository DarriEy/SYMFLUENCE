"""Unit tests for system dependency registry and platform detection."""

import os
import re
import subprocess
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from symfluence.cli.services.system_deps import (
    DepCheckResult,
    Platform,
    SystemDepsRegistry,
)

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


# ── Helpers ──────────────────────────────────────────────────────────────

def _fresh_registry(**env_overrides):
    """Create a fresh SystemDepsRegistry, bypassing the singleton.

    Resets the class-level singleton so each test gets a clean instance.
    Any *env_overrides* are patched into ``os.environ`` during construction.
    """
    SystemDepsRegistry._instance = None
    SystemDepsRegistry._registry = None
    with patch.dict(os.environ, env_overrides, clear=False):
        return SystemDepsRegistry()


# ── Platform enum ────────────────────────────────────────────────────────

class TestPlatformEnum:
    """Verify Platform enum contains all expected members."""

    def test_all_platforms_present(self):
        values = {p.value for p in Platform}
        assert values == {
            "apt", "dnf", "brew", "conda", "hpc_module",
            "msys2", "wsl", "unknown",
        }

    def test_enum_from_value(self):
        assert Platform("msys2") is Platform.MSYS2
        assert Platform("wsl") is Platform.WSL


# ── Platform detection ───────────────────────────────────────────────────

class TestDetectPlatform:
    """Test _detect_platform() under various simulated environments."""

    def test_hpc_lmod(self):
        """HPC detected via LMOD_CMD."""
        with patch.dict(os.environ, {"LMOD_CMD": "/usr/local/lmod/lmod/libexec/lmod"}):
            assert SystemDepsRegistry._detect_platform() == Platform.HPC_MODULE

    def test_hpc_moduleshome(self):
        """HPC detected via MODULESHOME."""
        with patch.dict(os.environ, {"MODULESHOME": "/usr/share/modules"}, clear=False):
            # Remove LMOD_CMD if present so we test MODULESHOME path
            env = os.environ.copy()
            env.pop("LMOD_CMD", None)
            env["MODULESHOME"] = "/usr/share/modules"
            with patch.dict(os.environ, env, clear=True):
                assert SystemDepsRegistry._detect_platform() == Platform.HPC_MODULE

    def test_conda_detected(self):
        """Conda wins over system package managers."""
        env = {"CONDA_PREFIX": "/opt/conda/envs/test"}
        with patch.dict(os.environ, env, clear=True):
            assert SystemDepsRegistry._detect_platform() == Platform.CONDA

    @patch("sys.platform", "win32")
    @patch("os.path.isfile")
    def test_msys2_detected_on_windows(self, mock_isfile):
        """MSYS2 detected when C:\\msys64 bash exists on win32."""
        mock_isfile.side_effect = lambda p: p == r"C:\msys64\usr\bin\bash.exe"
        env = {}  # No CONDA_PREFIX, no HPC vars
        with patch.dict(os.environ, env, clear=True):
            assert SystemDepsRegistry._detect_platform() == Platform.MSYS2

    @patch("sys.platform", "win32")
    @patch("os.path.isfile", return_value=False)
    @patch("shutil.which")
    @patch("subprocess.run")
    def test_wsl_detected_on_windows(self, mock_run, mock_which, mock_isfile):
        """WSL detected when wsl probe succeeds on win32."""
        mock_which.side_effect = lambda cmd: "/usr/bin/wsl" if cmd == "wsl" else None
        mock_run.return_value = Mock(returncode=0, stdout="ok\n", stderr="")
        env = {}
        with patch.dict(os.environ, env, clear=True):
            assert SystemDepsRegistry._detect_platform() == Platform.WSL

    @patch("sys.platform", "win32")
    @patch("os.path.isfile", return_value=False)
    @patch("shutil.which", return_value=None)
    def test_unknown_on_windows_no_tools(self, mock_which, mock_isfile):
        """UNKNOWN when nothing found on bare Windows."""
        env = {}
        with patch.dict(os.environ, env, clear=True):
            assert SystemDepsRegistry._detect_platform() == Platform.UNKNOWN

    @patch("sys.platform", "linux")
    @patch("shutil.which")
    def test_apt_detected_on_linux(self, mock_which):
        """APT detected when apt-get is on PATH."""
        mock_which.side_effect = lambda cmd: "/usr/bin/apt-get" if cmd == "apt-get" else None
        env = {}
        with patch.dict(os.environ, env, clear=True):
            assert SystemDepsRegistry._detect_platform() == Platform.APT

    @patch("sys.platform", "darwin")
    @patch("shutil.which")
    def test_brew_detected_on_macos(self, mock_which):
        """Brew detected on macOS."""
        mock_which.side_effect = lambda cmd: "/opt/homebrew/bin/brew" if cmd == "brew" else None
        env = {}
        with patch.dict(os.environ, env, clear=True):
            assert SystemDepsRegistry._detect_platform() == Platform.BREW


# ── YAML registry ────────────────────────────────────────────────────────

class TestRegistryYAML:
    """Validate the system_deps.yml registry contents."""

    @pytest.fixture(autouse=True)
    def _load_registry(self):
        self.registry = _fresh_registry()

    def test_yaml_loads_successfully(self):
        assert self.registry._registry is not None
        assert "dependencies" in self.registry._registry
        assert "tool_requirements" in self.registry._registry

    def test_all_deps_have_display_name(self):
        for dep_id, dep in self.registry._registry["dependencies"].items():
            assert "display_name" in dep, f"{dep_id} missing display_name"

    def test_msys2_entries_present(self):
        """Build toolchain deps should have msys2 package names."""
        for dep_id in ("cmake", "gcc", "gfortran", "make"):
            dep = self.registry._registry["dependencies"][dep_id]
            pkgs = dep.get("packages", {})
            assert "msys2" in pkgs, f"{dep_id} missing msys2 package entry"

    def test_conda_win_entries_present(self):
        """Key deps should have conda_win entries where they differ from conda."""
        for dep_id in ("cmake", "make", "gfortran"):
            dep = self.registry._registry["dependencies"][dep_id]
            pkgs = dep.get("packages", {})
            assert "conda_win" in pkgs, f"{dep_id} missing conda_win package entry"

    def test_make_conda_win_is_m2_prefixed(self):
        dep = self.registry._registry["dependencies"]["make"]
        assert dep["packages"]["conda_win"] == "m2-make"

    def test_msys2_cmake_is_mingw_prefixed(self):
        dep = self.registry._registry["dependencies"]["cmake"]
        assert dep["packages"]["msys2"].startswith("mingw-w64-")

    def test_core_library_msys2_entries(self):
        """Core libraries should have msys2 mingw-w64 package names."""
        for dep_id in ("netcdf", "hdf5", "gdal", "mpi", "blas"):
            dep = self.registry._registry["dependencies"][dep_id]
            pkg = dep.get("packages", {}).get("msys2", "")
            assert "mingw-w64" in pkg, f"{dep_id} msys2 should be mingw-w64 prefixed"


# ── Install command generation ───────────────────────────────────────────

class TestGetInstallCommand:
    """Test _get_install_command for all platforms."""

    def _dep(self, **pkg_overrides):
        """Create a minimal dep_info dict."""
        packages = {"apt": "libfoo-dev", "brew": "foo", "conda": "foo",
                     "dnf": "foo-devel", "hpc_module": "foo",
                     "msys2": "mingw-w64-x86_64-foo",
                     "conda_win": "m2-foo"}
        packages.update(pkg_overrides)
        return {"packages": packages}

    def test_apt(self):
        cmd = SystemDepsRegistry._get_install_command(self._dep(), Platform.APT)
        assert cmd == "sudo apt-get install -y libfoo-dev"

    def test_dnf(self):
        cmd = SystemDepsRegistry._get_install_command(self._dep(), Platform.DNF)
        assert cmd == "sudo dnf install -y foo-devel"

    def test_brew(self):
        cmd = SystemDepsRegistry._get_install_command(self._dep(), Platform.BREW)
        assert cmd == "brew install foo"

    def test_conda_unix(self):
        with patch("sys.platform", "linux"):
            cmd = SystemDepsRegistry._get_install_command(self._dep(), Platform.CONDA)
        assert cmd == "conda install -c conda-forge foo"

    @patch("sys.platform", "win32")
    def test_conda_windows_uses_conda_win(self):
        cmd = SystemDepsRegistry._get_install_command(self._dep(), Platform.CONDA)
        assert cmd == "conda install -c conda-forge m2-foo"

    @patch("sys.platform", "win32")
    def test_conda_windows_falls_back_to_conda(self):
        dep = self._dep()
        del dep["packages"]["conda_win"]
        cmd = SystemDepsRegistry._get_install_command(dep, Platform.CONDA)
        assert cmd == "conda install -c conda-forge foo"

    def test_hpc_module(self):
        cmd = SystemDepsRegistry._get_install_command(self._dep(), Platform.HPC_MODULE)
        assert cmd == "module load foo"

    def test_msys2(self):
        cmd = SystemDepsRegistry._get_install_command(self._dep(), Platform.MSYS2)
        assert cmd == "pacman -S --noconfirm mingw-w64-x86_64-foo"

    def test_wsl_uses_apt_packages(self):
        cmd = SystemDepsRegistry._get_install_command(self._dep(), Platform.WSL)
        assert cmd == "wsl -e sudo apt-get install -y libfoo-dev"

    def test_unknown_returns_none(self):
        cmd = SystemDepsRegistry._get_install_command(self._dep(), Platform.UNKNOWN)
        assert cmd is None

    def test_missing_package_returns_none(self):
        dep = {"packages": {}}
        assert SystemDepsRegistry._get_install_command(dep, Platform.APT) is None


# ── Script generation ────────────────────────────────────────────────────

class TestGenerateInstallScript:
    """Test generate_install_script for different platforms."""

    def _make_registry_with_missing(self, platform, dep_ids=("cmake", "make")):
        """Build a registry where specified deps are 'missing'."""
        reg = _fresh_registry()
        # Force the platform
        reg._platform = platform
        return reg

    def test_apt_script_has_bash_shebang(self):
        reg = _fresh_registry()
        reg._platform = Platform.APT
        # Force all deps to appear missing
        with patch.object(reg, "check_all_deps") as mock_check:
            mock_check.return_value = [
                DepCheckResult("cmake", "CMake", found=False,
                               install_command="sudo apt-get install -y cmake"),
            ]
            script = reg.generate_install_script()
        assert script is not None
        assert script.startswith("#!/usr/bin/env bash")
        assert "sudo apt-get update" in script
        assert "sudo apt-get install -y cmake" in script

    def test_msys2_script_uses_pacman(self):
        reg = _fresh_registry()
        reg._platform = Platform.MSYS2
        with patch.object(reg, "check_all_deps") as mock_check:
            mock_check.return_value = [
                DepCheckResult("cmake", "CMake", found=False,
                               install_command="pacman -S --noconfirm mingw-w64-x86_64-cmake"),
            ]
            script = reg.generate_install_script()
        assert script is not None
        assert "pacman -S --noconfirm" in script
        assert "mingw-w64-x86_64-cmake" in script
        # No bash shebang for MSYS2
        assert "#!/usr/bin/env bash" not in script

    def test_wsl_script_wraps_in_wsl_invocation(self):
        reg = _fresh_registry()
        reg._platform = Platform.WSL
        with patch.object(reg, "check_all_deps") as mock_check:
            mock_check.return_value = [
                DepCheckResult("cmake", "CMake", found=False,
                               install_command="wsl -e sudo apt-get install -y cmake"),
            ]
            script = reg.generate_install_script()
        assert script is not None
        assert 'wsl -e bash -c "sudo apt-get update' in script
        assert "sudo apt-get install -y cmake" in script

    @patch("shutil.which", return_value=None)
    @patch("sys.platform", "win32")
    def test_conda_windows_script_uses_conda_win_names(self, _mock_which):
        reg = _fresh_registry()
        reg._platform = Platform.CONDA
        with patch.object(reg, "check_all_deps") as mock_check:
            mock_check.return_value = [
                DepCheckResult("make", "Make", found=False,
                               install_command="conda install -c conda-forge m2-make"),
            ]
            script = reg.generate_install_script()
        assert script is not None
        assert "conda install -c conda-forge" in script
        assert "m2-make" in script

    def test_no_missing_returns_none(self):
        reg = _fresh_registry()
        with patch.object(reg, "check_all_deps") as mock_check:
            mock_check.return_value = [
                DepCheckResult("cmake", "CMake", found=True),
            ]
            assert reg.generate_install_script() is None


# ── WSL command probing ──────────────────────────────────────────────────

class TestCheckWslCommand:
    """Test _check_wsl_command static method."""

    @patch("subprocess.run")
    def test_command_found_in_wsl(self, mock_run):
        """Found command returns (True, version)."""
        mock_run.side_effect = [
            # First call: command -v cmake
            Mock(returncode=0, stdout="/usr/bin/cmake\n", stderr=""),
            # Second call: cmake --version
            Mock(returncode=0,
                 stdout="cmake version 3.28.1\n",
                 stderr=""),
        ]
        check = {
            "version_flag": "--version",
            "version_regex": r"cmake version (\d+\.\d+(?:\.\d+)?)",
        }
        found, version = SystemDepsRegistry._check_wsl_command("cmake", check)
        assert found is True
        assert version == "3.28.1"

    @patch("subprocess.run")
    def test_command_not_found_in_wsl(self, mock_run):
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="")
        check = {"version_flag": "--version", "version_regex": ""}
        found, version = SystemDepsRegistry._check_wsl_command("cmake", check)
        assert found is False
        assert version is None

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("wsl", 10))
    def test_timeout_returns_not_found(self, mock_run):
        check = {"version_flag": "--version", "version_regex": ""}
        found, version = SystemDepsRegistry._check_wsl_command("cmake", check)
        assert found is False

    @patch("subprocess.run")
    def test_found_but_no_version_regex(self, mock_run):
        """Found command but no version extraction configured."""
        mock_run.return_value = Mock(returncode=0, stdout="/usr/bin/make\n", stderr="")
        check = {"version_flag": "", "version_regex": ""}
        found, version = SystemDepsRegistry._check_wsl_command("make", check)
        assert found is True
        assert version is None


# ── check_dep with WSL fallback ──────────────────────────────────────────

class TestCheckDepWSLFallback:
    """Test that check_dep probes inside WSL when platform is WSL."""

    def test_wsl_fallback_triggers(self):
        """When platform is WSL and shutil.which fails, probe inside WSL."""
        reg = _fresh_registry()
        reg._platform = Platform.WSL

        with patch("shutil.which", return_value=None), \
             patch.object(SystemDepsRegistry, "_check_wsl_command",
                          return_value=(True, "3.28.1")) as mock_wsl, \
             patch.object(SystemDepsRegistry, "_check_pkg_config",
                          return_value=(False, None)):
            result = reg.check_dep("cmake")

        assert result.found is True
        assert result.version == "3.28.1"
        assert "(wsl:" in result.path
        mock_wsl.assert_called()

    def test_wsl_fallback_not_triggered_on_apt(self):
        """WSL fallback should not trigger when platform is APT."""
        reg = _fresh_registry()
        reg._platform = Platform.APT

        with patch("shutil.which", return_value=None), \
             patch.object(SystemDepsRegistry, "_check_wsl_command") as mock_wsl, \
             patch.object(SystemDepsRegistry, "_check_pkg_config",
                          return_value=(False, None)):
            reg.check_dep("cmake")

        mock_wsl.assert_not_called()


# ── Homebrew keg-only fallback ─────────────────────────────────────────────

class TestBrewKegPkgConfig:
    """Test _check_brew_keg_pkg_config and its integration in check_dep."""

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_finds_keg_only_openblas(self, mock_which, mock_run):
        """Detects openblas via brew --prefix when standard pkg-config fails."""
        mock_which.side_effect = lambda cmd: {
            "pkg-config": "/opt/homebrew/bin/pkg-config",
            "brew": "/opt/homebrew/bin/brew",
        }.get(cmd)

        mock_run.side_effect = [
            # brew --prefix openblas
            Mock(returncode=0, stdout="/opt/homebrew/opt/openblas\n", stderr=""),
            # pkg-config --modversion openblas (with augmented PKG_CONFIG_PATH)
            Mock(returncode=0, stdout="0.3.31\n", stderr=""),
        ]

        with patch("os.path.isdir", return_value=True):
            found, version = SystemDepsRegistry._check_brew_keg_pkg_config(
                "openblas", ["openblas"]
            )

        assert found is True
        assert version == "0.3.31"

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_returns_false_when_formula_not_installed(self, mock_which, mock_run):
        """Returns (False, None) when brew --prefix fails."""
        mock_which.side_effect = lambda cmd: {
            "pkg-config": "/usr/bin/pkg-config",
            "brew": "/opt/homebrew/bin/brew",
        }.get(cmd)
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="")

        found, version = SystemDepsRegistry._check_brew_keg_pkg_config(
            "openblas", ["openblas"]
        )
        assert found is False
        assert version is None

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_returns_false_when_no_pc_dir(self, mock_which, mock_run):
        """Returns (False, None) when keg exists but has no lib/pkgconfig."""
        mock_which.side_effect = lambda cmd: {
            "pkg-config": "/usr/bin/pkg-config",
            "brew": "/opt/homebrew/bin/brew",
        }.get(cmd)
        mock_run.return_value = Mock(
            returncode=0, stdout="/opt/homebrew/opt/openblas\n", stderr=""
        )

        with patch("os.path.isdir", return_value=False):
            found, version = SystemDepsRegistry._check_brew_keg_pkg_config(
                "openblas", ["openblas"]
            )
        assert found is False

    @patch("shutil.which", return_value=None)
    def test_returns_false_when_no_brew(self, mock_which):
        """Returns (False, None) when brew is not available."""
        found, version = SystemDepsRegistry._check_brew_keg_pkg_config(
            "openblas", ["openblas"]
        )
        assert found is False

    def test_multiple_formulas_probed(self):
        """Probes multiple brew formulas (e.g. 'openblas lapack')."""
        brew_calls = []

        def mock_run_side_effect(cmd, **kwargs):
            if cmd[0].endswith("brew"):
                brew_calls.append(cmd[2])  # formula name
                # First formula not found, second found
                if cmd[2] == "lapack":
                    return Mock(returncode=0,
                                stdout="/opt/homebrew/opt/lapack\n", stderr="")
                return Mock(returncode=1, stdout="", stderr="")
            # pkg-config call
            return Mock(returncode=0, stdout="3.12.0\n", stderr="")

        with patch("shutil.which", side_effect=lambda cmd: f"/usr/bin/{cmd}"), \
             patch("subprocess.run", side_effect=mock_run_side_effect), \
             patch("os.path.isdir", return_value=True):
            found, version = SystemDepsRegistry._check_brew_keg_pkg_config(
                "lapack", ["openblas", "lapack"]
            )

        assert found is True
        assert "openblas" in brew_calls
        assert "lapack" in brew_calls


class TestCheckDepBrewKegFallback:
    """Test that check_dep uses brew keg fallback on BREW platform."""

    def test_brew_keg_fallback_triggers_for_blas(self):
        """On BREW platform, blas dep falls back to keg detection."""
        reg = _fresh_registry()
        reg._platform = Platform.BREW

        with patch("shutil.which", return_value=None), \
             patch.object(SystemDepsRegistry, "_check_pkg_config",
                          return_value=(False, None)), \
             patch.object(SystemDepsRegistry, "_check_brew_keg_pkg_config",
                          return_value=(True, "0.3.31")) as mock_keg:
            result = reg.check_dep("blas")

        assert result.found is True
        assert result.version == "0.3.31"
        assert "brew keg" in result.path
        mock_keg.assert_called_once_with("openblas", ["openblas", "lapack"])

    def test_brew_keg_fallback_not_triggered_on_apt(self):
        """Keg fallback should not trigger on non-BREW platforms."""
        reg = _fresh_registry()
        reg._platform = Platform.APT

        with patch("shutil.which", return_value=None), \
             patch.object(SystemDepsRegistry, "_check_pkg_config",
                          return_value=(False, None)), \
             patch.object(SystemDepsRegistry, "_check_brew_keg_pkg_config") as mock_keg:
            reg.check_dep("blas")

        mock_keg.assert_not_called()

    def test_brew_keg_fallback_skipped_when_already_found(self):
        """Keg fallback should not trigger when standard pkg-config succeeds."""
        reg = _fresh_registry()
        reg._platform = Platform.BREW

        with patch("shutil.which", return_value=None), \
             patch.object(SystemDepsRegistry, "_check_pkg_config",
                          return_value=(True, "0.3.31")), \
             patch.object(SystemDepsRegistry, "_check_brew_keg_pkg_config") as mock_keg:
            result = reg.check_dep("blas")

        assert result.found is True
        mock_keg.assert_not_called()


# ── Version comparison ───────────────────────────────────────────────────

class TestVersionGe:
    """Test _version_ge helper."""

    def test_equal(self):
        assert SystemDepsRegistry._version_ge("3.20.0", "3.20.0") is True

    def test_greater(self):
        assert SystemDepsRegistry._version_ge("3.28.1", "3.20") is True

    def test_less(self):
        assert SystemDepsRegistry._version_ge("3.16.0", "3.20") is False

    def test_no_digits_compares_empty_tuple(self):
        # "unknown" has no digits → _parse returns () which is < (3, 20)
        assert SystemDepsRegistry._version_ge("unknown", "3.20") is False

    def test_truly_unparseable_returns_true(self):
        # Only truly unparseable values (raising exceptions) return True
        assert SystemDepsRegistry._version_ge(None, "3.20") is True


# ── Full integration: check_all_deps / check_deps_for_tool ──────────────

class TestRegistryIntegration:
    """Smoke tests that the registry loads and checks run without error."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.registry = _fresh_registry()

    def test_check_all_deps_returns_list(self):
        results = self.registry.check_all_deps()
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, DepCheckResult) for r in results)

    def test_check_deps_for_known_tool(self):
        results = self.registry.check_deps_for_tool("summa")
        assert len(results) > 0
        dep_ids = [r.dep_id for r in results]
        assert "cmake" in dep_ids
        assert "gfortran" in dep_ids

    def test_check_deps_for_unknown_tool(self):
        assert self.registry.check_deps_for_tool("nonexistent_tool") == []

    def test_get_tool_names(self):
        names = self.registry.get_tool_names()
        assert "summa" in names
        assert "mizuroute" in names

    def test_platform_is_valid(self):
        assert self.registry.platform in Platform
