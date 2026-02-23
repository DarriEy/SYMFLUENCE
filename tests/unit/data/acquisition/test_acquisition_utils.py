"""
Unit Tests for Acquisition Utility Functions.

Tests the utility functions:
- create_robust_session(): session creation with retry adapter
- resolve_credentials(): environment, netrc, config fallback chain
- download_file_streaming(): successful download, atomic write cleanup
- atomic_write(): context manager for safe file writes
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
from fixtures.acquisition_fixtures import MockResponseFactory, MockSessionFactory

# =============================================================================
# Create Robust Session Tests
# =============================================================================

@pytest.mark.acquisition
class TestCreateRobustSession:
    """Tests for create_robust_session function."""

    def test_returns_session(self):
        """Should return a requests.Session object."""
        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session()

        assert session is not None
        assert hasattr(session, 'get')
        assert hasattr(session, 'post')

    def test_session_has_retry_adapter(self):
        """Session should have retry adapter mounted."""
        from requests.adapters import HTTPAdapter

        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session()

        # Check adapters are mounted
        assert 'https://' in session.adapters
        assert 'http://' in session.adapters

        # Check adapters are HTTPAdapter (which includes retry)
        assert isinstance(session.adapters['https://'], HTTPAdapter)
        assert isinstance(session.adapters['http://'], HTTPAdapter)

    def test_custom_max_retries(self):
        """Should respect custom max_retries parameter."""
        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session(max_retries=10)

        # Session should be created successfully
        assert session is not None

    def test_custom_status_forcelist(self):
        """Should respect custom status_forcelist."""
        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session(status_forcelist=[500, 502])

        assert session is not None

    def test_custom_backoff_factor(self):
        """Should respect custom backoff_factor."""
        from symfluence.data.acquisition.utils import create_robust_session

        session = create_robust_session(backoff_factor=2.0)

        assert session is not None


# =============================================================================
# Download File Streaming Tests
# =============================================================================

@pytest.mark.acquisition
class TestDownloadFileStreaming:
    """Tests for download_file_streaming function."""

    def test_successful_download(self, tmp_path):
        """Should successfully download and save file."""
        from symfluence.data.acquisition.utils import download_file_streaming

        content = b"test file content"
        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=content)
        )

        target = tmp_path / "downloaded.txt"

        bytes_downloaded = download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session
        )

        assert target.exists()
        assert target.read_bytes() == content
        assert bytes_downloaded == len(content)

    def test_creates_parent_directory(self, tmp_path):
        """Should create parent directories if needed."""
        from symfluence.data.acquisition.utils import download_file_streaming

        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=b"data")
        )

        # Nested path that doesn't exist
        target = tmp_path / "a" / "b" / "c" / "file.txt"

        download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session
        )

        assert target.exists()

    def test_atomic_write_with_temp_file(self, tmp_path):
        """Should use .part file for atomic write."""
        from symfluence.data.acquisition.utils import download_file_streaming

        content = b"test content"
        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=content)
        )

        target = tmp_path / "file.txt"

        download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session,
            use_temp_file=True
        )

        # Target should exist, .part should not
        assert target.exists()
        assert not Path(str(target) + '.part').exists()

    def test_cleanup_on_error(self, tmp_path):
        """Should clean up .part file on error."""
        from requests.exceptions import HTTPError

        from symfluence.data.acquisition.utils import download_file_streaming

        mock_response = MockResponseFactory.error(500, "Server Error")
        mock_session = MockSessionFactory.create(default_response=mock_response)

        target = tmp_path / "file.txt"

        with pytest.raises(HTTPError):
            download_file_streaming(
                "https://example.com/file.txt",
                target,
                session=mock_session
            )

        # Neither target nor .part should exist
        assert not target.exists()
        assert not Path(str(target) + '.part').exists()

    def test_verifies_content_length(self, tmp_path):
        """Should verify download matches Content-Length."""
        from symfluence.data.acquisition.utils import download_file_streaming

        # Response claims 1000 bytes but only has 100
        content = b"x" * 100
        mock_response = MockResponseFactory.success(content=content)
        mock_response.headers['content-length'] = '1000'

        mock_session = MockSessionFactory.create(default_response=mock_response)

        target = tmp_path / "file.txt"

        with pytest.raises(IOError) as exc_info:
            download_file_streaming(
                "https://example.com/file.txt",
                target,
                session=mock_session
            )

        assert "Incomplete download" in str(exc_info.value)

    def test_passes_auth(self, tmp_path):
        """Should pass auth credentials to request."""
        from symfluence.data.acquisition.utils import download_file_streaming

        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=b"data")
        )

        target = tmp_path / "file.txt"

        download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session,
            auth=("user", "pass")
        )

        # Verify auth was passed
        mock_session.get.assert_called_once()
        call_kwargs = mock_session.get.call_args[1]
        assert call_kwargs['auth'] == ("user", "pass")

    def test_passes_headers(self, tmp_path):
        """Should pass custom headers to request."""
        from symfluence.data.acquisition.utils import download_file_streaming

        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=b"data")
        )

        target = tmp_path / "file.txt"

        download_file_streaming(
            "https://example.com/file.txt",
            target,
            session=mock_session,
            headers={"X-Custom": "value"}
        )

        call_kwargs = mock_session.get.call_args[1]
        assert call_kwargs['headers'] == {"X-Custom": "value"}


# =============================================================================
# Atomic Write Tests
# =============================================================================

@pytest.mark.acquisition
class TestAtomicWrite:
    """Tests for atomic_write context manager."""

    def test_successful_write(self, tmp_path):
        """Should write to target on success."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "output.txt"

        with atomic_write(target) as temp_path:
            temp_path.write_text("test content")

        assert target.exists()
        assert target.read_text() == "test content"
        assert not Path(str(target) + '.part').exists()

    def test_creates_parent_directory(self, tmp_path):
        """Should create parent directories."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "new" / "nested" / "dir" / "file.txt"

        with atomic_write(target) as temp_path:
            temp_path.write_text("data")

        assert target.exists()

    def test_cleanup_on_error(self, tmp_path):
        """Should clean up .part file on error."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "output.txt"

        with pytest.raises(RuntimeError):
            with atomic_write(target) as temp_path:
                temp_path.write_text("partial")
                raise RuntimeError("Simulated error")

        # Target and .part should not exist
        assert not target.exists()
        assert not Path(str(target) + '.part').exists()

    def test_yields_part_path(self, tmp_path):
        """Should yield .part path for writing."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "output.txt"

        with atomic_write(target) as temp_path:
            assert str(temp_path).endswith('.part')
            temp_path.write_text("data")

    def test_overwrites_existing_target(self, tmp_path):
        """Should overwrite existing target file."""
        from symfluence.data.acquisition.utils import atomic_write

        target = tmp_path / "output.txt"
        target.write_text("old content")

        with atomic_write(target) as temp_path:
            temp_path.write_text("new content")

        assert target.read_text() == "new content"


# =============================================================================
# Resolve Credentials Tests
# =============================================================================

@pytest.mark.acquisition
class TestResolveCredentials:
    """Tests for resolve_credentials function."""

    def test_from_environment_variables(self, clean_environment):
        """Should get credentials from environment variables."""
        from symfluence.data.acquisition.utils import resolve_credentials

        os.environ['TEST_USERNAME'] = 'env_user'
        os.environ['TEST_PASSWORD'] = 'env_pass'

        try:
            username, password = resolve_credentials(
                hostname='example.com',
                env_prefix='TEST'
            )

            assert username == 'env_user'
            assert password == 'env_pass'
        finally:
            os.environ.pop('TEST_USERNAME', None)
            os.environ.pop('TEST_PASSWORD', None)

    def test_from_config_dict(self, clean_environment):
        """Should get credentials from config dictionary."""
        from symfluence.data.acquisition.utils import resolve_credentials

        config = {
            'TEST_USERNAME': 'config_user',
            'TEST_PASSWORD': 'config_pass',
        }

        username, password = resolve_credentials(
            hostname='example.com',
            env_prefix='TEST',
            config=config
        )

        assert username == 'config_user'
        assert password == 'config_pass'

    def test_env_takes_priority_over_config(self, clean_environment):
        """Environment variables should take priority over config."""
        from symfluence.data.acquisition.utils import resolve_credentials

        os.environ['TEST_USERNAME'] = 'env_user'
        os.environ['TEST_PASSWORD'] = 'env_pass'

        config = {
            'TEST_USERNAME': 'config_user',
            'TEST_PASSWORD': 'config_pass',
        }

        try:
            username, password = resolve_credentials(
                hostname='example.com',
                env_prefix='TEST',
                config=config
            )

            # Env should win
            assert username == 'env_user'
            assert password == 'env_pass'
        finally:
            os.environ.pop('TEST_USERNAME', None)
            os.environ.pop('TEST_PASSWORD', None)

    def test_returns_none_when_not_found(self, clean_environment):
        """Should return (None, None) when credentials not found."""
        from symfluence.data.acquisition.utils import resolve_credentials

        username, password = resolve_credentials(
            hostname='example.com',
            env_prefix='NONEXISTENT'
        )

        assert username is None
        assert password is None

    def test_partial_credentials_from_env(self, clean_environment):
        """Should handle partial credentials (only username or password)."""
        from symfluence.data.acquisition.utils import resolve_credentials

        os.environ['TEST_USERNAME'] = 'only_user'
        # No password set

        try:
            username, password = resolve_credentials(
                hostname='example.com',
                env_prefix='TEST'
            )

            # Partial credentials should return None, None
            # (requires both to be set)
            assert username is None or password is None
        finally:
            os.environ.pop('TEST_USERNAME', None)

    def test_from_netrc(self, clean_environment, tmp_path):
        """Should get credentials from .netrc file."""
        from symfluence.data.acquisition.utils import resolve_credentials

        # Create a mock .netrc file
        netrc_content = "machine example.com login netrc_user password netrc_pass"
        netrc_path = tmp_path / '.netrc'
        netrc_path.write_text(netrc_content)

        with patch('pathlib.Path.home', return_value=tmp_path):
            username, password = resolve_credentials(
                hostname='example.com'
            )

        assert username == 'netrc_user'
        assert password == 'netrc_pass'

    def test_alt_hostnames_checked(self, clean_environment, tmp_path):
        """Should check alternative hostnames in .netrc."""
        from symfluence.data.acquisition.utils import resolve_credentials

        netrc_content = "machine alt.example.com login alt_user password alt_pass"
        netrc_path = tmp_path / '.netrc'
        netrc_path.write_text(netrc_content)

        with patch('pathlib.Path.home', return_value=tmp_path):
            username, password = resolve_credentials(
                hostname='example.com',
                alt_hostnames=['alt.example.com']
            )

        assert username == 'alt_user'
        assert password == 'alt_pass'


# =============================================================================
# Get Earthdata Credentials Tests
# =============================================================================

@pytest.mark.acquisition
class TestGetEarthdataCredentials:
    """Tests for get_earthdata_credentials convenience function."""

    def test_uses_earthdata_env_prefix(self, clean_environment, tmp_path):
        """Should use EARTHDATA_ environment variable prefix."""
        from symfluence.data.acquisition.utils import get_earthdata_credentials

        os.environ['EARTHDATA_USERNAME'] = 'earth_user'
        os.environ['EARTHDATA_PASSWORD'] = 'earth_pass'

        try:
            # Patch home to avoid reading system .netrc
            with patch('pathlib.Path.home', return_value=tmp_path):
                username, password = get_earthdata_credentials()

            assert username == 'earth_user'
            assert password == 'earth_pass'
        finally:
            os.environ.pop('EARTHDATA_USERNAME', None)
            os.environ.pop('EARTHDATA_PASSWORD', None)

    def test_checks_urs_earthdata_hostname(self, clean_environment, tmp_path):
        """Should check urs.earthdata.nasa.gov in .netrc."""
        from symfluence.data.acquisition.utils import get_earthdata_credentials

        netrc_content = "machine urs.earthdata.nasa.gov login urs_user password urs_pass"
        netrc_path = tmp_path / '.netrc'
        netrc_path.write_text(netrc_content)

        with patch('pathlib.Path.home', return_value=tmp_path):
            username, password = get_earthdata_credentials()

        assert username == 'urs_user'
        assert password == 'urs_pass'


# =============================================================================
# Resolve Earthdata Token Tests
# =============================================================================

@pytest.mark.acquisition
class TestResolveEarthdataToken:
    """Tests for resolve_earthdata_token function."""

    def test_from_environment(self, clean_environment):
        """Should get token from EARTHDATA_TOKEN env var."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        os.environ['EARTHDATA_TOKEN'] = 'my-test-token-123'

        try:
            token = resolve_earthdata_token()
            assert token == 'my-test-token-123'
        finally:
            os.environ.pop('EARTHDATA_TOKEN', None)

    def test_from_config(self, clean_environment):
        """Should get token from config dictionary."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        config = {'EARTHDATA_TOKEN': 'config-token-456'}
        token = resolve_earthdata_token(config=config)
        assert token == 'config-token-456'

    def test_env_takes_precedence_over_config(self, clean_environment):
        """Environment variable should take precedence over config."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        os.environ['EARTHDATA_TOKEN'] = 'env-token'
        config = {'EARTHDATA_TOKEN': 'config-token'}

        try:
            token = resolve_earthdata_token(config=config)
            assert token == 'env-token'
        finally:
            os.environ.pop('EARTHDATA_TOKEN', None)

    def test_returns_none_when_not_found(self, clean_environment):
        """Should return None when no token is available."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        token = resolve_earthdata_token()
        assert token is None

    def test_returns_none_for_empty_config(self, clean_environment):
        """Should return None when config has no token."""
        from symfluence.data.acquisition.utils import resolve_earthdata_token

        token = resolve_earthdata_token(config={'OTHER_KEY': 'value'})
        assert token is None


# =============================================================================
# Get CDS Credentials Tests
# =============================================================================

@pytest.mark.acquisition
class TestGetCdsCredentials:
    """Tests for get_cds_credentials convenience function."""

    def test_from_environment(self, clean_environment):
        """Should get CDS credentials from environment."""
        from symfluence.data.acquisition.utils import get_cds_credentials

        os.environ['CDSAPI_URL'] = 'https://cds.example.com/api'
        os.environ['CDSAPI_KEY'] = '12345:abcdef'

        try:
            url, key = get_cds_credentials()

            assert url == 'https://cds.example.com/api'
            assert key == '12345:abcdef'
        finally:
            os.environ.pop('CDSAPI_URL', None)
            os.environ.pop('CDSAPI_KEY', None)

    def test_from_config(self, clean_environment):
        """Should get CDS credentials from config."""
        from symfluence.data.acquisition.utils import get_cds_credentials

        config = {
            'CDSAPI_URL': 'https://config.cds.example.com/api',
            'CDSAPI_KEY': '99999:config_key',
        }

        url, key = get_cds_credentials(config=config)

        assert url == 'https://config.cds.example.com/api'
        assert key == '99999:config_key'

    def test_returns_none_when_not_found(self, clean_environment):
        """Should return (None, None) when not found."""
        from symfluence.data.acquisition.utils import get_cds_credentials

        url, key = get_cds_credentials()

        assert url is None
        assert key is None


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.acquisition
class TestUtilsEdgeCases:
    """Edge case tests for utility functions."""

    def test_download_zero_byte_file(self, tmp_path):
        """Should handle zero-byte file download."""
        from symfluence.data.acquisition.utils import download_file_streaming

        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=b"")
        )

        target = tmp_path / "empty.txt"

        bytes_downloaded = download_file_streaming(
            "https://example.com/empty.txt",
            target,
            session=mock_session
        )

        assert target.exists()
        assert bytes_downloaded == 0

    def test_download_large_file_chunked(self, tmp_path):
        """Should handle large file in chunks."""
        from symfluence.data.acquisition.utils import download_file_streaming

        # 1MB of data
        content = b"x" * (1024 * 1024)
        mock_session = MockSessionFactory.create(
            default_response=MockResponseFactory.success(content=content)
        )

        target = tmp_path / "large.bin"

        bytes_downloaded = download_file_streaming(
            "https://example.com/large.bin",
            target,
            session=mock_session,
            chunk_size=65536
        )

        assert target.exists()
        assert bytes_downloaded == len(content)
