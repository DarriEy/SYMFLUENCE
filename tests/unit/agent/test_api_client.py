"""
Tests for the APIClient class.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest


class TestAPIClientInitialization:
    """Tests for APIClient initialization."""

    def test_api_client_can_be_imported(self):
        """Test that APIClient can be imported."""
        from symfluence.agent.api_client import APIClient
        assert APIClient is not None

    @patch('symfluence.agent.api_client.OpenAI')
    def test_initialization_with_openai_key(self, mock_openai, mock_env_openai):
        """Test initialization with OpenAI API key."""
        from symfluence.agent.api_client import APIClient

        client = APIClient(verbose=False)

        assert client.provider == "OpenAI/Custom"
        assert client.api_key == 'test-key-12345'

    @patch('symfluence.agent.api_client.OpenAI')
    def test_initialization_with_groq_key(self, mock_openai):
        """Test initialization with Groq API key as fallback."""
        from symfluence.agent.api_client import APIClient

        with patch.dict('os.environ', {
            'OPENAI_API_KEY': '',
            'GROQ_API_KEY': 'gsk-test-key'
        }, clear=False):
            # Mock Ollama check to fail
            with patch.object(APIClient, '_is_ollama_available', return_value=False):
                client = APIClient(verbose=False)

                assert client.provider == "Groq"
                assert "groq.com" in client.api_base

    @patch('symfluence.agent.api_client.OpenAI')
    def test_initialization_verbose_mode(self, mock_openai, mock_env_openai, capsys):
        """Test verbose mode outputs provider info."""
        from symfluence.agent.api_client import APIClient

        client = APIClient(verbose=True)

        captured = capsys.readouterr()
        assert "Provider:" in captured.err or "API Client initialized" in captured.err

    def test_initialization_no_api_key_exits(self):
        """Test initialization exits when no API key configured."""
        from symfluence.agent.api_client import APIClient

        with patch.dict('os.environ', {
            'OPENAI_API_KEY': '',
            'GROQ_API_KEY': ''
        }, clear=False):
            with patch.object(APIClient, '_is_ollama_available', return_value=False):
                with pytest.raises(SystemExit):
                    APIClient()


class TestOllamaDetection:
    """Tests for Ollama availability detection."""

    @patch('symfluence.agent.api_client.OpenAI')
    def test_ollama_available(self, mock_openai):
        """Test Ollama detection when server is running."""
        from symfluence.agent.api_client import APIClient

        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value = MagicMock()

            with patch.dict('os.environ', {
                'OPENAI_API_KEY': '',
                'GROQ_API_KEY': ''
            }, clear=False):
                client = APIClient()

                assert client.provider == "Ollama (Local)"

    @patch('symfluence.agent.api_client.OpenAI')
    def test_ollama_not_available(self, mock_openai):
        """Test Ollama detection when server is not running."""
        from symfluence.agent.api_client import APIClient

        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = Exception("Connection refused")

            # Also need an API key since Ollama is not available
            with patch.dict('os.environ', {
                'OPENAI_API_KEY': 'test-key'
            }, clear=False):
                client = APIClient()

                assert client.provider == "OpenAI/Custom"


class TestChatCompletion:
    """Tests for chat completion API calls."""

    @patch('symfluence.agent.api_client.OpenAI')
    def test_chat_completion_success(self, mock_openai, mock_env_openai):
        """Test successful chat completion."""
        from symfluence.agent.api_client import APIClient

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = APIClient()
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}]
        )

        assert response.choices[0].message.content == "Test response"

    @patch('symfluence.agent.api_client.OpenAI')
    def test_chat_completion_with_tools(self, mock_openai, mock_env_openai):
        """Test chat completion with tool definitions."""
        from symfluence.agent.api_client import APIClient

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].finish_reason = "stop"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = APIClient()
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            tools=tools,
            tool_choice="auto"
        )

        # Verify tools were passed to API
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs


class TestConnectionTesting:
    """Tests for connection testing."""

    @patch('symfluence.agent.api_client.OpenAI')
    def test_test_connection_success(self, mock_openai, mock_env_openai):
        """Test successful connection test."""
        from symfluence.agent.api_client import APIClient

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        client = APIClient()
        result = client.test_connection()

        assert result is True

    @patch('symfluence.agent.api_client.OpenAI')
    def test_test_connection_failure(self, mock_openai, mock_env_openai):
        """Test failed connection test."""
        from symfluence.agent.api_client import APIClient

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Connection failed")
        mock_openai.return_value = mock_client

        client = APIClient()
        result = client.test_connection()

        assert result is False


class TestAPIClientConfiguration:
    """Tests for API client configuration."""

    @patch('symfluence.agent.api_client.OpenAI')
    def test_custom_timeout(self, mock_openai):
        """Test custom timeout configuration."""
        from symfluence.agent.api_client import APIClient

        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_TIMEOUT': '120'
        }, clear=False):
            client = APIClient()

            assert client.timeout == 120

    @patch('symfluence.agent.api_client.OpenAI')
    def test_custom_model(self, mock_openai):
        """Test custom model configuration."""
        from symfluence.agent.api_client import APIClient

        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_MODEL': 'gpt-3.5-turbo'
        }, clear=False):
            client = APIClient()

            assert client.model == 'gpt-3.5-turbo'

    @patch('symfluence.agent.api_client.OpenAI')
    def test_custom_base_url(self, mock_openai):
        """Test custom API base URL configuration."""
        from symfluence.agent.api_client import APIClient

        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key',
            'OPENAI_API_BASE': 'https://custom.api.com/v1'
        }, clear=False):
            client = APIClient()

            assert client.api_base == 'https://custom.api.com/v1'
