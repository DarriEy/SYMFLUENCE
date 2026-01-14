"""
Shared fixtures for agent tests.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('symfluence.agent.api_client.OpenAI') as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Set up default response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"

        mock_client.chat.completions.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_env_openai():
    """Mock environment with OpenAI API key."""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key-12345',
        'OPENAI_MODEL': 'gpt-4'
    }, clear=False):
        yield


@pytest.fixture
def mock_env_groq():
    """Mock environment with Groq API key."""
    with patch.dict('os.environ', {
        'GROQ_API_KEY': 'gsk-test-key-12345'
    }, clear=False):
        # Also clear OPENAI_API_KEY to test fallback
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}, clear=False):
            yield


@pytest.fixture
def sample_tool_response():
    """Create a sample tool call response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = None
    mock_response.choices[0].finish_reason = "tool_calls"

    # Create tool call
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "list_workflow_steps"
    mock_tool_call.function.arguments = "{}"

    mock_response.choices[0].message.tool_calls = [mock_tool_call]

    return mock_response


@pytest.fixture
def sample_config_yaml(temp_dir):
    """Create a sample SYMFLUENCE config file."""
    config_content = """
SYMFLUENCE_DATA_DIR: {data_dir}
SYMFLUENCE_CODE_DIR: {code_dir}
DOMAIN_NAME: test_domain
EXPERIMENT_ID: test_exp
EXPERIMENT_TIME_START: '2020-01-01 00:00'
EXPERIMENT_TIME_END: '2020-12-31 23:00'
DOMAIN_DEFINITION_METHOD: lumped
DOMAIN_DISCRETIZATION: GRUs
HYDROLOGICAL_MODEL: SUMMA
FORCING_DATASET: ERA5
""".format(data_dir=str(temp_dir / 'data'), code_dir=str(temp_dir / 'code'))

    config_path = temp_dir / 'test_config.yaml'
    config_path.write_text(config_content)

    return config_path
