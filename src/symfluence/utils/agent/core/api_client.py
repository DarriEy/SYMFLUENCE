"""
API client for OpenAI-compatible LLM providers.

This module provides a unified interface for calling any OpenAI-compatible API,
including OpenAI, Anthropic (OpenAI mode), and local LLMs like Ollama.
"""

import os
import sys
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, BadRequestError
except ImportError:
    print("Error: openai package not installed. Install it with: pip install openai>=1.0.0", file=sys.stderr)
    sys.exit(1)

from . import system_prompts


class APIClient:
    """
    Client for making API calls to OpenAI-compatible endpoints.

    Supports configuration via environment variables:
    - OPENAI_API_KEY: API authentication key (required)
    - OPENAI_API_BASE: Base URL for API (optional, default: https://api.openai.com/v1)
    - OPENAI_MODEL: Model name to use (optional, default: gpt-4-turbo-preview)
    - OPENAI_TIMEOUT: Request timeout in seconds (optional, default: 60)
    - OPENAI_MAX_RETRIES: Maximum retry attempts (optional, default: 2)
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the API client.

        Args:
            verbose: If True, print additional debug information

        Raises:
            SystemExit: If OPENAI_API_KEY is not set
        """
        self.verbose = verbose
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.timeout = int(os.getenv("OPENAI_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

        if not self.api_key:
            print(system_prompts.ERROR_MESSAGES["api_key_missing"], file=sys.stderr)
            sys.exit(1)

        # Initialize OpenAI client with custom base URL support
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

        if self.verbose:
            print(f"API Client initialized:", file=sys.stderr)
            print(f"  Base URL: {self.api_base}", file=sys.stderr)
            print(f"  Model: {self.model}", file=sys.stderr)
            print(f"  Timeout: {self.timeout}s", file=sys.stderr)

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto"
    ) -> Any:
        """
        Make a chat completion API call with function calling support.

        Args:
            messages: List of messages in OpenAI format
            tools: List of tool definitions for function calling (optional)
            tool_choice: How the model should use tools ("auto", "none", or specific tool)

        Returns:
            API response object with choices, message, and tool_calls

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit is exceeded
            APIConnectionError: If connection to API fails
            BadRequestError: If request is malformed
        """
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }

            # Only add tools if provided
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            if self.verbose:
                print(f"\n[API Call] Model: {self.model}, Messages: {len(messages)}, Tools: {len(tools) if tools else 0}", file=sys.stderr)

            response = self.client.chat.completions.create(**kwargs)

            if self.verbose:
                choice = response.choices[0]
                print(f"[API Response] Finish reason: {choice.finish_reason}", file=sys.stderr)
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    print(f"[API Response] Tool calls: {len(choice.message.tool_calls)}", file=sys.stderr)

            return response

        except AuthenticationError as e:
            error_msg = system_prompts.ERROR_MESSAGES["api_authentication_failed"].format(
                api_base=self.api_base,
                model=self.model
            )
            print(error_msg, file=sys.stderr)
            raise

        except RateLimitError as e:
            print(system_prompts.ERROR_MESSAGES["api_rate_limit"], file=sys.stderr)
            raise

        except APIConnectionError as e:
            error_msg = system_prompts.ERROR_MESSAGES["api_connection_failed"].format(
                api_base=self.api_base
            )
            print(error_msg, file=sys.stderr)
            raise

        except BadRequestError as e:
            print(f"Error: Invalid request - {str(e)}", file=sys.stderr)
            raise

        except Exception as e:
            print(f"Unexpected API error: {str(e)}", file=sys.stderr)
            raise

    def test_connection(self) -> bool:
        """
        Test the API connection with a simple request.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            if self.verbose:
                print(f"Connection test failed: {str(e)}", file=sys.stderr)
            return False
