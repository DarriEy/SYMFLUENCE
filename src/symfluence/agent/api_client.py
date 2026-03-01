# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
API client for OpenAI-compatible LLM providers.

This module provides a unified interface for calling any OpenAI-compatible API,
including OpenAI, Anthropic (OpenAI mode), and local LLMs like Ollama.
"""

import os
import sys
from typing import Any, Dict, List, Optional

try:
    from openai import APIConnectionError, AuthenticationError, BadRequestError, OpenAI, RateLimitError
except ImportError:
    print("Error: openai package not installed. Install it with: pip install openai>=1.0.0", file=sys.stderr)
    sys.exit(1)

from . import system_prompts


class APIClient:
    """
    Client for making API calls to OpenAI-compatible endpoints.

    Supports configuration via environment variables with auto-fallback:
    1. GROQ_API_KEY: Free Groq service (highest priority)
    2. OPENAI_API_KEY: OpenAI or custom endpoint (fallback)
    3. Error with setup instructions if neither is set

    Configuration variables:
    - GROQ_API_KEY: API authentication key for Groq (free, highest priority)
    - OPENAI_API_KEY: API authentication key for OpenAI/custom endpoint
    - OPENAI_API_BASE: Base URL for API (optional, default: https://api.openai.com/v1)
    - OPENAI_MODEL: Model name to use (optional, default: llama-3.3-70b-versatile for Groq, gpt-4o for OpenAI)
    - OPENAI_TIMEOUT: Request timeout in seconds (optional, default: 60)
    - OPENAI_MAX_RETRIES: Maximum retry attempts (optional, default: 2)
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the API client with auto-fallback.

        Priority order:
        1. GROQ_API_KEY (free Groq service)
        2. OPENAI_API_KEY (OpenAI or custom endpoint)
        3. Ollama (free local LLM if running)
        4. Error with setup instructions

        Args:
            verbose: If True, print additional debug information

        Raises:
            SystemExit: If no API key is configured (shows setup instructions)
        """
        self.verbose = verbose

        # Check for API keys in priority order
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if groq_key:
            # Use free Groq service (highest priority)
            self.api_key = groq_key
            self.api_base = "https://api.groq.com/openai/v1"
            self.model = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
            self.provider = "Groq"
        elif openai_key:
            # User provided OpenAI key or custom endpoint configuration
            self.api_key = openai_key
            self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
            self.provider = "OpenAI/Custom"
        elif self._is_ollama_available():
            # Use local Ollama if available
            self.api_key = "ollama"  # Dummy key, Ollama doesn't require authentication
            self.api_base = "http://localhost:11434/v1"
            self.model = os.getenv("OPENAI_MODEL", "llama2")
            self.provider = "Ollama (Local)"
        else:
            # No API key found and Ollama not running - show helpful error message
            print(system_prompts.ERROR_MESSAGES["api_key_missing"], file=sys.stderr)
            sys.exit(1)

        self.timeout = int(os.getenv("OPENAI_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))

        # Initialize OpenAI client with custom base URL support
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

        if self.verbose:
            print("API Client initialized:", file=sys.stderr)
            print(f"  Provider: {self.provider}", file=sys.stderr)
            print(f"  Base URL: {self.api_base}", file=sys.stderr)
            print(f"  Model: {self.model}", file=sys.stderr)
            print(f"  Timeout: {self.timeout}s", file=sys.stderr)

    def _is_ollama_available(self) -> bool:
        """
        Check if Ollama is running locally on the default port.

        Returns:
            True if Ollama is available at http://localhost:11434, False otherwise
        """
        try:
            import urllib.error
            import urllib.request
            urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)  # nosec B310
            return True
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

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

        except AuthenticationError:
            error_msg = system_prompts.ERROR_MESSAGES["api_authentication_failed"].format(
                api_base=self.api_base,
                model=self.model
            )
            print(error_msg, file=sys.stderr)
            raise

        except RateLimitError:
            print(system_prompts.ERROR_MESSAGES["api_rate_limit"], file=sys.stderr)
            raise

        except APIConnectionError:
            error_msg = system_prompts.ERROR_MESSAGES["api_connection_failed"].format(
                api_base=self.api_base
            )
            print(error_msg, file=sys.stderr)
            raise

        except BadRequestError as e:
            print(f"Error: Invalid request - {str(e)}", file=sys.stderr)
            raise

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            print(f"Unexpected API error: {str(e)}", file=sys.stderr)
            raise

    def test_connection(self) -> bool:
        """
        Test the API connection with a simple request.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except Exception as e:  # noqa: BLE001 — agent resilience
            if self.verbose:
                print(f"Connection test failed: {str(e)}", file=sys.stderr)
            return False
