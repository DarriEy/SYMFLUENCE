"""
SYMFLUENCE AI Agent Module

This module provides an AI-powered agent interface for SYMFLUENCE operations.
Users can interact with SYMFLUENCE workflows using natural language through any
OpenAI-compatible API (OpenAI, Anthropic, local LLMs, etc.).

Main Components:
- AgentManager: Main orchestration class
- APIClient: OpenAI-compatible API client
- ConversationManager: Conversation state management
- ToolRegistry: Tool definitions for function calling
- ToolExecutor: Tool execution engine

Usage:
    # Interactive mode
    $ symfluence agent start

    # Single prompt mode
    $ symfluence agent run "Install all modeling tools"

Environment Variables:
    OPENAI_API_KEY: API authentication key (required)
    OPENAI_API_BASE: Base URL for API (optional)
    OPENAI_MODEL: Model name to use (optional)
"""

from .agent_manager import AgentManager
from .api_client import APIClient
from .conversation_manager import ConversationManager
from .tool_registry import ToolRegistry
from .tool_executor import ToolExecutor, ToolResult

__all__ = [
    'AgentManager',
    'APIClient',
    'ConversationManager',
    'ToolRegistry',
    'ToolExecutor',
    'ToolResult',
]
