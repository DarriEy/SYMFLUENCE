"""
SYMFLUENCE AI Agent Module

This module provides an AI-powered agent interface for SYMFLUENCE operations.
Users can interact with SYMFLUENCE workflows using natural language through any
OpenAI-compatible API (OpenAI, Anthropic, local LLMs, etc.).

Main Components:
- AgentManager: Main orchestration class with parallel tool execution
- APIClient: OpenAI-compatible API client (supports OpenAI, Groq, Ollama)
- ConversationManager: Conversation state management
- ToolRegistry: Tool definitions for function calling
- ToolExecutor: Tool execution engine
- CodeSearch: Ripgrep-based code search capabilities
- PRManager: Automated PR creation via GitHub CLI

SOTA Features:
- Fuzzy code matching for self-modification
- Parallel tool execution for read-only operations
- Automated PR creation via gh CLI
- Code search with ripgrep integration

Usage:
    # Interactive mode
    $ symfluence agent start

    # Single prompt mode
    $ symfluence agent run "Install all modeling tools"

Environment Variables:
    OPENAI_API_KEY: API authentication key (or GROQ_API_KEY for free option)
    OPENAI_API_BASE: Base URL for API (optional)
    OPENAI_MODEL: Model name to use (optional)
"""

from .agent_manager import AgentManager
from .api_client import APIClient
from .code_search import CodeSearch, FuzzyMatcher
from .conversation_manager import ConversationManager
from .pr_manager import PRManager
from .tool_executor import ToolExecutor, ToolResult
from .tool_registry import ToolRegistry

__all__ = [
    'AgentManager',
    'APIClient',
    'ConversationManager',
    'ToolRegistry',
    'ToolExecutor',
    'ToolResult',
    'CodeSearch',
    'FuzzyMatcher',
    'PRManager',
]
