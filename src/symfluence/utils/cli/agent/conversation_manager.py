"""
Conversation history manager for the SYMFLUENCE AI agent.

This module manages the conversation state and message history using the
OpenAI message format, which is compatible with most LLM providers.
"""

from typing import List, Dict, Optional, Any
from . import system_prompts


class ConversationManager:
    """
    Manages conversation history and state for the agent.

    Uses the standard OpenAI message format:
    - role: "system" | "user" | "assistant" | "tool"
    - content: message text
    - tool_calls: (optional) list of tool calls from assistant
    - tool_call_id: (for tool messages) which call this result is for
    """

    def __init__(self, max_history: int = 50):
        """
        Initialize the conversation manager.

        Args:
            max_history: Maximum number of messages to keep (excluding system prompt)
        """
        self.max_history = max_history
        self.messages: List[Dict[str, Any]] = []
        self._initialize_with_system_prompt()

    def _initialize_with_system_prompt(self) -> None:
        """Initialize conversation with system prompt."""
        self.messages = [
            {
                "role": "system",
                "content": system_prompts.SYSTEM_PROMPT
            }
        ]

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            content: The user's message text
        """
        self.messages.append({
            "role": "user",
            "content": content
        })
        self._trim_history()

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None
    ) -> None:
        """
        Add an assistant message to the conversation.

        Args:
            content: The assistant's response text (can be None if only tool calls)
            tool_calls: List of tool calls the assistant wants to make
        """
        message: Dict[str, Any] = {
            "role": "assistant"
        }

        if content:
            message["content"] = content

        if tool_calls:
            message["tool_calls"] = tool_calls

        # If neither content nor tool_calls, add empty content
        if not content and not tool_calls:
            message["content"] = ""

        self.messages.append(message)
        self._trim_history()

    def add_tool_result(self, tool_call_id: str, result: str, tool_name: str) -> None:
        """
        Add a tool execution result to the conversation.

        Args:
            tool_call_id: ID of the tool call this result corresponds to
            result: The result string from tool execution
            tool_name: Name of the tool that was executed
        """
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        self._trim_history()

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get the full conversation history for API calls.

        Returns:
            List of message dictionaries in OpenAI format
        """
        return self.messages.copy()

    def clear_history(self, keep_system_prompt: bool = True) -> None:
        """
        Clear the conversation history.

        Args:
            keep_system_prompt: If True, keep the initial system prompt
        """
        if keep_system_prompt:
            self._initialize_with_system_prompt()
        else:
            self.messages = []

    def _trim_history(self) -> None:
        """
        Trim history to max_history, keeping system prompt and recent messages.

        This ensures we don't exceed context windows while preserving:
        1. The system prompt (always first)
        2. The most recent messages up to max_history
        """
        if len(self.messages) <= self.max_history + 1:  # +1 for system prompt
            return

        # Keep system prompt (first message) + most recent messages
        system_prompt = self.messages[0]
        recent_messages = self.messages[-(self.max_history):]
        self.messages = [system_prompt] + recent_messages

    def get_conversation_length(self) -> int:
        """
        Get the number of messages in the conversation.

        Returns:
            Number of messages (including system prompt)
        """
        return len(self.messages)

    def get_last_user_message(self) -> Optional[str]:
        """
        Get the content of the last user message.

        Returns:
            Content of last user message, or None if no user messages exist
        """
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return message.get("content")
        return None

    def get_last_assistant_message(self) -> Optional[str]:
        """
        Get the content of the last assistant message.

        Returns:
            Content of last assistant message, or None if no assistant messages exist
        """
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                return message.get("content")
        return None
