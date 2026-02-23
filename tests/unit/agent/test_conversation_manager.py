"""
Tests for the ConversationManager class.
"""

from unittest.mock import Mock, patch

import pytest


class TestConversationManagerInitialization:
    """Tests for ConversationManager initialization."""

    def test_conversation_manager_can_be_imported(self):
        """Test that ConversationManager can be imported."""
        from symfluence.agent.conversation_manager import ConversationManager
        assert ConversationManager is not None

    def test_conversation_manager_initialization(self):
        """Test ConversationManager initializes with system prompt."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()

        assert manager is not None
        assert len(manager.messages) == 1
        assert manager.messages[0]['role'] == 'system'

    def test_custom_max_history(self):
        """Test ConversationManager with custom max_history."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager(max_history=10)

        assert manager.max_history == 10


class TestMessageAddition:
    """Tests for adding messages to conversation."""

    def test_add_user_message(self):
        """Test adding user message."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        manager.add_user_message("Hello")

        assert len(manager.messages) == 2
        assert manager.messages[1]['role'] == 'user'
        assert manager.messages[1]['content'] == 'Hello'

    def test_add_assistant_message_with_content(self):
        """Test adding assistant message with content."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        manager.add_assistant_message(content="Hello, I'm here to help!")

        assert len(manager.messages) == 2
        assert manager.messages[1]['role'] == 'assistant'
        assert manager.messages[1]['content'] == "Hello, I'm here to help!"

    def test_add_assistant_message_with_tool_calls(self):
        """Test adding assistant message with tool calls."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        tool_calls = [{"id": "call_123", "function": {"name": "test_tool"}}]
        manager.add_assistant_message(tool_calls=tool_calls)

        assert len(manager.messages) == 2
        assert 'tool_calls' in manager.messages[1]
        assert manager.messages[1]['tool_calls'] == tool_calls

    def test_add_assistant_message_empty(self):
        """Test adding assistant message with no content or tool calls."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        manager.add_assistant_message()

        assert len(manager.messages) == 2
        assert manager.messages[1]['content'] == ''

    def test_add_tool_result(self):
        """Test adding tool result."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        manager.add_tool_result(
            tool_call_id="call_123",
            result="Tool executed successfully",
            tool_name="test_tool"
        )

        assert len(manager.messages) == 2
        assert manager.messages[1]['role'] == 'tool'
        assert manager.messages[1]['tool_call_id'] == 'call_123'
        assert manager.messages[1]['name'] == 'test_tool'
        assert manager.messages[1]['content'] == 'Tool executed successfully'


class TestMessageRetrieval:
    """Tests for retrieving messages."""

    def test_get_messages_returns_copy(self):
        """Test get_messages returns a copy."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        messages = manager.get_messages()

        # Modify the returned list
        messages.append({"role": "user", "content": "test"})

        # Original should be unchanged
        assert len(manager.messages) == 1

    def test_get_conversation_length(self):
        """Test getting conversation length."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        assert manager.get_conversation_length() == 1

        manager.add_user_message("Hello")
        assert manager.get_conversation_length() == 2

    def test_get_last_user_message(self):
        """Test getting last user message."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        manager.add_user_message("First")
        manager.add_assistant_message("Response")
        manager.add_user_message("Second")

        assert manager.get_last_user_message() == "Second"

    def test_get_last_user_message_none(self):
        """Test getting last user message when none exist."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        assert manager.get_last_user_message() is None

    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        manager.add_assistant_message("First response")
        manager.add_user_message("Question")
        manager.add_assistant_message("Second response")

        assert manager.get_last_assistant_message() == "Second response"

    def test_get_last_assistant_message_none(self):
        """Test getting last assistant message when none exist."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        manager.add_user_message("Hello")

        assert manager.get_last_assistant_message() is None


class TestHistoryManagement:
    """Tests for conversation history management."""

    def test_clear_history_keeps_system_prompt(self):
        """Test clearing history keeps system prompt."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there!")

        manager.clear_history(keep_system_prompt=True)

        assert len(manager.messages) == 1
        assert manager.messages[0]['role'] == 'system'

    def test_clear_history_removes_all(self):
        """Test clearing all history."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()
        manager.add_user_message("Hello")

        manager.clear_history(keep_system_prompt=False)

        assert len(manager.messages) == 0

    def test_trim_history(self):
        """Test history trimming when exceeding max_history."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager(max_history=5)

        # Add more messages than max_history
        for i in range(10):
            manager.add_user_message(f"Message {i}")

        # Should have system prompt + max_history messages
        assert len(manager.messages) <= manager.max_history + 1
        # System prompt should still be first
        assert manager.messages[0]['role'] == 'system'

    def test_trim_keeps_recent_messages(self):
        """Test trimming keeps most recent messages."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager(max_history=3)

        manager.add_user_message("Old message")
        manager.add_user_message("Middle message")
        manager.add_user_message("New message 1")
        manager.add_user_message("New message 2")
        manager.add_user_message("New message 3")

        # Get the last user message
        messages = manager.get_messages()
        user_messages = [m for m in messages if m['role'] == 'user']

        # Most recent should be preserved
        assert user_messages[-1]['content'] == "New message 3"


class TestConversationFlow:
    """Tests for typical conversation flows."""

    def test_complete_conversation_flow(self):
        """Test a complete conversation with tools."""
        from symfluence.agent.conversation_manager import ConversationManager

        manager = ConversationManager()

        # User asks question
        manager.add_user_message("What workflow steps are available?")

        # Assistant requests tool call
        tool_calls = [{"id": "call_001", "function": {"name": "list_workflow_steps"}}]
        manager.add_assistant_message(tool_calls=tool_calls)

        # Tool result comes back
        manager.add_tool_result(
            tool_call_id="call_001",
            result="Available steps: setup_project, acquire_attributes...",
            tool_name="list_workflow_steps"
        )

        # Assistant gives final response
        manager.add_assistant_message(content="The available workflow steps are...")

        # Verify conversation structure
        assert len(manager.messages) == 5  # system + 4 messages
        assert manager.messages[1]['role'] == 'user'
        assert manager.messages[2]['role'] == 'assistant'
        assert manager.messages[3]['role'] == 'tool'
        assert manager.messages[4]['role'] == 'assistant'
