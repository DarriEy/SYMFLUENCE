"""
Agent manager for the SYMFLUENCE AI agent.

This module provides the main orchestration for the agent, coordinating
API calls, tool execution, and conversation management.
"""

import sys

from .api_client import APIClient
from .conversation_manager import ConversationManager
from .tool_registry import ToolRegistry
from .tool_executor import ToolExecutor
from . import system_prompts


class AgentManager:
    """
    Main orchestration class for the SYMFLUENCE AI agent.

    Coordinates between the API client, conversation manager, tool registry,
    and tool executor to provide an intelligent assistant for SYMFLUENCE workflows.
    """

    def __init__(self, config_path: str, verbose: bool = False):
        """
        Initialize the agent manager.

        Args:
            config_path: Default config path for workflows
            verbose: Enable verbose output
        """
        self.config_path = config_path
        self.verbose = verbose

        # Initialize components
        self.api_client = APIClient(verbose=verbose)
        self.conversation_manager = ConversationManager(max_history=50)
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(tool_registry=self.tool_registry)

    def run_interactive_mode(self) -> int:
        """
        Run interactive chat mode with REPL loop.

        Provides a conversational interface where users can interact with
        SYMFLUENCE through natural language across multiple turns.

        Returns:
            Exit code (0 for success)
        """
        print(system_prompts.INTERACTIVE_WELCOME)

        # Try to enable readline for command history
        try:
            import readline  # noqa: F401 - imported for side effects (enables history)
        except ImportError:
            pass

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                    print(system_prompts.GOODBYE_MESSAGE)
                    return 0

                if user_input.lower() in ['/help', 'help']:
                    print(system_prompts.HELP_MESSAGE)
                    continue

                if user_input.lower() in ['/clear', 'clear']:
                    self.conversation_manager.clear_history()
                    print("Conversation history cleared.")
                    continue

                if user_input.lower() == '/tools':
                    self._print_available_tools()
                    continue

                # Process with agent
                print("\nAssistant: ", end="", flush=True)
                response = self._agent_loop(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type '/exit' to quit or continue chatting.")
                continue
            except EOFError:
                print("\n" + system_prompts.GOODBYE_MESSAGE)
                return 0
            except Exception as e:
                print(f"\nError: {str(e)}", file=sys.stderr)
                if self.verbose:
                    import traceback
                    traceback.print_exc()

        return 0

    def run_single_prompt(self, prompt: str) -> int:
        """
        Execute a single prompt and exit (non-interactive mode).

        Useful for scripting and automation where a single task needs to be
        performed via natural language.

        Args:
            prompt: The user's prompt/instruction

        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            response = self._agent_loop(prompt)
            print(response)
            return 0
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _agent_loop(self, user_message: str) -> str:
        """
        Core agent reasoning and execution loop.

        This implements the main agent logic:
        1. Add user message to conversation
        2. Call LLM with available tools
        3. If LLM requests tool calls:
           a. Execute each tool
           b. Add results to conversation
           c. Loop back to step 2
        4. Return final response when no more tool calls

        Args:
            user_message: The user's message/instruction

        Returns:
            The agent's final response

        Raises:
            Exception: If API calls fail or max iterations reached
        """
        # Add user message to conversation
        self.conversation_manager.add_user_message(user_message)

        max_iterations = 15  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            if self.verbose:
                print(f"\n[Agent Loop] Iteration {iteration}/{max_iterations}", file=sys.stderr)

            # Get messages and tools for API call
            messages = self.conversation_manager.get_messages()
            tools = self.tool_registry.get_tool_definitions()

            # Call LLM
            try:
                response = self.api_client.chat_completion(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            except Exception as e:
                return f"API error: {str(e)}"

            # Extract response
            message = response.choices[0].message

            # Check if LLM wants to use tools
            if hasattr(message, 'tool_calls') and message.tool_calls:
                if self.verbose:
                    print(f"[Agent Loop] {len(message.tool_calls)} tool call(s) requested", file=sys.stderr)

                # Add assistant message with tool calls
                tool_calls_data = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
                self.conversation_manager.add_assistant_message(
                    content=message.content,
                    tool_calls=tool_calls_data
                )

                # Execute each tool call
                for tool_call in message.tool_calls:
                    result = self._execute_tool_call(tool_call)

                    if self.verbose:
                        print(f"[Agent Loop] Tool {tool_call.function.name}: {'✓' if result.success else '✗'}", file=sys.stderr)

                    # Add tool result to conversation
                    self.conversation_manager.add_tool_result(
                        tool_call_id=tool_call.id,
                        result=result.to_string(),
                        tool_name=tool_call.function.name
                    )

                # Continue loop to get final response after tool execution
                continue

            else:
                # No tool calls - this is the final response
                final_response = message.content or "I completed the task."

                # Add final assistant message
                self.conversation_manager.add_assistant_message(content=final_response)

                return final_response

        # Max iterations reached
        error_msg = system_prompts.ERROR_MESSAGES["max_iterations_reached"]
        print(error_msg, file=sys.stderr)
        return "I apologize, but I encountered an issue completing this task. Please try breaking it into smaller steps."

    def _execute_tool_call(self, tool_call):
        """
        Execute a single tool call from the LLM.

        Args:
            tool_call: Tool call object from API response

        Returns:
            ToolResult with execution status and output
        """
        import json
        from .tool_executor import ToolResult

        tool_name = tool_call.function.name

        try:
            # Parse arguments
            arguments = json.loads(tool_call.function.arguments)

            if self.verbose:
                print(f"[Tool Execution] {tool_name}({arguments})", file=sys.stderr)

            # Execute tool
            result = self.tool_executor.execute_tool(tool_name, arguments)

            return result

        except json.JSONDecodeError as e:
            from .tool_executor import ToolResult
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid tool arguments: {str(e)}",
                exit_code=1
            )
        except Exception as e:
            from .tool_executor import ToolResult
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution failed: {str(e)}",
                exit_code=1
            )

    def _print_available_tools(self) -> None:
        """Print all available tools in a formatted manner."""
        print("\n" + "="*60)
        print("Available Tools")
        print("="*60 + "\n")

        tools_by_category = self.tool_registry.get_tools_by_category()

        for category, tools in tools_by_category.items():
            print(f"{category}:")
            print("-" * 60)
            for tool in tools:
                name = tool["function"]["name"]
                description = tool["function"]["description"]
                print(f"  • {name}")
                print(f"    {description}")
                print()
            print()

        print("="*60 + "\n")
