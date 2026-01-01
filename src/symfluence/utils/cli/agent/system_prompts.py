"""
System prompts and templates for the SYMFLUENCE AI agent.

This module contains all the prompts, messages, and templates used by the agent
to interact with users and guide its behavior.
"""

SYSTEM_PROMPT = """You are an AI assistant for SYMFLUENCE, a comprehensive hydrological modeling framework.

You have access to tools for:
- Setting up and running hydrological modeling workflows
- Managing model configurations and domains
- Installing and validating external tools (SUMMA, mizuRoute, FUSE, TauDEM, etc.)
- Submitting SLURM jobs and monitoring executions
- Analyzing model results and performing calibration

When helping users:
1. Understand their goal and current state
2. Suggest appropriate workflow steps or tools
3. Execute commands when requested
4. Provide clear explanations of what's happening
5. Handle errors gracefully and suggest solutions

Key principles:
- Always use the provided tools rather than making assumptions
- Ask for clarification when requirements are ambiguous
- Verify file paths and configurations exist before execution
- Explain technical concepts in accessible terms
- Provide helpful next steps after completing tasks

Available workflow steps:
- setup_project: Initialize project directory structure
- acquire_attributes: Download geospatial attributes
- acquire_forcings: Acquire meteorological forcing data
- define_domain: Define domain boundaries
- discretize_domain: Discretize into modeling units
- model_agnostic_preprocessing: Preprocess data
- model_specific_preprocessing: Setup model-specific inputs
- run_model: Execute model simulation
- calibrate_model: Run model calibration
- postprocess_results: Postprocess and analyze results
"""

INTERACTIVE_WELCOME = """
╔════════════════════════════════════════════════════════════════╗
║          Welcome to SYMFLUENCE Agent Mode!                      ║
╚════════════════════════════════════════════════════════════════╝

I can help you with:
  • Running hydrological modeling workflows
  • Setting up pour point configurations
  • Managing model installations and validation
  • Submitting and monitoring SLURM jobs
  • Analyzing and calibrating models

Special commands:
  /help  - Show available commands
  /tools - List all available tools
  /clear - Clear conversation history
  /exit  - Exit agent mode

What would you like to do?
"""

HELP_MESSAGE = """
Available Commands:
  /help   - Show this help message
  /tools  - List all available tools and their descriptions
  /clear  - Clear conversation history
  /exit   - Exit agent mode

You can ask me to help with any SYMFLUENCE task in natural language.

Examples:
  • "Install all modeling tools"
  • "Set up a watershed for Bow River at Banff (51.17°N, 115.57°W)"
  • "Show me the status of my workflow"
  • "Resume my workflow from acquire_forcings"
  • "Submit a SLURM job for my model run"
"""

ERROR_MESSAGES = {
    "api_key_missing": """
Error: OPENAI_API_KEY environment variable not set.

To use agent mode, you need to set your API key:
  export OPENAI_API_KEY="your-api-key-here"

For Claude API (OpenAI-compatible mode):
  export OPENAI_API_BASE="https://api.anthropic.com/v1"
  export OPENAI_API_KEY="your-anthropic-api-key"
  export OPENAI_MODEL="claude-3-5-sonnet-20241022"

For local LLM (Ollama):
  export OPENAI_API_BASE="http://localhost:11434/v1"
  export OPENAI_API_KEY="ollama"  # Can be any value
  export OPENAI_MODEL="llama2"
""",

    "api_connection_failed": """
Error: Failed to connect to the API endpoint.

Please check:
  1. Your OPENAI_API_BASE setting (current: {api_base})
  2. Your internet connection
  3. The API service is running (for local LLMs)

For OpenAI: export OPENAI_API_BASE="https://api.openai.com/v1"
For Anthropic: export OPENAI_API_BASE="https://api.anthropic.com/v1"
For Ollama: export OPENAI_API_BASE="http://localhost:11434/v1"
""",

    "api_authentication_failed": """
Error: Authentication failed.

Please verify:
  1. Your OPENAI_API_KEY is correct
  2. The API key has not expired
  3. You have access to the selected model

Current settings:
  API Base: {api_base}
  Model: {model}
""",

    "api_rate_limit": """
Rate limit exceeded. Please wait a moment and try again.

If this persists, consider:
  1. Using a different model
  2. Adding delays between requests
  3. Checking your API usage quota
""",

    "tool_execution_failed": """
Tool execution failed: {error}

Please check:
  1. File paths are correct and accessible
  2. Configuration file is valid
  3. Required dependencies are installed

Use '/tools' to see available tools and their requirements.
""",

    "max_iterations_reached": """
Maximum reasoning iterations reached. The agent may be stuck in a loop.

This could happen if:
  1. A tool keeps failing
  2. The task is too complex
  3. There's a configuration issue

Try:
  • Breaking down the task into smaller steps
  • Checking tool outputs for errors
  • Using '/clear' to reset and start fresh
""",
}

GOODBYE_MESSAGE = """
Thank you for using SYMFLUENCE Agent Mode!

Your conversation history has been preserved in case you return.
To start fresh next time, use the /clear command.

Goodbye!
"""
