AI Agent Guide
==============

SYMFLUENCE includes an AI-powered agent that provides natural language interaction
for managing hydrological modeling workflows. The agent can execute workflow steps,
manage configurations, analyze code, and automate complex tasks.

Overview
--------

The SYMFLUENCE agent is a conversational interface that:

- Executes workflow steps through natural language commands
- Manages model configurations interactively
- Provides intelligent assistance for troubleshooting
- Supports code analysis and modification proposals
- Integrates with multiple LLM providers (OpenAI, Groq, Ollama)

Getting Started
---------------

API Key Setup
^^^^^^^^^^^^^

The agent requires an API key for an OpenAI-compatible LLM provider. Configure
one of the following environment variables:

**OpenAI (Recommended for best performance)**::

    export OPENAI_API_KEY="sk-your-key-here"

**Groq (Free tier available)**::

    export GROQ_API_KEY="gsk_your-key-here"

**Ollama (Local, free)**::

    # Start Ollama server locally
    ollama serve
    # The agent will auto-detect Ollama on localhost:11434

Interactive Mode
^^^^^^^^^^^^^^^^

Start the agent in interactive mode for multi-turn conversations::

    symfluence agent

You'll see a welcome message and can start chatting:

.. code-block:: text

    SYMFLUENCE AI Agent
    Type '/help' for commands, '/exit' to quit.

    You: Help me set up a new watershed project
    Assistant: I'd be happy to help you set up a new watershed project...

Single Prompt Mode
^^^^^^^^^^^^^^^^^^

Execute a single command and exit (useful for scripts)::

    symfluence agent --prompt "Run the preprocessing step for my config.yaml"

Interactive Commands
--------------------

The following commands are available in interactive mode:

================  ================================================
Command           Description
================  ================================================
``/help``         Show help information
``/tools``        List all available tools
``/clear``        Clear conversation history
``/exit``         Exit the agent
================  ================================================

Example Workflows
-----------------

Setting Up a New Watershed
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    You: Set up a new watershed project for the Bow River at Banff
         with coordinates 51.1722, -115.5717

    Assistant: I'll set up a new watershed project for you...
    [Agent executes setup_pour_point_workflow tool]

Running Workflow Steps
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    You: Run the domain delineation step for config.yaml

    Assistant: Running domain delineation...
    [Agent executes define_domain tool]

Checking Workflow Status
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    You: What's the current status of my workflow?

    Assistant: Let me check the workflow status...
    [Agent executes show_workflow_status tool]

Installing Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    You: Install SUMMA and mizuRoute

    Assistant: Installing the requested tools...
    [Agent executes install_executables tool]

Configuration Options
---------------------

The agent can be configured via environment variables:

=======================  =====================================  ===================
Variable                 Description                            Default
=======================  =====================================  ===================
``OPENAI_API_KEY``       OpenAI API key                         (required)
``OPENAI_API_BASE``      Custom API base URL                    api.openai.com/v1
``OPENAI_MODEL``         Model to use                           gpt-4-turbo-preview
``OPENAI_TIMEOUT``       Request timeout (seconds)              60
``OPENAI_MAX_RETRIES``   Maximum retry attempts                 2
``GROQ_API_KEY``         Groq API key (fallback if no OpenAI)   -
=======================  =====================================  ===================

Provider Priority
^^^^^^^^^^^^^^^^^

When initializing, the agent checks for API keys in this order:

1. **OPENAI_API_KEY** - OpenAI or custom endpoint
2. **GROQ_API_KEY** - Groq free service
3. **Ollama** - Local LLM if running
4. **Error** - Shows setup instructions

Architecture
------------

The agent consists of several components:

- **AgentManager**: Main orchestration and REPL loop
- **APIClient**: Handles LLM API calls with provider fallback
- **ConversationManager**: Manages message history
- **ToolRegistry**: Defines available tools for function calling
- **ToolExecutor**: Executes tools and returns structured results

The agent uses OpenAI's function calling (tools) API to determine when to
execute SYMFLUENCE operations vs. provide conversational responses.

Troubleshooting
---------------

API Key Issues
^^^^^^^^^^^^^^

If you see "API key not configured":

1. Verify your environment variable is set: ``echo $OPENAI_API_KEY``
2. Ensure the key is valid and has sufficient credits
3. Try the Groq fallback: ``export GROQ_API_KEY="your-key"``

Connection Errors
^^^^^^^^^^^^^^^^^

If connections fail:

1. Check your internet connection
2. Verify the API endpoint is reachable
3. For Ollama, ensure the server is running: ``ollama serve``

Tool Execution Failures
^^^^^^^^^^^^^^^^^^^^^^^

If tools fail to execute:

1. Check that SYMFLUENCE is properly installed
2. Verify your configuration file exists and is valid
3. Use ``/tools`` to see available operations

Advanced Usage
--------------

Multi-Step Workflows
^^^^^^^^^^^^^^^^^^^^

The agent can execute complex multi-step workflows:

.. code-block:: text

    You: Set up a complete SUMMA project for the Bow River watershed,
         delineate the domain, acquire ERA5 forcing data, and run
         preprocessing

    Assistant: I'll set up your SUMMA project step by step...
    [Agent executes multiple tools in sequence]

Configuration Assistance
^^^^^^^^^^^^^^^^^^^^^^^^

Get help with configuration options:

.. code-block:: text

    You: What parameters should I calibrate for a snow-dominated basin
         using SUMMA?

    Assistant: For snow-dominated basins with SUMMA, I recommend...

Model Comparison
^^^^^^^^^^^^^^^^

Compare different models:

.. code-block:: text

    You: Compare SUMMA and FUSE for my watershed - which should I use?

    Assistant: Let me analyze your configuration and domain...

Capabilities
------------

The SYMFLUENCE agent can:

**Workflow Operations:**

- Execute any of the 16 workflow steps
- Run complete or partial workflows
- Monitor workflow status and progress
- Resume interrupted workflows
- Clean intermediate files

**Project Management:**

- Initialize new projects with presets
- Set up pour point workflows
- Configure domain definitions
- Manage experiment settings

**Model Operations:**

- Configure hydrological models (SUMMA, FUSE, GR, HBV, HYPE, NGEN, etc.)
- Set up routing (mizuRoute, dRoute)
- Manage model parameters
- Run calibration workflows

**Binary Management:**

- Install model executables
- Validate installations
- Run system diagnostics
- Check tool versions

**Code Analysis:**

- Browse the SYMFLUENCE codebase
- Propose code modifications
- Run tests
- Create PR proposals

Limitations
-----------

The agent has certain limitations:

- Cannot access external systems without explicit API keys
- Model execution times depend on domain size and complexity
- Some operations require HPC resources
- Code changes require human review before committing

Best Practices
--------------

1. **Be Specific**: Provide clear, specific instructions
2. **Include Context**: Mention your config file, domain name, or model
3. **Iterate**: If results aren't right, refine your request
4. **Check Output**: Review agent actions before proceeding
5. **Use Debug Mode**: Enable ``--verbose`` for troubleshooting

See Also
--------

- :doc:`agent_tools` - Complete tool reference
- :doc:`cli_reference` - CLI command reference
- :doc:`getting_started` - General SYMFLUENCE quickstart
- :doc:`configuration` - Configuration file reference
