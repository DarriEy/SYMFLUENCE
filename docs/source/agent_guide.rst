AI Agent Guide
==============

``symfluence agent launch`` hands off to an installed coding-agent CLI — Claude
Code (``claude``), OpenAI Codex (``codex``), Gemini CLI (``gemini``), or another —
primed as the *SYMFLUENCE agent*. SYMFLUENCE does not ship its own language
model: it detects whichever agent CLI you have installed, picks up whichever API
key is in your environment, primes the CLI with SYMFLUENCE context, and replaces
itself with that agent so it drives your project directly.

The priming has four provider-agnostic layers, each wired in only where the host
CLI supports it:

1. **Skills** — packaged domain guides for running and extending the platform.
2. **Identity & project context** — a system-prompt block that makes the session
   the SYMFLUENCE agent, including live context detected at launch (your config
   files, domain directories, experiment settings) and house rules (query the
   live registry, drive runs through ``symfluence workflow``, ...).
3. **The SYMFLUENCE MCP server** — structured tools for registry introspection,
   config validation, and workflow execution (``symfluence agent mcp``).
4. **Specialist subagents** — e.g. a calibration debugger the host CLI can
   delegate to.

This gives you a full, modern coding agent (with its own editing, search, and
git tooling) that already knows what platform it is embedded in and what project
it was launched into.

Prerequisites
-------------

Install one coding-agent CLI and set the matching API key:

================  ==========================================  ===========================
CLI               Install                                     API key
================  ==========================================  ===========================
Claude Code       https://docs.claude.com/claude-code         ``ANTHROPIC_API_KEY``
Codex CLI         https://github.com/openai/codex             ``OPENAI_API_KEY``
Gemini CLI        https://github.com/google-gemini/gemini-cli ``GEMINI_API_KEY``
================  ==========================================  ===========================

A CLI with a saved login (e.g. ``claude`` after ``claude login``) also works — the
API key is only used by the CLI itself, never read or forwarded by SYMFLUENCE.

Usage
-----

Open the Agent Command Center (run from your project directory)::

    symfluence agent launch

This is a dedicated screen in the SYMFLUENCE TUI: it shows the detected
runtimes (pick one with the arrow keys), the mission context (configs and
domains found in your directory), the capabilities that will prime the session,
and a preflight check panel. Press ``l`` to launch, ``p`` for a one-shot
prompt, ``k`` to toggle priming, ``r`` to refresh. The rest of the TUI
(dashboard, runs, workflow, calibration) is one keypress away; inside
``symfluence tui launch`` the same screen is mode ``7``.

The handoff always happens *after* the TUI exits — the screen never runs the
agent inside itself, which keeps the terminal clean for the host CLI (and
leaves room to embed the agent in the TUI later).

Hand off immediately without the command center::

    symfluence agent launch --direct

One-shot prompt (always direct; runs once and exits — useful in scripts)::

    symfluence agent launch "add an MSWEP forcing data handler"

Pick a specific CLI, or launch it bare without SYMFLUENCE priming::

    symfluence agent launch --cli codex
    symfluence agent launch --no-skills

Forward extra flags to the underlying CLI after ``--``::

    symfluence agent launch -- --model claude-sonnet-4-6

Sessions without a TTY, or installs without the TUI extra (``pip install
"symfluence[tui]"``), fall back to the direct handoff automatically.

Inspect the setup::

    symfluence agent list      # registered CLIs; which one launch would pick
    symfluence agent skills    # the packaged skills
    symfluence agent doctor    # full diagnosis (CLIs, keys, skills, MCP server)

How it works
------------

1. **Detect a CLI.** SYMFLUENCE looks for ``claude``, then ``codex``, then
   ``gemini`` on your ``PATH`` (first match wins). Override with ``--cli`` or
   ``SYMFLUENCE_AGENT_CLI=<command>``.
2. **Prime it.** Each priming layer is delivered through whatever mechanism the
   CLI offers, declared per-CLI in the launcher registry — nothing is
   provider-specific in the layers themselves:

   - *Skills*: Claude Code's native ``.claude/skills/`` discovery (via
     ``--add-dir``, without touching your project), or a generated ``AGENTS.md``
     (the cross-tool convention honoured by Codex, Gemini, and others), only if
     one is not already present.
   - *Identity & context*: the CLI's system-prompt flag where one exists
     (``--append-system-prompt`` for Claude Code), otherwise the top of the
     generated ``AGENTS.md``.
   - *MCP server*: an MCP config passed per-launch (``--mcp-config`` for Claude
     Code, ``-c mcp_servers...`` overrides for Codex). CLIs that only read MCP
     servers from their own settings (e.g. Gemini) get a printed one-line
     instruction for registering ``symfluence agent mcp`` once.
   - *Subagents*: the CLI's agent-definition flag where one exists
     (``--agents`` for Claude Code).
3. **Hand off.** SYMFLUENCE replaces its own process with the CLI, which then
   owns the terminal directly.

Skills
------

Skills are concise domain guides the agent consults when working on SYMFLUENCE
tasks — for both *running* the platform and *extending* it. The packaged skills:

==========================  =================================================
Skill                       Use it for
==========================  =================================================
``explore-platform``        Discovering available models/datasets/configs (live registry)
``run-workflow-locally``    Running the workflow end to end (or a single step)
``add-data-handler``        Adding a forcing / attribute / observation dataset
``add-model-handler``       Adding or wiring a hydrological model
``add-optimizer``           Adding a calibration/search algorithm
``debug-calibration``       Diagnosing calibration / optimizer problems
==========================  =================================================

The MCP server
--------------

``symfluence agent mcp`` serves the Model Context Protocol on stdio, exposing
structured tools backed by the live platform:

======================  =====================================================
Tool                    What it does
======================  =====================================================
``list_capabilities``   Registry catalogs: models, forcings, optimizers, ...
``validate_config``     Typed validation of a config file
``workflow_status``     Per-step pipeline status for a config
``run_workflow_step``   Run a single workflow step (long-running; set a timeout)
======================  =====================================================

``agent launch`` wires it in automatically where the host CLI supports
per-launch MCP configuration. To register it manually in any MCP-capable tool,
add a stdio server named ``symfluence`` running ``symfluence agent mcp`` — e.g.
for Gemini CLI::

    gemini mcp add symfluence symfluence agent mcp

Subagents
---------

Packaged specialist definitions are registered with host CLIs that support
custom subagents:

========================  ===================================================
Subagent                  Speciality
========================  ===================================================
``calibration-debugger``  Fault-tree diagnosis of misbehaving calibrations
``platform-scout``        Registry-backed "what does this install support?"
========================  ===================================================

Environment variables
---------------------

=========================  ===============================================================
Variable                   Effect
=========================  ===============================================================
``SYMFLUENCE_AGENT_CLI``   Force a specific CLI (e.g. ``codex``) instead of auto-detection
``SYMFLUENCE_NO_SKILLS``   Skip all priming and launch the bare CLI
=========================  ===============================================================

Troubleshooting
---------------

Run ``symfluence agent doctor`` first — it checks CLI detection, API keys,
packaged skills/subagents, the cache directory, the MCP server, and the project
context detected in your current directory.

**"No coding-agent CLI found on PATH."** Install one of the CLIs above and set its
API key, or point ``--cli`` / ``SYMFLUENCE_AGENT_CLI`` at an installed command.

**The agent doesn't seem to know about SYMFLUENCE.** Make sure you launched from
your project directory and that ``SYMFLUENCE_NO_SKILLS`` is not set. For Codex/Gemini,
check that an ``AGENTS.md`` was written (or already exists) in the working directory.

**The agent doesn't see my config.** Project context is detected from YAML files
in the working directory and ``0_config_files/`` that contain SYMFLUENCE keys
(``DOMAIN_NAME`` or ``HYDROLOGICAL_MODEL``). Launch from the project root, or
point ``SYMFLUENCE_DEFAULT_CONFIG`` at your config file.

Deprecation
-----------

``symfluence agent start`` and ``symfluence agent run "..."`` are deprecated aliases
for ``symfluence agent launch`` (interactive and one-shot respectively) and will be
removed in a future release.

See Also
--------

- :doc:`cli_reference` - CLI command reference
- :doc:`getting_started` - General SYMFLUENCE quickstart
- :doc:`configuration` - Configuration file reference
