AI Agent Guide
==============

``symfluence agent`` hands off to an installed coding-agent CLI — Claude Code
(``claude``), OpenAI Codex (``codex``), Gemini CLI (``gemini``), or another —
primed as the *SYMFLUENCE agent*. SYMFLUENCE does not ship its own language
model: it detects whichever agent CLI you have installed, picks up whichever API
key is in your environment, primes the CLI with SYMFLUENCE context, and replaces
itself with that agent so it drives your project directly.

There are two session modes, each with its own priming profile:

- ``symfluence agent model`` — a **modelling session**: run experiments
  conversationally (configs, runs, calibrations, results). Primed with the
  operational skills and the MCP tools; its house rules forbid editing platform
  source code and require driving everything through the workflow tools.
- ``symfluence agent code`` — a **coding session**: extend the platform
  (models, data handlers, optimizers) in the host CLI's full coding
  experience, primed with every packaged skill and subagent.

Bare ``symfluence agent`` opens the TUI agent screen to pick a mode.

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

================  =============================================  ===========================
CLI               Install                                        API key
================  =============================================  ===========================
Claude Code       https://docs.claude.com/claude-code            ``ANTHROPIC_API_KEY``
Codex CLI         https://github.com/openai/codex                ``OPENAI_API_KEY``
Gemini CLI        https://github.com/google-gemini/gemini-cli    ``GEMINI_API_KEY``
================  =============================================  ===========================

A CLI with a saved login (e.g. ``claude`` after ``claude login``) also works — the
API key is only used by the CLI itself, never read or forwarded by SYMFLUENCE.

Usage
-----

Start a session from your project directory::

    symfluence agent           # TUI agent screen: pick a mode
    symfluence agent model     # modelling session
    symfluence agent code      # coding session

The TUI agent screen (mode ``7`` inside ``symfluence tui launch``) is a
minimal home: two mode cards, the detected config, and runtime readiness;
``d`` opens the diagnostics detail. Starting a **coding** session from it
round-trips — the TUI suspends, the real coding-agent CLI runs full-screen in
the same terminal, and the home screen returns when the session ends.
Terminals that cannot suspend fall back to the classic exec handoff after the
TUI exits.

Starting a **modelling** session opens the native chat screen (Claude Code
only — it is driven headlessly over its stream-JSON protocol): assistant prose
streams into the conversation, tool invocations render as compact expandable
cards, and a run sidebar ticks independently (config, calibration progress,
background jobs, last log line) so long steps never make the screen feel
dead. ``esc`` interrupts a running turn or backs out when idle; sessions
resume automatically per project and mode. With Codex/Gemini (or if the
stream fails) modelling falls back to the suspend round-trip with modelling
priming.

One-shot prompts (run once and exit — useful in scripts)::

    symfluence agent model "validate my config and run the next step"
    symfluence agent code "add an MSWEP forcing data handler"

Skip the TUI, pick a specific CLI, or launch bare without SYMFLUENCE priming::

    symfluence agent code --direct
    symfluence agent code --cli codex
    symfluence agent model --no-skills

Forward extra flags to the underlying CLI after ``--``::

    symfluence agent code -- --model claude-sonnet-4-6

Sessions without a TTY, or installs without the TUI extra (``pip install
"symfluence[tui]"``), fall back to the direct handoff automatically.

Inspect the setup::

    symfluence agent doctor           # runtimes, keys, per-mode priming, MCP server
    symfluence agent doctor --json    # the same, machine-readable

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
tasks — for both *running* the platform and *extending* it. The packaged skills
and which mode carries them:

==========================  ==============================================  =========
Skill                       Use it for                                      Modes
==========================  ==============================================  =========
``explore-platform``        Discovering available models/datasets/configs   both
``run-workflow-locally``    Running the workflow end to end (or one step)   both
``debug-calibration``       Diagnosing calibration / optimizer problems     both
``add-data-handler``        Adding a forcing / attribute / obs dataset      code
``add-model-handler``       Adding or wiring a hydrological model           code
``add-optimizer``           Adding a calibration/search algorithm           code
==========================  ==============================================  =========

The MCP server
--------------

``symfluence agent mcp`` serves the Model Context Protocol on stdio, exposing
structured tools backed by the live platform:

=======================  ====================================================
Tool                     What it does
=======================  ====================================================
``list_capabilities``    Registry catalogs: models, forcings, optimizers, ...
``validate_config``      Typed validation of a config file
``workflow_status``      Per-step pipeline status for a config
``run_workflow_step``    Run a single workflow step, blocking until it ends
``start_workflow_job``   Start a run/step/resume as a detached background job
``get_job_status``       Poll a background job: state, runtime, log tail
``cancel_job``           Cancel a background job (TERM, then KILL)
``list_jobs``            All recorded background jobs, newest first
``read_run_log``         Tail / grep the newest run log of a config's domain
``list_domains``         The ``domain_*`` dirs under a data root, summarized
``calibration_status``   Iterations, best score, and progress of a calibration
``get_results_summary``  Headline metrics (KGE/NSE/...) and artifact paths
``get_plot_paths``       Figures (png/pdf/svg) produced for an experiment run
``compare_experiments``  All optimization runs of a domain, best score first
``update_config``        Guarded config edit: validated, backed up, in place
=======================  ====================================================

Background jobs survive server (and TUI) restarts: each runs detached under a
wrapper that records its exit code, with its record and log kept in the agent
cache. Long model runs and calibrations should go through
``start_workflow_job`` — ``run_workflow_step`` holds the MCP connection for
the whole step.

Interactive approvals
---------------------

In the native modelling chat, tools outside the modelling allowlist are not
denied outright: Claude Code routes them through a hidden ``approve_action``
permission bridge, and the chat pops an allow/deny modal (``y`` / ``n``). No
reply within the timeout is a denial — permission is never granted by
silence. Outside the chat (one-shot ``agent model "…"``, or any context with
no UI watching), the same tools are hard-denied instead. See ADR-0004 for the
policy.

In the chat, ``ctrl+e`` exports the conversation as Markdown into the working
directory, and each turn's footer line shows its duration, cost, and the
running session total.

The session verbs wire it in automatically where the host CLI supports
per-launch MCP configuration, passing ``--mode`` so the server serves that
mode's tool profile. To register it manually in any MCP-capable tool, add a
stdio server named ``symfluence`` running ``symfluence agent mcp`` — e.g. for
Gemini CLI::

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
in the working directory that contain SYMFLUENCE keys
(``DOMAIN_NAME`` or ``HYDROLOGICAL_MODEL``). Launch from the project root, or
point ``SYMFLUENCE_DEFAULT_CONFIG`` at your config file.

Deprecation
-----------

``symfluence agent launch`` is a deprecated alias for ``symfluence agent code``
and will be removed in a future release. The former ``start``, ``run``,
``list``, and ``skills`` verbs have been removed — ``doctor`` (and
``doctor --json``) covers what ``list`` and ``skills`` reported.

See Also
--------

- :doc:`cli_reference` - CLI command reference
- :doc:`getting_started` - General SYMFLUENCE quickstart
- :doc:`configuration` - Configuration file reference
