AI Agent Guide
==============

``symfluence agent launch`` hands off to an installed coding-agent CLI — Claude
Code (``claude``), OpenAI Codex (``codex``), Gemini CLI (``gemini``), or another —
primed with the SYMFLUENCE *skills*. SYMFLUENCE does not ship its own language
model: it detects whichever agent CLI you have installed, picks up whichever API
key is in your environment, exposes the SYMFLUENCE domain skills to it, and replaces
itself with that agent so it drives your project directly.

This gives you a full, modern coding agent (with its own editing, search, and
git tooling) that already knows how to work with SYMFLUENCE.

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

Interactive session (run from your project directory)::

    symfluence agent launch

One-shot prompt (runs once and exits — useful in scripts)::

    symfluence agent launch "add an MSWEP forcing data handler"

Forward extra flags to the underlying CLI after ``--``::

    symfluence agent launch -- --model claude-sonnet-4-6

How it works
------------

1. **Detect a CLI.** SYMFLUENCE looks for ``claude``, then ``codex``, then
   ``gemini`` on your ``PATH`` (first match wins). Override with
   ``SYMFLUENCE_AGENT_CLI=<command>``.
2. **Expose the skills.** The packaged SYMFLUENCE skills are made available to the
   CLI — via Claude Code's native ``.claude/skills/`` discovery (using
   ``--add-dir``, without touching your project's ``.claude/``), or via an
   ``AGENTS.md`` file for Codex/Gemini and other tools. Set
   ``SYMFLUENCE_NO_SKILLS=1`` to skip this.
3. **Hand off.** SYMFLUENCE replaces its own process with the CLI, which then owns
   the terminal directly.

Skills
------

Skills are concise domain guides the agent consults when working on SYMFLUENCE
tasks. The packaged skills are:

==========================  =================================================
Skill                       Use it for
==========================  =================================================
``add-data-handler``        Adding a forcing / attribute / observation dataset
``add-model-handler``       Adding or wiring a hydrological model
``debug-calibration``       Diagnosing calibration / optimizer problems
``run-workflow-locally``    Running the workflow end to end (or a single step)
==========================  =================================================

Environment variables
---------------------

=========================  ===============================================================
Variable                   Effect
=========================  ===============================================================
``SYMFLUENCE_AGENT_CLI``   Force a specific CLI (e.g. ``codex``) instead of auto-detection
``SYMFLUENCE_NO_SKILLS``   Skip skill materialization (manage your own ``.claude/``)
=========================  ===============================================================

Troubleshooting
---------------

**"No coding-agent CLI found on PATH."** Install one of the CLIs above and set its
API key, or point ``SYMFLUENCE_AGENT_CLI`` at an installed command.

**The agent doesn't seem to know about SYMFLUENCE.** Make sure you launched from
your project directory and that ``SYMFLUENCE_NO_SKILLS`` is not set. For Codex/Gemini,
check that an ``AGENTS.md`` was written (or already exists) in the working directory.

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
