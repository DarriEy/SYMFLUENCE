# ADR-0004: The agent interface keeps a human in the loop

- **Status:** Accepted (amended 2026-07-17; original 2026-06-05)
- **Date:** 2026-06-05, amended 2026-07-17
- **Related:** `docs/security.md` (threat model), `docs/source/agent_guide.rst`

## Context

SYMFLUENCE's agent surface has evolved twice since this ADR was first written:

1. The original in-house LLM agent (with its own `pr_manager.py` and an
   `auto_push` flag) was **removed** and replaced by a thin launcher that hands
   off to an installed coding-agent CLI (Claude Code, Codex, Gemini, ...).
2. The launcher grew into a two-mode interface: `agent code` (a primed session
   in the host CLI) and `agent model` (a modelling session, optionally driven
   headlessly behind the native chat screen in the TUI).

The threat that motivated this ADR is unchanged: an agent that processes
repository or issue content is exposed to prompt injection, so any path where
the agent can take an outward-facing or destructive action without a human
verdict is an exploitable surface. What changed is *where* that human
verdict lives.

## Decision

The agent interface stays **human-in-the-loop by default**, enforced at the
layer that actually executes actions:

- **Coding mode** delegates permissioning to the host CLI's own interactive
  permission system — a human sits in the session and confirms tool use,
  including anything outward-facing (pushes, PRs), exactly as they would in a
  bare Claude Code/Codex/Gemini session. SYMFLUENCE adds context (skills,
  identity, MCP tools) but no automatic execution paths.
- **Modelling mode, headless (native chat)** has no host permission UI, so
  SYMFLUENCE imposes its own: turns run under the modelling profile's tool
  allowlist (structured `mcp__symfluence__*` tools, read-only file access,
  `symfluence` CLI invocations). Anything outside the allowlist routes
  through the `approve_action` permission bridge
  (`--permission-prompt-tool`): the MCP server blocks, the TUI pops an
  explicit allow/deny modal, and **no reply is a denial** — permission is
  never granted by silence or timeout.
- **Modelling mode, non-interactive** (one-shot prompts, or any context with
  no UI watching the approvals directory) does not get the bridge: the
  profile's disallowed tools (file `Write`/`Edit`) are hard-denied instead.
- The only write path SYMFLUENCE itself offers the modelling agent is the
  `update_config` MCP tool, which is deliberately narrow: the user's
  experiment YAML only, typed-schema validation before writing, and a backup
  of the original next to itself.

## Consequences

- The default-safe posture survives the architecture change: an injected
  instruction cannot cause an unattended outward-facing action, because every
  execution path either has a human at the permission prompt (coding mode,
  chat approvals) or a hard denial (non-interactive modelling).
- Relaxations are visible and auditable: widening the modelling allowlist or
  the `approve_action` timeout semantics is a reviewable code change to
  `agent/modes.py` / `agent/approvals.py`, not a runtime flag.
- This remains a policy commitment at 1.0: flipping any of these defaults to
  autonomous-by-default requires a security review and a superseding ADR.

## References

- `src/symfluence/agent/modes.py` — per-mode tool allowlists and house rules
- `src/symfluence/agent/approvals.py` — the permission bridge (deny on timeout)
- `src/symfluence/agent/headless.py` — `interactive_approvals` gating
- `src/symfluence/agent/inspection.py` — `update_config` guardrails
- `docs/security.md` — agent trust boundary
