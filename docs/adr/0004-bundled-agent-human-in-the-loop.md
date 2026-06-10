# ADR-0004: The bundled AI agent does not push without a human

- **Status:** Accepted
- **Date:** 2026-06-05
- **Related:** `docs/security.md` (threat model)

## Context

SYMFLUENCE bundles an AI agent (`agent/`) that can create commits and open pull
requests. Its push-and-PR path originally defaulted to acting automatically.
That default is an exploitable surface: an agent that processes repository or
issue content is exposed to prompt injection, and under that threat an injected
instruction could trigger a push to a remote with no human in the loop.
Switching to a human-confirmed default costs essentially nothing in usability.

## Decision

The bundled agent is **human-in-the-loop by default.** Pushing to a remote and
opening a pull request require an explicit opt-in from the caller; they do not
happen automatically.

This is implemented in `agent/pr_manager.py`: the push/PR entry point takes
`auto_push: bool = False`. Autonomous push is available only when a caller
explicitly passes `auto_push=True`, which is a deliberate, auditable choice
rather than the default behavior.

## Consequences

- The default-safe posture closes the highest-likelihood prompt-injection
  path: an injected instruction cannot cause an unattended push because the
  default does not push.
- Workflows that genuinely want autonomous PR creation (e.g. a trusted CI
  context) opt in explicitly at the call site, where the decision is visible in
  code review.
- This is a policy commitment, not only a current default: the human-in-the-loop
  default for outward-facing actions (push, PR) is the intended behavior at 1.0
  and should not be flipped to autonomous-by-default without a corresponding
  security review (and a superseding ADR).

## References

- `agent/pr_manager.py` — `auto_push: bool = False`
- `docs/security.md` — agent trust boundary
