# Section 3.1.3 — Revised text

> **Notes for authors:**
> - Figure number (currently "Figure X") should be updated to match final numbering.
> - The original text listed five categories; the codebase defines six (Optimization was missing as a separate category). Fixed below.
> - The original Dependency Management paragraph contained a logging/environment-capture tangent ("For logging environment capture at runtime…"). That material belongs in Section 3.2.3 (Provenance and Logging) and has been removed here.
> - The figure's per-stage output artifacts are now referenced in the dependency discussion, tying the visual directly to the prose.
> - Execution Semantics, Completion Tracking, and Error Handling are tightened but retain all substantive content.

---

## 3.1.3 Workflow Orchestration

The workflow orchestrator transforms declarative configurations into executable sequences, managing the complete modeling lifecycle from project initialization through analysis.

**Pipeline Structure.** The orchestrator defines fifteen stages grouped into six categories: project initialization, domain definition, model-agnostic preprocessing, model-specific operations, optimization, and analysis (Figure X). Each stage produces a well-defined artifact—shapefiles, NetCDF datasets, processed CSV files, or optimised parameter sets—that serves as a precondition for downstream stages. Each stage corresponds to a method on the `WorkflowOrchestrator` class, which delegates to the appropriate domain manager, maintaining separation between orchestration logic (what to do and when) and domain logic (how to do it). The final two categories (optimization and analysis) are conditional: they execute only when the corresponding capabilities are enabled in the configuration.

**Dependency Management.** Stages encode explicit dependencies that the orchestrator enforces at runtime. Before executing any stage, the orchestrator verifies that all prerequisite stages have completed successfully—for example, domain discretization cannot proceed until the basin shapefile produced by domain definition exists, and model execution requires that preprocessing has generated the necessary input files (Figure X). Dependency violations produce informative errors identifying missing prerequisites rather than cryptic failures deep in execution. This explicit dependency tracking replaces the implicit ordering of procedural scripts, where execution sequence is encoded in line numbers rather than declared relationships.

**Execution Semantics.** The orchestrator supports three execution modes. *Sequential execution* processes stages in dependency order, blocking until each completes, ensuring deterministic ordering for interactive use. *Selective execution* allows users to specify individual stages or stage ranges; the orchestrator verifies that prerequisites have been satisfied before proceeding, supporting iterative workflows where users re-execute specific stages without reprocessing the entire pipeline. *Forced re-execution* overrides completion tracking, re-running stages regardless of prior state—essential when upstream data or configurations change in ways not captured by the completion system.

**Completion Tracking.** The orchestrator maintains persistent state recording which stages have completed successfully. Upon completion, a marker file is written encoding the stage name, timestamp, configuration hash, and framework version. Subsequent executions consult these markers to determine which stages can be skipped. Configuration hashing enables automatic invalidation: if parameters relevant to a stage change between runs, the marker is considered stale and the stage re-executes, balancing efficiency against correctness.

**Error Handling and Recovery.** Stage execution is wrapped in error handling that captures failures, logs diagnostics, and records partial state. *Fail-fast mode* (default) halts on first error, preserving system state for debugging. *Continue-on-error mode* logs failures but proceeds to subsequent stages where dependencies permit, useful for batch processing where partial results are preferable to complete failure. The orchestrator's resumption capability allows recovery from transient failures: after addressing the underlying issue, users re-invoke the workflow, and completed stages are skipped automatically.
