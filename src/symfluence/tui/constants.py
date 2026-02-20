"""
Constants for the SYMFLUENCE TUI.

Step lists, key bindings, color tokens, and status values.
"""

from symfluence.workflow_steps import WORKFLOW_STEP_ITEMS

# Workflow step definitions (matches WorkflowOrchestrator.define_workflow_steps)
WORKFLOW_STEPS = list(WORKFLOW_STEP_ITEMS)

# Run status values
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_RUNNING = "running"
STATUS_PARTIAL = "partial"
STATUS_PENDING = "pending"

# Status display icons (Unicode)
STATUS_ICONS = {
    STATUS_COMPLETED: "\u2714",  # check mark
    STATUS_FAILED: "\u2718",     # cross mark
    STATUS_RUNNING: "\u25b6",    # play triangle
    STATUS_PARTIAL: "\u25cb",    # circle
    STATUS_PENDING: "\u2500",    # dash
}

# Color tokens for status
STATUS_COLORS = {
    STATUS_COMPLETED: "green",
    STATUS_FAILED: "red",
    STATUS_RUNNING: "yellow",
    STATUS_PARTIAL: "yellow",
    STATUS_PENDING: "dim",
}

# Metric quality thresholds
METRIC_THRESHOLDS = {
    "KGE": {"good": 0.5, "poor": 0.0},
    "NSE": {"good": 0.5, "poor": 0.0},
    "RMSE": {"good": None, "poor": None},  # relative
    "PBIAS": {"good": 10.0, "poor": 25.0},
    "r": {"good": 0.7, "poor": 0.5},
}

# Sparkline characters (increasing block heights)
SPARKLINE_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
