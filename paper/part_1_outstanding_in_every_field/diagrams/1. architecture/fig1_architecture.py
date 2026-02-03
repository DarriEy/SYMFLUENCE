"""
SYMFLUENCE four-tier layered architecture diagram (Figure 1 replacement).

Verified against the codebase at /Users/darrieythorsson/compHydro/code/SYMFLUENCE.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── layout ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 10.0, 6.8
STACK_L, STACK_R = 1.95, 7.95
LAYER_GAP = 0.42
CORNER_R = 0.04

# colours
C_UI   = "#4A7FB5"
C_ORCH = "#5BA58B"
C_MGR  = "#C9943A"
C_CORE = "#8B6DAF"
C_CFG  = "#F2F2F2"
TEXT_DARK  = "#2D2D2D"
TEXT_WHITE = "#FFFFFF"
TEXT_GREY  = "#555555"
ARROW_CLR  = "#999999"
BRACKET_CLR = "#AAAAAA"

# ── helpers ─────────────────────────────────────────────────────────

def tier_box(ax, y_bot, height, colour, label, sublabels=None,
             label_size=10.5, sub_size=8, sub_rows=1):
    box = FancyBboxPatch(
        (STACK_L, y_bot), STACK_R - STACK_L, height,
        boxstyle=f"round,pad={CORNER_R}",
        facecolor=colour, edgecolor="white", linewidth=2,
        zorder=2,
    )
    ax.add_patch(box)

    label_y = y_bot + height - 0.20 if sublabels else y_bot + height / 2
    ax.text(
        (STACK_L + STACK_R) / 2, label_y, label,
        ha="center", va="center", fontsize=label_size,
        fontweight="bold", color=TEXT_WHITE, zorder=3,
    )

    if sublabels:
        if sub_rows == 1:
            _draw_sub_row(ax, sublabels, y_bot + 0.11, 0.32, sub_size)
        else:
            mid = (len(sublabels) + 1) // 2
            _draw_sub_row(ax, sublabels[:mid], y_bot + 0.48, 0.30, sub_size)
            _draw_sub_row(ax, sublabels[mid:], y_bot + 0.11, 0.30, sub_size)


def _draw_sub_row(ax, labels, sub_y, sub_h, sub_size):
    n = len(labels)
    pad = 0.15
    usable = (STACK_R - STACK_L) - 2 * pad
    gap = 0.07
    box_w = (usable - (n - 1) * gap) / n
    start_x = STACK_L + pad
    for i, sl in enumerate(labels):
        sx = start_x + i * (box_w + gap)
        sub_box = FancyBboxPatch(
            (sx, sub_y), box_w, sub_h,
            boxstyle="round,pad=0.02",
            facecolor="white", edgecolor="white",
            linewidth=0.8, alpha=0.88, zorder=3,
        )
        ax.add_patch(sub_box)
        ax.text(
            sx + box_w / 2, sub_y + sub_h / 2, sl,
            ha="center", va="center", fontsize=sub_size,
            color=TEXT_DARK, zorder=4,
        )


def tier_arrow(ax, y_top_of_lower, y_bot_of_upper):
    mid_x = (STACK_L + STACK_R) / 2
    margin = 0.06
    arrow = FancyArrowPatch(
        (mid_x, y_top_of_lower + margin),
        (mid_x, y_bot_of_upper - margin),
        arrowstyle="<->", color=ARROW_CLR, lw=1.3,
        zorder=5, mutation_scale=11,
    )
    ax.add_patch(arrow)


def bracket(ax, y_top, y_bot, labels, x_bracket, x_text):
    tick_w = 0.10

    ax.plot([x_bracket, x_bracket], [y_bot, y_top],
            color=BRACKET_CLR, lw=1.0, zorder=1, clip_on=False)
    ax.plot([x_bracket, x_bracket + tick_w], [y_top, y_top],
            color=BRACKET_CLR, lw=1.0, zorder=1, clip_on=False)
    ax.plot([x_bracket, x_bracket + tick_w], [y_bot, y_bot],
            color=BRACKET_CLR, lw=1.0, zorder=1, clip_on=False)

    if len(labels) == 1:
        positions = [(y_top + y_bot) / 2]
    else:
        inset = (y_top - y_bot) * 0.12
        span = y_top - y_bot - 2 * inset
        positions = [y_top - inset - i * span / (len(labels) - 1)
                     for i in range(len(labels))]

    for lbl, yy in zip(labels, positions):
        ax.text(x_text, yy, lbl,
                ha="right", va="center", fontsize=7.5, color=TEXT_GREY,
                fontstyle="italic", zorder=2, clip_on=False)


# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# tier heights
h_ui, h_orch, h_mgr, h_core = 0.82, 0.82, 1.18, 0.82

# y positions (bottom-up) — tight bottom margin
y_core = 0.30
y_mgr  = y_core + h_core + LAYER_GAP
y_orch = y_mgr  + h_mgr  + LAYER_GAP
y_ui   = y_orch + h_orch + LAYER_GAP

# ── draw tiers ──────────────────────────────────────────────────────
tier_box(ax, y_ui, h_ui, C_UI, "User Interface Layer",
         sublabels=["Python API", "CLI", "AI Assistant"])

tier_box(ax, y_orch, h_orch, C_ORCH, "Workflow Orchestration Layer",
         sublabels=["Orchestrator", "Step Sequencing", "State Tracking"])

tier_box(ax, y_mgr, h_mgr, C_MGR, "Manager Layer",
         sublabels=["Project\nManager", "Data\nManager", "Domain\nManager",
                     "Model\nManager",
                     "Optimization\nManager", "Analysis\nManager", "Reporting\nManager"],
         sub_size=7, sub_rows=2)

tier_box(ax, y_core, h_core, C_CORE, "Core Infrastructure Layer",
         sublabels=["Configuration", "Logging", "Path Resolver",
                     "Validation", "Profiling"],
         sub_size=7.5)

# inter-tier arrows
tier_arrow(ax, y_orch + h_orch, y_ui)
tier_arrow(ax, y_mgr + h_mgr, y_orch)
tier_arrow(ax, y_core + h_core, y_mgr)

# ── YAML config sidebar (right) ─────────────────────────────────────
cfg_x = STACK_R + 0.35
cfg_w = 1.20
cfg_bot = y_core - 0.06
cfg_top = y_ui + h_ui + 0.06
cfg_h = cfg_top - cfg_bot

cfg_box = FancyBboxPatch(
    (cfg_x, cfg_bot), cfg_w, cfg_h,
    boxstyle="round,pad=0.06",
    facecolor=C_CFG, edgecolor="#BBBBBB", linewidth=1.2,
    linestyle=(0, (5, 3)), zorder=1,
)
ax.add_patch(cfg_box)
ax.text(
    cfg_x + cfg_w / 2, cfg_bot + cfg_h / 2,
    "YAML\nConfiguration\nFile",
    ha="center", va="center", fontsize=8.5,
    color=TEXT_GREY, fontweight="bold", zorder=2,
    linespacing=1.5,
)

for y_t, h_t in [(y_ui, h_ui), (y_orch, h_orch),
                  (y_mgr, h_mgr), (y_core, h_core)]:
    y_mid = y_t + h_t / 2
    ax.annotate(
        "", xy=(STACK_R + 0.03, y_mid), xytext=(cfg_x - 0.03, y_mid),
        arrowprops=dict(arrowstyle="->", color="#BBBBBB", lw=0.9,
                        linestyle="--"),
        zorder=1,
    )

# ── pattern brackets (left) ─────────────────────────────────────────
bx = STACK_L - 0.28
tx = STACK_L - 0.40

bracket(ax,
        y_top=y_mgr + h_mgr - 0.06,
        y_bot=y_mgr + 0.06,
        labels=["Facade Pattern", "Registry Pattern", "Lazy Initialization"],
        x_bracket=bx, x_text=tx)

bracket(ax,
        y_top=y_core + h_core - 0.10,
        y_bot=y_core + 0.10,
        labels=["Mixin Pattern"],
        x_bracket=bx, x_text=tx)

# ── title ───────────────────────────────────────────────────────────
ax.text(
    (STACK_L + STACK_R) / 2, y_ui + h_ui + 0.25,
    "SYMFLUENCE System Architecture",
    ha="center", va="center", fontsize=13,
    fontweight="bold", color=TEXT_DARK,
)

# ── save ────────────────────────────────────────────────────────────
out = "/Users/darrieythorsson/compHydro/Papers/Article 2 - SYMFLUENCE/diagrams/architecture"
for fmt in ("pdf", "png"):
    fig.savefig(f"{out}/fig1_architecture.{fmt}",
                dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.1)
    print(f"Saved fig1_architecture.{fmt}")
plt.close(fig)
