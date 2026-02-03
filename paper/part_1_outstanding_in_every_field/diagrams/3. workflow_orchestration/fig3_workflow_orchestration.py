"""
SYMFLUENCE Workflow Orchestration Pipeline (Section 3.1.3).

Linear top-to-bottom layout: six category rows with stages flowing
left-to-right.  Each stage shows its key output artifact.
Enriched with section cross-references, manager annotations,
inter-category data-flow labels, configuration sidebar, and
execution-semantics note.
Verified against src/symfluence/project/workflow_orchestrator.py.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── layout ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 14.4, 10.6

# ── colours ─────────────────────────────────────────────────────────
C_CAT = [
    "#4A7FB5",  # Project Init       (blue)
    "#5BA58B",  # Domain Definition   (green)
    "#C9943A",  # Agnostic Preproc    (amber)
    "#8B6DAF",  # Model-Specific Ops  (purple)
    "#C0392B",  # Optimization        (red)
    "#2E86AB",  # Analysis            (teal)
]
TEXT_DARK  = "#2D2D2D"
TEXT_MID   = "#555555"
TEXT_GREY  = "#999999"
ARROW_CLR  = "#BBBBBB"
OUTPUT_CLR = "#777777"
C_CONFIG   = "#6C7A89"  # config sidebar colour
C_SECTION  = "#3A6EA5"  # section reference colour

# ── stage data: (label, output artifact) ────────────────────────────
# Each category: (display_name, stages, conditional, section_ref, managers)
categories = [
    ("Project\nInitialization", [
        ("Setup Project",             "Directory structure"),
    ], False, None, ["ProjectManager"]),
    ("Domain\nDefinition", [
        ("Create\nPour Point",       "Pour point shapefile"),
        ("Acquire\nAttributes",      "Geospatial rasters"),
        ("Define\nDomain",           "Basin shapefile"),
        ("Discretize\nDomain",       "HRU shapefile"),
    ], False, "\u00a73.4", ["ProjectManager", "DataManager", "DomainManager"]),
    ("Model-Agnostic\nPreprocessing", [
        ("Process\nObservations",    "Streamflow CSV"),
        ("Acquire\nForcings",        "Forcing NetCDF"),
        ("Agnostic\nPreprocessing",  "Basin-averaged forcing"),
    ], False, "\u00a73.3", ["DataManager"]),
    ("Model-Specific\nOperations", [
        ("Preprocess\nModels",       "Model input files"),
        ("Run\nModels",              "Model output NetCDF"),
        ("Postprocess\nResults",     "Standardised NetCDF"),
    ], False, "\u00a73.5", ["ModelManager"]),
    ("Optimization", [
        ("Calibrate\nModel",         "Optimised parameters"),
    ], True, "\u00a73.6", ["OptimizationManager"]),
    ("Analysis", [
        ("Run\nBenchmarking",        "Benchmark scores"),
        ("Decision\nAnalysis",       "Decision comparison"),
        ("Sensitivity\nAnalysis",    "Sensitivity indices"),
    ], True, "\u00a73.7", ["AnalysisManager"]),
]

# ── inter-category data flow labels ─────────────────────────────────
flow_labels = [
    "Project directory",                 # Init → Domain
    "HRU shapefile + basin attributes",  # Domain → Agnostic Preproc
    "Basin-averaged forcing + obs.",     # Agnostic → Model-Specific
    "Standardised model outputs",        # Model-Specific → Optimization
    "Optimised parameters + outputs",    # Optimization → Analysis
]

# ── sizing ──────────────────────────────────────────────────────────
STAGE_W    = 1.55
STAGE_H    = 0.82
STAGE_GAP  = 0.12
CAT_PAD_X  = 0.16
CAT_PAD_Y  = 0.14
ROW_GAP    = 0.50          # slightly larger for flow labels
LABEL_W    = 2.10
LABEL_GAP  = 0.30
MARGIN_T   = 0.55
BAR_W      = 0.055

ROW_H = STAGE_H + 2 * CAT_PAD_Y

# x positions
x_label     = 0.35
x_container = x_label + LABEL_W + LABEL_GAP

# ── config sidebar position ─────────────────────────────────────────
SIDEBAR_X  = 11.55
SIDEBAR_W  = 2.20


def lighten(hex_colour, factor=0.82):
    rgb = [int(hex_colour[i:i+2], 16) / 255 for i in (1, 3, 5)]
    return [1 - factor * (1 - c) for c in rgb]


def draw_group(ax, x_cont, y, cat_name, stages, colour, conditional,
               section_ref, managers):
    """Draw one category row with section ref and manager annotations."""
    n = len(stages)
    inner_w = (2 * CAT_PAD_X + BAR_W + 0.04
               + n * STAGE_W + max(0, n - 1) * STAGE_GAP)

    # ── category label ──────────────────────────────────────────
    lx = x_label + LABEL_W
    ly = y + ROW_H / 2
    ax.text(lx, ly, cat_name, ha="right", va="center",
            fontsize=8.8, fontweight="bold", color=colour,
            linespacing=1.25)

    # conditional tag
    label_lines = cat_name.count("\n") + 1
    base_off = 0.28 if label_lines > 1 else 0.22
    annotation_y = ly - base_off

    if conditional:
        ax.text(lx, annotation_y, "conditional", ha="right", va="center",
                fontsize=6.5, color=TEXT_GREY, fontstyle="italic")
        annotation_y -= 0.16

    # section reference
    if section_ref:
        ax.text(lx, annotation_y, section_ref, ha="right", va="center",
                fontsize=7.0, fontweight="bold", color=C_SECTION)
        annotation_y -= 0.16

    # manager names (stacked, smaller)
    for mgr in managers:
        ax.text(lx, annotation_y, mgr, ha="right", va="center",
                fontsize=5.5, color=TEXT_GREY, fontstyle="italic",
                fontfamily="monospace")
        annotation_y -= 0.12

    # ── container background ────────────────────────────────────
    bg = FancyBboxPatch(
        (x_cont, y), inner_w, ROW_H,
        boxstyle="round,pad=0.045",
        facecolor=lighten(colour), edgecolor=colour,
        linewidth=0.9, alpha=0.45, zorder=1)
    ax.add_patch(bg)

    # accent bar
    bar = FancyBboxPatch(
        (x_cont + 0.035, y + 0.07), BAR_W, ROW_H - 0.14,
        boxstyle="round,pad=0.012",
        facecolor=colour, edgecolor="none", alpha=0.85, zorder=2)
    ax.add_patch(bar)

    # ── stage pills ─────────────────────────────────────────────
    sx = x_cont + CAT_PAD_X + BAR_W + 0.04
    sy = y + CAT_PAD_Y
    for i, (label, output) in enumerate(stages):
        pill = FancyBboxPatch(
            (sx, sy), STAGE_W, STAGE_H,
            boxstyle="round,pad=0.04",
            facecolor="white", edgecolor=colour,
            linewidth=0.65, zorder=3)
        ax.add_patch(pill)

        # stage name (upper portion)
        ax.text(sx + STAGE_W / 2, sy + STAGE_H * 0.58, label,
                ha="center", va="center", fontsize=7.4,
                fontweight="medium", color=TEXT_DARK,
                linespacing=1.15, zorder=4)

        # thin separator line
        sep_y = sy + STAGE_H * 0.30
        ax.plot([sx + 0.12, sx + STAGE_W - 0.12], [sep_y, sep_y],
                color=colour, lw=0.4, alpha=0.45, zorder=4)

        # output artifact (lower portion)
        ax.text(sx + STAGE_W / 2, sy + STAGE_H * 0.14, output,
                ha="center", va="center", fontsize=5.8,
                color=OUTPUT_CLR, fontstyle="italic", zorder=4)

        # dot connector
        if i < len(stages) - 1:
            ax.plot(sx + STAGE_W + STAGE_GAP / 2, sy + STAGE_H / 2,
                    "o", color=colour, markersize=2.2, zorder=4)
        sx += STAGE_W + STAGE_GAP

    return inner_w


# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# title
ax.text(FIG_W / 2 - 0.6, FIG_H - 0.18,
        "SYMFLUENCE Workflow Orchestration Pipeline",
        ha="center", va="center", fontsize=12.5,
        fontweight="bold", color=TEXT_DARK)

# draw rows
y = FIG_H - MARGIN_T - ROW_H
row_ys = []

for idx, (cat_name, stages, conditional, section_ref, managers) in enumerate(categories):
    draw_group(ax, x_container, y, cat_name, stages,
               C_CAT[idx], conditional, section_ref, managers)
    row_ys.append(y)
    y -= ROW_H + ROW_GAP

# ── vertical flow arrows with data-flow labels ─────────────────────
arrow_x = x_container + 0.035 + BAR_W / 2
for i in range(len(row_ys) - 1):
    y_from = row_ys[i] - 0.04
    y_to   = row_ys[i + 1] + ROW_H + 0.04
    ax.annotate(
        "", xy=(arrow_x, y_to), xytext=(arrow_x, y_from),
        arrowprops=dict(arrowstyle="-|>", color=ARROW_CLR,
                        lw=0.9, shrinkA=0, shrinkB=0),
        zorder=5)

    # data-flow label to the right of the arrow
    mid_y = (y_from + y_to) / 2
    ax.text(arrow_x + 0.22, mid_y, flow_labels[i],
            ha="left", va="center", fontsize=5.8,
            color=TEXT_GREY, fontstyle="italic")

# ── configuration sidebar ───────────────────────────────────────────
sidebar_top = row_ys[0] + ROW_H
sidebar_bot = row_ys[-1]
sidebar_h   = sidebar_top - sidebar_bot

# sidebar background
sb_bg = FancyBboxPatch(
    (SIDEBAR_X, sidebar_bot), SIDEBAR_W, sidebar_h,
    boxstyle="round,pad=0.06",
    facecolor=lighten(C_CONFIG, factor=0.25), edgecolor=C_CONFIG,
    linewidth=0.8, alpha=0.35, zorder=1)
ax.add_patch(sb_bg)

# sidebar accent bar (top)
sb_bar = FancyBboxPatch(
    (SIDEBAR_X + 0.06, sidebar_top - 0.08), SIDEBAR_W - 0.12, 0.04,
    boxstyle="round,pad=0.008",
    facecolor=C_CONFIG, edgecolor="none", alpha=0.7, zorder=2)
ax.add_patch(sb_bar)

# sidebar title
ax.text(SIDEBAR_X + SIDEBAR_W / 2, sidebar_top - 0.20,
        "Configuration System", ha="center", va="center",
        fontsize=8.2, fontweight="bold", color=C_CONFIG)
ax.text(SIDEBAR_X + SIDEBAR_W / 2, sidebar_top - 0.38,
        "\u00a73.2", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=C_SECTION)

# sidebar content items
sidebar_items = [
    ("Declarative YAML", "\u00a73.2.1"),
    ("Validation & Type Safety", "\u00a73.2.2"),
    ("Provenance & Logging", "\u00a73.2.3"),
]
pill_w = SIDEBAR_W - 0.30
pill_h = 0.34
pill_gap = 0.12
pill_x = SIDEBAR_X + (SIDEBAR_W - pill_w) / 2  # horizontally centred

# vertically centre the three pills between title area and "feeds" label
n_items = len(sidebar_items)
block_h = n_items * pill_h + (n_items - 1) * pill_gap
avail_top = sidebar_top - 0.48   # below title + §3.2
avail_bot = sidebar_bot + 0.30   # above "feeds all stages"
item_y = (avail_top + avail_bot) / 2 + block_h / 2 - pill_h

for label, ref in sidebar_items:
    pill = FancyBboxPatch(
        (pill_x, item_y), pill_w, pill_h,
        boxstyle="round,pad=0.03",
        facecolor="white", edgecolor=C_CONFIG,
        linewidth=0.5, alpha=0.85, zorder=3)
    ax.add_patch(pill)
    ax.text(pill_x + pill_w / 2, item_y + pill_h * 0.62, label,
            ha="center", va="center", fontsize=6.5,
            fontweight="medium", color=TEXT_DARK, zorder=4)
    ax.text(pill_x + pill_w / 2, item_y + pill_h * 0.25, ref,
            ha="center", va="center", fontsize=6.0,
            fontweight="bold", color=C_SECTION, zorder=4)
    item_y -= pill_h + pill_gap

# "feeds all stages" annotation with arrow
feed_y = item_y + 0.05
ax.text(SIDEBAR_X + SIDEBAR_W / 2, feed_y,
        "feeds all stages", ha="center", va="center",
        fontsize=5.8, color=TEXT_GREY, fontstyle="italic")

# horizontal dashed arrows from sidebar to pipeline
for ry in row_ys:
    mid_row_y = ry + ROW_H / 2
    # find the rightmost stage pill end (approximate)
    n_stages = len(categories[row_ys.index(ry)][1])
    pills_end_x = (x_container + CAT_PAD_X + BAR_W + 0.04
                    + n_stages * STAGE_W + max(0, n_stages - 1) * STAGE_GAP + 0.15)
    ax.annotate(
        "", xy=(pills_end_x, mid_row_y),
        xytext=(SIDEBAR_X - 0.05, mid_row_y),
        arrowprops=dict(arrowstyle="-|>", color=C_CONFIG,
                        lw=0.45, linestyle=(0, (4, 4)),
                        shrinkA=0, shrinkB=0, alpha=0.45),
        zorder=0)

# ── execution semantics footer ──────────────────────────────────────
footer_y = row_ys[-1] - 0.55
footer_x = x_container

# footer box
footer_w = SIDEBAR_X + SIDEBAR_W - footer_x
footer_h = 0.42
fb = FancyBboxPatch(
    (footer_x, footer_y), footer_w, footer_h,
    boxstyle="round,pad=0.04",
    facecolor="#F5F5F5", edgecolor="#CCCCCC",
    linewidth=0.6, zorder=1)
ax.add_patch(fb)

ax.text(footer_x + 0.15, footer_y + footer_h / 2,
        "Execution modes:",
        ha="left", va="center", fontsize=6.5,
        fontweight="bold", color=TEXT_MID)

modes_text = ("Sequential  \u2022  Selective (per-stage)  \u2022  "
              "Forced re-execution   |   "
              "Completion tracking via marker files   |   "
              "Fail-fast / continue-on-error")
ax.text(footer_x + 1.85, footer_y + footer_h / 2,
        modes_text,
        ha="left", va="center", fontsize=5.8,
        color=TEXT_GREY)

# ── save ────────────────────────────────────────────────────────────
out = "/Users/darrieythorsson/compHydro/Papers/Article 2 - SYMFLUENCE/diagrams/3. workflow_orchestration"
for fmt in ("pdf", "png"):
    fig.savefig(f"{out}/fig3_workflow_orchestration.{fmt}",
                dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.15)
    print(f"Saved fig3_workflow_orchestration.{fmt}")
plt.close(fig)
