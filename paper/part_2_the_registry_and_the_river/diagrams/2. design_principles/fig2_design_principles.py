"""
SYMFLUENCE Design Principles figure (Section 3.1.1).

Three-column flow: Technical Barrier → Architectural Principle → SYMFLUENCE Mechanism.
Maps the five barriers from Section 2 to principles and concrete implementations.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── layout ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 13.0, 6.6
CORNER_R = 0.04

# column widths and positions
COL_W = [2.6, 2.6, 3.8]
COL_GAP = 0.70
total_w = sum(COL_W) + 2 * COL_GAP
x_start = (FIG_W - total_w) / 2
COL_X = []
_x = x_start
for w in COL_W:
    COL_X.append(_x)
    _x += w + COL_GAP

# row layout
N_ROWS = 5
ROW_H = 0.78
ROW_GAP = 0.26
ROWS_START = 0.50
ROW_Y = [ROWS_START + i * (ROW_H + ROW_GAP) for i in range(N_ROWS)]
ROW_Y.reverse()

# ── colours ─────────────────────────────────────────────────────────
C_BARRIER   = "#B83230"
C_PRINCIPLE = "#3A6FA5"
C_MECHANISM = "#438B73"
C_BG_STRIPE = "#F7F8FA"
TEXT_WHITE   = "#FFFFFF"
TEXT_DARK    = "#1E1E1E"
TEXT_MID     = "#4A4A4A"
ARROW_CLR    = "#9CA3AF"

# ── data ────────────────────────────────────────────────────────────
rows = [
    ("Glue-code\nfragmentation",
     "Declarative\nspecification",
     "Single YAML config as\nprimary experimental artifact"),
    ("Reproducibility\nfailure",
     "Automated\nprovenance",
     "Runtime metadata capture with\nunified logging and checksums"),
    ("Workflow\nbreakage",
     "End-to-end\norchestration",
     "Staged dependency graph with\nprecondition enforcement"),
    ("HPC\nbottleneck",
     "Native\nscalability",
     "Multi-level parallelism\nabstracted via configuration"),
    ("Platform\nlock-in",
     "Cross-platform\nportability",
     "Package management + CI\nacross OS & Python versions"),
]

# ── helpers ─────────────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, colour, text, fontsize=8.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={CORNER_R}",
        facecolor=colour, edgecolor="none", linewidth=0,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="semibold", color=TEXT_WHITE, zorder=3,
        linespacing=1.35,
    )


def draw_arrow(ax, x_from, x_to, y_mid):
    """Clean simple arrow using annotate."""
    ax.annotate(
        "", xy=(x_to, y_mid), xytext=(x_from, y_mid),
        arrowprops=dict(
            arrowstyle="->,head_width=0.25,head_length=0.18",
            color=ARROW_CLR, lw=1.6,
            shrinkA=4, shrinkB=4,
        ),
        zorder=1,
    )


# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# alternating row stripes
for r in range(N_ROWS):
    if r % 2 == 0:
        stripe = FancyBboxPatch(
            (x_start - 0.30, ROW_Y[r] - 0.08),
            total_w + 0.60, ROW_H + 0.16,
            boxstyle="round,pad=0.06",
            facecolor=C_BG_STRIPE, edgecolor="none", zorder=0,
        )
        ax.add_patch(stripe)

# column headers with underline accent
header_y = ROW_Y[0] + ROW_H + 0.28
headers = ["Technical barrier", "Architectural principle", "SYMFLUENCE mechanism"]
header_colours = [C_BARRIER, C_PRINCIPLE, C_MECHANISM]
for i, (hdr, hc) in enumerate(zip(headers, header_colours)):
    cx = COL_X[i] + COL_W[i] / 2
    ax.text(
        cx, header_y, hdr,
        ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=hc, zorder=3,
    )
    ul_w = len(hdr) * 0.065
    ax.plot(
        [cx - ul_w, cx + ul_w],
        [header_y - 0.17, header_y - 0.17],
        color=hc, lw=1.8, solid_capstyle="round", alpha=0.45, zorder=3,
    )

# draw rows
colours = [C_BARRIER, C_PRINCIPLE, C_MECHANISM]
sizes = [8.8, 9.2, 8.2]

for r, (barrier, principle, mechanism) in enumerate(rows):
    y = ROW_Y[r]
    texts = [barrier, principle, mechanism]

    for c in range(3):
        draw_box(ax, COL_X[c], y, COL_W[c], ROW_H,
                 colours[c], texts[c], fontsize=sizes[c])

    for c in range(2):
        draw_arrow(ax, COL_X[c] + COL_W[c], COL_X[c + 1], y + ROW_H / 2)

# P1–P5 row labels
for r in range(N_ROWS):
    ax.text(
        COL_X[0] - 0.30, ROW_Y[r] + ROW_H / 2, f"P{r+1}",
        ha="center", va="center", fontsize=8,
        fontweight="bold", color="#BBBBBB",
        fontstyle="italic", zorder=3,
    )

# title + subtitle (tightened gap)
title_y = header_y + 0.42
ax.text(
    FIG_W / 2, title_y,
    "SYMFLUENCE Design Principles",
    ha="center", va="center", fontsize=14,
    fontweight="bold", color=TEXT_DARK,
)

# ── save ────────────────────────────────────────────────────────────
out = "/Users/darrieythorsson/compHydro/Papers/Article 2 - SYMFLUENCE/diagrams/2. design_principles"
for fmt in ("pdf", "png"):
    fig.savefig(f"{out}/fig2_design_principles.{fmt}",
                dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.2)
    print(f"Saved fig2_design_principles.{fmt}")
plt.close(fig)
