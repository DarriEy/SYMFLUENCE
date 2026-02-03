"""
SYMFLUENCE minimal configuration visualization (Figure 4).

Shows a minimal YAML configuration example alongside the hierarchical
section structure, illustrating the "declarative experiment" concept
described in Section 3.2.

Verified against the codebase at /Users/darrieythorsson/compHydro/code/SYMFLUENCE.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── layout ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 10.0, 7.8
CORNER_R = 0.04

# colours — consistent with other figures
C_SYSTEM = "#4A7FB5"
C_DOMAIN = "#5BA58B"
C_FORCING = "#B07D1A"
C_MODEL  = "#8B6DAF"
C_OPT    = "#D46A6A"
C_EVAL   = "#5B8FB9"
C_YAML_BG = "#FAFAFA"
C_YAML_BORDER = "#CCCCCC"
C_HEADER_BG = "#2D2D2D"

TEXT_DARK  = "#2D2D2D"
TEXT_WHITE = "#FFFFFF"
TEXT_GREY  = "#555555"
TEXT_LIGHT = "#888888"
ARROW_CLR  = "#AAAAAA"

SECTION_COLOURS = [C_SYSTEM, C_DOMAIN, C_FORCING, C_MODEL, C_OPT, C_EVAL]

# ── monospace YAML content ──────────────────────────────────────────
# Each entry: (section_key, [(line, is_required), ...], colour_index)
# Required parameters match the 10 in root.py required_flat
YAML_SECTIONS = [
    ("system:", [
        ("  data_dir:          /project/data",        True),
        ("  code_dir:          /project/SYMFLUENCE",  True),
    ], 0),
    ("domain:", [
        ("  name:              Bow_at_Banff",         True),
        ("  experiment_id:     Bow_lumped_ERA5",      True),
        ("  definition_method: lumped",               True),
        ("  discretization:    elevation",            True),
        ("  pour_point_coords: 51.17/-115.57",        False),
        ("  time_start:        2004-01-01",           True),
        ("  time_end:          2007-12-31",           True),
    ], 1),
    ("forcing:", [
        ("  dataset:           ERA5",                 True),
        ("  time_step_size:    3600",                 False),
        ("  pet_method:        oudin",                False),
    ], 2),
    ("model:", [
        ("  hydrological_model: SUMMA",               True),
        ("  routing_model:      mizuRoute",           False),
        ("  calibrate:  [tempCritRain,",              False),
        ("               k_soil, theta_sat]",         False),
    ], 3),
    ("optimization:", [
        ("  algorithm:  DDS",                         False),
        ("  metric:     KGE",                         False),
        ("  iterations: 1000",                        False),
    ], 4),
    ("evaluation:", [
        ("  targets:    [streamflow]",                False),
        ("  streamflow:",                             False),
        ("    station_id: 05BB001",                   False),
    ], 5),
]

# ── section summary cards (right side) ──────────────────────────────
SECTION_CARDS = [
    ("System",       "Paths, parallelism,\nlogging, execution"),
    ("Domain",       "Spatial extent, time\nperiod, discretization"),
    ("Forcing",      "Meteorological data\nsource & processing"),
    ("Model",        "Hydrological & routing\nmodel selection"),
    ("Optimization", "Calibration algorithm\n& objective function"),
    ("Evaluation",   "Observation data &\nperformance metrics"),
]

# ── pre-compute YAML content height to size the panel ─────────────
line_h = 0.195
sec_gap = 0.10
hdr_h = 0.38
content_pad_top = 0.20
content_pad_bot = 0.25

n_lines = sum(1 + len(lines) for _, lines, _ in YAML_SECTIONS)
n_gaps = len(YAML_SECTIONS) - 1
content_h = n_lines * line_h + n_gaps * sec_gap

yaml_l, yaml_r = 0.6, 5.5
yaml_w = yaml_r - yaml_l
yaml_top = FIG_H - 0.90
yaml_bot = yaml_top - hdr_h - content_pad_top - content_h - content_pad_bot
yaml_h = yaml_top - yaml_bot

# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── title ───────────────────────────────────────────────────────────
ax.text(
    FIG_W / 2, FIG_H - 0.25,
    "SYMFLUENCE Declarative Configuration",
    ha="center", va="center", fontsize=13,
    fontweight="bold", color=TEXT_DARK,
)
ax.text(
    FIG_W / 2, FIG_H - 0.60,
    "A single YAML file defines the complete experiment — 8 sections, 450+ available parameters",
    ha="center", va="center", fontsize=8.5,
    color=TEXT_LIGHT, fontstyle="italic",
)

# ── YAML panel (left) ──────────────────────────────────────────────
# background
yaml_bg = FancyBboxPatch(
    (yaml_l, yaml_bot), yaml_w, yaml_h,
    boxstyle=f"round,pad={CORNER_R}",
    facecolor=C_YAML_BG, edgecolor=C_YAML_BORDER,
    linewidth=1.5, zorder=1,
)
ax.add_patch(yaml_bg)

# header bar
hdr = FancyBboxPatch(
    (yaml_l, yaml_top - hdr_h), yaml_w, hdr_h,
    boxstyle=f"round,pad={CORNER_R}",
    facecolor=C_HEADER_BG, edgecolor=C_HEADER_BG,
    linewidth=0, zorder=2,
)
ax.add_patch(hdr)
# cover bottom rounding of header
ax.add_patch(plt.Rectangle(
    (yaml_l, yaml_top - hdr_h), yaml_w, 0.08,
    facecolor=C_HEADER_BG, edgecolor="none", zorder=2,
))

ax.text(
    yaml_l + 0.25, yaml_top - hdr_h / 2,
    "config.yaml",
    ha="left", va="center", fontsize=9,
    fontweight="bold", color=TEXT_WHITE, zorder=3,
    fontfamily="monospace",
)

# file icon (simple document indicator)
ax.text(
    yaml_r - 0.25, yaml_top - hdr_h / 2,
    "YAML",
    ha="right", va="center", fontsize=7,
    color=TEXT_LIGHT, zorder=3, fontfamily="monospace",
    bbox=dict(facecolor="#444444", edgecolor="#666666",
              boxstyle="round,pad=0.15", alpha=0.8),
)

# ── render YAML lines ──────────────────────────────────────────────
content_top = yaml_top - hdr_h - content_pad_top
y_cursor = content_top

# track section vertical positions for arrows
section_y_ranges = []  # (y_top, y_bot, colour_idx)

for sec_key, sec_lines, cidx in YAML_SECTIONS:
    sec_y_top = y_cursor

    # section key (bold, coloured)
    colour = SECTION_COLOURS[cidx]
    ax.text(
        yaml_l + 0.25, y_cursor, sec_key,
        ha="left", va="center", fontsize=8.5,
        fontweight="bold", color=colour, zorder=3,
        fontfamily="monospace",
    )
    y_cursor -= line_h

    # section values — render as single line for proper monospace alignment
    for line, is_required in sec_lines:
        ax.text(
            yaml_l + 0.25, y_cursor, line,
            ha="left", va="center", fontsize=7.8,
            color=TEXT_DARK if is_required else TEXT_GREY,
            fontweight="bold" if is_required else "normal",
            zorder=3, fontfamily="monospace",
        )
        y_cursor -= line_h

    sec_y_bot = y_cursor + line_h * 0.3

    # colour accent bar on left edge
    bar_x = yaml_l + 0.08
    bar_w = 0.04
    bar = plt.Rectangle(
        (bar_x, sec_y_bot), bar_w, sec_y_top - sec_y_bot,
        facecolor=colour, edgecolor="none", alpha=0.7, zorder=3,
    )
    ax.add_patch(bar)

    section_y_ranges.append((sec_y_top, sec_y_bot, cidx))
    y_cursor -= sec_gap  # gap between sections

# ── section cards (right) ──────────────────────────────────────────
card_l = 6.9
card_w = 2.6
card_h = 0.72
card_gap = 0.20

total_cards_h = len(SECTION_CARDS) * card_h + (len(SECTION_CARDS) - 1) * card_gap
cards_top = (yaml_top + yaml_bot) / 2 + total_cards_h / 2

for i, (title, desc) in enumerate(SECTION_CARDS):
    cy = cards_top - i * (card_h + card_gap)
    colour = SECTION_COLOURS[i]

    # card background
    card = FancyBboxPatch(
        (card_l, cy - card_h), card_w, card_h,
        boxstyle="round,pad=0.04",
        facecolor=colour, edgecolor="white",
        linewidth=2, alpha=0.92, zorder=2,
    )
    ax.add_patch(card)

    # title
    ax.text(
        card_l + card_w / 2, cy - 0.18, title,
        ha="center", va="center", fontsize=9,
        fontweight="bold", color=TEXT_WHITE, zorder=3,
    )
    # description
    ax.text(
        card_l + card_w / 2, cy - card_h + 0.20, desc,
        ha="center", va="center", fontsize=7,
        color=TEXT_WHITE, alpha=0.90, zorder=3,
        linespacing=1.3,
    )

    # ── connecting arrow from YAML section to card ──────────────────
    if i < len(section_y_ranges):
        src_y_top, src_y_bot, _ = section_y_ranges[i]
        src_y_mid = (src_y_top + src_y_bot) / 2
        dst_y_mid = cy - card_h / 2

        arrow = FancyArrowPatch(
            (yaml_r + 0.08, src_y_mid),
            (card_l - 0.08, dst_y_mid),
            arrowstyle="->,head_length=5,head_width=3.5",
            color=colour, lw=1.5, alpha=0.7,
            connectionstyle="arc3,rad=0.12",
            zorder=1,
        )
        ax.add_patch(arrow)

# ── bottom annotation ──────────────────────────────────────────────
ax.text(
    FIG_W / 2, yaml_bot - 0.30,
    "Minimal working example  —  10 required parameters shown (bold keys); all others use validated defaults",
    ha="center", va="center", fontsize=8,
    color=TEXT_LIGHT, fontstyle="italic",
)

# ── save ────────────────────────────────────────────────────────────
out_dir = "/Users/darrieythorsson/compHydro/Papers/Article 2 - SYMFLUENCE/diagrams/4. configuration"
for fmt in ("pdf", "png"):
    fig.savefig(
        f"{out_dir}/fig4_configuration.{fmt}",
        dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1,
    )
    print(f"Saved fig4_configuration.{fmt}")
plt.close(fig)
