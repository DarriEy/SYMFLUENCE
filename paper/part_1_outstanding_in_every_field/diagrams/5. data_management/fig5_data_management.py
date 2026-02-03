"""
SYMFLUENCE Data Management Pipelines (Section 3.3).

Layout:
  DataManager on the left as orchestrator sidebar.
  Two clearly separated zones:
    Acquisition Layer — modes, handlers → Raw Data Store
    Processing Layer  — panels (a), (b), (c) → Model-Ready Data Store
  DM dispatches into both zones; Raw Data Store is the interface.

Colour palette consistent with other SYMFLUENCE figures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── layout ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 13.4, 13.5

# ── colours ─────────────────────────────────────────────────────────
C_FORCING = "#C9943A"
C_OBS     = "#4A7FB5"
C_ATTR    = "#5BA58B"
C_CORE    = "#8B6DAF"
C_OUTPUT  = "#2E86AB"
C_ACQ     = "#7A7A7A"
C_RAW     = "#A0522D"

TEXT_DARK  = "#2D2D2D"
TEXT_GREY  = "#666666"
TEXT_WHITE = "#FFFFFF"
ARR_DARK   = "#3A3A3A"

# ── DataManager column ────────────────────────────────────────────
DM_X = 0.30
DM_W = 1.80

# ── main content area ─────────────────────────────────────────────
MAIN_L = DM_X + DM_W + 0.90          # ≈ 3.00

# ── shared sizing ─────────────────────────────────────────────────
BW       = 1.70
BH       = 0.72
OUT_W    = 1.50
APAD     = 0.08
HGAP     = 0.28
PNL_GAP  = 0.44
PNL_PAD  = 0.24
TITLE_H  = 0.28
BAR_W    = 0.055

# Chain geometry (shifted right to leave room for DM column)
CHAIN_X0 = MAIN_L + 0.18             # ≈ 3.18
OUT_X    = CHAIN_X0 + 4 * BW + 3 * HGAP + HGAP
OUT_R    = OUT_X + OUT_W
PNL_L    = CHAIN_X0 - 0.22
PNL_R    = OUT_R + 0.22
RAIL_R   = OUT_R + 0.42

# ── helpers ───────────────────────────────────────────────────────

def lighten(hex_colour, factor=0.82):
    rgb = [int(hex_colour[i:i+2], 16) / 255 for i in (1, 3, 5)]
    return [1 - factor * (1 - c) for c in rgb]


def box(ax, x, y, w, h, colour, label, sublabel=None,
        fontsize=8.5, sub_fontsize=7.0, text_colour=TEXT_WHITE,
        bold=True, edge_colour=None, lw=1.0, zorder=3,
        linestyle="-"):
    ec = edge_colour or colour
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        facecolor=colour, edgecolor=ec, linewidth=lw,
        linestyle=linestyle, zorder=zorder))
    if sublabel:
        ax.text(x + w / 2, y + h * 0.62, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                color=text_colour, zorder=zorder + 1, linespacing=1.15)
        ax.text(x + w / 2, y + h * 0.25, sublabel,
                ha="center", va="center", fontsize=sub_fontsize,
                color=text_colour, alpha=0.85, zorder=zorder + 1,
                fontstyle="italic", linespacing=1.1)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                color=text_colour, zorder=zorder + 1, linespacing=1.15)


def h_arrow(ax, x1, x2, y, colour=None, lw=1.8):
    colour = colour or ARR_DARK
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.2",
                                color=colour, lw=lw,
                                shrinkA=0, shrinkB=0), zorder=5)


def v_arrow(ax, x, y1, y2, colour=None, lw=1.8):
    colour = colour or ARR_DARK
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.2",
                                color=colour, lw=lw,
                                shrinkA=0, shrinkB=0), zorder=5)


def panel_bg(ax, x, y, w, h, colour):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        facecolor=lighten(colour, 0.18), edgecolor=colour,
        linewidth=0.8, alpha=0.30, zorder=0))


def accent_bar(ax, x, y, h, colour):
    bar = FancyBboxPatch(
        (x + 0.035, y + 0.07), BAR_W, h - 0.14,
        boxstyle="round,pad=0.012",
        facecolor=colour, edgecolor="none", alpha=0.85, zorder=2)
    ax.add_patch(bar)


def draw_chain(ax, x0, y, items, colour, bw=BW, bh=BH, hgap=HGAP):
    sx = x0
    for i, (label, sub) in enumerate(items):
        box(ax, sx, y, bw, bh, colour, label, sublabel=sub)
        if i < len(items) - 1:
            h_arrow(ax, sx + bw + APAD, sx + bw + hgap - APAD,
                    y + bh / 2, ARR_DARK)
        sx += bw + hgap
    return sx - hgap


# ── figure ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

ax.text((PNL_L + PNL_R) / 2, FIG_H - 0.35,
        "SYMFLUENCE Data Management Pipelines",
        ha="center", va="center", fontsize=13,
        fontweight="bold", color=TEXT_DARK)

# ════════════════════════════════════════════════════════════════════
# ACQUISITION LAYER  (zone contents — modes, handlers)
# ════════════════════════════════════════════════════════════════════

# -- Acquisition mode strip (wrapped in sub-panel) --
ACQ_H  = 0.55
ACQ_BW = 2.05
acq_gap = 0.28
acq_strip_w = 3 * ACQ_BW + 2 * acq_gap
acq_x0 = PNL_L + ((PNL_R - PNL_L) - acq_strip_w) / 2
acq_y  = FIG_H - 1.35 - ACQ_H          # bottom of mode boxes

# Sub-panel background around mode boxes
MODE_PAD = 0.18
MODE_TITLE_H = 0.26
mode_bg_l = acq_x0 - MODE_PAD
mode_bg_r = acq_x0 + acq_strip_w + MODE_PAD
mode_bg_bot = acq_y - MODE_PAD
mode_bg_top = acq_y + ACQ_H + MODE_PAD + MODE_TITLE_H
ax.add_patch(FancyBboxPatch(
    (mode_bg_l, mode_bg_bot), mode_bg_r - mode_bg_l, mode_bg_top - mode_bg_bot,
    boxstyle="round,pad=0.08",
    facecolor=lighten(C_ACQ, 0.12), edgecolor=C_ACQ,
    linewidth=0.8, alpha=0.45, zorder=0))
ax.text((mode_bg_l + mode_bg_r) / 2, acq_y + ACQ_H + MODE_PAD + MODE_TITLE_H * 0.45,
        "Acquisition Modes", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=TEXT_DARK, zorder=1)

modes = [
    ("Cloud Mode",     "Zarr / S3 / API"),
    ("MAF Mode",       "gistool / datatool / SLURM"),
    ("User Supplied",  "Pre-staged local files"),
]
for i, (lbl, sub) in enumerate(modes):
    mx = acq_x0 + i * (ACQ_BW + acq_gap)
    box(ax, mx, acq_y, ACQ_BW, ACQ_H, C_ACQ,
        lbl, sublabel=sub, fontsize=7.5, sub_fontsize=6.0)

# Fan line below modes
mode_fan_y = acq_y - 0.12
acq_mode_cxs = [acq_x0 + i * (ACQ_BW + acq_gap) + ACQ_BW / 2 for i in range(3)]
for cx in acq_mode_cxs:
    v_arrow(ax, cx, acq_y - APAD, mode_fan_y + 0.02, C_ACQ, lw=1.0)
ax.plot([acq_mode_cxs[0], acq_mode_cxs[-1]], [mode_fan_y, mode_fan_y],
        color=C_ACQ, lw=1.2, zorder=4)

# -- Acquisition handler cards --
AH_H  = 0.70
AH_BW = 2.60
AH_GAP = 0.26
ah_strip_w = 3 * AH_BW + 2 * AH_GAP
ah_x0 = PNL_L + ((PNL_R - PNL_L) - ah_strip_w) / 2
ah_y  = mode_fan_y - 0.22 - AH_H       # bottom of handler boxes

handler_groups = [
    ("Forcing Handlers",     "ERA5 \u00b7 AORC \u00b7 CARRA\nRDRS \u00b7 HRRR \u00b7 NEX-GDDP",    C_FORCING),
    ("Observation Handlers", "USGS \u00b7 WSC \u00b7 SNOTEL\nGRACE \u00b7 SMAP \u00b7 MODIS",       C_OBS),
    ("Attribute Handlers",   "MERIT-Hydro \u00b7 SoilGrids\nMODIS LC \u00b7 Copernicus DEM",         C_ATTR),
]
handler_cxs = [ah_x0 + i * (AH_BW + AH_GAP) + AH_BW / 2 for i in range(3)]

for i, (lbl, sub, clr) in enumerate(handler_groups):
    hx = ah_x0 + i * (AH_BW + AH_GAP)
    box(ax, hx, ah_y, AH_BW, AH_H, clr,
        lbl, sublabel=sub, fontsize=7.5, sub_fontsize=6.0)

# Vertical lines from mode fan → handler cards
for cx in handler_cxs:
    v_arrow(ax, cx, mode_fan_y - 0.02, ah_y + AH_H + APAD, ARR_DARK, lw=1.1)

# Fan line below handlers
handler_fan_y = ah_y - 0.12
for cx in handler_cxs:
    v_arrow(ax, cx, ah_y - APAD, handler_fan_y + 0.02, ARR_DARK, lw=1.1)
ax.plot([handler_cxs[0], handler_cxs[-1]], [handler_fan_y, handler_fan_y],
        color=ARR_DARK, lw=1.3, zorder=4)

# Acquisition zone bounds (for background)
acq_zone_top = mode_bg_top + 0.34
acq_zone_bot = handler_fan_y - 0.15

# ════════════════════════════════════════════════════════════════════
# RAW DATA STORE  (interface between acquisition & processing)
# ════════════════════════════════════════════════════════════════════
RAW_W, RAW_H = 4.80, 0.70
content_cx = (PNL_L + PNL_R) / 2
raw_x = content_cx - RAW_W / 2
raw_y = acq_zone_bot - 0.40 - RAW_H

box(ax, raw_x, raw_y, RAW_W, RAW_H, C_RAW,
    "Raw Data Store",
    sublabel="forcing/raw_data/ \u00b7 observations/ \u00b7 attributes/",
    fontsize=9.0, sub_fontsize=6.5)

# Arrow from handler fan → raw data store
v_arrow(ax, content_cx, handler_fan_y - 0.02, raw_y + RAW_H + APAD, C_RAW, lw=1.6)

# ════════════════════════════════════════════════════════════════════
# DATA FLOW: Raw Data Store → left rail → Processing panels
# ════════════════════════════════════════════════════════════════════
# Single line from raw store left edge going down, then branching right
# into each processing panel (drawn after panels are laid out below).
RAW_RAIL_X = PNL_L - 0.28           # x-position of the vertical rail (outside zone backgrounds)
raw_out_y = raw_y - APAD

# ════════════════════════════════════════════════════════════════════
# PANEL (a) — Forcing Preprocessing Pipeline
# ════════════════════════════════════════════════════════════════════
PA_H   = BH + 2 * PNL_PAD + TITLE_H
PA_TOP = raw_y - 0.78
PA_BOT = PA_TOP - PA_H
panel_bg(ax, PNL_L, PA_BOT, PNL_R - PNL_L, PA_H, C_FORCING)
accent_bar(ax, PNL_L, PA_BOT, PA_H, C_FORCING)

ax.text(PNL_L + 0.16, PA_TOP - TITLE_H / 2,
        "(a)  Forcing Preprocessing Pipeline",
        ha="left", va="center", fontsize=8.5, fontweight="bold",
        color=TEXT_DARK)

r_a = PA_BOT + PNL_PAD
stages_a = [
    ("Spatial\nRemapping",        "EASYMORE W\u1d40 \u00d7 grid \u2192 HRU"),
    ("Temporal\nProcessing",      "Agg / disagg / alignment"),
    ("Variable\nStandardisation", "Name + unit mapping (Pint)"),
    ("Elevation\nCorrection",     "Lapse rates + precip scaling"),
]
last_a = draw_chain(ax, CHAIN_X0, r_a, stages_a, C_FORCING)
box(ax, OUT_X, r_a, OUT_W, BH, C_OUTPUT,
    "Processed\nForcing", sublabel="Per-HRU time series",
    fontsize=7.5, sub_fontsize=6.0)
h_arrow(ax, last_a + APAD, OUT_X - APAD, r_a + BH / 2, ARR_DARK)
pa_out_cy = r_a + BH / 2

# (raw data rail drawn after all panels are laid out)

# ════════════════════════════════════════════════════════════════════
# PANEL (b) — Observation Processing Pipeline
# ════════════════════════════════════════════════════════════════════
SRC_W   = 1.70
SRC_GAP = 0.20
SRC_H   = 0.72

sources_merged = [
    ("Streamflow",    "USGS \u00b7 WSC \u00b7 SMHI\nAPI + Units"),
    ("Snow",          "SNOTEL + MODIS\nMerge Terra/Aqua"),
    ("Soil Moisture", "SMAP \u00b7 ESA CCI \u00b7 ISMN\nQC Flags + Depth"),
    ("ET",            "MODIS MOD16 \u00b7 FLUXNET\nComposite + TZ"),
    ("Storage",       "GRACE/GRACE-FO\nSpatial Extraction"),
]
src_total = len(sources_merged) * SRC_W + (len(sources_merged) - 1) * SRC_GAP
src_x0 = PNL_L + ((PNL_R - PNL_L) - src_total) / 2

GATHER_H  = 0.26
PB_VGAP   = 0.26
PB_CONTENT = SRC_H + GATHER_H + PB_VGAP + BH
PB_H = PB_CONTENT + 2 * PNL_PAD + TITLE_H
PB_TOP = PA_BOT - PNL_GAP
PB_BOT = PB_TOP - PB_H
panel_bg(ax, PNL_L, PB_BOT, PNL_R - PNL_L, PB_H, C_OBS)
accent_bar(ax, PNL_L, PB_BOT, PB_H, C_OBS)

ax.text(PNL_L + 0.16, PB_TOP - TITLE_H / 2,
        "(b)  Observation Processing Pipeline",
        ha="left", va="center", fontsize=8.5, fontweight="bold",
        color=TEXT_DARK)

# Source cards
src_y = PB_TOP - TITLE_H - PNL_PAD - SRC_H
for i, (label, sub) in enumerate(sources_merged):
    sx = src_x0 + i * (SRC_W + SRC_GAP)
    box(ax, sx, src_y, SRC_W, SRC_H, lighten(C_OBS, 0.55),
        label, sublabel=sub, fontsize=7.0, sub_fontsize=6.0,
        text_colour=TEXT_DARK, edge_colour=C_OBS, lw=0.9)

# Gathering line
line_y = src_y - GATHER_H / 2
left_cx  = src_x0 + SRC_W / 2
right_cx = src_x0 + (len(sources_merged) - 1) * (SRC_W + SRC_GAP) + SRC_W / 2
ax.plot([left_cx, right_cx], [line_y, line_y],
        color=ARR_DARK, lw=1.4, zorder=4)
for i in range(len(sources_merged)):
    cx = src_x0 + i * (SRC_W + SRC_GAP) + SRC_W / 2
    v_arrow(ax, cx, src_y - APAD, line_y + 0.02, ARR_DARK, lw=1.2)

bb_y = PB_BOT + PNL_PAD
v_arrow(ax, content_cx, line_y - 0.02, bb_y + BH + APAD, ARR_DARK, lw=1.6)

backbone = [
    ("Unit Conversion",    "Provider \u2192 SI units"),
    ("Quality Control",    "Flag filtering + validation"),
    ("Gap Handling",       "Detect / interpolate / mask"),
    ("Temporal Alignment", "TZ + calendar + stamps"),
]
last_bb = draw_chain(ax, CHAIN_X0, bb_y, backbone, C_OBS)
box(ax, OUT_X, bb_y, OUT_W, BH, C_OUTPUT,
    "Evaluation-Ready\nTime Series", sublabel="Standardised format",
    fontsize=7.0, sub_fontsize=5.8)
h_arrow(ax, last_bb + APAD, OUT_X - APAD, bb_y + BH / 2, ARR_DARK)
pb_out_cy = bb_y + BH / 2

# (raw data rail drawn after all panels are laid out)

# ════════════════════════════════════════════════════════════════════
# PANEL (c) — Attribute Processing Pipeline
# ════════════════════════════════════════════════════════════════════
PC_H = BH + 2 * PNL_PAD + TITLE_H
PC_TOP = PB_BOT - PNL_GAP
PC_BOT = PC_TOP - PC_H
panel_bg(ax, PNL_L, PC_BOT, PNL_R - PNL_L, PC_H, C_ATTR)
accent_bar(ax, PNL_L, PC_BOT, PC_H, C_ATTR)

ax.text(PNL_L + 0.16, PC_TOP - TITLE_H / 2,
        "(c)  Attribute Processing Pipeline",
        ha="left", va="center", fontsize=8.5, fontweight="bold",
        color=TEXT_DARK)

r_c = PC_BOT + PNL_PAD
attrs = [
    ("Acquire\nAttributes",          "DEM \u00b7 Soil \u00b7 Land Cover"),
    ("Mosaic & Terrain\nDerivation", "Tile merge + slope/aspect"),
    ("Zonal Statistics",             "Mean / mode per catchment"),
    ("Format &\nWrite",              "Standardised output"),
]
last_c = draw_chain(ax, CHAIN_X0, r_c, attrs, C_ATTR)
box(ax, OUT_X, r_c, OUT_W, BH, C_OUTPUT,
    "HRU\nAttributes", sublabel="Parameter templates",
    fontsize=7.5, sub_fontsize=6.0)
h_arrow(ax, last_c + APAD, OUT_X - APAD, r_c + BH / 2, ARR_DARK)
pc_out_cy = r_c + BH / 2


# (raw data rail drawn after all panels are laid out)

# Processing zone bounds
proc_zone_top = PA_TOP + 0.46
proc_zone_bot = PC_BOT - 0.15

# ════════════════════════════════════════════════════════════════════
# MODEL-READY DATA STORE
# ════════════════════════════════════════════════════════════════════
REPO_W, REPO_H = 3.80, 0.70
repo_x = content_cx - REPO_W / 2
repo_y = PC_BOT - 1.30
box(ax, repo_x, repo_y, REPO_W, REPO_H, C_OUTPUT,
    "Model-Ready Data Store",
    sublabel="NetCDF-4 | CF-1.8 conventions | per-HRU",
    fontsize=8.5, sub_fontsize=6.5)
repo_top = repo_y + REPO_H

# ════════════════════════════════════════════════════════════════════
# RIGHT RAIL — output collection
# ════════════════════════════════════════════════════════════════════
for cy in [pa_out_cy, pb_out_cy, pc_out_cy]:
    ax.plot([OUT_R + APAD, RAIL_R], [cy, cy],
            color=C_OUTPUT, lw=1.1, zorder=5)
    ax.plot(RAIL_R, cy, "o", color=C_OUTPUT, ms=3.5, zorder=6)

ax.plot([RAIL_R, RAIL_R], [pa_out_cy, pc_out_cy],
        color=C_OUTPUT, lw=1.4, solid_capstyle="round", zorder=4)

repo_cx = repo_x + REPO_W / 2
elbow_y = repo_top + 0.22
ax.plot([RAIL_R, RAIL_R], [pc_out_cy, elbow_y],
        color=C_OUTPUT, lw=1.4, zorder=4)
ax.plot([RAIL_R, repo_cx], [elbow_y, elbow_y],
        color=C_OUTPUT, lw=1.4, zorder=4)
v_arrow(ax, repo_cx, elbow_y, repo_top + APAD, C_OUTPUT, lw=1.4)

# ════════════════════════════════════════════════════════════════════
# ZONE BACKGROUNDS  (very subtle, behind everything)
# ════════════════════════════════════════════════════════════════════
zone_l = PNL_L - 0.12
zone_r = PNL_R + 0.12
zone_w = zone_r - zone_l

# Acquisition zone
ax.add_patch(FancyBboxPatch(
    (zone_l, acq_zone_bot), zone_w, acq_zone_top - acq_zone_bot,
    boxstyle="round,pad=0.10",
    facecolor="#F5F0E8", edgecolor="#B8AE9A",
    linewidth=1.2, alpha=0.55, zorder=-1))
ax.text(zone_l + 0.20, acq_zone_top - 0.10,
        "Acquisition\nLayer", ha="left", va="top",
        fontsize=9.5, fontweight="bold", color="#6B5F4F",
        fontstyle="italic", zorder=1, linespacing=1.1)

# Processing zone
ax.add_patch(FancyBboxPatch(
    (zone_l, proc_zone_bot), zone_w, proc_zone_top - proc_zone_bot,
    boxstyle="round,pad=0.10",
    facecolor="#EBF0F5", edgecolor="#94A6B8",
    linewidth=1.2, alpha=0.55, zorder=-1))
ax.text(zone_l + 0.20, proc_zone_top - 0.14,
        "Processing Layer", ha="left", va="top",
        fontsize=9.5, fontweight="bold", color="#4F5F6B",
        fontstyle="italic", zorder=1)

# ════════════════════════════════════════════════════════════════════
# DATAMANAGER — full-height sidebar (fig8 convention)
# ════════════════════════════════════════════════════════════════════
C_SIDE = C_CORE
C_CFG  = "#4A6741"
side_x = DM_X
side_w = DM_W

# ── config.yaml container (anchored at bottom of old sidebar span) ──
cfg_outer_bot = repo_y - 0.06
cfg_outer_h   = 1.10
cfg_outer_top = cfg_outer_bot + cfg_outer_h

ax.add_patch(FancyBboxPatch(
    (side_x, cfg_outer_bot), side_w, cfg_outer_h,
    boxstyle="round,pad=0.08",
    facecolor=lighten(C_CFG, 0.10), edgecolor=C_CFG,
    linewidth=1.2, zorder=0))

# Config header bar
cfg_th_h = 0.32
cfg_th_y = cfg_outer_top - 0.08 - cfg_th_h
ax.add_patch(FancyBboxPatch(
    (side_x + 0.08, cfg_th_y), side_w - 0.16, cfg_th_h,
    boxstyle="round,pad=0.05",
    facecolor=C_CFG, edgecolor="white", linewidth=0, zorder=2))
ax.text(side_x + side_w / 2, cfg_th_y + cfg_th_h / 2,
        "config.yaml", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=TEXT_WHITE, zorder=3)

# Config key tags inside container
cfg_key_w = side_w - 0.30
cfg_key_h = 0.18
cfg_key_x = side_x + 0.15
cfg_key_gap = 0.04
cky = cfg_th_y - 0.10 - cfg_key_h
for k in ["data_sources:", "processing:", "output:"]:
    box(ax, cfg_key_x, cky, cfg_key_w, cfg_key_h,
        "#FFFFFF", k, fontsize=5.5,
        text_colour=TEXT_DARK, edge_colour=C_CFG, lw=0.6, bold=False)
    cky -= cfg_key_h + cfg_key_gap

# Arrow from config container up to DataManager sidebar
cfg_cx = side_x + side_w / 2

# ── DataManager sidebar (shortened, stops above config box) ─────
dm_gap = 0.22
side_top = acq_zone_top + 0.06
side_bot = cfg_outer_top + dm_gap
side_h = side_top - side_bot

v_arrow(ax, cfg_cx, cfg_outer_top + APAD, side_bot - APAD, C_SIDE, lw=1.3)

# Outer container
ax.add_patch(FancyBboxPatch(
    (side_x, side_bot), side_w, side_h,
    boxstyle="round,pad=0.08",
    facecolor=lighten(C_SIDE, 0.12), edgecolor=C_SIDE,
    linewidth=1.2, zorder=0))

# ── shared pill geometry ──────────────────────────────────────────
pill_w   = side_w - 0.28
pill_x   = side_x + 0.14
th_h     = 0.38          # header bar height
pill_h   = 0.52          # workflow pill
pill_gap = 0.16
reg_h    = 0.44          # registry pill
reg_gap  = 0.12
inset    = 0.08          # inset from sidebar edge

# ── title header (pinned to top) ────────────────────────────────
th_y = side_top - inset - th_h
ax.add_patch(FancyBboxPatch(
    (side_x + inset, th_y), side_w - 2 * inset, th_h,
    boxstyle="round,pad=0.05",
    facecolor=C_SIDE, edgecolor="white", linewidth=0, zorder=2))
ax.text(side_x + side_w / 2, th_y + th_h / 2,
        "DataManager", ha="center", va="center",
        fontsize=9.0, fontweight="bold", color=TEXT_WHITE, zorder=3)

# ── subtitle (just below header) ────────────────────────────────
sub_y = th_y - 0.24
ax.text(side_x + side_w / 2, sub_y,
        "Registry-based\nOrchestration",
        ha="center", va="center", fontsize=6.5,
        fontweight="bold", color=TEXT_DARK, zorder=2, linespacing=1.3)
ax.text(side_x + side_w / 2, sub_y - 0.38,
        "Dispatches acquisition\nand preprocessing\nworkflows sequentially",
        ha="center", va="center", fontsize=5.8,
        color=TEXT_GREY, zorder=2, linespacing=1.4)

# ── separator below subtitle ─────────────────────────────────────
sep1_y = sub_y - 0.68
ax.plot([side_x + 0.14, side_x + side_w - 0.14], [sep1_y, sep1_y],
        color=C_SIDE, lw=0.5, alpha=0.5, zorder=2)

# ── workflow phase pills (centred in middle of sidebar) ──────────
phases = [
    ("1. Acquire",    "Modes \u2192 Handlers\n\u2192 Raw Data Store"),
    ("2. Preprocess", "Raw Store \u2192 Pipelines\n\u2192 Model-Ready"),
]
pills_block_h = 2 * pill_h + pill_gap
# Vertical centre of the usable area between sep1 and registry zone
reg_block_h = 2 * reg_h + reg_gap
reg_zone_top = side_bot + inset + reg_block_h + 0.50   # top of registry zone
mid_zone = (sep1_y + reg_zone_top) / 2
py = mid_zone + pills_block_h / 2 - pill_h   # top pill top-edge

# Ensure pills don't overlap sep1
py = min(py, sep1_y - 0.18 - pill_h)

for label, desc in phases:
    ax.add_patch(FancyBboxPatch(
        (pill_x, py), pill_w, pill_h,
        boxstyle="round,pad=0.05",
        facecolor="#FFFFFF", edgecolor=C_SIDE,
        linewidth=0.7, zorder=4))
    ax.text(pill_x + pill_w / 2, py + pill_h * 0.68,
            label, ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=TEXT_DARK, zorder=5)
    ax.text(pill_x + pill_w / 2, py + pill_h * 0.28,
            desc, ha="center", va="center",
            fontsize=5.5, color=TEXT_GREY, fontstyle="italic",
            zorder=5, linespacing=1.3)
    py -= pill_h + pill_gap

# ── separator above registries ──────────────────────────────────
sep2_y = side_bot + inset + reg_block_h + 0.30
ax.plot([side_x + 0.14, side_x + side_w - 0.14], [sep2_y, sep2_y],
        color=C_SIDE, lw=0.5, alpha=0.5, zorder=2)

# ── registry pills (pinned near bottom) ─────────────────────────
ry = sep2_y - 0.18 - reg_h
for label, sub in [
    ("Pipeline Registry",    "@register_pipeline"),
    ("Acquisition Registry", "@register_handler"),
]:
    ax.add_patch(FancyBboxPatch(
        (pill_x, ry), pill_w, reg_h,
        boxstyle="round,pad=0.05",
        facecolor=lighten(C_SIDE, 0.15), edgecolor=C_SIDE,
        linewidth=0.7, linestyle=(0, (4, 3)), zorder=4))
    ax.text(pill_x + pill_w / 2, ry + reg_h * 0.62,
            label, ha="center", va="center",
            fontsize=6.5, fontweight="bold", color=TEXT_DARK, zorder=5)
    ax.text(pill_x + pill_w / 2, ry + reg_h * 0.28,
            sub, ha="center", va="center",
            fontsize=5.5, color=TEXT_GREY, fontstyle="italic", zorder=5)
    ry -= reg_h + reg_gap

# ── dashed connector arrows from sidebar to key rows ─────────────
side_r = side_x + side_w
acq_zone_mid = (acq_zone_top + acq_zone_bot) / 2
proc_zone_mid = (proc_zone_top + proc_zone_bot) / 2
connect_rows = [
    acq_zone_mid,
    proc_zone_mid,
]
for y_t in connect_rows:
    ax.annotate(
        "", xy=(zone_l + 0.05, y_t),
        xytext=(side_r + 0.03, y_t),
        arrowprops=dict(arrowstyle="-|>", color=C_SIDE, lw=0.9,
                        linestyle=(0, (4, 3)), mutation_scale=10,
                        alpha=0.50),
        zorder=1)

# ── Raw data feed rail (left-side rail branching to each panel) ──
# Elbow from raw store left edge → down to rail → branch right
raw_elbow_y = raw_y + RAW_H / 2       # start at raw store centre-left
rail_top_y  = raw_elbow_y
rail_bot_y  = r_c + BH / 2            # bottom branch level (panel c)

# Horizontal segment from raw store left edge to rail x
ax.plot([raw_x - APAD, RAW_RAIL_X], [raw_elbow_y, raw_elbow_y],
        color=C_RAW, lw=1.5, solid_capstyle="round", zorder=4)

# Vertical rail from elbow down to lowest panel
ax.plot([RAW_RAIL_X, RAW_RAIL_X], [rail_top_y, rail_bot_y],
        color=C_RAW, lw=1.5, solid_capstyle="round", zorder=4)

# Branch arrows from rail into each panel, with labels
raw_branches = [
    (r_a + BH / 2,          CHAIN_X0 - APAD, "forcing"),
    (src_y + SRC_H / 2,     src_x0 - APAD,   "observations"),
    (r_c + BH / 2,          CHAIN_X0 - APAD, "attributes"),
]
for branch_y, target_x, lbl in raw_branches:
    ax.plot(RAW_RAIL_X, branch_y, "o", color=C_RAW, ms=4, zorder=6)
    h_arrow(ax, RAW_RAIL_X + 0.02, target_x - APAD, branch_y, C_RAW, lw=1.3)

# Rail label
rail_mid = (rail_top_y + rail_bot_y) / 2
ax.text(RAW_RAIL_X - 0.06, rail_mid, "raw data feed",
        ha="right", va="center", fontsize=6.0, color=C_RAW,
        rotation=90, fontstyle="italic", zorder=6)



# ── save ──────────────────────────────────────────────────────────
out = "/Users/darrieythorsson/compHydro/Papers/Article 2 - SYMFLUENCE/diagrams/5. data_management"
for fmt in ("pdf", "png"):
    fig.savefig(f"{out}/fig5_data_management.{fmt}",
                dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.05)
    print(f"Saved fig5_data_management.{fmt}")
plt.close(fig)
