"""
SYMFLUENCE Analysis Layer (Section 3.7).

Shows the four pillars of the analysis layer — Performance Metrics,
Multi-Variable Evaluation, Visualization & Diagnostics, and Benchmarking
& Sensitivity — plus Calibration Integration (§3.6), with their internal
registries, concrete implementations, and interconnections.

Verified against src/symfluence/analysis/ and
src/symfluence/evaluation/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── layout ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 14.8, 13.0          # match fig8 width; reduced after eval simplification
LABEL_W = 0.90                      # left-margin label zone
STACK_L = LABEL_W + 0.18            # main content left edge
STACK_R = 13.50                     # main content right edge
STACK_W = STACK_R - STACK_L
MID_X   = (STACK_L + STACK_R) / 2
PAD     = 0.04
BAR_W   = 0.055                     # accent bar width (fig3/fig5)

# vertical gaps
GAP_TITLE_REG  = 0.20               # title → registry banner
GAP_REG_CAT    = 0.28               # registry → metric categories
GAP_CAT_EVAL   = 0.38               # metric categories → eval banner
GAP_EVAL_CARDS = 0.28               # eval banner → evaluator cards
GAP_CARDS_BOT  = 0.45               # evaluator cards → bottom panels

# ── colours — consistent with paper palette ─────────────────────────
C_MET   = "#4A7FB5"   # blue   — metrics
C_EVAL  = "#5BA58B"   # green  — evaluators
C_VIS   = "#C9943A"   # gold   — visualization
C_BENCH = "#8B6DAF"   # purple — benchmarking / sensitivity
C_CALIB = "#C0392B"   # red    — calibration integration (fig8 loop-step)
C_REG   = "#2E86AB"   # teal   — registries

TEXT_D  = "#2D2D2D"
TEXT_W  = "#FFFFFF"
TEXT_G  = "#555555"
TEXT_L  = "#888888"
CLR_A   = "#999999"
CLR_B   = "#BBBBBB"

# ── helpers ─────────────────────────────────────────────────────────

def lighten(c, f=0.35):
    r, g, b = (int(c[i:i+2], 16) for i in (1, 3, 5))
    return "#{:02X}{:02X}{:02X}".format(
        int(r + (255 - r) * f), int(g + (255 - g) * f), int(b + (255 - b) * f))


def _dark(c):
    r, g, b = (int(c[i:i+2], 16) for i in (1, 3, 5))
    return r * 0.299 + g * 0.587 + b * 0.114 < 160


def rbox(ax, x, y, w, h, fc, label=None, sub=None, *,
         ec="white", lw=1.5, ls="-", fs=9.5, sfs=7.5,
         tc=None, zorder=3, bold=True, lo=0.14, so=0.17):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={PAD}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        zorder=zorder, clip_on=False)
    ax.add_patch(patch)
    if label is None:
        return
    tc = tc or (TEXT_W if _dark(fc) else TEXT_D)
    if sub:
        ax.text(x + w / 2, y + h / 2 + lo, label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal",
                color=tc, zorder=zorder + 1, clip_on=False)
        ax.text(x + w / 2, y + h / 2 - so, sub,
                ha="center", va="center", fontsize=sfs,
                color=tc, alpha=0.82, fontstyle="italic",
                zorder=zorder + 1, clip_on=False)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal",
                color=tc, zorder=zorder + 1, clip_on=False)


def accent_bar(ax, x, y, h, colour):
    """Thin coloured accent bar on the left edge (fig3/fig5 convention)."""
    bar = FancyBboxPatch(
        (x + 0.035, y + 0.07), BAR_W, h - 0.14,
        boxstyle="round,pad=0.012",
        facecolor=colour, edgecolor="none", alpha=0.85, zorder=2)
    ax.add_patch(bar)


def section_label(ax, y_center, label, colour, fs=7.5):
    """Left-margin section label (fig3 convention)."""
    ax.text(STACK_L - 0.14, y_center, label,
            ha="right", va="center", fontsize=fs,
            fontweight="bold", color=colour, linespacing=1.3)


def harr(ax, x1, x2, y, c=CLR_A, lw=1.3, st="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle=st, color=c, lw=lw,
        zorder=5, mutation_scale=ms))


def varr(ax, x, y1, y2, c=CLR_A, lw=1.3, st="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(
        (x, y1), (x, y2), arrowstyle=st, color=c, lw=lw,
        zorder=5, mutation_scale=ms))


def flow_label(ax, x, y, text):
    """Small italic flow annotation next to an arrow."""
    ax.text(x, y, text, ha="left", va="center",
            fontsize=6.5, color=TEXT_G, fontstyle="italic")


# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_clip_on(False)
ax.axis("off")

# ── title ───────────────────────────────────────────────────────────
title_y = FIG_H - 0.36
ax.text(MID_X, title_y,
        "Analysis Layer Architecture",
        ha="center", va="center", fontsize=14,
        fontweight="bold", color=TEXT_D)
ax.text(MID_X, title_y - 0.36,
        "Performance evaluation, multi-variable assessment, "
        "visualization, and systematic analysis  (\u00a73.7)",
        ha="center", va="center", fontsize=8.5,
        color=TEXT_G, fontstyle="italic")

# ════════════════════════════════════════════════════════════════════
# ROW 1 — METRIC_REGISTRY banner
# ════════════════════════════════════════════════════════════════════
reg_h = 0.48
reg_y = title_y - 0.36 - GAP_TITLE_REG - reg_h

rbox(ax, STACK_L, reg_y, STACK_W, reg_h, C_REG,
     "METRIC_REGISTRY",
     sub="Centralised metric store  \u2022  Function refs + metadata "
         "(optimal, range, direction, units)  \u2022  Multi-alias lookup",
     lo=0.10, so=0.11, fs=10, sfs=7)

section_label(ax, reg_y + reg_h / 2, "\u00a73.7.1\nMetrics", C_MET)

# ════════════════════════════════════════════════════════════════════
# ROW 2 — Four metric categories (proportional widths, FIXED sizing)
# ════════════════════════════════════════════════════════════════════
cat_gap = 0.14
cat_pad = 0.04
cat_h = 1.84           # FIXED: was 1.05 — 4 pills need 1.36 zone
cat_y = reg_y - GAP_REG_CAT - cat_h

categories = [
    (C_MET, "Efficiency", ["NSE", "logNSE", "KGE / KGE\u2032", "KGEnp"]),
    (C_MET, "Error",      ["RMSE", "MAE", "NRMSE", "MARE"]),
    (C_MET, "Bias",       ["Abs. Bias", "PBIAS"]),
    (C_MET, "Correlation", ["Pearson r / R\u00b2", "Spearman \u03c1"]),
]

# Proportional widths: 30% / 30% / 20% / 20% of usable space
cat_usable = STACK_W - 2 * cat_pad - 3 * cat_gap
cat_fracs = [0.30, 0.30, 0.20, 0.20]
cat_widths = [cat_usable * f for f in cat_fracs]

cx_pos = STACK_L + cat_pad
for i, (clr, title, items) in enumerate(categories):
    cat_w = cat_widths[i]
    cx = cx_pos

    # dashed container
    rbox(ax, cx, cat_y, cat_w, cat_h, lighten(clr, 0.75),
         ec=clr, lw=0.7, ls=(0, (4, 3)), zorder=1)
    accent_bar(ax, cx, cat_y, cat_h, clr)

    # header
    hdr_h = 0.26
    hdr_top = cat_y + cat_h - hdr_h - 0.06
    rbox(ax, cx + 0.06, hdr_top, cat_w - 0.12, hdr_h, clr, title,
         fs=8, ec="white", lw=0)

    # pills — single-column, vertically centred in available zone
    pill_h = 0.28       # FIXED: was 0.24
    pill_gap_v = 0.08   # FIXED: was 0.06
    pw = cat_w - 0.16
    px = cx + 0.08

    zone_top = hdr_top - 0.08
    zone_bot = cat_y + 0.08
    zone_h = zone_top - zone_bot
    n = len(items)
    block_h = n * pill_h + (n - 1) * pill_gap_v
    block_top = zone_bot + zone_h / 2 + block_h / 2

    for j, label in enumerate(items):
        py = block_top - j * (pill_h + pill_gap_v) - pill_h
        rbox(ax, px, py, pw, pill_h, "#FFFFFF",
             label, ec=clr, lw=0.7,
             fs=7, bold=True, tc=TEXT_D, zorder=4)

    # arrow from registry → category
    varr(ax, cx + cat_w / 2, reg_y - 0.02, cat_y + cat_h + 0.02,
         c=C_MET, lw=1.1, ms=11)

    cx_pos += cat_w + cat_gap

# ════════════════════════════════════════════════════════════════════
# ROW 3 — EvaluationRegistry banner + evaluator cards (3×2 grid)
# ════════════════════════════════════════════════════════════════════
eval_banner_h = 0.44
eval_banner_y = cat_y - GAP_CAT_EVAL - eval_banner_h

# arrow: metrics → evaluators (simple italic label)
arrow_top = cat_y - 0.04
arrow_bot = eval_banner_y + eval_banner_h + 0.04
arrow_mid = (arrow_top + arrow_bot) / 2
varr(ax, MID_X, arrow_top, arrow_bot, c="#666666", lw=1.4, ms=13)
flow_label(ax, MID_X + 0.14, arrow_mid, "metrics used by evaluators")

rbox(ax, STACK_L, eval_banner_y, STACK_W, eval_banner_h, C_EVAL,
     "EvaluationRegistry",
     sub="@register decorator  \u2022  ModelEvaluator base class  "
         "\u2022  Multi-alias resolution (e.g. ET \u2192 evapotranspiration, et)",
     lo=0.09, so=0.11, fs=10, sfs=6.8)

section_label(ax, eval_banner_y + eval_banner_h / 2, "\u00a73.7.2\nEvaluators", C_EVAL)

# evaluator cards — 2 themed groups side by side
ev_pad = 0.04
ev_card_gap = 0.20
ev_usable_w = STACK_W - 2 * ev_pad - ev_card_gap
ev_w = ev_usable_w / 2

# Card height: hdr(0.26)+top_pad(0.06)+gap(0.08)+3pills(3×0.28+2×0.08=1.00)+bot(0.08)=1.48
ev_pill_h = 0.28
ev_pill_gap = 0.08
ev_hdr_h = 0.26
ev_h = 1.48
ev_y_top = eval_banner_y - GAP_EVAL_CARDS

eval_groups = [
    ("Surface Water", C_EVAL, [
        ("Streamflow", "Unit conversion \u2022 Spatial aggregation"),
        ("Evapotranspiration", "MODIS \u2022 FLUXCOM \u2022 FluxNet \u2022 GLEAM"),
        ("Snow", "SWE & SCA metrics \u2022 MODIS / Landsat"),
    ]),
    ("Sub-surface & Storage", C_EVAL, [
        ("Soil Moisture", "SMAP \u2022 ESA CCI \u2022 ISMN towers"),
        ("Groundwater", "Well-based & GRACE modes"),
        ("Total Water Storage", "GRACE satellite \u2022 \u03a3 storage components"),
    ]),
]

for i, (theme, clr, evals) in enumerate(eval_groups):
    ex = STACK_L + ev_pad + i * (ev_w + ev_card_gap)
    ey = ev_y_top - ev_h

    # dashed container
    rbox(ax, ex, ey, ev_w, ev_h, lighten(clr, 0.75),
         ec=clr, lw=0.7, ls=(0, (4, 3)), zorder=1)
    accent_bar(ax, ex, ey, ev_h, clr)

    # theme header
    rbox(ax, ex + 0.06, ey + ev_h - ev_hdr_h - 0.06,
         ev_w - 0.12, ev_hdr_h, clr, theme,
         fs=8, ec="white", lw=0)

    # evaluator pills — name bold + detail italic on one line
    pw = ev_w - 0.16
    px = ex + 0.08
    zone_top = ey + ev_h - ev_hdr_h - 0.06 - 0.08
    n = len(evals)
    block_h = n * ev_pill_h + (n - 1) * ev_pill_gap
    pill_top = zone_top - (zone_top - ey - 0.08) / 2 + block_h / 2

    for j, (name, detail) in enumerate(evals):
        py = pill_top - j * (ev_pill_h + ev_pill_gap) - ev_pill_h
        rbox(ax, px, py, pw, ev_pill_h, "#FFFFFF",
             ec=clr, lw=0.7, zorder=4)
        # bold name on left, italic detail on right
        ax.text(px + 0.10, py + ev_pill_h / 2, name,
                ha="left", va="center", fontsize=7,
                fontweight="bold", color=TEXT_D, zorder=5)
        ax.text(px + pw - 0.10, py + ev_pill_h / 2, detail,
                ha="right", va="center", fontsize=6,
                color=TEXT_G, fontstyle="italic", zorder=5)

    # arrow from registry
    varr(ax, ex + ev_w / 2, eval_banner_y - 0.02, ey + ev_h + 0.02,
         c=C_EVAL, lw=1.0, ms=11)

# Remember bottom of evaluator cards for layout
ev_y_bottom = ev_y_top - ev_h

# ════════════════════════════════════════════════════════════════════
# ROW 4 — Three bottom panels: Vis | Bench | Calibration
# ════════════════════════════════════════════════════════════════════
bot_gap = 0.20          # gap between each pair of panels
bot_pad = 0.04
bot_usable = STACK_W - 2 * bot_pad - 2 * bot_gap   # 2 gaps for 3 panels
bot_w = bot_usable / 3  # ≈ 3.98
bot_h = 3.20            # slightly taller for narrower panels
bot_y = ev_y_bottom - GAP_CARDS_BOT - bot_h

# ── Three-way flow arrow split ──────────────────────────────────────
arr2_top = ev_y_bottom - 0.04
varr(ax, MID_X, arr2_top, arr2_top - 0.18, c="#666666", lw=1.4, ms=13)
flow_label(ax, MID_X + 0.14, arr2_top - 0.10,
           "results feed visualization, benchmarking & calibration")

vis_x   = STACK_L + bot_pad
bench_x = vis_x + bot_w + bot_gap
calib_x = bench_x + bot_w + bot_gap
left_cx   = vis_x + bot_w / 2
centre_cx = bench_x + bot_w / 2
right_cx  = calib_x + bot_w / 2
split_y   = arr2_top - 0.32

# central stem
ax.plot([MID_X, MID_X], [arr2_top - 0.18, split_y],
        color="#666666", lw=1.2, zorder=5, solid_capstyle="round")
# horizontal bar spanning all three panels
ax.plot([left_cx, right_cx], [split_y, split_y],
        color="#666666", lw=0.9, zorder=5, solid_capstyle="round")
# three arrows down
varr(ax, left_cx,   split_y, bot_y + bot_h + 0.02, c=C_VIS,   lw=1.2, ms=12)
varr(ax, centre_cx, split_y, bot_y + bot_h + 0.02, c=C_BENCH, lw=1.2, ms=12)
varr(ax, right_cx,  split_y, bot_y + bot_h + 0.02, c=C_CALIB, lw=1.2, ms=12)

# ── LEFT: Visualization & Diagnostics ─────────────────────────────
rbox(ax, vis_x, bot_y, bot_w, bot_h, lighten(C_VIS, 0.85),
     ec=C_VIS, lw=1.0, ls=(0, (4, 3)), zorder=0)
accent_bar(ax, vis_x, bot_y, bot_h, C_VIS)

vis_hdr_h = 0.28
rbox(ax, vis_x + 0.10, bot_y + bot_h - vis_hdr_h - 0.08,
     bot_w - 0.20, vis_hdr_h, C_VIS,
     "Visualization & Diagnostics", fs=8, ec="white", lw=0)

section_label(ax, bot_y + bot_h * 0.78, "\u00a73.7.3\nVisualization", C_VIS)

# sub-column widths (narrower panel — adjusted)
inner_pad  = 0.10
sub_gap    = 0.10
sub_col_w  = (bot_w - 2 * inner_pad - sub_gap) / 2
plotter_x  = vis_x + inner_pad
panel_x    = plotter_x + sub_col_w + sub_gap

# column headers
col_hdr_y = bot_y + bot_h - vis_hdr_h - 0.24
ax.text(plotter_x + sub_col_w / 2, col_hdr_y,
        "Plotter Classes (12)", ha="center", va="center",
        fontsize=6.5, fontweight="bold", color=TEXT_G)
ax.text(panel_x + sub_col_w / 2, col_hdr_y,
        "Reusable Panels (7)", ha="center", va="center",
        fontsize=6.5, fontweight="bold", color=TEXT_G)

plotters = [
    "DomainPlotter",
    "ForcingComparison\u2026",
    "OptimizationPlotter",
    "ModelComparison\u2026",
    "BenchmarkPlotter",
    "AnalysisPlotter",
    "WorkflowDiagnostic\u2026",
]
panels = [
    "TimeSeriesPanel",
    "FDCPanel",
    "ScatterPanel",
    "MetricsTablePanel",
    "MonthlyBoxplotPanel",
    "ResidualAnalysis\u2026",
    "SpatialPanel",
]

pp_h   = 0.22           # reduced from default for narrower panel
pp_gap = 0.04
pp_top = col_hdr_y - 0.16

for col_items, col_x, col_w in [(plotters, plotter_x, sub_col_w),
                                  (panels,   panel_x,   sub_col_w)]:
    py = pp_top
    for label in col_items:
        rbox(ax, col_x, py - pp_h, col_w, pp_h, "#FFFFFF",
             label, ec=C_VIS, lw=0.5, fs=6.5, bold=False, tc=TEXT_D, zorder=4)
        py -= pp_h + pp_gap

# "+5 additional" badge below plotter list
badge_y = pp_top - len(plotters) * (pp_h + pp_gap) - 0.02
rbox(ax, plotter_x + 0.06, badge_y - 0.20, sub_col_w - 0.12, 0.20,
     lighten(C_VIS, 0.40),
     "+5 additional", ec=C_VIS, lw=0.5,
     fs=6.5, bold=False, tc=TEXT_W, zorder=4)

# compose arrow between columns
compose_y = pp_top - 3 * (pp_h + pp_gap)
harr(ax, plotter_x + sub_col_w + 0.02, panel_x - 0.02,
     compose_y, c=C_VIS, lw=0.9, st="<|-|>", ms=9)
ax.text((plotter_x + sub_col_w + panel_x) / 2, compose_y + 0.10,
        "compose", ha="center", va="center",
        fontsize=6.5, color=TEXT_G, fontstyle="italic")

# output note
ax.text(vis_x + bot_w / 2, bot_y + 0.14,
        "PNG 300 DPI  \u2022  Structured output  \u2022  Consistent styling",
        ha="center", va="center", fontsize=6.5,
        color=TEXT_L, fontstyle="italic", zorder=4)

# ── CENTRE: Benchmarking & Sensitivity ──────────────────────────────
rbox(ax, bench_x, bot_y, bot_w, bot_h, lighten(C_BENCH, 0.85),
     ec=C_BENCH, lw=1.0, ls=(0, (4, 3)), zorder=0)
accent_bar(ax, bench_x, bot_y, bot_h, C_BENCH)

rbox(ax, bench_x + 0.10, bot_y + bot_h - vis_hdr_h - 0.08,
     bot_w - 0.20, vis_hdr_h, C_BENCH,
     "Benchmarking & Sensitivity", fs=8, ec="white", lw=0)

section_label(ax, bot_y + bot_h * 0.50, "\u00a73.7.4\nBenchmarking", C_BENCH)

# Three sub-sections (adjusted for narrower panel)
bench_sections = [
    ("Benchmarker", [
        ("Time-invariant", "Mean, Median, Annual"),
        ("Time-variant", "Monthly, Daily climatol."),
        ("Rainfall-runoff", "Long & short-term ratio"),
    ]),
    ("Sensitivity\nAnalyzer", [
        ("Sobol\u2019 indices", "Total-order var. decomp."),
        ("Param. ranking", "Fix low-sensitivity params"),
        ("Uncertainty", "Min ~60 samples required"),
    ]),
    ("Structure\nEnsemble", [
        ("Generate combos", "All structural choices"),
        ("Execute & eval.", "Full metric comparison"),
        ("Quantify uncert.", "Structure\u2013performance"),
    ]),
]

sec_x = bench_x + 0.10
sec_w = bot_w - 0.20
sec_h = 0.60            # balanced for 3 sections in narrower panel

bench_zone_top = bot_y + bot_h - vis_hdr_h - 0.18
bench_zone_bot = bot_y + 0.30
n_sec = len(bench_sections)
total_sec_h = n_sec * sec_h
avail = bench_zone_top - bench_zone_bot - total_sec_h
sec_gap = avail / (n_sec - 1) if n_sec > 1 else 0
sec_gap = min(sec_gap, 0.18)

actual_block = total_sec_h + (n_sec - 1) * sec_gap
block_start = bench_zone_bot + (bench_zone_top - bench_zone_bot - actual_block) / 2 + actual_block

for idx, (title, items) in enumerate(bench_sections):
    sec_y = block_start - idx * (sec_h + sec_gap) - sec_h

    # header block (left)
    hdr_w = sec_w * 0.26
    rbox(ax, sec_x, sec_y, hdr_w, sec_h, C_BENCH,
         title, fs=6.5, ec="white", lw=0, zorder=4)

    # item pills (right)
    pz_x = sec_x + hdr_w + 0.06
    pz_w = sec_w - hdr_w - 0.06
    n = len(items)
    ph = 0.15            # slightly smaller for narrower panel
    pg = 0.04
    total_pill = n * ph + (n - 1) * pg
    py = sec_y + sec_h / 2 + total_pill / 2 - ph

    for label, detail in items:
        rbox(ax, pz_x, py, pz_w, ph, "#FFFFFF",
             f"{label}  \u2014  {detail}",
             ec=C_BENCH, lw=0.5, fs=6.5, bold=False, tc=TEXT_D, zorder=4)
        py -= ph + pg

# registry note
ax.text(bench_x + bot_w / 2, bot_y + 0.14,
        "AnalysisRegistry  \u2022  @register  \u2022  Model fallback",
        ha="center", va="center", fontsize=6.5,
        color=TEXT_L, fontstyle="italic", zorder=4)

# ── RIGHT: Calibration Integration (§3.6) ──────────────────────────
rbox(ax, calib_x, bot_y, bot_w, bot_h, lighten(C_CALIB, 0.85),
     ec=C_CALIB, lw=1.0, ls=(0, (4, 3)), zorder=0)
accent_bar(ax, calib_x, bot_y, bot_h, C_CALIB)

rbox(ax, calib_x + 0.10, bot_y + bot_h - vis_hdr_h - 0.08,
     bot_w - 0.20, vis_hdr_h, C_CALIB,
     "Calibration Integration", fs=8, ec="white", lw=0)

section_label(ax, bot_y + bot_h * 0.22, "\u00a73.6\nCalibration", C_CALIB)

# Three sub-sections for calibration
calib_sections = [
    ("Objective\nRegistry", [
        ("Metric \u2192 cost", "cost = 1 \u2212 metric"),
        ("KGE, NSE, RMSE, MAE", "Standard objective set"),
        ("Multi-alias resolution", "Consistent interface"),
    ]),
    ("Multi-Variable\nTargets", [
        ("Weighted objectives", "\u03a3 w\u1d62 \u00d7 cost\u1d62"),
        ("Q + ET + Snow + SM", "Joint calibration"),
        ("User-configurable", "Per-target weights"),
    ]),
    ("Final\nEvaluation", [
        ("Calib. vs Validation", "Split-period comparison"),
        ("Full metric suite", "All registered metrics"),
        ("Diagnostic plots", "Time-series, FDC, scatter"),
    ]),
]

cal_sec_x = calib_x + 0.10
cal_sec_w = bot_w - 0.20
cal_sec_h = 0.60        # match benchmarking — now 3 sections

calib_zone_top = bot_y + bot_h - vis_hdr_h - 0.18
calib_zone_bot = bot_y + 0.30
n_cal = len(calib_sections)
total_cal_h = n_cal * cal_sec_h
cal_avail = calib_zone_top - calib_zone_bot - total_cal_h
cal_sec_gap = cal_avail / (n_cal - 1) if n_cal > 1 else 0
cal_sec_gap = min(cal_sec_gap, 0.24)

cal_block = total_cal_h + (n_cal - 1) * cal_sec_gap
cal_block_start = calib_zone_bot + (calib_zone_top - calib_zone_bot - cal_block) / 2 + cal_block

for idx, (title, items) in enumerate(calib_sections):
    sec_y = cal_block_start - idx * (cal_sec_h + cal_sec_gap) - cal_sec_h

    # header block (left) — same ratio as benchmarking
    hdr_w = cal_sec_w * 0.26
    rbox(ax, cal_sec_x, sec_y, hdr_w, cal_sec_h, C_CALIB,
         title, fs=6.5, ec="white", lw=0, zorder=4)

    # item pills (right) — match bench pill dims
    pz_x = cal_sec_x + hdr_w + 0.06
    pz_w = cal_sec_w - hdr_w - 0.06
    n = len(items)
    ph = 0.15            # match benchmarking pill height
    pg = 0.04            # match benchmarking pill gap
    total_pill = n * ph + (n - 1) * pg
    py = sec_y + cal_sec_h / 2 + total_pill / 2 - ph

    for label, detail in items:
        rbox(ax, pz_x, py, pz_w, ph, "#FFFFFF",
             f"{label}  \u2014  {detail}",
             ec=C_CALIB, lw=0.5, fs=6.5, bold=False, tc=TEXT_D, zorder=4)
        py -= ph + pg

# calibration bottom note
ax.text(calib_x + bot_w / 2, bot_y + 0.14,
        "Feeds \u00a73.6 calibration loop  \u2022  Cost minimisation",
        ha="center", va="center", fontsize=6.5,
        color=TEXT_L, fontstyle="italic", zorder=4)

# ── crop axes to content extent, then save ──────────────────────────
content_bot = bot_y - 0.15
content_top = FIG_H
ax.set_ylim(content_bot, content_top)

out = Path(__file__).resolve().parent
for fmt in ("pdf", "png"):
    fig.savefig(out / f"fig10_analysis_layer.{fmt}",
                dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.15)
    print(f"Saved fig10_analysis_layer.{fmt}")
plt.close(fig)
