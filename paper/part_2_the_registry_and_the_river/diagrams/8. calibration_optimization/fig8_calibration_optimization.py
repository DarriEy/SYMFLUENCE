"""
SYMFLUENCE Calibration and Optimization Framework (Section 3.6).

Shows the OptimizationManager orchestrating decoupled algorithm, objective,
and parallelisation components through the calibration loop, with normalised
parameter space and results persistence.

Verified against src/symfluence/optimisation/ and
src/symfluence/project/optimization_manager.py.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── layout ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 14.8, 9.8
STACK_L = 1.40
STACK_R = 11.60
STACK_W = STACK_R - STACK_L
MID_X   = (STACK_L + STACK_R) / 2
PAD     = 0.04

# ── colours — consistent with paper palette ─────────────────────────
C_MGR   = "#C9943A"   # gold   — OptimizationManager
C_ALGO  = "#4A7FB5"   # blue   — algorithms
C_OBJ   = "#5BA58B"   # green  — objectives
C_TGT   = "#8B6DAF"   # purple — calibration targets
C_PAR   = "#8B6DAF"   # purple — execution distribution (sidebar)
C_LOOP  = "#FAFAFA"   # loop background
C_STEP  = "#C0392B"   # red    — loop steps
C_NORM  = "#2E86AB"   # teal   — parameter normalisation
C_RES   = "#F5F0E6"   # result artefacts

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


def harr(ax, x1, x2, y, c=CLR_A, lw=1.3, st="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle=st, color=c, lw=lw,
        zorder=5, mutation_scale=ms))


def varr(ax, x, y1, y2, c=CLR_A, lw=1.3, st="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(
        (x, y1), (x, y2), arrowstyle=st, color=c, lw=lw,
        zorder=5, mutation_scale=ms))


# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_clip_on(False)
ax.axis("off")

# ── title ───────────────────────────────────────────────────────────
ax.text(MID_X, FIG_H - 0.28,
        "Calibration and Optimization Framework",
        ha="center", va="center", fontsize=13,
        fontweight="bold", color=TEXT_D)

# ════════════════════════════════════════════════════════════════════
# ROW 1 — OptimizationManager
# ════════════════════════════════════════════════════════════════════
mgr_h = 0.58
mgr_y = FIG_H - 1.10

rbox(ax, STACK_L, mgr_y, STACK_W, mgr_h, C_MGR,
     "OptimizationManager",
     sub="Orchestrates calibration workflows via OptimizerRegistry",
     lo=0.10, so=0.12)

# ════════════════════════════════════════════════════════════════════
# ROW 2 — Three decoupled component columns
# ════════════════════════════════════════════════════════════════════
col_gap = 0.28
col_pad = 0.10
col_usable = STACK_W - 2 * col_pad - 2 * col_gap
col_w = col_usable / 3
col_top = mgr_y - 0.40
col_h = 2.60
col_y = col_top - col_h

columns = [
    (C_ALGO, "Algorithm Library", [
        ("Local Search", "DDS"),
        ("Population", "PSO, DE, SCE-UA, GA"),
        ("Gradient", "ADAM, L-BFGS"),
        ("Multi-Objective", "NSGA-II, MOEA/D"),
        ("Bayesian / MCMC", "DREAM, ABC"),
    ]),
    (C_OBJ, "ObjectiveRegistry", [
        ("Efficiency", "KGE, KGE', NSE"),
        ("Error", "RMSE, MAE, PBIAS"),
        ("Correlation", "R\u00b2, Spearman"),
        ("Transform", "cost = 1 \u2212 metric"),
    ]),
    (C_TGT, "Calibration Targets", [
        ("Streamflow", "Model-specific overrides"),
        ("Snow / SWE", "SWE & snow-covered area"),
        ("ET", "MODIS, FLUXCOM, FluxNet"),
        ("Soil Moisture", "SMAP, CCI, ISMN"),
        ("Multivariate", "Weighted multi-variable"),
    ]),
]

for i, (clr, title, items) in enumerate(columns):
    cx = STACK_L + col_pad + i * (col_w + col_gap)

    # column background
    rbox(ax, cx, col_y, col_w, col_h, lighten(clr, 0.65),
         ec=clr, lw=1.0, zorder=1)

    # column header
    header_h = 0.36
    rbox(ax, cx + 0.06, col_y + col_h - header_h - 0.08,
         col_w - 0.12, header_h, clr, title,
         fs=8.5, ec="white", lw=0)

    # item pills
    pill_h = 0.32
    pill_gap_y = 0.10
    pill_x = cx + 0.10
    pill_w = col_w - 0.20
    py = col_y + col_h - header_h - 0.22 - pill_h

    for label, detail in items:
        rbox(ax, pill_x, py, pill_w, pill_h, "#FFFFFF",
             label, sub=detail, ec=clr, lw=0.6,
             fs=7.2, sfs=6.0, bold=True, tc=TEXT_D,
             lo=0.06, so=0.07, zorder=4)
        py -= pill_h + pill_gap_y

    # arrow from manager
    col_cx = cx + col_w / 2
    varr(ax, col_cx, mgr_y - 0.02, col_y + col_h + 0.02,
         c=clr, lw=1.8, ms=16)

# ════════════════════════════════════════════════════════════════════
# ROW 3 — Parameter Normalisation bar
# ════════════════════════════════════════════════════════════════════
norm_h = 0.52
norm_y = col_y - 0.38 - norm_h

rbox(ax, STACK_L, norm_y, STACK_W, norm_h, C_NORM,
     "BaseParameterManager  \u2014  Normalised [0, 1] Parameter Space",
     sub="Bounds from model sources \u2022 Bidirectional transform \u2022 Algorithm-portable",
     lo=0.10, so=0.12, fs=9, sfs=7)

# arrow: columns → normalisation
varr(ax, MID_X, col_y - 0.02, norm_y + norm_h + 0.02, c=C_NORM, lw=1.8, ms=16)

# ════════════════════════════════════════════════════════════════════
# ROW 4 — Calibration Loop
# ════════════════════════════════════════════════════════════════════
loop_h = 1.50
loop_y = norm_y - 0.38 - loop_h

# loop background
rbox(ax, STACK_L, loop_y, STACK_W, loop_h, C_LOOP,
     ec=CLR_B, lw=1.0, ls=(0, (5, 3)), zorder=0)

ax.text(STACK_L + 0.18, loop_y + loop_h - 0.14,
        "Calibration Loop",
        ha="left", va="center", fontsize=9,
        fontweight="bold", color=TEXT_G, fontstyle="italic", zorder=1)

# loop steps — circular flow
steps = [
    "Propose\nCandidate(s)",
    "Denormalize\nParameters",
    "Execute\nModel",
    "Evaluate\nObjective",
    "Update\nAlgorithm State",
]
n_steps = len(steps)
step_w = 1.50
step_h = 0.78
step_gap = 0.20
total_steps_w = n_steps * step_w + (n_steps - 1) * step_gap
step_x0 = STACK_L + (STACK_W - total_steps_w) / 2
step_y = loop_y + 0.22

step_colors = [C_ALGO, C_NORM, "#7A8B99", C_OBJ, C_ALGO]

for i, (label, clr) in enumerate(zip(steps, step_colors)):
    sx = step_x0 + i * (step_w + step_gap)
    rbox(ax, sx, step_y, step_w, step_h, clr, label,
         fs=7.8, ec="white", lw=1.2, zorder=4)
    # step number badge
    bw, bh = 0.22, 0.18
    rbox(ax, sx + 0.06, step_y + step_h - bh - 0.06,
         bw, bh, lighten(clr, 0.20), str(i + 1),
         ec="white", lw=0, fs=6.5, tc=TEXT_W, zorder=5)

# arrows between steps
for i in range(n_steps - 1):
    x1 = step_x0 + i * (step_w + step_gap) + step_w + 0.04
    x2 = step_x0 + (i + 1) * (step_w + step_gap) - 0.04
    harr(ax, x1, x2, step_y + step_h / 2, c="#555555", lw=1.5, ms=13)

# return arrow (step 5 back to step 1) — manual L-shaped path
ret_margin = 0.18
ret_y_line = step_y + step_h + ret_margin
x_last_center = step_x0 + (n_steps - 1) * (step_w + step_gap) + step_w / 2
x_first_center = step_x0 + step_w / 2

# vertical up from last step
ax.plot([x_last_center, x_last_center],
        [step_y + step_h + 0.02, ret_y_line],
        color="#555555", lw=1.3, zorder=5, solid_capstyle="round")
# horizontal across the top
ax.plot([x_last_center, x_first_center],
        [ret_y_line, ret_y_line],
        color="#555555", lw=1.3, zorder=5, solid_capstyle="round")
# arrowhead back down into first step
ax.add_patch(FancyArrowPatch(
    (x_first_center, ret_y_line),
    (x_first_center, step_y + step_h + 0.02),
    arrowstyle="-|>", color="#555555", lw=1.3,
    zorder=5, mutation_scale=14))

ax.text(MID_X, ret_y_line + 0.10,
        "iterate until convergence or budget exhaustion",
        ha="center", va="center", fontsize=6.5, color=TEXT_G,
        fontstyle="italic", zorder=6)

# arrow: normalisation → loop
varr(ax, MID_X, norm_y - 0.02, loop_y + loop_h + 0.02, c="#666666", lw=1.8, ms=16)

# ════════════════════════════════════════════════════════════════════
# ROW 5 — Final Evaluation & Results
# ════════════════════════════════════════════════════════════════════
res_h = 0.72
res_y = loop_y - 0.38 - res_h

# background
rbox(ax, STACK_L, res_y, STACK_W, res_h, C_RES,
     ec=CLR_B, lw=0.8, zorder=1)

ax.text(STACK_L + 0.18, res_y + res_h - 0.12,
        "Final Evaluation & Results Persistence",
        ha="left", va="center", fontsize=8.5,
        fontweight="bold", color=TEXT_G, zorder=2)

result_items = [
    "Best Parameters\n(JSON)",
    "Full-Period\nSimulation",
    "Calibration vs\nValidation Metrics",
    "Convergence\nTrajectory",
    "Optimization\nHistory (CSV)",
]
ri_gap = 0.14
ri_pad = 0.16
ri_usable = STACK_W - 2 * ri_pad
ri_w = (ri_usable - (len(result_items) - 1) * ri_gap) / len(result_items)
ri_h = 0.38
ri_y = res_y + 0.08
ri_x0 = STACK_L + ri_pad

for i, label in enumerate(result_items):
    rx = ri_x0 + i * (ri_w + ri_gap)
    rbox(ax, rx, ri_y, ri_w, ri_h, "#FFFFFF", label,
         ec=CLR_B, lw=0.7, fs=6.5, bold=False, tc=TEXT_G, zorder=4)

# arrow: loop → results
varr(ax, MID_X, loop_y - 0.02, res_y + res_h + 0.02, c="#666666", lw=1.8, ms=16)
ax.text(MID_X + 0.14, (loop_y + res_y + res_h) / 2,
        "best solution", ha="left", va="center",
        fontsize=7.5, color=TEXT_G, fontstyle="italic")

# ════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Execution Distribution & Process Isolation
# Full-height sidebar spanning all main rows
# ════════════════════════════════════════════════════════════════════
C_SIDE = "#7A8B99"
side_x = STACK_R + 0.38
side_w = 1.70
side_top = mgr_y + mgr_h + 0.06       # align with top of manager
side_bot = res_y - 0.06               # align with bottom of results
side_h = side_top - side_bot

# outer container
rbox(ax, side_x, side_bot, side_w, side_h, lighten(C_SIDE, 0.78),
     ec=C_SIDE, lw=1.2, zorder=0)

# ── title header ───────────────────────────────────────────────────
th_h = 0.38
rbox(ax, side_x + 0.08, side_top - th_h - 0.10,
     side_w - 0.16, th_h, C_SIDE, "Execution Distribution",
     fs=8.5, ec="white", lw=0)

# ── process isolation description (subtitle area) ──────────────────
desc_y = side_top - th_h - 0.24
ax.text(side_x + side_w / 2, desc_y - 0.08,
        "Process-Isolated\nParallel Evaluation",
        ha="center", va="center", fontsize=6.8,
        fontweight="bold", color=TEXT_G, zorder=2, clip_on=False,
        linespacing=1.45)
ax.text(side_x + side_w / 2, desc_y - 0.42,
        "Each candidate evaluation runs\nin a dedicated directory with\nisolated config and output files",
        ha="center", va="center", fontsize=6.0,
        color=TEXT_L, zorder=2, linespacing=1.5, clip_on=False)

# thin separator
sep1_y = desc_y - 0.60
ax.plot([side_x + 0.14, side_x + side_w - 0.14], [sep1_y, sep1_y],
        color=C_SIDE, lw=0.5, alpha=0.5, zorder=2, clip_on=False)

# ── three execution strategy pills (centered vertically) ──────────
strat_data = [
    ("MPI Strategy", "Distributed memory\nHPC clusters"),
    ("ProcessPool", "Shared memory\nMulti-core workstations"),
    ("Sequential", "Single process\nFallback / debugging"),
]
sp_w = side_w - 0.28
sp_h = 0.52
sp_gap = 0.14
total_strat_h = len(strat_data) * sp_h + (len(strat_data) - 1) * sp_gap
# center the pills between the separators
strat_zone_top = sep1_y - 0.14
strat_zone_bot = side_bot + 0.60
strat_center = (strat_zone_top + strat_zone_bot) / 2
sp_y_start = strat_center + total_strat_h / 2 - sp_h
sp_x = side_x + 0.14

for label, desc in strat_data:
    rbox(ax, sp_x, sp_y_start, sp_w, sp_h, "#FFFFFF",
         label, sub=desc, ec=C_SIDE, lw=0.7,
         fs=7.5, sfs=5.8, bold=True, tc=TEXT_D,
         lo=0.10, so=0.10, zorder=4)
    sp_y_start -= sp_h + sp_gap

# thin separator below strategies
sep2_y = strat_zone_bot - 0.06
ax.plot([side_x + 0.14, side_x + side_w - 0.14], [sep2_y, sep2_y],
        color=C_SIDE, lw=0.5, alpha=0.5, zorder=2, clip_on=False)

# ── bottom: automatic strategy selection note ──────────────────────
ax.text(side_x + side_w / 2, sep2_y - 0.14,
        "Auto-selected at runtime",
        ha="center", va="center", fontsize=6.5,
        fontweight="bold", color=TEXT_G, zorder=2, clip_on=False)
ax.text(side_x + side_w / 2, sep2_y - 0.32,
        "MPI \u2192 ProcessPool \u2192 Sequential",
        ha="center", va="center", fontsize=6.2,
        color=TEXT_L, zorder=2, clip_on=False)

# ── dashed connector arrows from each main row to sidebar ──────────
connect_rows = [
    mgr_y + mgr_h / 2,         # OptimizationManager
    col_y + col_h / 2,         # component columns
    norm_y + norm_h / 2,       # parameter normalisation
    loop_y + loop_h / 2,       # calibration loop
    res_y + res_h / 2,         # results
]
for y_t in connect_rows:
    ax.annotate(
        "", xy=(side_x - 0.03, y_t),
        xytext=(STACK_R + 0.03, y_t),
        arrowprops=dict(arrowstyle="-|>", color=C_SIDE, lw=0.9,
                        linestyle=(0, (4, 3)), mutation_scale=10,
                        alpha=0.6),
        zorder=1, clip_on=False)

# ── save ────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent
for fmt in ("pdf", "png"):
    fig.savefig(out / f"fig8_calibration_optimization.{fmt}",
                dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.20)
    print(f"Saved fig8_calibration_optimization.{fmt}")
plt.close(fig)
