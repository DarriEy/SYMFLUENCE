"""
SYMFLUENCE user interfaces and deployment diagram (Section 3.8).

Redesigned layout featuring:
  - Numbered workflow pipeline with flow arrows showing step sequence
  - Phase-colored regions behind workflow step groups
  - Step number badges for clear sequencing
  - Wrap connector from step 8 → step 9 showing pipeline continuation
  - Compact infrastructure section
  - Three vertical colour bands for cross-cutting access modalities

Verified against the codebase at /Users/darrieythorsson/compHydro/code/SYMFLUENCE.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── global font setup ────────────────────────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

# ── global constants ─────────────────────────────────────────────────
FIG_W, FIG_H = 14.0, 11.0
PAD = 0.04

# ── colours ──────────────────────────────────────────────────────────
C_API   = "#4A7FB5"   # blue   — Python API
C_CLI   = "#5BA58B"   # green  — CLI
C_AGENT = "#8B6DAF"   # purple — AI Agent
C_CFG   = "#B08968"   # mocha  — Configuration layer
C_ORCH  = "#C9943A"   # gold   — Orchestrator
C_CI    = "#2E86AB"   # teal   — CI/CD
C_COMM  = "#D4764E"   # rust   — Community
C_DIST  = "#6B8E6B"   # sage   — Distribution

TEXT_D = "#2D2D2D"
TEXT_W = "#FFFFFF"
TEXT_G = "#555555"

# Workflow step group tints
C_GRP_DOMAIN = "#E6D5A8"   # warm sand
C_GRP_DATA   = "#C8DDB8"   # pale green
C_GRP_MODEL  = "#E0C9A0"   # wheat
C_GRP_ANAL   = "#D8C0C8"   # pale mauve


# ── colour helpers ───────────────────────────────────────────────────

def lighten(c, f=0.35):
    r, g, b = (int(c[i:i+2], 16) for i in (1, 3, 5))
    return "#{:02X}{:02X}{:02X}".format(
        int(r + (255 - r) * f), int(g + (255 - g) * f), int(b + (255 - b) * f))


def darken(c, f=0.25):
    r, g, b = (int(c[i:i+2], 16) for i in (1, 3, 5))
    return "#{:02X}{:02X}{:02X}".format(
        int(r * (1 - f)), int(g * (1 - f)), int(b * (1 - f)))


def _dark(c):
    r, g, b = (int(c[i:i+2], 16) for i in (1, 3, 5))
    return r * 0.299 + g * 0.587 + b * 0.114 < 160


# ── drawing helpers ──────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc, label=None, sub=None, *,
         ec="white", lw=1.5, ls="-", fs=9.5, sfs=7.5,
         tc=None, zorder=5, bold=True, lo=0.14, so=0.17, alpha=1.0):
    """Rounded box primitive."""
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={PAD}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        zorder=zorder, alpha=alpha)
    ax.add_patch(patch)
    if label is None:
        return
    tc = tc or (TEXT_W if _dark(fc) else TEXT_D)
    if sub:
        ax.text(x + w / 2, y + h / 2 + lo, label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal",
                color=tc, zorder=zorder + 1)
        ax.text(x + w / 2, y + h / 2 - so, sub,
                ha="center", va="center", fontsize=sfs,
                color=tc, alpha=0.90, fontstyle="italic",
                zorder=zorder + 1)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal",
                color=tc, zorder=zorder + 1)


def varr(ax, x, y1, y2, color=TEXT_G, lw=1.2, zorder=4, head_w=0.12):
    """Vertical arrow from y1 down to y2."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle=f"->,head_width={head_w}",
                                color=color, lw=lw),
                zorder=zorder)


def harr(ax, x1, x2, y, color=TEXT_G, lw=1.0, zorder=4, head_w=0.08):
    """Horizontal arrow from x1 to x2 at height y."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle=f"->,head_width={head_w}",
                                color=color, lw=lw),
                zorder=zorder)


# ── build figure ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── layout constants ─────────────────────────────────────────────────
LEFT    = 0.70
RIGHT   = 13.30
STACK_W = RIGHT - LEFT

# Y positions (bottom to top)
L1_Y   = 0.35;   L1_H   = 0.80    # distribution row
L2_Y   = 1.40;   L2_H   = 1.30    # CI/CD + Community (compact)
SEP_Y  = 2.90                      # "supported by" separator
HERO_Y = 3.15;   HERO_H = 3.65    # workflow pipeline hero area
L4_Y   = 7.10;   L4_H   = 0.70    # orchestrator bar (more gap from hero)
CFG_Y  = 8.00;   CFG_H  = 0.55    # configuration layer
MOD_Y  = 8.80;   MOD_H  = 0.85    # modality caps

BAND_BOT = 3.00                    # below workflow area
BAND_TOP = 9.80                    # above modality caps

TITLE_Y = 10.25

# ── title ────────────────────────────────────────────────────────────
ax.text(FIG_W / 2, TITLE_Y,
        "User Interfaces, Deployment, and Community Infrastructure",
        ha="center", va="center", fontsize=15,
        fontweight="bold", color=TEXT_D)

ax.text(FIG_W / 2, TITLE_Y - 0.30,
        "Access Modalities",
        ha="center", va="center", fontsize=10.5,
        fontstyle="italic", color=TEXT_G)

# ════════════════════════════════════════════════════════════════════
# THREE VERTICAL COLOUR BANDS (behind workflow + orchestrator + config)
# ════════════════════════════════════════════════════════════════════
band_w = STACK_W * 0.30
band_gap = STACK_W * 0.05
band_total = 3 * band_w + 2 * band_gap
band_x0 = LEFT + (STACK_W - band_total) / 2

band_colors = [C_API, C_CLI, C_AGENT]

for i, bc in enumerate(band_colors):
    bx = band_x0 + i * (band_w + band_gap)
    patch = FancyBboxPatch(
        (bx, BAND_BOT), band_w, BAND_TOP - BAND_BOT,
        boxstyle=f"round,pad={PAD}",
        facecolor=bc, edgecolor="none", alpha=0.08,
        zorder=0.5)
    ax.add_patch(patch)

# ════════════════════════════════════════════════════════════════════
# MODALITY CAPS (top row) + arrows down into config layer
# ════════════════════════════════════════════════════════════════════
mod_w = 3.2
mod_data = [
    (C_API,   "Python API",  "SYMFLUENCE class\nfrom_file \u00b7 from_preset \u00b7 from_minimal"),
    (C_CLI,   "CLI",         "7 command categories\n16 steps \u00b7 Rich output"),
    (C_AGENT, "AI Agent",    "AgentManager\nmulti-provider \u00b7 50+ tools"),
]

for i, (mc, mn, ms) in enumerate(mod_data):
    mx = band_x0 + i * (band_w + band_gap) + (band_w - mod_w) / 2
    rbox(ax, mx, MOD_Y, mod_w, MOD_H, mc, mn, sub=ms,
         fs=11.5, sfs=7.5, lo=0.22, so=0.22, ec="white", lw=1.5,
         tc=TEXT_W, zorder=6)
    # Arrow from cap down to config layer
    varr(ax, mx + mod_w / 2, MOD_Y - 0.04, CFG_Y + CFG_H + 0.04,
         color=mc, lw=1.4, zorder=5)

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION LAYER
# ════════════════════════════════════════════════════════════════════
rbox(ax, LEFT, CFG_Y, STACK_W, CFG_H, C_CFG,
     "SymfluenceConfig",
     sub="346+ params  \u2022  YAML / preset / minimal factories  \u2022  LazyManagerDict",
     fs=11.5, sfs=7.5, lo=0.12, so=0.14, ec="white", lw=1.5, tc=TEXT_W, zorder=3)

# Arrow from config down to orchestrator
varr(ax, FIG_W / 2, CFG_Y - 0.04, L4_Y + L4_H + 0.04,
     color=C_CFG, lw=1.6, zorder=5, head_w=0.15)

# ════════════════════════════════════════════════════════════════════
# L4: ORCHESTRATOR BAR
# ════════════════════════════════════════════════════════════════════
rbox(ax, LEFT, L4_Y, STACK_W, L4_H, C_ORCH,
     "WorkflowOrchestrator",
     sub="Step sequencing  \u2022  Dependency resolution  \u2022  State persistence  \u2022  Caching",
     fs=11.5, sfs=8, lo=0.14, so=0.16, ec="white", lw=1.5, tc=TEXT_W, zorder=3)

ax.text(LEFT - 0.35, L4_Y + L4_H / 2, "L4",
        ha="center", va="center", fontsize=10,
        fontweight="bold", color=C_ORCH, zorder=7)

# ════════════════════════════════════════════════════════════════════
# L3: WORKFLOW PIPELINE — numbered steps with flow arrows
# ════════════════════════════════════════════════════════════════════

# Dashed container
rbox(ax, LEFT, HERO_Y, STACK_W, HERO_H, lighten(C_ORCH, 0.55),
     ec=C_ORCH, lw=1.2, ls="--", zorder=2, alpha=0.45)

# Title inside top of container
ax.text(LEFT + STACK_W / 2, HERO_Y + HERO_H - 0.16,
        "Workflow Steps (16)",
        ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=darken(C_ORCH, 0.15), zorder=4)

ax.text(LEFT - 0.35, HERO_Y + HERO_H / 2, "L3",
        ha="center", va="center", fontsize=10,
        fontweight="bold", color=C_ORCH, zorder=7)

# 16 step labels (two lines each)
steps = [
    "setup\nproject", "create\npour point", "acquire\nattributes",
    "define\ndomain", "discretize\ndomain", "process\nobserved",
    "acquire\nforcings", "agnostic\npreproc",
    "specific\npreproc", "run\nmodel", "calibrate\nmodel",
    "run\nemulation", "run\nbenchmark", "decision\nanalysis",
    "sensitivity\nanalysis", "post-\nprocess",
]

# Grid parameters
sp_pad   = 0.28     # padding inside hero container
sp_gap_x = 0.22     # horizontal gap (room for flow arrows)
sp_usable_w = STACK_W - 2 * sp_pad
sp_w = (sp_usable_w - 7 * sp_gap_x) / 8
sp_h = 0.90         # taller pills for readability

# Row positions — row 2 at bottom, row 1 at top
row_gap = 0.95       # vertical gap between rows (for labels + wrap arrow)
y_row2 = HERO_Y + 0.25                        # bottom row (steps 9-16)
y_row1 = y_row2 + sp_h + row_gap              # top row (steps 1-8)
row_ys = [y_row2, y_row1]                      # index 0 = bottom, 1 = top

# ── Phase group backgrounds and labels ──
# (row_index, start_col, n_cols, colour, label)
groups = [
    (1, 0, 5, C_GRP_DOMAIN, "Domain Setup"),
    (1, 5, 3, C_GRP_DATA,   "Data Acquisition"),
    (0, 0, 4, C_GRP_MODEL,  "Modeling"),
    (0, 4, 4, C_GRP_ANAL,   "Analysis"),
]

grp_inset = 0.08
for row, col0, ncols, gc, glabel in groups:
    gx = LEFT + sp_pad + col0 * (sp_w + sp_gap_x) - grp_inset
    gy = row_ys[row] - grp_inset
    gw = ncols * sp_w + (ncols - 1) * sp_gap_x + 2 * grp_inset
    gh = sp_h + 2 * grp_inset
    rbox(ax, gx, gy, gw, gh, gc,
         ec="none", lw=0, zorder=3, alpha=0.55)
    # Group label above the background
    ax.text(gx + gw / 2, gy + gh + 0.12, glabel,
            ha="center", va="center", fontsize=9,
            fontweight="bold", color=darken(gc, 0.55),
            zorder=4)

# ── Step pills with number badges ──
pill_colors = [lighten(C_ORCH, 0.30), lighten(C_ORCH, 0.40),
               lighten(C_ORCH, 0.35), lighten(C_ORCH, 0.45)]
badge_color = darken(C_ORCH, 0.15)

for row in range(2):
    for col in range(8):
        idx = (1 - row) * 8 + col   # top row → 0-7, bottom row → 8-15
        if idx >= len(steps):
            break
        px = LEFT + sp_pad + col * (sp_w + sp_gap_x)
        py = row_ys[row]

        # Pill (no label — we place text manually to avoid badge)
        rbox(ax, px, py, sp_w, sp_h,
             pill_colors[idx % 4],
             ec="white", lw=0.8, zorder=4)

        # Step text (offset below center to clear the badge area)
        ax.text(px + sp_w / 2, py + sp_h * 0.40, steps[idx],
                ha="center", va="center", fontsize=8,
                color=TEXT_D, zorder=5)

        # Step number badge (top-left corner, smaller)
        bx = px + 0.13
        by = py + sp_h - 0.13
        ax.plot(bx, by, 'o', color=badge_color,
                markersize=10, zorder=5, markeredgecolor="white",
                markeredgewidth=0.5)
        ax.text(bx, by, str(idx + 1),
                ha="center", va="center", fontsize=5.5,
                fontweight="bold", color=TEXT_W, zorder=6)

# ── Flow arrows between consecutive pills ──
arrow_color = darken(C_ORCH, 0.20)

for row in range(2):
    for col in range(7):
        idx = (1 - row) * 8 + col
        if idx + 1 >= len(steps):
            break
        # Arrow from right edge of pill to left edge of next pill
        x1 = LEFT + sp_pad + col * (sp_w + sp_gap_x) + sp_w + 0.03
        x2 = LEFT + sp_pad + (col + 1) * (sp_w + sp_gap_x) - 0.03
        y  = row_ys[row] + sp_h / 2
        harr(ax, x1, x2, y, color=arrow_color, lw=1.3, zorder=4, head_w=0.06)

# ── Wrap connector: step 8 → step 9 ──
# Squared path through the gap between rows
step8_rx  = LEFT + sp_pad + 7 * (sp_w + sp_gap_x) + sp_w   # right edge of step 8
step8_my  = y_row1 + sp_h / 2                               # mid-height row 1
step9_lx  = LEFT + sp_pad                                   # left edge of step 9
step9_my  = y_row2 + sp_h / 2                               # mid-height row 2

r_turn = step8_rx + 0.18    # right turn point
l_turn = step9_lx - 0.18    # left turn point
gap_y  = (y_row1 + y_row2 + sp_h) / 2   # midpoint of inter-row gap

wrap_c = arrow_color
wrap_lw = 1.2
wrap_dash = (4, 3)  # dashed to distinguish from inline flow arrows
# Horizontal right from step 8
ax.plot([step8_rx + 0.03, r_turn], [step8_my, step8_my],
        color=wrap_c, lw=wrap_lw, ls="--", dashes=wrap_dash,
        zorder=4, solid_capstyle="round")
# Down to mid-gap
ax.plot([r_turn, r_turn], [step8_my, gap_y],
        color=wrap_c, lw=wrap_lw, ls="--", dashes=wrap_dash,
        zorder=4, solid_capstyle="round")
# Across the gap to left margin
ax.plot([r_turn, l_turn], [gap_y, gap_y],
        color=wrap_c, lw=wrap_lw, ls="--", dashes=wrap_dash,
        zorder=4, solid_capstyle="round")
# Down to step 9 height
ax.plot([l_turn, l_turn], [gap_y, step9_my],
        color=wrap_c, lw=wrap_lw, ls="--", dashes=wrap_dash,
        zorder=4, solid_capstyle="round")
# Arrow into step 9
harr(ax, l_turn, step9_lx - 0.03, step9_my,
     color=wrap_c, lw=wrap_lw, zorder=4, head_w=0.06)

# ════════════════════════════════════════════════════════════════════
# "supported by" separator
# ════════════════════════════════════════════════════════════════════
ax.plot([LEFT + 0.3, LEFT + STACK_W - 0.3], [SEP_Y, SEP_Y],
        color=TEXT_G, linewidth=0.8, alpha=0.45, zorder=1)
ax.text(FIG_W / 2, SEP_Y + 0.10, "supported by",
        ha="center", va="center", fontsize=8,
        fontstyle="italic", color=TEXT_G, alpha=0.65, zorder=2)

# ════════════════════════════════════════════════════════════════════
# L2: CI/CD PIPELINE + COMMUNITY & QUALITY (compact)
# ════════════════════════════════════════════════════════════════════
l2_gap = 0.25
ci_w   = (STACK_W - l2_gap) / 2
comm_w = ci_w
l2_pad = 0.16
l2_pill_gap = 0.10
l2_pill_h = 0.55
l2_pill_y = L2_Y + 0.14

# CI/CD container
rbox(ax, LEFT, L2_Y, ci_w, L2_H, C_CI,
     ec="white", lw=1.5, zorder=2)

ax.text(LEFT + ci_w / 2, L2_Y + L2_H - 0.13,
        "CI/CD Pipeline",
        ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=TEXT_W, zorder=3)

# CI/CD inner pills
ci_pills = [
    ("Lint &\nType Check", "Ruff + MyPy"),
    ("Matrix\nTesting", "3 OS \u00d7 Py 3.11"),
    ("Binary\nBuilds", "SUMMA, mizuRoute +"),
]
ci_inner = ci_w - 2 * l2_pad
ci_pill_w = (ci_inner - 2 * l2_pill_gap) / 3
for i, (name, desc) in enumerate(ci_pills):
    cx = LEFT + l2_pad + i * (ci_pill_w + l2_pill_gap)
    rbox(ax, cx, l2_pill_y, ci_pill_w, l2_pill_h,
         lighten(C_CI, 0.38 + i * 0.06), name, sub=desc,
         fs=7.5, sfs=6, lo=0.10, so=0.12, ec="white", lw=0.7,
         tc=TEXT_D, zorder=4)

# Docs deploy bar
docs_y = l2_pill_y + l2_pill_h + l2_pill_gap
rbox(ax, LEFT + l2_pad, docs_y,
     ci_inner, 0.26,
     lighten(C_CI, 0.50), "Docs deploy",
     ec="white", lw=0.5, fs=7, bold=False, tc=TEXT_D, zorder=4)

# Community container
comm_x = LEFT + ci_w + l2_gap
rbox(ax, comm_x, L2_Y, comm_w, L2_H, C_COMM,
     ec="white", lw=1.5, zorder=2)

ax.text(comm_x + comm_w / 2, L2_Y + L2_H - 0.13,
        "Community & Quality",
        ha="center", va="center", fontsize=10.5,
        fontweight="bold", color=TEXT_W, zorder=3)

# Community inner pills
comm_pills = [
    ("Pre-commit\nHooks", "Ruff, Bandit, MyPy"),
    ("Test Suite", "99+ files\n70+ markers"),
]
comm_inner = comm_w - 2 * l2_pad
comm_pill_w = (comm_inner - l2_pill_gap) / 2
for i, (name, desc) in enumerate(comm_pills):
    cx = comm_x + l2_pad + i * (comm_pill_w + l2_pill_gap)
    rbox(ax, cx, l2_pill_y, comm_pill_w, l2_pill_h,
         lighten(C_COMM, 0.35 + i * 0.08), name, sub=desc,
         fs=7.5, sfs=6, lo=0.10, so=0.12, ec="white", lw=0.7,
         tc=TEXT_D, zorder=4)

# Fork & branch bar
rbox(ax, comm_x + l2_pad, docs_y,
     comm_inner, 0.26,
     lighten(C_COMM, 0.48), "Fork & branch workflow",
     ec="white", lw=0.5, fs=7, bold=False, tc=TEXT_D, zorder=4)

ax.text(LEFT - 0.35, L2_Y + L2_H / 2, "L2",
        ha="center", va="center", fontsize=10,
        fontweight="bold", color=C_CI, zorder=7)

# ════════════════════════════════════════════════════════════════════
# L1: DISTRIBUTION ROW
# ════════════════════════════════════════════════════════════════════
rbox(ax, LEFT, L1_Y, STACK_W, L1_H, lighten(C_DIST, 0.50),
     ec=C_DIST, lw=1.0, zorder=1, alpha=0.6)

ax.text(LEFT + STACK_W / 2, L1_Y + L1_H - 0.10,
        "Distribution",
        ha="center", va="center", fontsize=9.5,
        fontweight="bold", color=TEXT_D, alpha=0.70, zorder=3)

dist_channels = [
    ("NPM", "Pre-compiled binaries"),
    ("pip", "Python package"),
    ("uv", "External pkg mgmt"),
    ("Dev bootstrap", "Source compilation"),
]
d_pad = 0.25
d_gap = 0.18
d_inner = STACK_W - 2 * d_pad
d_w = (d_inner - 3 * d_gap) / 4
d_h = 0.45
d_y = L1_Y + 0.08

for i, (name, desc) in enumerate(dist_channels):
    dx = LEFT + d_pad + i * (d_w + d_gap)
    rbox(ax, dx, d_y, d_w, d_h,
         lighten(C_DIST, 0.30 + i * 0.08), name, sub=desc,
         fs=8, sfs=6, lo=0.08, so=0.09, ec="white", lw=0.8,
         tc=TEXT_D, zorder=3)

ax.text(LEFT - 0.35, L1_Y + L1_H / 2, "L1",
        ha="center", va="center", fontsize=10,
        fontweight="bold", color=C_DIST, zorder=7)

# ── save ─────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent
for fmt in ("pdf", "png"):
    fig.savefig(out / f"fig9_user_interfaces_deployment.{fmt}",
                dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.12)
    print(f"Saved fig9_user_interfaces_deployment.{fmt}")
plt.close(fig)
