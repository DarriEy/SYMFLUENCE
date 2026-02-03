"""
SYMFLUENCE model component interface diagram (Section 3.5).

Shows the registry-driven pipeline: PreProcessor -> Runner -> PostProcessor
-> ResultExtractor, the ABC base classes and mixin architecture, and how
ModelManager orchestrates per-model component instances resolved via
ComponentRegistry.

Verified against the codebase at /Users/darrieythorsson/compHydro/code/SYMFLUENCE.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── global grid ─────────────────────────────────────────────────────
FIG_W, FIG_H = 12.0, 9.4
STACK_L = 1.20
STACK_R = 10.80
STACK_W = STACK_R - STACK_L  # 9.60
MID_X   = (STACK_L + STACK_R) / 2
PAD     = 0.04
ROW_GAP = 0.24

# ── colours — consistent with fig1 palette ─────────────────────────
C_MGR    = "#C9943A"   # gold  — ModelManager
C_PRE    = "#5BA58B"   # green — PreProcessor
C_RUN    = "#4A7FB5"   # blue  — Runner
C_POST   = "#8B6DAF"   # purple — PostProcessor
C_EXT    = "#2E86AB"   # teal  — ResultExtractor
C_REG    = "#EAEAEA"   # registry bg
C_DATA   = "#F5F0E6"   # data artifact bg
C_MIXIN  = "#F2F2F2"   # mixin row bg
C_MODEL  = "#FBF6EE"   # implementations bg
C_PIPE   = "#FAFAFA"   # pipeline bg

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
    """Rounded box primitive — every visual element uses this."""
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={PAD}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        zorder=zorder)
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
                color=tc, alpha=0.82, fontstyle="italic",
                zorder=zorder + 1)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal",
                color=tc, zorder=zorder + 1)


def harr(ax, x1, x2, y, c=CLR_A, lw=1.3, st="->"):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle=st, color=c, lw=lw,
        zorder=5, mutation_scale=11))


def varr(ax, x, y1, y2, c=CLR_A, lw=1.3, st="->"):
    ax.add_patch(FancyArrowPatch(
        (x, y1), (x, y2), arrowstyle=st, color=c, lw=lw,
        zorder=5, mutation_scale=11))


# ── build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── title ───────────────────────────────────────────────────────────
ax.text(MID_X, FIG_H - 0.32,
        "Model Integration: Component Interface Architecture",
        ha="center", va="center", fontsize=13,
        fontweight="bold", color=TEXT_D)

# ════════════════════════════════════════════════════════════════════
# ROW 1 — ModelManager (facade tier)
# ════════════════════════════════════════════════════════════════════
mgr_h = 0.58
mgr_y = FIG_H - 1.20

rbox(ax, STACK_L, mgr_y, STACK_W, mgr_h, C_MGR,
     "ModelManager",
     sub="Facade — resolves workflow order, iterates models",
     lo=0.10, so=0.12)

# ════════════════════════════════════════════════════════════════════
# ROW 2 — ComponentRegistry
# ════════════════════════════════════════════════════════════════════
reg_h = 0.72
reg_y = mgr_y - 0.40 - reg_h

rbox(ax, STACK_L, reg_y, STACK_W, reg_h, C_REG,
     ec=CLR_B, lw=1.0, ls=(0, (4, 3)), zorder=2)

ax.text(MID_X, reg_y + reg_h - 0.13, "ComponentRegistry",
        ha="center", va="center", fontsize=9.5,
        fontweight="bold", color=TEXT_G, zorder=3)

# four registry slots
slot_labels = [
    ("preprocessors", C_PRE),
    ("runners", C_RUN),
    ("postprocessors", C_POST),
    ("result_extractors", C_EXT),
]
n_slots = len(slot_labels)
slot_h = 0.28
slot_gap = 0.14
slot_pad = 0.18
slot_usable = STACK_W - 2 * slot_pad
slot_w = (slot_usable - (n_slots - 1) * slot_gap) / n_slots
slot_x0 = STACK_L + slot_pad
slot_y = reg_y + 0.10

for i, (lbl, clr) in enumerate(slot_labels):
    sx = slot_x0 + i * (slot_w + slot_gap)
    rbox(ax, sx, slot_y, slot_w, slot_h, lighten(clr, 0.55),
         lbl, ec=clr, lw=1.0, fs=7.5, bold=False, tc=TEXT_G)

# arrow: Manager → Registry
varr(ax, MID_X, mgr_y - 0.02, reg_y + reg_h + 0.02, c=C_MGR, lw=1.4)
ax.text(MID_X + 0.14, (mgr_y + reg_y + reg_h) / 2, "lookup",
        ha="left", va="center", fontsize=7.5, color=TEXT_G, fontstyle="italic")

# ════════════════════════════════════════════════════════════════════
# ROW 3 — Pipeline (4 components + RoutingDecider + data artifacts)
# ════════════════════════════════════════════════════════════════════

# ── pipeline components ───────────────────────────────────────────
pipe_pad = 0.22                          # inset from STACK edges
pipe_left  = STACK_L + pipe_pad
pipe_right = STACK_R - pipe_pad
pipe_inner = pipe_right - pipe_left

comp_h   = 1.00
comp_gap = 0.24
n_comp   = 4
comp_w   = (pipe_inner - (n_comp - 1) * comp_gap) / n_comp

comp_top = reg_y - 0.58
comp_y   = comp_top - comp_h
comp_cy  = comp_y + comp_h / 2

components = [
    ("BaseModel\nPreProcessor",   "run_preprocessing()",  C_PRE),
    ("BaseModel\nRunner",         "run()",                C_RUN),
    ("BaseModel\nPostProcessor",  "extract_streamflow()", C_POST),
    ("ModelResult\nExtractor",    "extract_variable()",   C_EXT),
]

comp_positions = []   # left-x of each component
comp_centers   = []   # center-x of each component
for i, (name, method, clr) in enumerate(components):
    cx = pipe_left + i * (comp_w + comp_gap)
    rbox(ax, cx, comp_y, comp_w, comp_h, clr, name, sub=method,
         fs=8.5, sfs=7, lo=0.14, so=0.18)
    # ABC badge — top-right, well inside the box
    bw, bh = 0.34, 0.15
    rbox(ax, cx + comp_w - bw - 0.12, comp_y + comp_h - bh - 0.10,
         bw, bh, lighten(clr, 0.25), "ABC",
         ec="white", lw=0.5, fs=5.5, tc=TEXT_W)
    comp_positions.append(cx)
    comp_centers.append(cx + comp_w / 2)

# horizontal arrows between components
for i in range(n_comp - 1):
    x1 = comp_positions[i] + comp_w + 0.04
    x2 = comp_positions[i + 1] - 0.04
    harr(ax, x1, x2, comp_cy, c="#666666", lw=1.3)

# coloured arrows from registry slots down to components
for i, clr in enumerate([C_PRE, C_RUN, C_POST, C_EXT]):
    varr(ax, comp_centers[i], reg_y - 0.02, comp_top + 0.02, c=clr, lw=1.1)

# ── data artifacts — aligned directly below each component ────────
art_h = 0.38
art_y = comp_y - 0.30 - art_h

art_labels = [
    "Standardised inputs\n(basin_averaged_data/)",
    "Model-specific forcing\n(forcing/{MODEL}_input/)",
    "Raw simulations\n(simulations/{EXP}/)",
    "Extracted variables\n(standardised CSV/NC)",
]

art_positions = []
for i, label in enumerate(art_labels):
    ax_pos = comp_positions[i]
    rbox(ax, ax_pos, art_y, comp_w, art_h, C_DATA, label,
         ec=CLR_B, lw=0.7, fs=6.5, bold=False, tc=TEXT_G, zorder=6)
    art_positions.append((ax_pos, comp_w))

# flow arrows between artifact pills
art_cy = art_y + art_h / 2
for i in range(len(art_labels) - 1):
    x1 = art_positions[i][0] + art_positions[i][1] + 0.03
    x2 = art_positions[i + 1][0] - 0.03
    harr(ax, x1, x2, art_cy, c="#999999", lw=0.8)

# ── pipeline background — flush with STACK_L / STACK_R ───────────
pipe_bot = art_y - 0.16
pipe_top = comp_top + 0.36
rbox(ax, STACK_L, pipe_bot,
     STACK_W, pipe_top - pipe_bot,
     C_PIPE, ec=CLR_B, lw=0.8, ls=(0, (5, 3)), zorder=0)

# label centred in the top margin of the pipeline box
ax.text(MID_X, pipe_top - 0.16,
        "Per-model pipeline instance",
        ha="center", va="center", fontsize=7.5, color=TEXT_G,
        fontstyle="italic", fontweight="bold", zorder=1)

# ════════════════════════════════════════════════════════════════════
# ROW 4 — Shared Infrastructure (Mixins)
# ════════════════════════════════════════════════════════════════════
mix_h = 0.84
mix_y = pipe_bot - 0.42 - mix_h

rbox(ax, STACK_L, mix_y, STACK_W, mix_h, C_MIXIN,
     ec=CLR_B, lw=0.8, zorder=2)
ax.text(MID_X, mix_y + mix_h - 0.12,
        "Shared Infrastructure (Mixins)",
        ha="center", va="center", fontsize=9,
        fontweight="bold", color=TEXT_G, zorder=3)

mixin_data = [
    ("ModelComponentMixin",  "_init_model_component()"),
    ("PathResolverMixin",    "resolve paths, config access"),
    ("ShapefileAccessMixin", "catchment geometry"),
    ("ConfigMixin",          "typed config coercion"),
]
mp_h = 0.40
mp_gap = 0.14
mp_pad = 0.20
mp_usable = STACK_W - 2 * mp_pad
mp_w = (mp_usable - 3 * mp_gap) / 4
mp_y = mix_y + 0.10
mp_x0 = STACK_L + mp_pad

for i, (name, desc) in enumerate(mixin_data):
    mx = mp_x0 + i * (mp_w + mp_gap)
    rbox(ax, mx, mp_y, mp_w, mp_h, "#FFFFFF", name, sub=desc,
         ec=CLR_B, lw=0.7, fs=7, sfs=6, tc=TEXT_D,
         lo=0.07, so=0.09)

# arrow: pipeline → mixins
varr(ax, MID_X, pipe_bot - 0.02, mix_y + mix_h + 0.02,
     c=CLR_B, lw=0.9)
ax.text(MID_X + 0.12, (pipe_bot + mix_y + mix_h) / 2,
        "base classes inherit via",
        ha="left", va="center", fontsize=7.5, color=TEXT_G,
        fontstyle="italic")

# ════════════════════════════════════════════════════════════════════
# ROW 5 — Registered implementations
# ════════════════════════════════════════════════════════════════════
impl_h = 0.72
impl_y = mix_y - 0.40 - impl_h

rbox(ax, STACK_L, impl_y, STACK_W, impl_h, C_MODEL,
     ec=CLR_B, lw=0.7, ls=(0, (4, 3)), zorder=1)
ax.text(MID_X, impl_y + impl_h - 0.10,
        "Registered implementations  (@ComponentRegistry.register_*)",
        ha="center", va="center", fontsize=7.5,
        fontweight="bold", color=TEXT_L, zorder=2)

models = ["SUMMA", "MESH", "FUSE", "GR", "HYPE", "HBV", "VIC", "NextGen", "LSTM", "GNN"]
pill_h = 0.30
pill_gap = 0.12
pill_pad = 0.22
pill_usable = STACK_W - 2 * pill_pad
pill_gap = 0.10
pill_w = (pill_usable - (len(models) - 1) * pill_gap) / len(models)
pill_y = impl_y + 0.10
pill_x0 = STACK_L + pill_pad

C_IMPL = "#D6CEBE"
C_IMPL_E = "#B0A694"

for i, m in enumerate(models):
    px = pill_x0 + i * (pill_w + pill_gap)
    rbox(ax, px, pill_y, pill_w, pill_h, C_IMPL, m,
         ec=C_IMPL_E, lw=0.8, fs=7.5, bold=True, tc=TEXT_D)

# arrow: mixins → implementations
varr(ax, MID_X, mix_y - 0.02, impl_y + impl_h + 0.02,
     c=CLR_B, lw=0.9)
ax.text(MID_X + 0.12, (mix_y + impl_y + impl_h) / 2,
        "inherit", ha="left", va="center", fontsize=7.5,
        color=TEXT_G, fontstyle="italic")

# ── save ────────────────────────────────────────────────────────────
out = Path(__file__).resolve().parent
for fmt in ("pdf", "png"):
    fig.savefig(out / f"fig7_model_integration.{fmt}",
                dpi=300, bbox_inches="tight", facecolor="white",
                pad_inches=0.12)
    print(f"Saved fig7_model_integration.{fmt}")
plt.close(fig)
