#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Paper 3, Figure 9: calibration performance across the model x algorithm matrix
(the ~130 combinations of the calibration ensemble).

Panel (a) — calibration-period KGE heatmap. Rows are the eight hydrological
models (five JAX-native above the divider, three external below). Columns are the
optimization algorithms, **ordered and bracketed by algorithmic family**, with the
family named above its span: the caption claims a family grouping, so the figure
has to show where one family stops and the next starts (review comment, Wouter
Knoben, 7 Jul). A per-model mean closes the panel, set off by a gap so it does not
read as another algorithm.

Panel (b) — calibration vs evaluation KGE for the same combinations, one marker
shape+colour per model, against a 1:1 line. Distance below the line is
calibration-to-evaluation degradation.

Each cell/point is one calibration run at a single fixed seed. Gradient methods
(ADAM, L-BFGS) require differentiable implementations and so do not exist for the
three external models; those cells are intentionally blank.

Colour is documented at the constants below: a sequential, colour-vision-safe
ramp for panel (a), and two well-separated hues plus eight marker shapes for
panel (b). Every cell also carries its numeric KGE, so colour is never the only
channel in either panel.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[4]
DATA_DIR = Path(os.getenv("SYMFLUENCE_DATA_DIR", str(_REPO_ROOT.parent / "SYMFLUENCE_data")))
DOMAIN = "Bow_at_Banff_lumped_calibration_ensemble"
OPT = DATA_DIR / f"domain_{DOMAIN}" / "optimization"
OUT = _HERE.parents[1] / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Row order: 5 JAX-native, then 3 external (matches the paper's divider).
JAX_MODELS = ["HBV", "SACSMA", "XINANJIANG", "HECHMS", "TOPMODEL"]
EXTERNAL_MODELS = ["SUMMA", "FUSE", "HYPE"]
MODELS = JAX_MODELS + EXTERNAL_MODELS
MODEL_LABELS = {
    "HBV": "HBV", "SACSMA": "SAC-SMA", "XINANJIANG": "Xinanjiang",
    "HECHMS": "HEC-HMS", "TOPMODEL": "TOPMODEL",
    "SUMMA": "SUMMA", "FUSE": "FUSE", "HYPE": "HYPE",
}

# Column order IS the family grouping: each entry is (family label, algorithms).
# Families follow each algorithm's own implementation, not its acronym — ABC here
# is Approximate Bayesian Computation (ABC-SMC, likelihood-free inference), not
# Artificial Bee Colony, so it sits with DREAM (MCMC) and GLUE (Monte Carlo)
# rather than with the population searches.
ALGO_FAMILIES = [
    ("Sampling",        ["DDS"]),
    ("Evolutionary",    ["SCE-UA", "DE", "PSO", "GA", "CMA-ES"]),
    ("Gradient",        ["ADAM", "L-BFGS"]),
    ("Direct search",   ["Nelder-Mead"]),
    ("Stochastic",      ["SA", "Basin Hop."]),
    ("Surrogate",       ["Bayes. Opt."]),
    ("Bayesian / MC",   ["DREAM", "ABC", "GLUE"]),
    ("Multi-objective", ["NSGA-II", "MOEA/D"]),
]
ALGO_ORDER = [a for _, algos in ALGO_FAMILIES for a in algos]
# Normalise the many spellings the run JSONs use to the labels above.
ALIAS = {
    "DDS": "DDS", "SCE-UA": "SCE-UA", "SCEUA": "SCE-UA", "DE": "DE", "PSO": "PSO",
    "GA": "GA", "CMA-ES": "CMA-ES", "CMAES": "CMA-ES", "DREAM": "DREAM",
    "ABC": "ABC", "GLUE": "GLUE", "NSGA-II": "NSGA-II", "NSGA2": "NSGA-II",
    "MOEA/D": "MOEA/D", "MOEAD": "MOEA/D", "NELDER-MEAD": "Nelder-Mead",
    "NELDER_MEAD": "Nelder-Mead", "SA": "SA", "SIMULATED_ANNEALING": "SA",
    "BASIN-HOPPING": "Basin Hop.", "BASIN_HOPPING": "Basin Hop.",
    "BAYESIAN_OPT": "Bayes. Opt.", "BAYESIAN-OPT": "Bayes. Opt.",
    "ADAM": "ADAM", "LBFGS": "L-BFGS", "L-BFGS": "L-BFGS",
}


def _cal_kge_from_dir(algo_dir: Path):
    """Calibration KGE for one run dir, tolerant of the three storage conventions.

    Different model families write the metric differently:
      * ``final_evaluation.json`` -> ``calibration_metrics.KGE`` (SUMMA/FUSE/HYPE/SAC-SMA)
      * ``final_evaluation.json`` -> ``calibration_metrics.Calib_KGE`` (HBV/TOPMODEL)
      * ``best_params.json`` -> ``best_score`` with ``metric == 'KGE'`` (HEC-HMS, and a
        universal fallback when a run has no final-evaluation file).
    Returns (algorithm_label, kge) or (label, None).
    """
    label = None
    kge = None
    fe = list(algo_dir.glob("*final_evaluation.json"))
    if fe:
        try:
            d = json.load(open(fe[0]))
            label = ALIAS.get(str(d.get("algorithm", "")).strip().upper())
            cm = d.get("calibration_metrics") or {}
            for key in ("KGE", "Calib_KGE"):
                if cm.get(key) is not None:
                    kge = float(cm[key])
                    break
        except (OSError, ValueError):
            pass
    if kge is None:
        bp = list(algo_dir.glob("*best_params.json"))
        if bp:
            try:
                b = json.load(open(bp[0]))
                if label is None:
                    label = ALIAS.get(str(b.get("algorithm", "")).strip().upper())
                if b.get("best_score") is not None and "KGE" in str(b.get("metric", "")).upper():
                    kge = float(b["best_score"])
            except (OSError, ValueError):
                pass
    if kge is None:
        # Last resort: the calibration ran but never wrote a final-evaluation file.
        # The best (max) objective in the iteration log is the calibration KGE
        # (the optimiser maximises KGE); ignore penalty scores (<= -100).
        csvs = list(algo_dir.glob("*parallel_iteration_results.csv"))
        if csvs:
            try:
                import csv as _csv
                scores = []
                for row in _csv.DictReader(open(csvs[0])):
                    try:
                        s = float(row.get("score", ""))
                    except (TypeError, ValueError):
                        continue
                    if s > -100:
                        scores.append(s)
                if scores:
                    kge = max(scores)
            except OSError:
                pass
    if label is None:
        label = ALIAS.get(algo_dir.name.split("_cal_ensemble")[0].upper())
    return label, kge


def _load_cal_kge(model: str) -> dict:
    """Map algorithm-label -> calibration KGE for one model."""
    out = {}
    mdir = OPT / model
    if not mdir.is_dir():
        return out
    for algo_dir in sorted(mdir.iterdir()):
        if not algo_dir.is_dir():
            continue
        label, kge = _cal_kge_from_dir(algo_dir)
        if label is not None and kge is not None:
            out[label] = kge
    return out


def _eval_kge_from_dir(algo_dir: Path):
    """Evaluation-period KGE for one run dir, or (label, None).

    Only ``final_evaluation.json`` carries the held-out score — there is no
    iteration-log fallback, because the optimiser never sees the evaluation
    period. A run that predates the final-evaluation step therefore has a
    calibration KGE (panel a) but no evaluation KGE (panel b); panel (b) states
    how many of the combinations it could pair.
    """
    label = None
    fe = list(algo_dir.glob("*final_evaluation.json"))
    if not fe:
        return None, None
    try:
        d = json.load(open(fe[0]))
    except (OSError, ValueError):
        return None, None
    label = ALIAS.get(str(d.get("algorithm", "")).strip().upper())
    if label is None:
        label = ALIAS.get(algo_dir.name.split("_cal_ensemble")[0].upper())
    em = d.get("evaluation_metrics") or {}
    for key in ("KGE", "Eval_KGE", "Evaluation_KGE"):
        if em.get(key) is not None:
            try:
                return label, float(em[key])
            except (TypeError, ValueError):
                return label, None
    return label, None


def _load_eval_kge(model: str) -> dict:
    """Map algorithm-label -> evaluation KGE for one model."""
    out = {}
    mdir = OPT / model
    if not mdir.is_dir():
        return out
    for algo_dir in sorted(mdir.iterdir()):
        if not algo_dir.is_dir():
            continue
        label, kge = _eval_kge_from_dir(algo_dir)
        if label is not None and kge is not None:
            out[label] = kge
    return out


# Panel (b) encoding. Colour carries the implementation tier (the same split
# panel (a) divides its rows on); marker SHAPE carries the individual model.
#
# Colour cannot carry all eight models: in a scatter any two points can sit side
# by side, so the palette must clear colour-vision separation on ALL pairs, and
# eight hues cannot. The previous eight-hue version measured ΔE 1.9 between the
# HBV and TOPMODEL blues/purples under protanopia (indistinguishable) and 6.8
# between SUMMA and HYPE under NORMAL vision. Two well-separated hues (ΔE 24.7
# protan) plus eight shapes encode the same information and survive both
# colour-vision deficiency and greyscale printing.
TIER_COLORS = {"jax": "#2a78d6", "external": "#eb6834"}
MODEL_MARKERS = {
    "HBV": "o", "SACSMA": "s", "XINANJIANG": "^", "HECHMS": "D", "TOPMODEL": "v",
    "SUMMA": "P", "FUSE": "X", "HYPE": "*",
}

# Panel (a) colour scale. A SEQUENTIAL ramp (viridis: monotone lightness,
# perceptually uniform, colour-vision safe) rather than the earlier red-yellow-blue
# — that is a diverging map, and diverging implies a meaningful midpoint that KGE
# does not have here.
#
# The scale spans VMIN..VMAX rather than 0..1 because the values are not uniformly
# spread: all but a handful sit between 0.6 and 0.95, so a full 0..1 ramp spent
# most of its range on empty territory and rendered the entire interesting band as
# near-identical dark blue. Cells below VMIN keep a distinct under-colour and the
# colourbar is drawn with an arrow at that end, so clipping is visible rather than
# silent — and every cell is annotated with its exact value regardless.
VMIN, VMAX = 0.3, 0.95
GAP = 0.6          # blank columns between the matrix and the per-model mean

# Type scale. One place to tune, and deliberately stepped so the hierarchy is
# visible: panel title > family band > axis title > row/tick labels > cell value.
FS_PANEL_TITLE = 13.0
FS_FAMILY = 7.5     # fits the narrowest (one-column) family span
FS_AXIS_TITLE = 10.5
FS_ROWLABEL = 10.0
FS_TICK = 9.0
FS_TIER = 9.0       # rotated JAX-native / External labels
FS_CELL = 7.8       # the numbers are the panel's actual payload
FS_LEGEND = 8.0
FS_NOTE = 7.5

# Family header geometry, in data (cell) units above the matrix.
#
# No bracket and no box. Both were tried and both looked wrong: a filled box read
# like a spreadsheet, and a bracket drawn to the family's span is NARROWER than
# the word sitting on it whenever a family owns a single column, so every
# one-column group looked misaligned. The vertical dividers already mark exactly
# where each family starts and stops — extending them up through the header strip
# does the boundary work, and the label just sits centred in the space between
# them.
DIVIDER_TOP = -0.78      # boundary ticks stop here, BELOW the labels
LABEL_Y_HI = -1.44       # staggered header: upper baseline (unused)
LABEL_Y_LO = -0.90       # header baseline, clear above the rule
DIVIDER_COLOR = "#343a40"

# The header labels alternate between two baselines.
#
# Setting them on one line caps the type size at whatever fits the NARROWEST
# family: three families own a single column, and "Surrogate" in a one-column
# cell held the whole header row to ~7pt. Widening the figure does not help —
# the cell and the glyphs scale together, so the ratio is fixed. Alternating
# rows means a label only has to clear the next label on ITS row, two families
# away, which is roughly three columns of room instead of one.
FAMILY_ROW_ALTERNATES = False
FS_FAMILY_MAX = 10.0     # cap: bigger than this and the header outshouts the data

# Names shortened only where the full name still cannot fit; full names live in
# the caption.
# One shared size is used for the whole header row, so the row is only ever as
# large as its tightest label allows. Measured at 10pt against the gap each label
# has to its neighbours, the ceilings are: Direct search 6.6pt, Stochastic 8.4,
# Surrogate 8.7, Bayesian/MC 9.2, Gradient 10.0, everything else 13-22. So the
# single 13-character name was holding every other label to 6.6pt; shortening just
# that one lifts the row to ~8.7. Full names are in the caption.
FAMILY_DISPLAY = {
    "Direct search": "Direct",
    "Multi-objective": "Multi-obj.",
    "Bayesian / MC": "Bayesian/MC",
}


def _place_family_labels(fig, ax, fam_spans, x_mean) -> None:
    """Draw the family headers at the largest size that fits every header cell.

    The size is derived from the RENDERED width of each label against the cell
    the dividers give it, not from a character-count estimate. Estimating was
    tried and was wrong by enough that the dividers sliced through the "S" of
    "Sampling" and "Surrogate" — a one-column family has about half an inch to
    work with and a nine-character name does not fit it at header size.

    One size is then used for all of them: a header row whose labels are each a
    different size reads as broken rather than as a hierarchy, so the tightest
    label sets the size for everyone.

    Labels alternate between two baselines, so the room a label has is the
    distance to the next label on ITS OWN row — two families away — rather than
    its own cell. That is what lets the type be legible: constrained to single
    cells the header was stuck near 7pt, because three families own one column
    each.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    cells = [(FAMILY_DISPLAY.get(fam, fam), (lo + hi) / 2)
             for fam, lo, hi in fam_spans]
    cells.append(("Mean", x_mean))

    rows = [i % 2 if FAMILY_ROW_ALTERNATES else 1 for i in range(len(cells))]
    # "Mean" is a separate column past the gap; keep it on the lower row.
    rows[-1] = 1

    def _px(x_data):
        return ax.transData.transform((x_data, 0))[0]

    size = FS_FAMILY_MAX
    for i, (text, centre) in enumerate(cells):
        probe = ax.text(0, 0, text, fontsize=FS_FAMILY_MAX, fontweight="semibold")
        width_px = probe.get_window_extent(renderer).width
        probe.remove()
        # Half-distance to the nearest neighbour sharing this row, in pixels.
        same_row = [c for j, (_, c) in enumerate(cells) if rows[j] == rows[i] and j != i]
        gaps = [abs(_px(centre) - _px(c)) for c in same_row]
        available = (min(gaps) if gaps else 1e9) * 0.88   # half each side + air
        if width_px > available:
            size = min(size, FS_FAMILY_MAX * available / width_px)

    placed = []
    for i, (text, centre) in enumerate(cells):
        placed.append(ax.text(
            centre, LABEL_Y_HI if rows[i] == 0 else LABEL_Y_LO, text,
            ha="center", va="bottom", fontsize=size, fontweight="semibold",
            color="#212529", clip_on=False, zorder=4))

    # A label wider than its own family (the first and last especially) can hang
    # past the edge of the matrix. Nudge those back inside, in data units.
    fig.canvas.draw()
    x_lo, x_hi = ax.get_xlim()
    edge_lo = ax.transData.transform((x_lo, 0))[0]
    edge_hi = ax.transData.transform((x_hi, 0))[0]
    unit = abs(ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])
    for label in placed:
        bb = label.get_window_extent(renderer)
        shift = 0.0
        if bb.x0 < edge_lo:
            shift = (edge_lo - bb.x0) / unit
        elif bb.x1 > edge_hi:
            shift = -(bb.x1 - edge_hi) / unit
        if shift:
            label.set_x(label.get_position()[0] + shift)


def _place_tier_labels(fig, ax) -> None:
    """Add the rotated JAX-native / External labels clear of the row labels.

    Measured rather than guessed: the labels go to the left of wherever the row
    labels actually end, which depends on the longest model name, the font and
    the figure width. Requires a completed layout, so it runs after everything
    else is drawn.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks = [t for t in ax.get_yticklabels() if t.get_text()]
    if not ticks:
        return
    left_px = min(t.get_window_extent(renderer).x0 for t in ticks)
    # One label height of clearance, converted to the axes' x fraction.
    gap_px = FS_TIER * fig.dpi / 72.0 * 0.9
    x_axes = ax.transAxes.inverted().transform((left_px - gap_px, 0))[0]

    tr = ax.get_yaxis_transform()
    for label, y in (
        ("JAX-native", (len(JAX_MODELS) - 1) / 2),
        ("External", len(JAX_MODELS) + (len(EXTERNAL_MODELS) - 1) / 2),
    ):
        ax.text(x_axes, y, label, rotation=90, va="center", ha="center",
                fontsize=FS_TIER, fontweight="bold", transform=tr, clip_on=False)


def _draw_heatmap(ax, grid, cols, fam_spans):
    """Panel (a): the model x algorithm matrix, bracketed by algorithm family."""
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e0e0e0")     # combination does not exist (gradient x external)
    cmap.set_under("#5b0f26")   # scores below VMIN, flagged by the colourbar arrow
    im = ax.imshow(grid, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=FS_TICK)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in MODELS], fontsize=FS_ROWLABEL)
    ax.set_xlabel("Optimization algorithm", fontsize=FS_AXIS_TITLE, labelpad=4)

    # Viridis runs dark->light with magnitude, so the text flips once, near the
    # point where the background crosses mid-lightness.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                v = grid[i, j]
                # Flip where viridis crosses mid-lightness, i.e. halfway along the
                # VMIN..VMAX span - not at a raw KGE value, or the light greens
                # near the top of the ramp end up carrying white text.
                frac = (v - VMIN) / (VMAX - VMIN)
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_CELL,
                        color="black" if frac > 0.5 else "white")

    # Divider between the five JAX-native models and the three external ones.
    ax.axhline(len(JAX_MODELS) - 0.5, color="black", linewidth=1.8)
    # The tier labels are added after layout by _place_tier_labels(), which
    # measures where the row labels actually end. A fixed offset guessed at here
    # cannot work: the gap depends on the longest model name, the font and the
    # figure size, and the previous guess put "JAX-native" straight through
    # "Xinanjiang".

    # --- Family bands ---------------------------------------------------------
    # A filled, outlined band above each family's columns rather than a hairline
    # rule with floating text: the caption claims a family grouping, so the
    # grouping has to be the most legible thing above the matrix. The band also
    # gives short names room to breathe and long ones a place to wrap, which a
    # bare rule did not — "Direct search", "Stochastic" and "Surrogate" used to
    # run into each other.
    n_rows = grid.shape[0]
    # The boundary marks stop BELOW the labels — a short tick above the matrix,
    # closed by one rule the labels sit on. Running them up through the header
    # strip put vertical lines straight through the lettering: the labels are
    # sized against the space up to their NEIGHBOUR, so a wide name legitimately
    # overhangs its own cell, and any line at that boundary struck the glyphs.
    # Below the text there is nothing left to collide with, and the tick plus
    # rule still say exactly where each family begins and ends.
    for k, (fam, lo, hi) in enumerate(fam_spans):
        if k:
            ax.plot([lo - 0.5, lo - 0.5], [DIVIDER_TOP, n_rows - 0.5],
                    color=DIVIDER_COLOR, linewidth=1.4, clip_on=False, zorder=4)
    for x in (-0.5, len(cols) - 0.5):
        ax.plot([x, x], [DIVIDER_TOP, -0.5], color=DIVIDER_COLOR,
                linewidth=1.4, clip_on=False, zorder=4)
    ax.plot([-0.5, len(cols) - 0.5], [DIVIDER_TOP] * 2, color=DIVIDER_COLOR,
            linewidth=1.0, clip_on=False, zorder=4)
    return im


def main():
    cal = {m: _load_cal_kge(m) for m in MODELS}
    ev = {m: _load_eval_kge(m) for m in MODELS}
    n_found = sum(len(v) for v in cal.values())
    n_pairs = sum(1 for m in MODELS for a in cal[m] if a in ev[m])

    # Keep only algorithm columns some model actually has, preserving family order.
    cols, fam_spans = [], []
    for fam, algos in ALGO_FAMILIES:
        present = [a for a in algos if any(a in cal[m] for m in MODELS)]
        if not present:
            continue
        fam_spans.append((fam, len(cols), len(cols) + len(present) - 1))
        cols.extend(present)

    grid = np.full((len(MODELS), len(cols)), np.nan)
    for i, m in enumerate(MODELS):
        for j, a in enumerate(cols):
            if a in cal[m]:
                grid[i, j] = cal[m][a]
    means = np.nanmean(grid, axis=1)

    print(f"Loaded {n_found} model-algorithm KGE values "
          f"({len(MODELS)} models x {len(cols)} algorithms) "
          f"in {len(fam_spans)} families; {n_pairs} have a paired evaluation KGE.")

    # Layout: the colourbar sits hard against panel (a) (it describes only that
    # panel — stranded in the middle it read as belonging to neither), and the
    # two panels are sized so neither dominates.
    # An explicit empty spacer column separates the colourbar from panel (b):
    # gridspec applies one wspace to every gap, so without it the gap that keeps
    # the colourbar attached to (a) also jams its label into (b)'s y-axis label.
    fig = plt.figure(figsize=(17.2, 5.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[len(cols) + 2.2, 0.30, 2.6, 7.6],
                          wspace=0.05, left=0.075, right=0.985,
                          top=0.845, bottom=0.185)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 3])

    im = _draw_heatmap(ax_a, grid, cols, fam_spans)

    # Per-model mean, offset by a visible gap so it does not read as an algorithm.
    x_mean = len(cols) - 0.5 + GAP + 0.5
    for i, mu in enumerate(means):
        if np.isfinite(mu):
            ax_a.add_patch(plt.Rectangle((x_mean - 0.5, i - 0.5), 1, 1,
                                         facecolor="white", edgecolor="black",
                                         linewidth=0.6, clip_on=False, zorder=3))
            ax_a.text(x_mean, i, f"{mu:.2f}", ha="center", va="center",
                      fontsize=FS_CELL, fontweight="bold", clip_on=False, zorder=4)
    for x in (x_mean - 0.5, x_mean + 0.5):
        ax_a.plot([x, x], [DIVIDER_TOP, -0.5], color=DIVIDER_COLOR,
                  linewidth=1.4, clip_on=False, zorder=4)
    ax_a.plot([x_mean - 0.5, x_mean + 0.5], [DIVIDER_TOP] * 2,
              color=DIVIDER_COLOR, linewidth=1.0, clip_on=False, zorder=4)
    ax_a.set_xlim(-0.5, x_mean + 0.5)
    ax_a.set_title("(a) Calibration-period KGE across model × algorithm combinations",
                   fontsize=FS_PANEL_TITLE, fontweight="bold", pad=36, loc="left")

    cb = fig.colorbar(im, cax=ax_cb, extend="min")
    cb.set_label("Calibration KGE", fontsize=FS_AXIS_TITLE)
    cb.ax.tick_params(labelsize=FS_TICK)

    # --- Panel (b): calibration vs evaluation --------------------------------
    # A diverged calibration can land far below the rest and squash every other
    # point into a corner. Window the axes to FLOOR, but never drop a point
    # silently: anything outside is named on the panel itself.
    FLOOR = -0.5
    inside, outside = [], []
    for m in MODELS:
        for a in cal[m]:
            if a not in ev[m]:
                continue
            (inside if min(cal[m][a], ev[m][a]) >= FLOOR else outside).append(
                (m, a, cal[m][a], ev[m][a]))

    lo = min([min(x, y) for *_, x, y in inside], default=0.0)
    hi = max([max(x, y) for *_, x, y in inside], default=1.0)
    pad = 0.05 * (hi - lo) if hi > lo else 0.05
    lim = (lo - pad, hi + pad)

    # Shade the region below the 1:1 line: every point there lost skill out of
    # sample, which is the pattern the panel exists to show.
    ax_b.fill_between(lim, lim, [lim[0]] * 2, color="#c9302c", alpha=0.045, zorder=1)
    ax_b.plot(lim, lim, ls="--", color="#555555", linewidth=1.0, zorder=2)

    for m in MODELS:
        xs = [x for mm, _, x, _ in inside if mm == m]
        ys = [y for mm, _, _, y in inside if mm == m]
        if not xs:
            continue
        tier = "jax" if m in JAX_MODELS else "external"
        ax_b.scatter(xs, ys, s=52, marker=MODEL_MARKERS.get(m, "o"),
                     facecolor=TIER_COLORS[tier], edgecolor="white",
                     linewidth=0.8, alpha=0.85, zorder=3,
                     label=MODEL_LABELS.get(m, m))

    footnotes = [f"{n_pairs} of {n_found} combinations"]
    if outside:
        footnotes.append("Outside axes: " + "; ".join(
            f"{MODEL_LABELS.get(m, m)}/{a} ({x:.2f}, {y:.2f})"
            for m, a, x, y in outside))
    ax_b.text(0.0, -0.135, ". ".join(footnotes) + ".", transform=ax_b.transAxes,
              ha="left", va="top", fontsize=FS_NOTE, color="#555555")
    ax_b.text(lim[1], lim[1], " 1:1", fontsize=7.5, color="#555555",
              ha="left", va="center")
    ax_b.set_xlim(*lim)
    ax_b.set_ylim(*lim)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("Calibration KGE", fontsize=FS_AXIS_TITLE)
    ax_b.set_ylabel("Evaluation KGE", fontsize=FS_AXIS_TITLE)
    ax_b.tick_params(labelsize=FS_TICK)
    ax_b.grid(alpha=0.25, linewidth=0.5)
    ax_b.set_axisbelow(True)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    # Two legends: shape = which model, colour = which implementation tier. The
    # shape entries are drawn in neutral grey so they read as "shape means this"
    # rather than implying a colour.
    from matplotlib.lines import Line2D
    shape_keys = [Line2D([], [], ls="", marker=MODEL_MARKERS.get(m, "o"),
                         markerfacecolor="#9a9a9a", markeredgecolor="white",
                         markersize=7, label=MODEL_LABELS.get(m, m))
                  for m in MODELS]
    tier_keys = [Line2D([], [], ls="", marker="o", markerfacecolor=TIER_COLORS[k],
                        markeredgecolor="white", markersize=8, label=lab)
                 for k, lab in (("jax", "JAX-native"), ("external", "External"))]
    leg1 = ax_b.legend(handles=shape_keys, title="Model", fontsize=FS_LEGEND,
                       title_fontsize=FS_LEGEND, loc="upper left", frameon=True,
                       framealpha=0.95, borderpad=0.45, labelspacing=0.32,
                       handletextpad=0.5, ncol=2, columnspacing=0.9)
    ax_b.add_artist(leg1)
    ax_b.legend(handles=tier_keys, title="Implementation", fontsize=FS_LEGEND,
                title_fontsize=FS_LEGEND, loc="lower right", frameon=True,
                framealpha=0.95, borderpad=0.45, labelspacing=0.32,
                handletextpad=0.5)
    # The pair count lives in the footnote, not the title — spelled out up here it
    # made the title wider than the panel. It still has to appear somewhere: the
    # panel does not show all 130 combinations, and a reader must not have to
    # assume it does.
    ax_b.set_title("(b) Calibration vs evaluation KGE", fontsize=FS_PANEL_TITLE,
                   fontweight="bold", pad=36, loc="left")

    # Both need the finished layout, so they come after both panels.
    _place_family_labels(fig, ax_a, fam_spans, x_mean)
    _place_tier_labels(fig, ax_a)

    for ext in ("png", "pdf"):
        p = OUT / f"figure_09_calibration_heatmap.{ext}"
        fig.savefig(p, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"Saved: {p.name}")


if __name__ == "__main__":
    main()
