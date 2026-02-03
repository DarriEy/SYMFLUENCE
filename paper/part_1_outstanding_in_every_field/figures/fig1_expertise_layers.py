"""
Figure 1: The Impossible Generalist — Layers of Required Expertise
in Modern Computational Hydrology

Generates a concentric-ring / stacked-layer diagram showing the
breadth of knowledge domains a single researcher is expected to master,
with representative skills, tools, and languages annotated per layer.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Configuration ──────────────────────────────────────────────────
FIGSIZE = (14, 10)
DPI = 300
OUTPUT = "fig1_expertise_layers.png"
OUTPUT_PDF = "fig1_expertise_layers.pdf"

# Layers from core (bottom) to periphery (top)
# Each: (label, color, representative_skills, representative_tools)
LAYERS = [
    (
        "Field Observation &\nPhysical Intuition",
        "#2E6B4F",
        [
            "Snowpack structure",
            "Soil horizon identification",
            "Stream gauging",
            "Lysimeter operation",
            "Flux tower maintenance",
        ],
        ["Field notebooks", "Sensors", "Boots"],
    ),
    (
        "Process Understanding &\nMathematical Foundations",
        "#3A7D5C",
        [
            "Richards' equation (3D)",
            "Energy balance closure",
            "Saint-Venant equations",
            "Numerical discretization",
            "PDE stability analysis",
            "Snow & canopy physics",
        ],
        ["Calculus", "PDEs", "Numerical methods"],
    ),
    (
        "Parameter Estimation &\nPhysical Meaning",
        "#4A8F6E",
        [
            "Ksat, porosity, field capacity",
            "Conceptual vs. physical parameters",
            "Equifinality & identifiability",
            "Spatial parameter transfer",
            "Parameter-structure interactions",
        ],
        ["ASCII", "Fortran namelists", "YAML", "JSON", "PyTorch tensors"],
    ),
    (
        "Legacy Numerical Codes",
        "#5B7B9A",
        [
            "Fortran 77/90/2003 compilation",
            "C/C++ build systems",
            "Makefile / CMake debugging",
            "NetCDF-Fortran / LAPACK / BLAS",
            "Linker errors & implicit typing",
        ],
        ["gfortran", "gcc", "Make", "CMake", "ifort"],
    ),
    (
        "Data Science at Scale",
        "#6B8DAE",
        [
            "ERA5, AORC, RDRS, CHIRPS, ...",
            "GRACE, MODIS, SMAP, Sentinel",
            "Spatial remapping & regridding",
            "Unit conversion & QC filtering",
            "API authentication & rate limits",
        ],
        ["Python", "R", "NumPy", "Xarray", "GDAL", "GeoPandas", "Shell"],
    ),
    (
        "Optimization &\nCalibration Algorithms",
        "#8B6B8A",
        [
            "DDS, SCE-UA, DE, PSO, GA",
            "NSGA-II, MOEA/D",
            "ADAM, L-BFGS (gradient-based)",
            "Objective function design",
            "Parallel population evaluation",
        ],
        ["SciPy", "Platypus", "DEAP", "JAX", "Optuna"],
    ),
    (
        "Differentiable Modeling &\nAutomatic Differentiation",
        "#A07DA0",
        [
            "AD through legacy solvers",
            "LLVM-level gradient kernels",
            "End-to-end differentiable physics",
            "Flux derivatives: canopy → aquifer",
            "Hybrid physics-ML graphs",
        ],
        ["JAX", "Julia/Zygote", "Enzyme", "PyTorch", "LLVM"],
    ),
    (
        "Machine Learning &\nGPU Computing",
        "#B0677A",
        [
            "LSTM, GNN, Transformers",
            "Loss design & regularization",
            "CUDA / cuDNN / mixed precision",
            "Transfer learning & fine-tuning",
            "Explainability & attribution",
        ],
        ["PyTorch", "TensorFlow", "CUDA", "W&B"],
    ),
    (
        "High-Performance Computing",
        "#C05A5A",
        [
            "SLURM job scripting",
            "MPI parallelization",
            "Shared filesystem I/O",
            "Memory & queue management",
            "Cross-node debugging",
        ],
        ["SLURM", "MPI", "PBS", "Singularity", "SSH"],
    ),
    (
        "Reproducibility &\nCommunity Infrastructure",
        "#D04E4E",
        [
            "Version control & CI/CD",
            "Containerization",
            "Documentation & provenance",
            "Package management",
            "Cross-platform portability",
        ],
        ["Git", "Docker", "Conda", "pip", "GitHub Actions"],
    ),
]

N = len(LAYERS)


def create_figure():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.8, N + 0.5)
    ax.axis("off")

    # Title
    ax.text(
        5.0,
        N + 0.3,
        "The Impossible Generalist",
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="bold",
        color="#2C2C2C",
    )
    ax.text(
        5.0,
        N + 0.05,
        "Layers of required expertise in modern computational hydrology",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#555555",
        style="italic",
    )

    # Draw layers as horizontal bars
    bar_left = 0.8
    bar_right = 6.0
    bar_height = 0.78
    skill_x = 6.4
    tool_x = 9.8

    # Column headers
    ax.text(
        (bar_left + bar_right) / 2,
        N - 0.25,
        "Knowledge Domain",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#333",
    )
    ax.text(
        skill_x + 0.0,
        N - 0.25,
        "Representative Skills",
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#333",
    )

    for i, (label, color, skills, tools) in enumerate(LAYERS):
        y = N - 1 - i  # top to bottom

        # Main bar
        rect = FancyBboxPatch(
            (bar_left, y + (1 - bar_height) / 2),
            bar_right - bar_left,
            bar_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.92,
        )
        ax.add_patch(rect)

        # Layer label
        ax.text(
            (bar_left + bar_right) / 2,
            y + 0.5,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground=color)],
        )

        # Skills list (compact)
        skill_text = " · ".join(skills[:3])
        if len(skills) > 3:
            skill_text += " ..."
        ax.text(
            skill_x,
            y + 0.58,
            skill_text,
            ha="left",
            va="center",
            fontsize=7.2,
            color="#333",
        )

        # Tools as small tags
        tool_str = "  ".join(tools[:5])
        ax.text(
            skill_x,
            y + 0.32,
            tool_str,
            ha="left",
            va="center",
            fontsize=6.5,
            color="#777",
            family="monospace",
        )

        # Connecting bracket line
        ax.plot(
            [bar_right + 0.1, skill_x - 0.15],
            [y + 0.5, y + 0.5],
            color=color,
            linewidth=1.0,
            alpha=0.5,
        )

    # Bottom annotation
    ax.text(
        5.0,
        -0.55,
        "Each layer is individually learnable. The compound expectation that a single researcher\n"
        "will master all of them simultaneously is the structural problem this paper identifies.",
        ha="center",
        va="top",
        fontsize=10,
        color="#555",
        style="italic",
    )

    # Vertical arrow on left side indicating "Single researcher"
    arrow_x = 0.25
    ax.annotate(
        "",
        xy=(arrow_x, -0.1),
        xytext=(arrow_x, N - 0.15),
        arrowprops=dict(
            arrowstyle="<->",
            color="#333",
            lw=2.0,
        ),
    )
    ax.text(
        arrow_x - 0.15,
        N / 2 - 0.1,
        "One\nresearcher",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#333",
        rotation=90,
    )

    plt.tight_layout()
    fig.savefig(OUTPUT, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT}, {OUTPUT_PDF}")
    plt.close()


if __name__ == "__main__":
    create_figure()
