#!/usr/bin/env python3
"""
Generate configuration files for the Section 4.11 Parallel Scaling Study.

Creates YAML configs for all six sub-experiments by modifying the base
template with experiment-specific parallelism and domain settings.

Usage:
    python generate_configs.py                    # All experiments, laptop scale
    python generate_configs.py --hpc              # Include HPC-scale configs
    python generate_configs.py --experiment 1     # ProcessPool configs only
    python generate_configs.py --experiment 1 3 4 # Multiple sub-experiments
    python generate_configs.py --dry-run          # Preview without writing
"""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    yaml = None

# Paths
STUDY_DIR = Path(__file__).parent.parent
CONFIG_DIR = STUDY_DIR / "configs"
BASE_CONFIG = CONFIG_DIR / "base_bow_hbv_dds.yaml"


def load_base_config() -> Dict[str, Any]:
    """Load the base YAML configuration."""
    if yaml is not None:
        with open(BASE_CONFIG) as f:
            return yaml.safe_load(f)
    else:
        # Fallback: parse simple key-value YAML without pyyaml
        config = {}
        with open(BASE_CONFIG) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line and not line.startswith("-"):
                    key, _, value = line.partition(":")
                    value = value.strip()
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    config[key.strip()] = value
        return config


def write_config(config: Dict[str, Any], filepath: Path, header: str = "") -> None:
    """Write a configuration dictionary to a YAML file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        if header:
            for line in header.strip().split("\n"):
                f.write(f"# {line}\n")
            f.write("\n")

        for key, value in config.items():
            if isinstance(value, list):
                f.write(f"{key}:\n")
                for item in value:
                    f.write(f"- {item}\n")
            elif isinstance(value, bool):
                f.write(f"{key}: {'true' if value else 'false'}\n")
            else:
                f.write(f"{key}: {value}\n")


def make_variant(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Create a config variant by applying overrides to the base."""
    config = copy.deepcopy(base)
    config.update(overrides)
    return config


# =============================================================================
# Experiment 1: Strong Scaling (ProcessPool)
# =============================================================================

def generate_exp1_configs(
    base: Dict[str, Any], hpc: bool = False
) -> List[Dict[str, Any]]:
    """Generate ProcessPool strong-scaling configs."""
    configs = []
    worker_counts = [1, 2, 4, 8]
    if hpc:
        worker_counts.extend([16, 32, 64])

    for np in worker_counts:
        name = f"strong_processpool_np{np}"
        config = make_variant(base, {
            "EXPERIMENT_ID": name,
            "NUM_PROCESSES": np,
            "MPI_PROCESSES": 1,
            "NUMBER_OF_ITERATIONS": 1000,
        })
        filepath = CONFIG_DIR / f"{name}.yaml"
        header = (
            f"4.11.1 Strong Scaling (ProcessPool) - {np} workers\n"
            f"HBV + DDS 1000 iterations on Bow at Banff (lumped)"
        )
        configs.append((config, filepath, header))

    return configs


# =============================================================================
# Experiment 2: Strong Scaling (MPI)
# =============================================================================

def generate_exp2_configs(
    base: Dict[str, Any], hpc: bool = False
) -> List[Dict[str, Any]]:
    """Generate MPI strong-scaling configs."""
    configs = []
    rank_counts = [1, 2, 4, 8]
    if hpc:
        rank_counts.extend([16, 32, 64, 128])

    for np in rank_counts:
        name = f"strong_mpi_np{np}"
        config = make_variant(base, {
            "EXPERIMENT_ID": name,
            "NUM_PROCESSES": np,
            "MPI_PROCESSES": np,
            "NUMBER_OF_ITERATIONS": 1000,
        })
        filepath = CONFIG_DIR / f"{name}.yaml"
        header = (
            f"4.11.2 Strong Scaling (MPI) - {np} ranks\n"
            f"HBV + DDS 1000 iterations on Bow at Banff (lumped)"
        )
        configs.append((config, filepath, header))

    return configs


# =============================================================================
# Experiment 3: Async DDS vs Synchronous DDS
# =============================================================================

def generate_exp3_configs(
    base: Dict[str, Any], hpc: bool = False
) -> List[Dict[str, Any]]:
    """Generate Async-DDS and sync baseline configs."""
    configs = []
    iterations = 4000  # Match 4.4 budget for comparability

    # Synchronous baselines at different worker counts
    sync_workers = [4]
    if hpc:
        sync_workers.extend([16, 64])

    for np in sync_workers:
        name = f"sync_dds_np{np}"
        config = make_variant(base, {
            "EXPERIMENT_ID": name,
            "NUM_PROCESSES": np,
            "MPI_PROCESSES": 1,
            "ITERATIVE_OPTIMIZATION_ALGORITHM": "DDS",
            "NUMBER_OF_ITERATIONS": iterations,
        })
        filepath = CONFIG_DIR / f"{name}.yaml"
        header = (
            f"4.11.3 Synchronous DDS baseline - {np} workers\n"
            f"Standard DDS with synchronous batch evaluation"
        )
        configs.append((config, filepath, header))

    # Async-DDS configurations with varying pool and batch sizes
    async_configs = [
        (5, 4, 4),    # small pool, small batch, laptop
        (10, 8, 8),   # medium pool, medium batch, laptop/HPC
        (20, 16, 16), # large pool, large batch, HPC
    ]

    for pool_size, batch_size, workers in async_configs:
        if workers > 8 and not hpc:
            continue
        name = f"async_dds_pool{pool_size}_batch{batch_size}"
        config = make_variant(base, {
            "EXPERIMENT_ID": name,
            "NUM_PROCESSES": workers,
            "MPI_PROCESSES": 1,
            "ITERATIVE_OPTIMIZATION_ALGORITHM": "ASYNC_DDS",
            "NUMBER_OF_ITERATIONS": iterations,
            "ASYNC_DDS_POOL_SIZE": pool_size,
            "ASYNC_DDS_BATCH_SIZE": batch_size,
        })
        filepath = CONFIG_DIR / f"{name}.yaml"
        header = (
            f"4.11.3 Async-DDS - pool={pool_size}, batch={batch_size}, "
            f"workers={workers}\n"
            f"Asynchronous parallel DDS with decoupled evaluation"
        )
        configs.append((config, filepath, header))

    return configs


# =============================================================================
# Experiment 4: JAX Acceleration
# =============================================================================

def generate_exp4_configs(
    base: Dict[str, Any], hpc: bool = False
) -> List[Dict[str, Any]]:
    """Generate JAX backend comparison configs."""
    configs = []
    iterations = 1000

    # NumPy backend (baseline)
    config = make_variant(base, {
        "EXPERIMENT_ID": "jax_numpy",
        "HBV_BACKEND": "numpy",
        "HBV_SMOOTHING": False,
        "NUMBER_OF_ITERATIONS": iterations,
        "NUM_PROCESSES": 1,
    })
    configs.append((
        config,
        CONFIG_DIR / "jax_numpy.yaml",
        "4.11.4 JAX Acceleration - NumPy backend (baseline)\n"
        "Standard NumPy HBV without JIT or GPU"
    ))

    # JAX CPU, no JIT
    config = make_variant(base, {
        "EXPERIMENT_ID": "jax_cpu_nojit",
        "HBV_BACKEND": "jax",
        "HBV_JIT_COMPILE": False,
        "HBV_USE_GPU": False,
        "HBV_SMOOTHING": False,
        "NUMBER_OF_ITERATIONS": iterations,
        "NUM_PROCESSES": 1,
    })
    configs.append((
        config,
        CONFIG_DIR / "jax_cpu_nojit.yaml",
        "4.11.4 JAX Acceleration - JAX CPU without JIT\n"
        "JAX backend on CPU, JIT compilation disabled"
    ))

    # JAX CPU, JIT enabled
    config = make_variant(base, {
        "EXPERIMENT_ID": "jax_cpu_jit",
        "HBV_BACKEND": "jax",
        "HBV_JIT_COMPILE": True,
        "HBV_USE_GPU": False,
        "HBV_SMOOTHING": False,
        "NUMBER_OF_ITERATIONS": iterations,
        "NUM_PROCESSES": 1,
    })
    configs.append((
        config,
        CONFIG_DIR / "jax_cpu_jit.yaml",
        "4.11.4 JAX Acceleration - JAX CPU with JIT\n"
        "JAX backend on CPU, JIT compilation enabled"
    ))

    # JAX GPU (HPC only)
    if hpc:
        config = make_variant(base, {
            "EXPERIMENT_ID": "jax_gpu_jit",
            "HBV_BACKEND": "jax",
            "HBV_JIT_COMPILE": True,
            "HBV_USE_GPU": True,
            "HBV_SMOOTHING": False,
            "NUMBER_OF_ITERATIONS": iterations,
            "NUM_PROCESSES": 1,
        })
        configs.append((
            config,
            CONFIG_DIR / "jax_gpu_jit.yaml",
            "4.11.4 JAX Acceleration - JAX GPU with JIT\n"
            "JAX backend on GPU (CUDA), JIT compilation enabled"
        ))

    # Composability: JAX JIT + multiprocessing (demonstrates orthogonality)
    for np in [1, 4, 8]:
        config = make_variant(base, {
            "EXPERIMENT_ID": f"jax_jit_np{np}",
            "HBV_BACKEND": "jax",
            "HBV_JIT_COMPILE": True,
            "HBV_USE_GPU": False,
            "HBV_SMOOTHING": False,
            "NUMBER_OF_ITERATIONS": iterations,
            "NUM_PROCESSES": np,
        })
        configs.append((
            config,
            CONFIG_DIR / f"jax_jit_np{np}.yaml",
            f"4.11.4 JAX + ProcessPool composability - JIT + {np} workers\n"
            f"Demonstrates orthogonal composition of acceleration and parallelism"
        ))

    return configs


# =============================================================================
# Experiment 5: Weak Scaling (Domain Complexity)
# =============================================================================

def generate_exp5_configs(
    base: Dict[str, Any], hpc: bool = False
) -> List[Dict[str, Any]]:
    """Generate weak-scaling configs with increasing domain complexity."""
    configs = []
    np = 8  # Fixed worker count

    # Lumped (1 GRU, 1 HRU)
    config = make_variant(base, {
        "EXPERIMENT_ID": "weak_lumped",
        "DOMAIN_NAME": "Bow_at_Banff_lumped_era5",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "lumped",
        "ROUTING_MODEL": "none",
        "HYDROLOGICAL_MODEL": "FUSE",
        "NUMBER_OF_ITERATIONS": 500,
        "NUM_PROCESSES": np,
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "DDS",
    })
    configs.append((
        config,
        CONFIG_DIR / "weak_lumped.yaml",
        "4.11.5 Weak Scaling - Lumped domain (1 GRU, 1 HRU)\n"
        "FUSE + DDS, 8 workers, minimal spatial complexity"
    ))

    # Lumped with elevation bands (1 GRU, 12 HRUs)
    config = make_variant(base, {
        "EXPERIMENT_ID": "weak_elevation",
        "DOMAIN_NAME": "Bow_at_Banff_elevation",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "elevation",
        "ROUTING_MODEL": "none",
        "HYDROLOGICAL_MODEL": "FUSE",
        "NUMBER_OF_ITERATIONS": 500,
        "NUM_PROCESSES": np,
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "DDS",
    })
    configs.append((
        config,
        CONFIG_DIR / "weak_elevation.yaml",
        "4.11.5 Weak Scaling - Elevation bands (1 GRU, 12 HRUs)\n"
        "FUSE + DDS, 8 workers, moderate spatial complexity"
    ))

    # Semi-distributed (49 GRUs, 379 HRUs)
    config = make_variant(base, {
        "EXPERIMENT_ID": "weak_semidist",
        "DOMAIN_NAME": "Bow_at_Banff_semi_distributed",
        "DOMAIN_DEFINITION_METHOD": "subset",
        "SUB_GRID_DISCRETIZATION": "elevation",
        "ROUTING_MODEL": "mizuroute",
        "HYDROLOGICAL_MODEL": "FUSE",
        "NUMBER_OF_ITERATIONS": 500,
        "NUM_PROCESSES": np,
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "DDS",
    })
    configs.append((
        config,
        CONFIG_DIR / "weak_semidist.yaml",
        "4.11.5 Weak Scaling - Semi-distributed (49 GRUs, 379 HRUs)\n"
        "FUSE + mizuRoute + DDS, 8 workers, high spatial complexity"
    ))

    # Distributed (2335 grid cells) -- HPC only
    if hpc:
        config = make_variant(base, {
            "EXPERIMENT_ID": "weak_distributed",
            "DOMAIN_NAME": "Bow_at_Banff_distributed_era5",
            "DOMAIN_DEFINITION_METHOD": "distributed",
            "SUB_GRID_DISCRETIZATION": "none",
            "ROUTING_MODEL": "mizuroute",
            "HYDROLOGICAL_MODEL": "FUSE",
            "NUMBER_OF_ITERATIONS": 500,
            "NUM_PROCESSES": np,
            "ITERATIVE_OPTIMIZATION_ALGORITHM": "DDS",
        })
        configs.append((
            config,
            CONFIG_DIR / "weak_distributed.yaml",
            "4.11.5 Weak Scaling - Distributed (2,335 grid cells)\n"
            "FUSE + mizuRoute + DDS, 8 workers, full spatial resolution"
        ))

    return configs


# =============================================================================
# Experiment 6: Task-Level Ensemble Parallelism
# =============================================================================

def generate_exp6_configs(
    base: Dict[str, Any], hpc: bool = False
) -> List[Dict[str, Any]]:
    """Generate ensemble parallelism configs for multi-model execution."""
    configs = []
    models = ["HBV", "GR4J", "FUSE", "JFUSE"]
    iterations = 500

    # Per-model configs (used in all execution modes)
    for model in models:
        model_overrides = {
            "EXPERIMENT_ID": f"ensemble_{model.lower()}",
            "HYDROLOGICAL_MODEL": model,
            "NUMBER_OF_ITERATIONS": iterations,
            "ITERATIVE_OPTIMIZATION_ALGORITHM": "DDS",
            "NUM_PROCESSES": 1,  # Overridden by orchestrator
        }

        # Model-specific backend settings
        if model == "HBV":
            model_overrides["HBV_BACKEND"] = "jax"
            model_overrides["HBV_TIMESTEP_HOURS"] = 24
            model_overrides["HBV_WARMUP_DAYS"] = 365
        elif model == "GR4J":
            model_overrides["ROUTING_MODEL"] = "none"
        elif model == "FUSE":
            model_overrides["ROUTING_MODEL"] = "none"
        elif model == "JFUSE":
            model_overrides["ROUTING_MODEL"] = "none"

        config = make_variant(base, model_overrides)
        filepath = CONFIG_DIR / f"ensemble_{model.lower()}.yaml"
        header = (
            f"4.11.6 Ensemble Parallelism - {model} model\n"
            f"{model} + DDS {iterations} iterations for ensemble timing"
        )
        configs.append((config, filepath, header))

    return configs


# =============================================================================
# Main
# =============================================================================

EXPERIMENT_GENERATORS = {
    1: ("4.11.1 Strong Scaling (ProcessPool)", generate_exp1_configs),
    2: ("4.11.2 Strong Scaling (MPI)", generate_exp2_configs),
    3: ("4.11.3 Async vs Sync DDS", generate_exp3_configs),
    4: ("4.11.4 JAX Acceleration", generate_exp4_configs),
    5: ("4.11.5 Weak Scaling", generate_exp5_configs),
    6: ("4.11.6 Ensemble Parallelism", generate_exp6_configs),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate configs for Section 4.11 Parallel Scaling Study"
    )
    parser.add_argument(
        "--experiment", type=int, nargs="*", default=None,
        help="Sub-experiment numbers to generate (default: all)"
    )
    parser.add_argument(
        "--hpc", action="store_true",
        help="Include HPC-scale configurations (>8 cores, GPU)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config paths without writing files"
    )
    args = parser.parse_args()

    experiments = args.experiment or list(EXPERIMENT_GENERATORS.keys())
    base = load_base_config()

    total_configs = 0
    for exp_num in experiments:
        if exp_num not in EXPERIMENT_GENERATORS:
            print(f"WARNING: Unknown experiment {exp_num}, skipping")
            continue

        name, generator = EXPERIMENT_GENERATORS[exp_num]
        configs = generator(base, hpc=args.hpc)

        print(f"\n{'=' * 60}")
        print(f"  Experiment {exp_num}: {name}")
        print(f"  Generating {len(configs)} configurations")
        print(f"{'=' * 60}")

        for config, filepath, header in configs:
            if args.dry_run:
                print(f"  [DRY RUN] Would write: {filepath.name}")
            else:
                write_config(config, filepath, header)
                print(f"  Created: {filepath.name}")
            total_configs += 1

    print(f"\n{'=' * 60}")
    print(f"  Total: {total_configs} configuration files")
    if args.dry_run:
        print("  (dry run -- no files written)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
