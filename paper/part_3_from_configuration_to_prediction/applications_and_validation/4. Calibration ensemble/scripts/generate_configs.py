#!/usr/bin/env python3
"""
Generate configuration files for the 4.4 Calibration Ensemble Study.

Creates one YAML config per optimization algorithm, all sharing the same
domain, model (HBV), forcing (ERA5), and time periods. The total function
evaluation budget is normalized across algorithms for fair comparison.

Algorithm families represented:
  - Sampling-based:    DDS
  - Evolutionary:      SCE-UA, DE, PSO, GA, CMA-ES
  - Gradient-based:    ADAM, L-BFGS  (require JAX smoothing)
  - Direct search:     Nelder-Mead
  - Stochastic:        Simulated Annealing, Basin Hopping
  - Surrogate-based:   Bayesian Optimization

Usage:
    python generate_configs.py                  # Generate all configs
    python generate_configs.py --seeds 3        # Generate multi-seed configs
    python generate_configs.py --budget 8000    # Custom function eval budget
"""

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None
    print("Warning: PyYAML not installed. Using manual YAML writing.")

# Paths
SCRIPT_DIR = Path(__file__).parent
CONFIGS_DIR = SCRIPT_DIR.parent / "configs"

# Resolve SYMFLUENCE paths from environment or defaults
_DEFAULT_DATA_DIR = "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/"
_DEFAULT_CODE_DIR = "/Users/darrieythorsson/compHydro/code/SYMFLUENCE"
SYMFLUENCE_DATA_DIR = os.getenv("SYMFLUENCE_DATA_DIR", _DEFAULT_DATA_DIR)
SYMFLUENCE_CODE_DIR = os.getenv("SYMFLUENCE_CODE_DIR", _DEFAULT_CODE_DIR)

# =============================================================================
# Base configuration template (shared across all experiments)
# =============================================================================
BASE_CONFIG: Dict[str, Any] = {
    # --- Paths ---
    "SYMFLUENCE_DATA_DIR": SYMFLUENCE_DATA_DIR,
    "SYMFLUENCE_CODE_DIR": SYMFLUENCE_CODE_DIR,

    # --- Domain ---
    "DOMAIN_NAME": "Bow_at_Banff_lumped_era5",
    "POUR_POINT_COORDS": "51.1722/-115.5717",
    "BOUNDING_BOX_COORDS": "51.76/-116.55/50.95/-115.5",
    "DOMAIN_DEFINITION_METHOD": "lumped",
    "SUB_GRID_DISCRETIZATION": "lumped",
    "DOMAIN_DISCRETIZATION": "GRUs",
    "ROUTING_DELINEATION": "lumped",
    "GEOFABRIC_TYPE": "TDX",
    "STREAM_THRESHOLD": 10000,
    "LUMPED_WATERSHED_METHOD": "TauDEM",

    # --- Time periods ---
    "EXPERIMENT_TIME_START": "2002-01-01 01:00",
    "EXPERIMENT_TIME_END": "2009-12-31 23:00",
    "CALIBRATION_PERIOD": "2004-01-01, 2007-12-31",
    "EVALUATION_PERIOD": "2008-01-01, 2009-12-31",
    "SPINUP_PERIOD": "2002-01-01, 2003-12-31",

    # --- Forcing ---
    "FORCING_DATASET": "ERA5",
    "FORCING_VARIABLES": "default",
    "FORCING_MEASUREMENT_HEIGHT": 2,
    "FORCING_TIME_STEP_SIZE": 3600,

    # --- Model ---
    "HYDROLOGICAL_MODEL": "HBV",
    "ROUTING_MODEL": "none",
    "HBV_TIMESTEP_HOURS": 24,
    "HBV_WARMUP_DAYS": 365,
    "HBV_BACKEND": "jax",

    # --- Observations ---
    "STREAMFLOW_DATA_PROVIDER": "WSC",
    "STATION_ID": "05BB001",
    "DOWNLOAD_WSC_DATA": True,

    # --- Calibration common ---
    "OPTIMIZATION_METHODS": ["iteration"],
    "OPTIMIZATION_METRIC": "KGE",
    "TARGET_METRIC": "KGE",
    "MPI_PROCESSES": 1,
    "FORCE_RUN_ALL_STEPS": False,
}

# =============================================================================
# Algorithm-specific configurations
# =============================================================================
# Total budget: ~4000 function evaluations for fair comparison.
# - Single-solution methods (DDS, Nelder-Mead): 4000 iterations (~1 eval/iter)
# - Population-based (DE, PSO, GA): pop=20, 200 generations => 4000 evals
# - SCE-UA: ~28 iterations (~145 evals/iter => ~4000 evals)
# - CMA-ES: pop=20, 200 generations => 4000 evals
# - ADAM (native grad): 2000 iters * ~2 evals => ~4000 evals
# - L-BFGS (native grad + line search): 500 iters * ~8 avg evals => ~4000 evals
# - Simulated Annealing: 400 temps * 10 steps/temp => 4000 evals
# - Basin Hopping: 80 hops * ~50 local evals/hop => ~4000 evals
# - Surrogate-based (Bayesian Opt): 200 iterations (GP fit + evaluate)

ALGORITHMS: Dict[str, Dict[str, Any]] = {
    # --- Sampling-based ---
    "dds": {
        "label": "DDS",
        "description": "Dynamically Dimensioned Search",
        "family": "Sampling",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "DDS",
        "NUMBER_OF_ITERATIONS": 4000,
        "DDS_R": 0.2,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },

    # --- Evolutionary ---
    "sceua": {
        "label": "SCE-UA",
        "description": "Shuffled Complex Evolution - University of Arizona",
        "family": "Evolutionary",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "SCE-UA",
        # SCE-UA evaluates pop_size points per iteration.
        # With 3 complexes and 14 params: pop = 3*(2*14+1) ~ 87-145.
        # Target ~4000 evals => ~28 iterations.
        "NUMBER_OF_ITERATIONS": 28,
        "NUMBER_OF_COMPLEXES": 3,
        "POINTS_PER_SUBCOMPLEX": 5,
        "NUMBER_OF_EVOLUTION_STEPS": 20,
        "EVOLUTION_STAGNATION": 5,
        "PERCENT_CHANGE_THRESHOLD": 0.01,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },
    "de": {
        "label": "DE",
        "description": "Differential Evolution",
        "family": "Evolutionary",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "DE",
        "NUMBER_OF_ITERATIONS": 200,
        "POPULATION_SIZE": 20,
        "DE_SCALING_FACTOR": 0.7,
        "DE_CROSSOVER_RATE": 0.7,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },
    "pso": {
        "label": "PSO",
        "description": "Particle Swarm Optimization",
        "family": "Evolutionary",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "PSO",
        "NUMBER_OF_ITERATIONS": 200,
        "POPULATION_SIZE": 20,
        "SWRMSIZE": 20,
        "PSO_COGNITIVE_PARAM": 1.5,
        "PSO_SOCIAL_PARAM": 1.5,
        "PSO_INERTIA_WEIGHT": 0.7,
        "PSO_INERTIA_REDUCTION_RATE": 0.99,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },
    "ga": {
        "label": "GA",
        "description": "Genetic Algorithm",
        "family": "Evolutionary",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "GA",
        "NUMBER_OF_ITERATIONS": 200,
        "POPULATION_SIZE": 20,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },
    "cmaes": {
        "label": "CMA-ES",
        "description": "Covariance Matrix Adaptation Evolution Strategy",
        "family": "Evolutionary",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "CMA-ES",
        # CMA-ES evaluates lambda (pop_size) per generation.
        # 200 generations * 20 pop = 4000 evaluations.
        "NUMBER_OF_ITERATIONS": 200,
        "POPULATION_SIZE": 20,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },

    # --- Gradient-based (require smoothing for differentiability) ---
    "adam": {
        "label": "ADAM",
        "description": "Adaptive Moment Estimation (gradient-based)",
        "family": "Gradient",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "ADAM",
        "NUMBER_OF_ITERATIONS": 2000,
        "ADAM_LR": 0.01,
        "ADAM_BETA1": 0.9,
        "ADAM_BETA2": 0.999,
        "ADAM_EPSILON": 1e-8,
        "HBV_SMOOTHING": True,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },
    "lbfgs": {
        "label": "L-BFGS",
        "description": "Limited-memory BFGS (quasi-Newton gradient)",
        "family": "Gradient",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "LBFGS",
        # L-BFGS with native gradients: ~2 evals/gradient + line search
        # (~4-12 evals/iter). 500 iters * ~8 avg = ~4000 evals.
        "NUMBER_OF_ITERATIONS": 500,
        "HBV_SMOOTHING": True,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },

    # --- Direct search ---
    "nelder_mead": {
        "label": "Nelder-Mead",
        "description": "Nelder-Mead Simplex",
        "family": "Direct Search",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "NELDER-MEAD",
        "NUMBER_OF_ITERATIONS": 4000,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },

    # --- Stochastic global ---
    "sa": {
        "label": "SA",
        "description": "Simulated Annealing",
        "family": "Stochastic",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "SIMULATED_ANNEALING",
        # SA evaluates steps_per_temp (default=10) neighbors per temperature.
        # 400 temperatures * 10 steps = 4000 evaluations.
        "NUMBER_OF_ITERATIONS": 400,
        "SA_STEPS_PER_TEMP": 10,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },
    "basin_hopping": {
        "label": "Basin Hopping",
        "description": "Basin Hopping (multi-start local optimization)",
        "family": "Stochastic",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "BASIN-HOPPING",
        # Each hop runs local_steps (default=50) local optimizer evals.
        # 80 hops * ~50 local evals = ~4000 evaluations.
        "NUMBER_OF_ITERATIONS": 80,
        "BH_LOCAL_STEPS": 50,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },

    # --- Surrogate-based ---
    "bayesian_opt": {
        "label": "Bayesian Opt.",
        "description": "Bayesian Optimization (Gaussian Process surrogate)",
        "family": "Surrogate",
        "ITERATIVE_OPTIMIZATION_ALGORITHM": "BAYESIAN_OPT",
        "NUMBER_OF_ITERATIONS": 200,
        "HBV_SMOOTHING": False,
        "HBV_SMOOTHING_FACTOR": 15.0,
    },
}

# Algorithm family ordering for consistent display
FAMILY_ORDER = [
    "Sampling", "Evolutionary", "Gradient",
    "Direct Search", "Stochastic", "Surrogate",
]


def build_config(algorithm_key: str, seed: int = 42,
                 budget: int = None) -> Dict[str, Any]:
    """Build a complete configuration for a given algorithm and seed."""
    algo = ALGORITHMS[algorithm_key]
    config = copy.deepcopy(BASE_CONFIG)

    # Experiment identity
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    config["EXPERIMENT_ID"] = f"cal_ensemble_{algorithm_key}{seed_suffix}"
    config["RANDOM_SEED"] = seed

    # Algorithm-specific settings
    for key, value in algo.items():
        if key not in ("label", "description", "family"):
            config[key] = value

    # Override iteration count if custom budget specified
    if budget is not None:
        pop_size = algo.get("POPULATION_SIZE", 1)
        if pop_size > 1:
            config["NUMBER_OF_ITERATIONS"] = budget // pop_size
        elif algo["family"] == "Gradient":
            config["NUMBER_OF_ITERATIONS"] = budget // 2
        elif algo["family"] == "Surrogate":
            config["NUMBER_OF_ITERATIONS"] = budget // 20
        else:
            config["NUMBER_OF_ITERATIONS"] = budget

    return config


def write_yaml(config: Dict[str, Any], path: Path):
    """Write configuration to YAML file."""
    if yaml is not None:
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        # Fallback: manual YAML writing
        with open(path, 'w') as f:
            for key, value in config.items():
                if isinstance(value, list):
                    f.write(f"{key}:\n")
                    for item in value:
                        f.write(f"- {item}\n")
                elif isinstance(value, bool):
                    f.write(f"{key}: {'true' if value else 'false'}\n")
                elif isinstance(value, str):
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate configs for the Calibration Ensemble Study"
    )
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Number of random seeds per algorithm (default: 1, seed=42)"
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help="Total function evaluation budget (default: ~4000 per algorithm)"
    )
    parser.add_argument(
        "--algorithms", type=str, default="all",
        help="Comma-separated algorithm keys, or 'all' (default: all)"
    )
    args = parser.parse_args()

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Select algorithms
    if args.algorithms == "all":
        algo_keys = list(ALGORITHMS.keys())
    else:
        algo_keys = [k.strip() for k in args.algorithms.split(",")]

    # Generate seeds
    base_seeds = [42 + i * 1000 for i in range(args.seeds)]

    print("=" * 60)
    print("Calibration Ensemble Study - Config Generator")
    print("=" * 60)
    print(f"  Algorithms: {len(algo_keys)}")
    print(f"  Seeds per algorithm: {args.seeds}")
    print(f"  Total configs: {len(algo_keys) * args.seeds}")
    if args.budget:
        print(f"  Function evaluation budget: {args.budget}")
    print()

    generated = 0
    for algo_key in algo_keys:
        if algo_key not in ALGORITHMS:
            print(f"  WARNING: Unknown algorithm '{algo_key}', skipping")
            continue

        algo = ALGORITHMS[algo_key]
        for seed in base_seeds:
            config = build_config(algo_key, seed=seed, budget=args.budget)
            seed_suffix = f"_seed{seed}" if seed != 42 else ""
            filename = f"config_bow_hbv_{algo_key}{seed_suffix}.yaml"
            filepath = CONFIGS_DIR / filename

            write_yaml(config, filepath)
            generated += 1

            iters = config["NUMBER_OF_ITERATIONS"]
            pop = config.get("POPULATION_SIZE", 1)
            smooth = "smooth" if config.get("HBV_SMOOTHING", False) else "no-smooth"
            evals = iters * pop if algo["family"] == "Evolutionary" else iters

            print(f"  {filename}")
            print(f"    {algo['label']} ({algo['family']}) | "
                  f"iters={iters}, pop={pop}, ~evals={evals}, "
                  f"{smooth}, seed={seed}")

    print(f"\nGenerated {generated} configuration files in {CONFIGS_DIR}")


if __name__ == "__main__":
    main()
