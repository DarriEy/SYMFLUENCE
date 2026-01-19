#!/usr/bin/env python
"""
Comprehensive benchmark comparing all HBV optimization methods.

Compares:
- Adam with Finite Differences
- Adam with Native Gradients (JAX autodiff)
- DDS (Dynamically Dimensioned Search)
- DE (Differential Evolution)

Usage:
    python scripts/benchmark_hbv_all_optimizers.py
    python scripts/benchmark_hbv_all_optimizers.py --iterations 500
    python scripts/benchmark_hbv_all_optimizers.py --quick  # 100 iterations
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('hbv_benchmark')


def run_optimization(
    config: Dict[str, Any],
    method: str,
    n_iterations: int,
    gradient_mode: str = 'auto'
) -> Dict[str, Any]:
    """
    Run a single optimization method.

    Args:
        config: Base configuration dictionary
        method: Optimization method ('adam', 'lbfgs', 'dds', 'de', 'pso', 'sce')
        n_iterations: Number of iterations/function evaluations
        gradient_mode: For gradient methods: 'native', 'finite_difference', or 'auto'

    Returns:
        Dictionary with results
    """
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.optimization.registry import OptimizerRegistry

    # Create a fresh config for this run
    run_config = config.copy()
    run_config['GRADIENT_MODE'] = gradient_mode

    # Configure based on method (don't set ITERATIVE_OPTIMIZATION_ALGORITHM for gradient methods)
    run_config['NUMBER_OF_ITERATIONS'] = n_iterations

    if method.lower() == 'adam':
        run_config['ADAM_STEPS'] = n_iterations
        run_config['ADAM_LR'] = 0.02
    elif method.lower() == 'lbfgs':
        run_config['LBFGS_STEPS'] = n_iterations
        run_config['LBFGS_LR'] = 0.1
    elif method.lower() == 'dds':
        run_config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'DDS'
        run_config['DDS_R'] = 0.2
    elif method.lower() == 'de':
        run_config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'DE'
        run_config['POPULATION_SIZE'] = 20
        run_config['DE_SCALING_FACTOR'] = 0.5
        run_config['DE_CROSSOVER_RATE'] = 0.9
    elif method.lower() == 'pso':
        run_config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'PSO'
        run_config['SWRMSIZE'] = 20
    elif method.lower() == 'sce':
        run_config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = 'SCE-UA'
        run_config['NUMBER_OF_COMPLEXES'] = 3
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create typed config
    try:
        typed_config = SymfluenceConfig(**run_config)
    except Exception as e:
        logger.warning(f"Could not create typed config: {e}")
        typed_config = run_config

    # Create optimizer via registry
    HBVOptimizer = OptimizerRegistry.get_optimizer('HBV')
    optimizer = HBVOptimizer(typed_config, logger)

    # Run optimization
    start_time = time.time()

    try:
        if method.lower() == 'adam':
            result_path = optimizer.run_adam(steps=n_iterations, lr=0.02)
        elif method.lower() == 'lbfgs':
            result_path = optimizer.run_lbfgs(steps=n_iterations, lr=0.1)
        elif method.lower() == 'dds':
            result_path = optimizer.run_dds(iterations=n_iterations)
        elif method.lower() == 'de':
            result_path = optimizer.run_de(iterations=n_iterations)
        elif method.lower() == 'pso':
            result_path = optimizer.run_pso(iterations=n_iterations)
        elif method.lower() == 'sce':
            result_path = optimizer.run_sce(iterations=n_iterations)

        elapsed_time = time.time() - start_time

        # Get results
        best_score = optimizer.best_score if hasattr(optimizer, 'best_score') else None
        best_params = optimizer.best_params if hasattr(optimizer, 'best_params') else None

        # Get iteration history if available
        iteration_history = []
        if hasattr(optimizer, 'iteration_results'):
            iteration_history = [
                {'iteration': r.get('iteration', i), 'score': r.get('score', r.get('best_score'))}
                for i, r in enumerate(optimizer.iteration_results)
            ]

        return {
            'method': method,
            'gradient_mode': gradient_mode if method.lower() in ['adam', 'lbfgs'] else 'n/a',
            'n_iterations': n_iterations,
            'elapsed_time': elapsed_time,
            'best_score': float(best_score) if best_score is not None else None,
            'best_params': best_params,
            'result_path': str(result_path) if result_path else None,
            'iteration_history': iteration_history,
            'success': True,
            'error': None
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        import traceback
        logger.error(f"{method} optimization failed: {e}")
        logger.debug(traceback.format_exc())
        return {
            'method': method,
            'gradient_mode': gradient_mode if method.lower() in ['adam', 'lbfgs'] else 'n/a',
            'n_iterations': n_iterations,
            'elapsed_time': elapsed_time,
            'best_score': None,
            'best_params': None,
            'result_path': None,
            'iteration_history': [],
            'success': False,
            'error': str(e)
        }


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a formatted summary table of results."""
    logger.info("")
    logger.info("=" * 90)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 90)
    logger.info("")

    # Header
    header = f"{'Method':<25} {'Iterations':>10} {'Time (s)':>12} {'Best KGE':>12} {'Status':>10}"
    logger.info(header)
    logger.info("-" * 90)

    # Sort by best score (descending, since higher KGE is better)
    sorted_results = sorted(
        results,
        key=lambda x: x['best_score'] if x['best_score'] is not None else -999,
        reverse=True
    )

    for r in sorted_results:
        method_name = r['method']
        if r['gradient_mode'] != 'n/a':
            method_name += f" ({r['gradient_mode'][:6]})"

        score_str = f"{r['best_score']:.4f}" if r['best_score'] is not None else "FAILED"
        status = "OK" if r['success'] else "FAILED"

        row = f"{method_name:<25} {r['n_iterations']:>10} {r['elapsed_time']:>12.2f} {score_str:>12} {status:>10}"
        logger.info(row)

    logger.info("-" * 90)

    # Find best result
    successful = [r for r in results if r['success'] and r['best_score'] is not None]
    if successful:
        best = max(successful, key=lambda x: x['best_score'])
        logger.info(f"Best method: {best['method']} ({best['gradient_mode']}) with KGE = {best['best_score']:.4f}")

        # Compute speedups relative to slowest
        slowest_time = max(r['elapsed_time'] for r in successful)
        logger.info("")
        logger.info("Relative timing (speedup vs slowest):")
        for r in sorted(successful, key=lambda x: x['elapsed_time']):
            speedup = slowest_time / r['elapsed_time'] if r['elapsed_time'] > 0 else 0
            method_name = r['method']
            if r['gradient_mode'] != 'n/a':
                method_name += f" ({r['gradient_mode'][:6]})"
            logger.info(f"  {method_name:<25}: {r['elapsed_time']:>8.2f}s ({speedup:>5.1f}x faster)")


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'optimizer_benchmark_{timestamp}.json'

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    # Clean results for serialization
    clean_results = []
    for r in results:
        clean_r = {}
        for k, v in r.items():
            if k == 'best_params' and v is not None:
                clean_r[k] = {pk: convert_for_json(pv) for pk, pv in v.items()}
            elif k == 'iteration_history':
                clean_r[k] = [
                    {hk: convert_for_json(hv) for hk, hv in h.items()}
                    for h in v
                ]
            else:
                clean_r[k] = convert_for_json(v)
        clean_results.append(clean_r)

    output_data = {
        'timestamp': timestamp,
        'results': clean_results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark HBV optimization methods'
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=PROJECT_ROOT / "0_config_files" / "config_Bow_lumped_casr_em_earth.yaml",
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=300,
        help='Number of iterations for each method'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with 100 iterations'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=PROJECT_ROOT / 'benchmark_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    n_iterations = 100 if args.quick else args.iterations

    logger.info("=" * 90)
    logger.info("HBV OPTIMIZER BENCHMARK")
    logger.info("=" * 90)
    logger.info(f"Config: {args.config}")
    logger.info(f"Iterations per method: {n_iterations}")
    logger.info("")

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info(f"Domain: {config['DOMAIN_NAME']}")
    logger.info(f"Calibration period: {config.get('CALIBRATION_PERIOD', 'Not specified')}")

    # Define methods to benchmark
    methods = [
        ('Adam', 'finite_difference'),
        ('Adam', 'native'),
        ('DDS', 'n/a'),
        ('DE', 'n/a'),
    ]

    results = []

    for method, gradient_mode in methods:
        logger.info("")
        logger.info("=" * 90)
        logger.info(f"Running: {method.upper()}" + (f" ({gradient_mode})" if gradient_mode != 'n/a' else ""))
        logger.info("=" * 90)

        result = run_optimization(
            config=config,
            method=method,
            n_iterations=n_iterations,
            gradient_mode=gradient_mode
        )
        results.append(result)

        if result['success']:
            logger.info(f"Completed in {result['elapsed_time']:.2f}s, Best KGE: {result['best_score']:.4f}")
        else:
            logger.error(f"Failed: {result['error']}")

    # Print summary
    print_summary_table(results)

    # Save results
    save_results(results, args.output)

    logger.info("")
    logger.info("Benchmark complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
