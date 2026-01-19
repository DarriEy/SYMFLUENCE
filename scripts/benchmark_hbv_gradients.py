#!/usr/bin/env python
"""
Benchmark script comparing finite-difference vs native gradient optimization for HBV.

This script runs Adam and L-BFGS optimization on an HBV calibration problem using
both finite-difference and native (JAX autodiff) gradients, measuring:
- Convergence quality (final KGE/NSE)
- Computation time
- Number of function evaluations

Usage:
    python scripts/benchmark_hbv_gradients.py --config /path/to/config.yaml

    # Quick test with fewer iterations
    python scripts/benchmark_hbv_gradients.py --config /path/to/config.yaml --quick

    # Specify output directory
    python scripts/benchmark_hbv_gradients.py --config /path/to/config.yaml --output ./benchmark_results
"""

import argparse
import logging
import time
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for benchmark."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger('hbv_gradient_benchmark')


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Ensure HBV is the model
    if config.get('HYDROLOGICAL_MODEL', '').upper() != 'HBV':
        logging.warning(
            f"Config specifies {config.get('HYDROLOGICAL_MODEL')}, "
            f"overriding to HBV for benchmark"
        )
        config['HYDROLOGICAL_MODEL'] = 'HBV'

    return config


def run_optimization(
    config: Dict[str, Any],
    algorithm: str,
    gradient_mode: str,
    n_iterations: int,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run optimization with specified settings.

    Args:
        config: Configuration dictionary
        algorithm: 'adam' or 'lbfgs'
        gradient_mode: 'native' or 'finite_difference'
        n_iterations: Number of optimization steps
        logger: Logger instance

    Returns:
        Dictionary with results including timing and scores
    """
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.optimization.model_optimizers import get_optimizer

    # Update config for this run
    run_config = config.copy()
    run_config['GRADIENT_MODE'] = gradient_mode
    run_config['ITERATIVE_OPTIMIZATION_ALGORITHM'] = algorithm.upper()

    if algorithm.lower() == 'adam':
        run_config['ADAM_STEPS'] = n_iterations
        run_config['ADAM_LR'] = 0.01
    else:  # lbfgs
        run_config['LBFGS_STEPS'] = n_iterations
        run_config['LBFGS_LR'] = 0.1

    # Create typed config
    try:
        typed_config = SymfluenceConfig(**run_config)
    except Exception as e:
        logger.warning(f"Could not create typed config: {e}, using dict")
        typed_config = run_config

    # Create optimizer
    optimizer = get_optimizer('HBV', typed_config, logger)

    # Track function evaluations
    eval_count = {'count': 0}
    original_evaluate = optimizer._evaluate_solution

    def counting_evaluate(*args, **kwargs):
        eval_count['count'] += 1
        return original_evaluate(*args, **kwargs)

    optimizer._evaluate_solution = counting_evaluate

    # Run optimization
    logger.info(f"Running {algorithm.upper()} with {gradient_mode} gradients...")
    start_time = time.time()

    try:
        if algorithm.lower() == 'adam':
            results_path = optimizer.run_adam(steps=n_iterations, lr=0.01)
        else:
            results_path = optimizer.run_lbfgs(steps=n_iterations, lr=0.1)

        elapsed_time = time.time() - start_time

        return {
            'algorithm': algorithm,
            'gradient_mode': gradient_mode,
            'n_iterations': n_iterations,
            'elapsed_time': elapsed_time,
            'function_evaluations': eval_count['count'],
            'best_score': optimizer.best_score if hasattr(optimizer, 'best_score') else None,
            'best_params': optimizer.best_params if hasattr(optimizer, 'best_params') else None,
            'results_path': str(results_path) if results_path else None,
            'success': True,
            'error': None
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Optimization failed: {e}")
        return {
            'algorithm': algorithm,
            'gradient_mode': gradient_mode,
            'n_iterations': n_iterations,
            'elapsed_time': elapsed_time,
            'function_evaluations': eval_count['count'],
            'best_score': None,
            'best_params': None,
            'results_path': None,
            'success': False,
            'error': str(e)
        }


def run_synthetic_benchmark(
    n_params: int = 11,
    n_iterations: int = 100,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run benchmark on synthetic HBV-like problem (no data required).

    This creates a mock HBV-like objective function with known optimum
    for testing gradient methods without requiring actual forcing/obs data.

    Args:
        n_params: Number of parameters (default 11 for HBV)
        n_iterations: Number of optimization steps
        logger: Logger instance

    Returns:
        Dictionary with benchmark results
    """
    if logger is None:
        logger = logging.getLogger('synthetic_benchmark')

    try:
        import jax
        import jax.numpy as jnp
        HAS_JAX = True
    except ImportError:
        HAS_JAX = False
        logger.warning("JAX not available, skipping native gradient benchmark")

    # HBV-like parameter bounds (normalized to [0, 1])
    param_names = ['tt', 'cfmax', 'fc', 'lp', 'beta', 'k0', 'k1', 'k2', 'uzl', 'perc', 'maxbas'][:n_params]

    # Synthetic objective: Rosenbrock-like function (challenging)
    # Optimal around 0.6 for all params
    optimal = 0.6

    def objective(x: np.ndarray, step_id: int = 0) -> float:
        """Synthetic objective for benchmarking (maximization)."""
        # Rosenbrock-like with added noise
        val = 0.0
        for i in range(len(x) - 1):
            val += (1 - x[i])**2 + 10 * (x[i+1] - x[i]**2)**2

        # Convert to KGE-like scale [0, 1] and negate for maximization
        kge_like = 1.0 - np.tanh(val / 100)
        return kge_like

    results = {
        'n_params': n_params,
        'n_iterations': n_iterations,
        'param_names': param_names,
        'runs': []
    }

    # Import algorithms
    from symfluence.optimization.optimizers.algorithms.adam import AdamAlgorithm
    from symfluence.optimization.optimizers.algorithms.lbfgs import LBFGSAlgorithm

    base_config = {
        'GRADIENT_MODE': 'auto',
        'GRADIENT_EPSILON': 1e-5,
        'GRADIENT_CLIP_VALUE': 5.0,
        'ADAM_STEPS': n_iterations,
        'ADAM_LR': 0.02,
        'ADAM_BETA1': 0.9,
        'ADAM_BETA2': 0.999,
        'LBFGS_STEPS': n_iterations,
        'LBFGS_LR': 0.5,
        'NUMBER_OF_ITERATIONS': n_iterations,
    }

    # Create JAX gradient function if available
    if HAS_JAX:
        def jax_loss(x):
            val = 0.0
            for i in range(len(x) - 1):
                val = val + (1 - x[i])**2 + 10 * (x[i+1] - x[i]**2)**2
            return val

        jax_grad_fn = jax.grad(jax_loss)

        def native_gradient(x: np.ndarray) -> Tuple[float, np.ndarray]:
            x_jax = jnp.array(x)
            loss = float(jax_loss(x_jax))
            grad = np.array(jax_grad_fn(x_jax))
            return loss, grad

    # Run benchmarks
    algorithms = [
        ('adam', AdamAlgorithm),
        ('lbfgs', LBFGSAlgorithm),
    ]

    gradient_modes = ['finite_difference']
    if HAS_JAX:
        gradient_modes.append('native')

    for alg_name, AlgClass in algorithms:
        for mode in gradient_modes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {alg_name.upper()} with {mode} gradients")
            logger.info(f"{'='*60}")

            # Count evaluations
            eval_count = {'fd': 0, 'native': 0}

            def counting_objective(x, step_id=0):
                eval_count['fd'] += 1
                return objective(x, step_id)

            if HAS_JAX and mode == 'native':
                def counting_native_grad(x):
                    eval_count['native'] += 1
                    return native_gradient(x)
                grad_callback = counting_native_grad
            else:
                grad_callback = None

            algo = AlgClass(base_config.copy(), logger)

            start_time = time.time()
            result = algo.optimize(
                n_params=n_params,
                evaluate_solution=counting_objective,
                evaluate_population=lambda p, i: np.array([counting_objective(x, i) for x in p]),
                denormalize_params=lambda x: {param_names[i]: v for i, v in enumerate(x)},
                record_iteration=lambda *args, **kwargs: None,
                update_best=lambda *args, **kwargs: None,
                log_progress=lambda *args, **kwargs: None,
                compute_gradient=grad_callback,
                gradient_mode=mode
            )
            elapsed = time.time() - start_time

            run_result = {
                'algorithm': alg_name,
                'gradient_mode': mode,
                'elapsed_time': elapsed,
                'best_score': float(result['best_score']),
                'best_solution': result['best_solution'].tolist(),
                'fd_evaluations': eval_count['fd'],
                'native_evaluations': eval_count['native'],
                'gradient_method_used': result.get('gradient_method', 'unknown')
            }

            results['runs'].append(run_result)

            logger.info(f"  Best score: {result['best_score']:.6f}")
            logger.info(f"  Time: {elapsed:.2f}s")
            logger.info(f"  FD evaluations: {eval_count['fd']}")
            if HAS_JAX:
                logger.info(f"  Native evaluations: {eval_count['native']}")

    # Compute speedups
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    for alg in ['adam', 'lbfgs']:
        fd_run = next((r for r in results['runs'] if r['algorithm'] == alg and r['gradient_mode'] == 'finite_difference'), None)
        native_run = next((r for r in results['runs'] if r['algorithm'] == alg and r['gradient_mode'] == 'native'), None)

        if fd_run and native_run:
            speedup = fd_run['elapsed_time'] / native_run['elapsed_time']
            eval_reduction = fd_run['fd_evaluations'] / max(native_run['native_evaluations'], 1)

            logger.info(f"\n{alg.upper()}:")
            logger.info(f"  FD time: {fd_run['elapsed_time']:.2f}s, score: {fd_run['best_score']:.4f}")
            logger.info(f"  Native time: {native_run['elapsed_time']:.2f}s, score: {native_run['best_score']:.4f}")
            logger.info(f"  Speedup: {speedup:.1f}x")
            logger.info(f"  Evaluation reduction: {eval_reduction:.1f}x")

            results[f'{alg}_speedup'] = speedup
            results[f'{alg}_eval_reduction'] = eval_reduction

    return results


def save_results(results: Dict[str, Any], output_dir: Path, logger: logging.Logger):
    """Save benchmark results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'gradient_benchmark_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark HBV optimization with FD vs native gradients'
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Path to configuration YAML file (optional for synthetic benchmark)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('./benchmark_results'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=100,
        help='Number of optimization iterations'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with fewer iterations'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Run synthetic benchmark (no data required)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    logger.info("HBV Gradient Benchmark")
    logger.info("="*60)

    n_iterations = 20 if args.quick else args.iterations

    if args.synthetic or args.config is None:
        # Run synthetic benchmark
        logger.info("Running synthetic benchmark (no data required)")
        results = run_synthetic_benchmark(
            n_params=11,
            n_iterations=n_iterations,
            logger=logger
        )
    else:
        # Run real benchmark with config
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)

        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
        logger.info(f"Domain: {config.get('DOMAIN_NAME')}")

        results = {
            'config_file': str(args.config),
            'domain': config.get('DOMAIN_NAME'),
            'n_iterations': n_iterations,
            'runs': []
        }

        # Run Adam with both gradient modes
        for mode in ['finite_difference', 'native']:
            result = run_optimization(
                config, 'adam', mode, n_iterations, logger
            )
            results['runs'].append(result)

        # Run L-BFGS with both gradient modes
        for mode in ['finite_difference', 'native']:
            result = run_optimization(
                config, 'lbfgs', mode, n_iterations, logger
            )
            results['runs'].append(result)

    # Save results
    output_file = save_results(results, args.output, logger)

    logger.info("\nBenchmark complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
