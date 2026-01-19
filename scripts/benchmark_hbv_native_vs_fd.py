#!/usr/bin/env python
"""
Benchmark HBV native vs FD gradients using real Bow at Banff data.

This runs actual HBV model evaluations with the real forcing and observation data.
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import yaml

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('hbv_benchmark')

def main():
    config_path = PROJECT_ROOT / "0_config_files" / "config_Bow_lumped_casr_em_earth.yaml"

    logger.info("="*70)
    logger.info("HBV Native vs FD Gradient Benchmark - Real Data")
    logger.info("="*70)

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Config: {config_path.name}")
    logger.info(f"Domain: {config['DOMAIN_NAME']}")

    # Import HBV optimizer via registry
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.optimization.registry import OptimizerRegistry

    # Override for benchmark
    config['NUMBER_OF_ITERATIONS'] = 30
    config['GRADIENT_MODE'] = 'auto'

    # Get HBV optimizer class from registry
    HBVOptimizer = OptimizerRegistry.get_optimizer('HBV')

    results = {}

    n_steps = 20  # Number of Adam steps for benchmark

    # Test 1: Adam with FD
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Adam with Finite Differences")
    logger.info("="*70)

    config_fd = config.copy()
    config_fd['GRADIENT_MODE'] = 'finite_difference'
    config_fd['ADAM_STEPS'] = n_steps
    config_fd['ADAM_LR'] = 0.01

    try:
        typed_config_fd = SymfluenceConfig(**config_fd)
        optimizer_fd = HBVOptimizer(config_fd, logger)

        start = time.time()
        result_path = optimizer_fd.run_adam(steps=n_steps, lr=0.01)
        elapsed_fd = time.time() - start

        results['adam_fd'] = {
            'time': elapsed_fd,
            'best_score': optimizer_fd.best_score if hasattr(optimizer_fd, 'best_score') else None,
        }
        logger.info(f"Adam FD completed in {elapsed_fd:.2f}s")
        logger.info(f"Best score: {results['adam_fd']['best_score']}")
    except Exception as e:
        logger.error(f"Adam FD failed: {e}")
        import traceback
        traceback.print_exc()
        results['adam_fd'] = {'error': str(e)}

    # Test 2: Adam with Native gradients
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Adam with Native Gradients (JAX)")
    logger.info("="*70)

    config_native = config.copy()
    config_native['GRADIENT_MODE'] = 'native'
    config_native['ADAM_STEPS'] = n_steps
    config_native['ADAM_LR'] = 0.01

    try:
        typed_config_native = SymfluenceConfig(**config_native)
        optimizer_native = HBVOptimizer(config_native, logger)

        start = time.time()
        result_path = optimizer_native.run_adam(steps=n_steps, lr=0.01)
        elapsed_native = time.time() - start

        results['adam_native'] = {
            'time': elapsed_native,
            'best_score': optimizer_native.best_score if hasattr(optimizer_native, 'best_score') else None,
        }
        logger.info(f"Adam Native completed in {elapsed_native:.2f}s")
        logger.info(f"Best score: {results['adam_native']['best_score']}")
    except Exception as e:
        logger.error(f"Adam Native failed: {e}")
        import traceback
        traceback.print_exc()
        results['adam_native'] = {'error': str(e)}

    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)

    if 'adam_fd' in results and 'adam_native' in results:
        if 'time' in results['adam_fd'] and 'time' in results['adam_native']:
            speedup = results['adam_fd']['time'] / results['adam_native']['time']
            logger.info(f"Adam FD time:     {results['adam_fd']['time']:.2f}s")
            logger.info(f"Adam Native time: {results['adam_native']['time']:.2f}s")
            logger.info(f"Speedup:          {speedup:.1f}x")

    return 0

if __name__ == '__main__':
    sys.exit(main())
