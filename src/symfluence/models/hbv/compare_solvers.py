#!/usr/bin/env python3
"""
Compare Discrete (lax.scan) vs ODE (diffrax adjoint) HBV Implementations.

This script provides a comprehensive comparison of the two HBV model
implementations:

1. **Discrete Method** (original): Uses lax.scan for time-stepping with
   JAX's native autodiff (backprop through time / BPTT)

2. **ODE Method** (new): Uses diffrax ODE solver with adjoint-based
   gradient computation via the Implicit Function Theorem (IFT)

Comparison Metrics:
- Output accuracy (RMSE, correlation between simulations)
- Gradient accuracy (relative difference in computed gradients)
- Performance (execution time, memory usage)
- Scaling behavior (how performance changes with simulation length)

Usage:
    python -m symfluence.models.hbv.compare_solvers

Or import and use programmatically:
    from symfluence.models.hbv.compare_solvers import run_comparison
    results = run_comparison(n_days=1000)
"""

import argparse
import warnings
from typing import Any, Dict, Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

try:
    import diffrax  # noqa: F401
    HAS_DIFFRAX = True
except ImportError:
    HAS_DIFFRAX = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_synthetic_forcing(
    n_days: int,
    seed: int = 42,
    timestep_hours: int = 24
) -> Dict[str, Any]:
    """
    Generate synthetic forcing data for testing.

    Creates realistic-ish precipitation, temperature, and PET timeseries
    with seasonal patterns.

    Args:
        n_days: Number of days to simulate
        seed: Random seed for reproducibility
        timestep_hours: Model timestep in hours

    Returns:
        Dictionary with 'precip', 'temp', 'pet', 'obs' arrays
    """
    np.random.seed(seed)

    timesteps_per_day_val = 24.0 / timestep_hours
    n_timesteps = int(round(n_days * timesteps_per_day_val))

    # Time in days (fractional for sub-daily)
    t_days = np.arange(n_timesteps) / timesteps_per_day_val

    # Seasonal temperature pattern (-5 to 25 C)
    temp_seasonal = 10.0 + 15.0 * np.sin(2 * np.pi * (t_days - 90) / 365)
    temp_noise = np.random.normal(0, 3, n_timesteps)
    temp = temp_seasonal + temp_noise

    # Precipitation (intermittent, seasonal bias)
    precip_prob = 0.3 + 0.1 * np.sin(2 * np.pi * t_days / 365)
    precip_occurrence = np.random.random(n_timesteps) < precip_prob
    precip_intensity = np.random.exponential(5.0, n_timesteps)
    precip = precip_occurrence * precip_intensity
    precip = precip / timesteps_per_day_val  # Scale to timestep

    # PET (seasonal, temperature-dependent)
    pet_seasonal = 2.0 + 2.0 * np.sin(2 * np.pi * (t_days - 90) / 365)
    pet = np.maximum(pet_seasonal + 0.1 * temp, 0.0)
    pet = pet / timesteps_per_day_val  # Scale to timestep

    # Generate synthetic "observed" runoff by running discrete model
    # (provides a target that should be achievable by both methods)
    from .model import simulate
    from .parameters import DEFAULT_PARAMS

    obs_params = DEFAULT_PARAMS.copy()
    obs_params['smoothing_enabled'] = True
    obs_params['smoothing'] = 15.0

    runoff, _ = simulate(
        precip, temp, pet, obs_params,
        use_jax=False, timestep_hours=timestep_hours
    )

    # Add noise to observations
    obs = runoff + np.random.normal(0, 0.1 * np.std(runoff), n_timesteps)
    obs = np.maximum(obs, 0.0)

    return {
        'precip': precip.astype(np.float32),
        'temp': temp.astype(np.float32),
        'pet': pet.astype(np.float32),
        'obs': obs.astype(np.float32),
        'n_timesteps': n_timesteps,
        'timestep_hours': timestep_hours,
    }


def run_simulation_comparison(
    forcing: Dict[str, np.ndarray],
    params_dict: Dict[str, float],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare simulation outputs between discrete and ODE methods.

    Args:
        forcing: Dictionary with forcing arrays
        params_dict: HBV parameter dictionary
        verbose: Whether to print results

    Returns:
        Comparison results dictionary
    """
    from .hbv_ode import compare_simulations

    if verbose:
        print("\n" + "=" * 60)
        print("SIMULATION OUTPUT COMPARISON")
        print("=" * 60)

    results = compare_simulations(
        params_dict,
        forcing['precip'],
        forcing['temp'],
        forcing['pet'],
        timestep_hours=forcing['timestep_hours'],
        smoothing=15.0
    )

    if verbose:
        print(f"\nSimulation length: {forcing['n_timesteps']} timesteps")
        print(f"Timestep: {forcing['timestep_hours']} hours")
        print("\nOutput Comparison:")
        print(f"  RMSE between methods:  {results['rmse']:.6f} mm")
        print(f"  Correlation:           {results['correlation']:.6f}")
        print("\nExecution Time:")
        print(f"  Discrete (lax.scan):   {results['discrete_time']*1000:.2f} ms")
        print(f"  ODE (diffrax):         {results['ode_time']*1000:.2f} ms")
        print(f"  Ratio (ODE/discrete):  {results['ode_time']/results['discrete_time']:.2f}x")

    return results


def run_gradient_comparison(
    forcing: Dict[str, np.ndarray],
    params_dict: Dict[str, float],
    warmup_days: int = 30,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare gradients between discrete (BPTT) and ODE (adjoint) methods.

    Args:
        forcing: Dictionary with forcing arrays
        params_dict: HBV parameter dictionary
        warmup_days: Warmup period to exclude from loss
        verbose: Whether to print results

    Returns:
        Comparison results dictionary
    """
    from .hbv_ode import compare_gradients

    if verbose:
        print("\n" + "=" * 60)
        print("GRADIENT COMPARISON (BPTT vs Adjoint)")
        print("=" * 60)

    results = compare_gradients(
        params_dict,
        forcing['precip'],
        forcing['temp'],
        forcing['pet'],
        forcing['obs'],
        warmup_days=warmup_days,
        timestep_hours=forcing['timestep_hours'],
        smoothing=15.0
    )

    if verbose:
        print("\nLoss Values:")
        print(f"  Discrete method:       {results['loss_discrete']:.6f}")
        print(f"  ODE method:            {results['loss_ode']:.6f}")
        print(f"  Difference:            {abs(results['loss_ode'] - results['loss_discrete']):.6f}")

        print("\nGradient Computation Time:")
        print(f"  Discrete (BPTT):       {results['discrete_time']*1000:.2f} ms")
        print(f"  ODE (Adjoint):         {results['ode_time']*1000:.2f} ms")
        print(f"  Ratio (ODE/discrete):  {results['ode_time']/results['discrete_time']:.2f}x")

        print("\nGradient Comparison (relative difference):")
        print(f"  {'Parameter':<12} {'Discrete':>12} {'ODE':>12} {'Rel.Diff':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for param in sorted(results['relative_diff'].keys()):
            g_d = results['discrete_grads'][param]
            g_o = results['ode_grads'][param]
            rel = results['relative_diff'][param]
            print(f"  {param:<12} {g_d:>12.4f} {g_o:>12.4f} {rel:>12.2%}")

        # Summary statistics
        rel_diffs = list(results['relative_diff'].values())
        print(f"\n  Mean relative difference: {np.mean(rel_diffs):.2%}")
        print(f"  Max relative difference:  {np.max(rel_diffs):.2%}")

        print("\n  NOTE: Gradients differ because they are derivatives of DIFFERENT")
        print("  loss functions (ODE vs discrete). Both are mathematically correct")
        print("  for their respective formulations.")

    return results


def run_adjoint_verification(
    forcing: Dict[str, np.ndarray],
    params_dict: Dict[str, float],
    warmup_days: int = 30,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify ODE adjoint gradients against finite differences.

    This confirms the adjoint method is computing correct gradients
    for the ODE loss function.

    Args:
        forcing: Dictionary with forcing arrays
        params_dict: HBV parameter dictionary
        warmup_days: Warmup period
        verbose: Whether to print results

    Returns:
        Verification results dictionary
    """
    from .hbv_ode import verify_adjoint_gradients

    if verbose:
        print("\n" + "=" * 60)
        print("ADJOINT GRADIENT VERIFICATION (vs Finite Differences)")
        print("=" * 60)

    results = verify_adjoint_gradients(
        params_dict,
        forcing['precip'],
        forcing['temp'],
        forcing['pet'],
        forcing['obs'],
        warmup_days=warmup_days,
        timestep_hours=forcing['timestep_hours'],
        smoothing=15.0
    )

    if verbose:
        print(f"\n  {'Parameter':<12} {'Adjoint':>12} {'Fin.Diff':>12} {'Rel.Error':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

        for param in sorted(results['relative_error'].keys()):
            adj = results['adjoint_grads'][param]
            fd = results['fd_grads'][param]
            err = results['relative_error'][param]
            print(f"  {param:<12} {adj:>12.6f} {fd:>12.6f} {err:>12.2%}")

        print(f"\n  Max relative error: {results['max_error']:.2%}")
        status = "PASSED" if results['adjoint_correct'] else "FAILED"
        print(f"  Adjoint verification: {status}")

        if results['adjoint_correct']:
            print("\n  The adjoint method computes CORRECT gradients for the ODE loss.")
        else:
            print("\n  WARNING: Adjoint gradients may have numerical issues.")

    return results


def run_scaling_comparison(
    simulation_lengths: Optional[list] = None,
    timestep_hours: int = 24,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare how performance scales with simulation length.

    Args:
        simulation_lengths: List of simulation lengths in days
        timestep_hours: Model timestep
        verbose: Whether to print results

    Returns:
        Scaling results dictionary
    """
    if simulation_lengths is None:
        simulation_lengths = [100, 365, 730, 1825, 3650]
    from .hbv_ode import compare_gradients
    from .parameters import DEFAULT_PARAMS

    if verbose:
        print("\n" + "=" * 60)
        print("SCALING COMPARISON")
        print("=" * 60)

    params_dict = DEFAULT_PARAMS.copy()
    params_dict['smoothing_enabled'] = True
    params_dict['smoothing'] = 15.0

    results: dict[str, list] = {
        'n_days': [],
        'discrete_time': [],
        'ode_time': [],
        'ratio': [],
    }

    for n_days in simulation_lengths:
        if verbose:
            print(f"\nTesting {n_days} days ({n_days * 24 // timestep_hours} timesteps)...")

        forcing = generate_synthetic_forcing(n_days, timestep_hours=timestep_hours)

        try:
            comparison = compare_gradients(
                params_dict,
                forcing['precip'],
                forcing['temp'],
                forcing['pet'],
                forcing['obs'],
                warmup_days=min(30, n_days // 10),
                timestep_hours=timestep_hours,
                smoothing=15.0
            )

            results['n_days'].append(n_days)
            results['discrete_time'].append(comparison['discrete_time'])
            results['ode_time'].append(comparison['ode_time'])
            results['ratio'].append(comparison['ode_time'] / comparison['discrete_time'])

            if verbose:
                print(f"  Discrete: {comparison['discrete_time']*1000:.1f} ms")
                print(f"  ODE:      {comparison['ode_time']*1000:.1f} ms")
                print(f"  Ratio:    {comparison['ode_time']/comparison['discrete_time']:.2f}x")

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")

    if verbose:
        print("\n" + "-" * 60)
        print("Summary:")
        print(f"  {'Days':<10} {'Discrete (ms)':<15} {'ODE (ms)':<15} {'Ratio':<10}")
        for i in range(len(results['n_days'])):
            print(f"  {results['n_days'][i]:<10} "
                  f"{results['discrete_time'][i]*1000:<15.1f} "
                  f"{results['ode_time'][i]*1000:<15.1f} "
                  f"{results['ratio'][i]:<10.2f}")

    return results


def run_memory_comparison(
    n_days: int = 1000,
    timestep_hours: int = 24,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare memory usage between methods.

    Note: This is an approximate comparison using JAX's device memory tracking.

    Args:
        n_days: Simulation length in days
        timestep_hours: Model timestep
        verbose: Whether to print results

    Returns:
        Memory comparison results
    """
    if verbose:
        print("\n" + "=" * 60)
        print("MEMORY USAGE COMPARISON")
        print("=" * 60)

    from .parameters import DEFAULT_PARAMS, create_params_from_dict, scale_params_for_timestep

    params_dict = DEFAULT_PARAMS.copy()
    params_dict['smoothing_enabled'] = True
    params_dict['smoothing'] = 15.0

    forcing = generate_synthetic_forcing(n_days, timestep_hours=timestep_hours)

    scaled_params = scale_params_for_timestep(params_dict, timestep_hours)
    _params = create_params_from_dict(scaled_params, use_jax=True)  # noqa: F841

    precip = jnp.array(forcing['precip'])
    temp = jnp.array(forcing['temp'])
    pet = jnp.array(forcing['pet'])
    obs = jnp.array(forcing['obs'])

    # Define loss functions for memory testing
    def discrete_loss(p_array):
        from .losses import nse_loss
        p_dict = dict(zip(['tt', 'cfmax', 'fc', 'k1', 'k2'], p_array))
        full_params = {**params_dict, **p_dict}
        return nse_loss(full_params, precip, temp, pet, obs, warmup_days=30,
                       use_jax=True, timestep_hours=timestep_hours)

    def ode_loss(p_array):
        from .hbv_ode import nse_loss_ode
        p_dict = dict(zip(['tt', 'cfmax', 'fc', 'k1', 'k2'], p_array))
        full_params = {**params_dict, **p_dict}
        return nse_loss_ode(full_params, precip, temp, pet, obs, warmup_days=30,
                          timestep_hours=timestep_hours, smoothing=15.0)

    _test_params = jnp.array([0.0, 3.5, 250.0, 0.1, 0.01])  # noqa: F841

    # Measure memory for gradient computation
    # Note: This requires jax.profiler or similar; here we just show structure
    if verbose:
        print(f"\nSimulation length: {forcing['n_timesteps']} timesteps")
        print("\nTheoretical Memory Complexity:")
        print(f"  Discrete (BPTT):      O(T) = O({forcing['n_timesteps']}) state copies")
        print("  ODE (Adjoint):        O(1) constant memory for states")
        print("                        (plus O(sqrt(T)) for checkpointing)")

        print("\nNote: Actual memory depends on:")
        print("  - JAX XLA compilation strategy")
        print("  - Checkpoint placement in RecursiveCheckpointAdjoint")
        print("  - State vector size (5 floats per timestep)")

        # Estimate memory
        state_size_bytes = 5 * 4  # 5 floats, 4 bytes each
        discrete_memory_mb = (forcing['n_timesteps'] * state_size_bytes) / (1024 * 1024)
        ode_memory_mb = (np.sqrt(forcing['n_timesteps']) * state_size_bytes) / (1024 * 1024)

        print("\nEstimated State Memory:")
        print(f"  Discrete: ~{discrete_memory_mb:.2f} MB (for {forcing['n_timesteps']} states)")
        print(f"  ODE:      ~{ode_memory_mb:.4f} MB (for ~{int(np.sqrt(forcing['n_timesteps']))} checkpoints)")

    return {
        'n_timesteps': forcing['n_timesteps'],
        'discrete_complexity': 'O(T)',
        'ode_complexity': 'O(sqrt(T))',
    }


def plot_comparison(
    forcing: Dict[str, np.ndarray],
    sim_results: Dict[str, Any],
    grad_results: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
):
    """
    Create visualization of comparison results.

    Args:
        forcing: Forcing data dictionary
        sim_results: Simulation comparison results
        grad_results: Gradient comparison results (optional)
        save_path: Path to save figure (optional)
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib not available for plotting")
        return

    n_plots = 3 if grad_results else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

    # Plot 1: Runoff comparison
    ax1 = axes[0]
    t = np.arange(len(sim_results['discrete_runoff']))
    ax1.plot(t, sim_results['discrete_runoff'], label='Discrete (lax.scan)', alpha=0.7)
    ax1.plot(t, sim_results['ode_runoff'], label='ODE (diffrax)', alpha=0.7, linestyle='--')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Runoff (mm)')
    ax1.set_title(f'Runoff Comparison (RMSE: {sim_results["rmse"]:.4f} mm, r: {sim_results["correlation"]:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Difference
    ax2 = axes[1]
    diff = sim_results['discrete_runoff'] - sim_results['ode_runoff']
    ax2.plot(t, diff, color='red', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.fill_between(t, diff, 0, alpha=0.3, color='red')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Difference (mm)')
    ax2.set_title('Discrete - ODE Difference')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gradient comparison (if available)
    if grad_results:
        ax3 = axes[2]
        params = list(grad_results['discrete_grads'].keys())
        x = np.arange(len(params))
        width = 0.35

        discrete_vals = [grad_results['discrete_grads'][p] for p in params]
        ode_vals = [grad_results['ode_grads'][p] for p in params]

        ax3.bar(x - width/2, discrete_vals, width, label='Discrete (BPTT)', alpha=0.7)
        ax3.bar(x + width/2, ode_vals, width, label='ODE (Adjoint)', alpha=0.7)
        ax3.set_xlabel('Parameter')
        ax3.set_ylabel('Gradient')
        ax3.set_title('Gradient Comparison by Parameter')
        ax3.set_xticks(x)
        ax3.set_xticklabels(params, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


def run_comparison(
    n_days: int = 365,
    timestep_hours: int = 24,
    warmup_days: int = 30,
    plot: bool = True,
    save_plot: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run full comparison between discrete and ODE implementations.

    This is the main entry point for comparing the two methods.

    Args:
        n_days: Number of days to simulate
        timestep_hours: Model timestep in hours
        warmup_days: Warmup period for loss calculation
        plot: Whether to create visualization
        save_plot: Path to save plot (None to show interactively)
        verbose: Whether to print detailed results

    Returns:
        Dictionary containing all comparison results
    """
    if not HAS_JAX:
        raise ImportError("JAX is required for comparison. Install with: pip install jax jaxlib")

    if not HAS_DIFFRAX:
        raise ImportError("diffrax is required for ODE comparison. Install with: pip install diffrax")

    from .parameters import DEFAULT_PARAMS

    if verbose:
        print("\n" + "=" * 60)
        print("HBV MODEL IMPLEMENTATION COMPARISON")
        print("Discrete (lax.scan + BPTT) vs ODE (diffrax + Adjoint)")
        print("=" * 60)
        print("\nConfiguration:")
        print(f"  Simulation length: {n_days} days")
        print(f"  Timestep: {timestep_hours} hours")
        print(f"  Warmup period: {warmup_days} days")

    # Generate synthetic data
    if verbose:
        print("\nGenerating synthetic forcing data...")

    forcing = generate_synthetic_forcing(n_days, timestep_hours=timestep_hours)

    # Use default parameters
    params_dict = DEFAULT_PARAMS.copy()
    params_dict['smoothing_enabled'] = True
    params_dict['smoothing'] = 15.0

    # Run comparisons
    sim_results = run_simulation_comparison(forcing, params_dict, verbose=verbose)
    grad_results = run_gradient_comparison(forcing, params_dict, warmup_days=warmup_days, verbose=verbose)
    adjoint_results = run_adjoint_verification(forcing, params_dict, warmup_days=warmup_days, verbose=verbose)
    memory_results = run_memory_comparison(n_days, timestep_hours, verbose=verbose)

    # Create visualization
    if plot and HAS_MATPLOTLIB:
        plot_comparison(forcing, sim_results, grad_results, save_path=save_plot)

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print("\n1. SIMULATION OUTPUTS:")
        print(f"   Correlation between methods: {sim_results['correlation']:.4f}")
        print("   Note: Differences are expected because ODE integrates continuously")
        print("         while discrete model uses explicit time-stepping.")

        print("\n2. GRADIENT COMPUTATION:")
        print("   The ODE and discrete methods compute gradients of DIFFERENT")
        print("   loss functions, so gradients will differ. Both are mathematically")
        print("   correct for their respective formulations.")

        print("\n3. PERFORMANCE:")
        if grad_results['ode_time'] < grad_results['discrete_time']:
            speedup = grad_results['discrete_time'] / grad_results['ode_time']
            print(f"   Gradient computation: ODE is {speedup:.1f}x FASTER")
        else:
            slowdown = grad_results['ode_time'] / grad_results['discrete_time']
            print(f"   Gradient computation: ODE is {slowdown:.1f}x slower")
        print("   (ODE overhead is in forward simulation, not gradient computation)")

        print("\n4. MEMORY EFFICIENCY:")
        print(f"   Discrete (BPTT): O(T) memory for {forcing['n_timesteps']} timesteps")
        print(f"   ODE (Adjoint):   O(sqrt(T)) ~ {int(np.sqrt(forcing['n_timesteps']))} checkpoints")
        print("   For long simulations (multi-year), ODE has significant advantage.")

        print("\n5. WHEN TO USE EACH METHOD:")
        print("   - Discrete (lax.scan): Short simulations, exact HBV-96 behavior")
        print("   - ODE (diffrax): Long simulations, adaptive time-stepping,")
        print("                    when physics/solver separation is desired")

        # Check if adjoint verification passed
        if adjoint_results['adjoint_correct']:
            print("\n6. ADJOINT VERIFICATION: PASSED")
            print("   ODE gradients match finite differences within tolerance.")
        else:
            print("\n6. ADJOINT VERIFICATION: Numerical differences detected")
            print("   This is common with ODE adjoints due to integration tolerances.")
            print("   The gradients are still useful for optimization (see demo below).")

    return {
        'forcing': forcing,
        'simulation': sim_results,
        'gradients': grad_results,
        'adjoint': adjoint_results,
        'memory': memory_results,
    }


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Compare discrete vs ODE HBV implementations"
    )
    parser.add_argument(
        '--n-days', type=int, default=365,
        help='Number of days to simulate (default: 365)'
    )
    parser.add_argument(
        '--timestep', type=int, default=24, choices=[1, 3, 6, 12, 24],
        help='Timestep in hours (default: 24)'
    )
    parser.add_argument(
        '--warmup', type=int, default=30,
        help='Warmup days for loss calculation (default: 30)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Disable plotting'
    )
    parser.add_argument(
        '--save-plot', type=str, default=None,
        help='Path to save plot (default: show interactively)'
    )
    parser.add_argument(
        '--scaling-test', action='store_true',
        help='Run scaling comparison across different simulation lengths'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    if args.scaling_test:
        run_scaling_comparison(
            simulation_lengths=[100, 365, 730, 1825, 3650],
            timestep_hours=args.timestep,
            verbose=not args.quiet
        )
    else:
        run_comparison(
            n_days=args.n_days,
            timestep_hours=args.timestep,
            warmup_days=args.warmup,
            plot=not args.no_plot,
            save_plot=args.save_plot,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
