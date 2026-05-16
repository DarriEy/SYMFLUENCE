#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
SCE-UA (Shuffled Complex Evolution - University of Arizona) Algorithm

A global optimization algorithm that combines the strengths of the simplex
procedure with competitive evolution. Widely used in hydrological modeling.

Reference:
    Duan, Q., Sorooshian, S., and Gupta, V.K. (1992). Effective and Efficient
    Global Optimization for Conceptual Rainfall-Runoff Models.
    Water Resources Research, 28(4), 1015-1031.

Parallel Execution Design
--------------------------
The original serial SCE-UA processes each complex sequentially, issuing one
``evaluate_solution`` call at a time.  This implementation replaces that serial
pattern with batched evaluation via ``evaluate_population``, which dispatches
all candidates in a batch to the underlying parallel executor (ProcessPool or
MPI) in a single call.

**Why the parallelisation is algorithmically valid**

The CCE procedure evolves each complex independently.  After the population is
partitioned into complexes, no information flows *between* complexes until the
shuffle step that reunites them at the end of the iteration.  This inter-complex
independence is an explicit, load-bearing property of the SCE algorithm: it is
what gives the S (Shuffled) in the name its meaning.  Exploiting it for
parallelism therefore does not alter the algorithm's search behaviour — it only
changes the order in which the model is called.

**What is preserved**

The *intra*-complex dependency chain is fully respected.  Within a single
complex, step k+1 uses the complex state that was updated by step k — this
serial chain cannot be parallelised without changing the algorithm, so it is
not.

**Batching strategy per evolution step**

At each CCE evolution step every complex independently selects a random
simplex, generates a reflection point, and may fall back to a contraction or a
random replacement.  Since the candidates across complexes are independent, we
batch-evaluate them in up to three rounds per step:

  Round 1  — Reflection batch (always n_complexes candidates)
             Complexes whose reflection improves the worst simplex point accept
             it immediately.  The rest proceed to round 2.

  Round 2  — Contraction batch (only complexes that failed reflection)
             A contraction point is formed by pulling the worst point halfway
             towards the centroid.  Complexes that improve accept it.  The rest
             proceed to round 3.

  Round 3  — Random replacement batch (only complexes that failed both)
             A uniformly random point replaces the worst point if it improves
             the fitness; otherwise the worst point is left unchanged (the
             original SCE behaviour).

**Worst-case model calls per shuffle iteration**

  Serial original : n_complexes × n_evolution_steps × 3  (all fallback paths)
  Parallel new    : n_evolution_steps × 3 batches          (same calls, fewer wall-clock rounds)

With default settings (5 complexes, 33 evolution steps) this reduces the
number of sequential evaluation rounds from up to 495 to at most 99, giving an
ideal 5× wall-clock speedup when using 5 or more parallel workers.
"""

import csv
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base_algorithm import OptimizationAlgorithm


class SCEUAAlgorithm(OptimizationAlgorithm):
    """Shuffled Complex Evolution algorithm."""

    @property
    def name(self) -> str:
        """Algorithm identifier for logging and result tracking."""
        return "SCE-UA"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_reflection_candidates(
        self,
        sub_complexes: List[np.ndarray],
        sub_fitnesses: List[np.ndarray],
        complex_sizes: List[int],
        n_params: int,
    ) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
        """Generate one reflection candidate per complex and collect simplex state.

        For each complex a random simplex of size ``n_params + 1`` is chosen
        from the complex members.  The worst point in the simplex is identified,
        and the centroid of the remaining points is used to compute the classic
        Nelder–Mead reflection:

            reflection = 2 * centroid − worst_point

        The result is clipped to [0, 1] to keep it inside the normalised
        parameter space.

        Args:
            sub_complexes: List of parameter arrays, one per complex
                           (shape: [n_members_in_complex, n_params]).
            sub_fitnesses: Corresponding fitness vectors.
            complex_sizes: Number of members in each complex.
            n_params: Number of optimisation parameters.

        Returns:
            reflection_batch:
                Array of shape (n_complexes, n_params) ready for batch
                evaluation.
            per_step_state:
                List of (worst_idx_in_sub_complex, centroid) tuples, one per
                complex.  These are reused in the contraction stage without
                needing to recompute them.
        """
        n_complexes = len(sub_complexes)
        simplex_size = n_params + 1
        reflection_batch = np.empty((n_complexes, n_params))
        per_step_state: List[Tuple[int, np.ndarray]] = []

        for c in range(n_complexes):
            sub_complex = sub_complexes[c]
            sub_fitness = sub_fitnesses[c]
            n_members = complex_sizes[c]

            # Choose a random simplex (n_params+1 unique members)
            simplex_idx = np.random.choice(n_members, simplex_size, replace=False)

            # The worst point in the simplex is the reflection target
            worst_in_simplex = simplex_idx[np.argmin(sub_fitness[simplex_idx])]
            others = [i for i in simplex_idx if i != worst_in_simplex]
            centroid = np.mean(sub_complex[others], axis=0)

            # Reflection: move the worst point through the centroid
            reflection = 2.0 * centroid - sub_complex[worst_in_simplex]
            reflection = np.clip(reflection, 0.0, 1.0)

            reflection_batch[c] = reflection
            per_step_state.append((worst_in_simplex, centroid))

        return reflection_batch, per_step_state

    def _apply_cce_step(
        self,
        sub_complexes: List[np.ndarray],
        sub_fitnesses: List[np.ndarray],
        per_step_state: List[Tuple[int, np.ndarray]],
        reflection_batch: np.ndarray,
        reflection_fitness: np.ndarray,
        n_params: int,
        iteration: int,
        evaluate_population: Callable,
        log_evolution_batch: Optional[Callable[[List[int], np.ndarray, np.ndarray, str], None]] = None,
    ) -> None:
        """Apply one CCE step to all complexes using batched contraction/random fallbacks.

        Accepts reflection candidates where they improve fitness.  For complexes
        where the reflection fails, batches contraction candidates and evaluates
        them in parallel.  For complexes where contraction also fails, batches
        random replacements and evaluates those too.

        All updates are performed in-place on ``sub_complexes`` and
        ``sub_fitnesses``.

        Args:
            sub_complexes: List of parameter arrays per complex (modified in-place).
            sub_fitnesses: Corresponding fitness vectors (modified in-place).
            per_step_state: (worst_idx, centroid) tuples from
                            ``_generate_reflection_candidates``.
            reflection_batch: Candidate array (n_complexes, n_params).
            reflection_fitness: Fitness scores for each reflection candidate.
            n_params: Number of optimisation parameters.
            iteration: Current shuffle iteration (passed to evaluate_population
                       for logging/tracking).
            evaluate_population: Batch evaluation callback.
            log_evolution_batch: Optional callback used to persist per-batch
                evolution diagnostics to CSV.
        """
        n_complexes = len(sub_complexes)

        if log_evolution_batch:
            log_evolution_batch(
                list(range(n_complexes)),
                reflection_batch,
                reflection_fitness,
                'reflection',
            )

        # ---- Stage 1: Accept reflections; queue contractions for failures ----
        # Complexes whose reflection improved the worst simplex point are updated
        # immediately.  For the rest, a contraction point is formed and queued for
        # parallel evaluation.
        contraction_complex_ids: List[int] = []
        contraction_candidates: List[np.ndarray] = []

        for c in range(n_complexes):
            worst_idx, centroid = per_step_state[c]

            if reflection_fitness[c] > sub_fitnesses[c][worst_idx]:
                # Reflection improved the worst point — accept it
                sub_complexes[c][worst_idx] = reflection_batch[c]
                sub_fitnesses[c][worst_idx] = reflection_fitness[c]
            else:
                # Reflection failed — compute contraction: move worst halfway
                # towards the centroid (midpoint contraction from Nelder-Mead)
                contraction = (
                    sub_complexes[c][worst_idx]
                    + 0.5 * (centroid - sub_complexes[c][worst_idx])
                )
                contraction_complex_ids.append(c)
                contraction_candidates.append(contraction)

        if not contraction_complex_ids:
            # All complexes accepted their reflections; nothing left to do
            return

        # ---- Stage 2: Batch-evaluate all contraction candidates ----
        contraction_batch = np.array(contraction_candidates)
        contraction_fitness = evaluate_population(contraction_batch, iteration)

        if log_evolution_batch:
            log_evolution_batch(
                contraction_complex_ids,
                contraction_batch,
                contraction_fitness,
                'contraction',
            )

        # Accept contractions where they improve; queue random replacements for failures
        random_complex_ids: List[int] = []
        random_candidates: List[np.ndarray] = []

        for batch_i, c in enumerate(contraction_complex_ids):
            worst_idx, _ = per_step_state[c]

            if contraction_fitness[batch_i] > sub_fitnesses[c][worst_idx]:
                # Contraction improved the worst point — accept it
                sub_complexes[c][worst_idx] = contraction_batch[batch_i]
                sub_fitnesses[c][worst_idx] = contraction_fitness[batch_i]
            else:
                # Contraction also failed — fall back to a uniformly random point.
                # This is the original SCE-UA final fallback.
                random_point = np.random.uniform(0.0, 1.0, n_params)
                random_complex_ids.append(c)
                random_candidates.append(random_point)

        if not random_complex_ids:
            # All remaining complexes accepted their contractions
            return

        # ---- Stage 3: Batch-evaluate all random replacement candidates ----
        random_batch = np.array(random_candidates)
        random_fitness = evaluate_population(random_batch, iteration)

        if log_evolution_batch:
            log_evolution_batch(
                random_complex_ids,
                random_batch,
                random_fitness,
                'random',
            )

        for batch_i, c in enumerate(random_complex_ids):
            worst_idx, _ = per_step_state[c]

            if random_fitness[batch_i] > sub_fitnesses[c][worst_idx]:
                # Random point is better — accept it
                sub_complexes[c][worst_idx] = random_batch[batch_i]
                sub_fitnesses[c][worst_idx] = random_fitness[batch_i]
            # If the random point is also not better, the worst point is left
            # unchanged.  This matches the original serial SCE-UA behaviour.

    # ------------------------------------------------------------------
    # Main optimisation entry point
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_params: int,
        evaluate_solution: Callable[[np.ndarray, int], float],
        evaluate_population: Callable[[np.ndarray, int], np.ndarray],
        denormalize_params: Callable[[np.ndarray], Dict],
        record_iteration: Callable,
        update_best: Callable,
        log_progress: Callable,
        evaluate_population_objectives: Optional[Callable] = None,
        compute_gradient: Optional[Callable] = None,
        gradient_mode: str = 'auto',
        log_initial_population: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the SCE-UA optimisation algorithm with parallel CCE evaluation.

        The algorithm proceeds in three phases:

        1. **Initialisation** — A random population of ``pop_size`` points is
           drawn and evaluated in one parallel batch.  Points are ranked by
           fitness (descending), establishing the initial complex partition.

        2. **Shuffle iterations** — Each iteration:

           a. Partition the ranked population into ``n_complexes`` complexes
              using the standard interleaved assignment
              (complex c owns ranks c, c+n_complexes, c+2*n_complexes, …).

           b. Run ``n_evolution_steps`` CCE steps.  At each step:
              - Generate one reflection candidate per complex (cheap, CPU-only).
              - Batch-evaluate all reflections in parallel via
                ``evaluate_population``.
              - Accept successful reflections; batch-evaluate contractions for
                failures; accept successful contractions; batch-evaluate random
                replacements for remaining failures.
              All three evaluation rounds involve only complexes that reached
              that stage, minimising total model calls.

           c. Merge the evolved complexes back into the population array and
              re-sort by fitness (the "shuffle" step).

        3. **Early stopping** — If the best fitness changes by less than
           ``percent_change_threshold`` for ``evolution_stagnation`` consecutive
           shuffle iterations, optimisation terminates.

        Parallel execution is handled entirely by ``evaluate_population``, which
        dispatches batches to the configured strategy (ProcessPool or MPI) in the
        ``PopulationEvaluator``.  This method uses ``evaluate_solution`` only if
        called externally; the internal CCE loop exclusively uses
        ``evaluate_population`` to keep all model evaluations on workers.

        Args:
            n_params: Number of parameters to optimise.
            evaluate_solution: Callback to evaluate a single normalised solution.
                               Not used internally; present for interface
                               compatibility.
            evaluate_population: Callback to evaluate a batch of solutions.
                                  Signature: (population: ndarray[N, n_params],
                                  iteration: int) -> ndarray[N].
            denormalize_params: Callback to convert a normalised parameter vector
                                to a human-readable dictionary.
            record_iteration: Callback to persist the result of each iteration.
            update_best: Callback to update the tracked global best.
            log_progress: Callback to emit a progress log line.
            evaluate_population_objectives: Unused by SCE-UA (single-objective
                                            algorithm).
            compute_gradient: Unused by SCE-UA (gradient-free algorithm).
            gradient_mode: Unused by SCE-UA.
            log_initial_population: Optional callback invoked after the initial
                                    population is evaluated.
            **kwargs: Reserved for future algorithm-specific extensions.

        Returns:
            dict with keys:
                ``best_solution``  — Best normalised parameter vector found.
                ``best_score``     — Best fitness score.
                ``best_params``    — Best parameters as a denormalised dict.
        """
        self.logger.info(f"Starting SCE-UA optimization with {n_params} parameters")

        # Evolution tracking output config
        run_id = kwargs.get('experiment_id')
        if run_id is None:
            run_id = self._get_config_value(
                lambda: self.config.domain.experiment_id,
                default='optimization',
                dict_key='EXPERIMENT_ID'
            )

        results_dir_arg = kwargs.get('results_dir')
        tracking_csv_path: Optional[Path] = None
        if results_dir_arg is not None:
            try:
                results_dir = Path(results_dir_arg)
                results_dir.mkdir(parents=True, exist_ok=True)
                tracking_csv_path = results_dir / f"sce_{run_id}_evolution_tracking.csv"
            except Exception as e:  # noqa: BLE001 - best-effort tracking path setup
                self.logger.warning(f"Failed to initialize SCE evolution tracking path: {e}")
        else:
            self.logger.warning(
                "SCE evolution tracking disabled: results_dir was not provided to optimizer callbacks"
            )

        # ---- Read SCE-UA parameters from config ----
        n_complexes = self._get_config_value(
            lambda: self.config.optimization.sce_ua.number_of_complexes,
            default=max(2, self.population_size // 10),
            dict_key='NUMBER_OF_COMPLEXES'
        )
        n_evolution_steps = self._get_config_value(
            lambda: self.config.optimization.sce_ua.number_of_evolution_steps,
            default=2 * n_params + 1,
            dict_key='NUMBER_OF_EVOLUTION_STEPS'
        )

        # SCE-UA convention: each complex holds 2*n_params+1 members so that a
        # simplex of n_params+1 points can always be drawn from it with room
        # to spare, giving the simplex operations enough geometric coverage.
        n_per_complex = 2 * n_params + 1
        pop_size = n_complexes * n_per_complex

        # Early stopping parameters from config
        stagnation_limit = self._get_config_value(
            lambda: self.config.optimization.sce_ua.evolution_stagnation,
            default=5,
            dict_key='EVOLUTION_STAGNATION'
        )
        pct_change_threshold = self._get_config_value(
            lambda: self.config.optimization.sce_ua.percent_change_threshold,
            default=0.01,
            dict_key='PERCENT_CHANGE_THRESHOLD'
        )

        self.logger.info(
            f"SCE-UA structure: {n_complexes} complexes × {n_per_complex} points/complex "
            f"= {pop_size} total, {n_evolution_steps} evolution steps per shuffle"
        )

        # ---- Phase 1: Initialise and evaluate the full population ----
        self.logger.info(f"Evaluating initial population ({pop_size} individuals)...")
        population = np.random.uniform(0.0, 1.0, (pop_size, n_params))
        fitness = evaluate_population(population, 0)

        # Rank population: index 0 is the best individual (descending fitness)
        sorted_idx = np.argsort(-fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        best_pos = population[0].copy()
        best_fit = fitness[0]

        params_dict = denormalize_params(best_pos)
        record_iteration(0, best_fit, params_dict)
        update_best(best_fit, params_dict, 0)

        # Initialize CSV header once parameter names are available.
        param_names = list(params_dict.keys())
        tracking_headers = [
            'timestamp',
            'run_id',
            'metric_name',
            'shuffle_iteration',
            'evolution_id',
            'complex_id',
            'stage',
            'score',
            *param_names,
        ]

        if tracking_csv_path is not None:
            with tracking_csv_path.open('w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=tracking_headers)
                writer.writeheader()

        tracking_rows_buffer: List[Dict[str, Any]] = []

        metric_name = self._get_config_value(
            lambda: self.config.optimization.metric,
            default='KGE',
            dict_key='OPTIMIZATION_METRIC'
        )

        evolution_id = 0

        def _to_csv_scalar(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                if value.ndim == 0 or value.size == 1:
                    return float(value.ravel()[0])
                return ';'.join(str(float(v)) for v in value.ravel())
            if isinstance(value, (np.floating, np.integer)):
                return float(value)
            return value

        def log_evolution_batch(
            complex_ids: List[int],
            candidate_batch: np.ndarray,
            candidate_fitness: np.ndarray,
            stage: str,
            shuffle_iteration: int,
            evolution_step: int,
        ) -> None:
            if tracking_csv_path is None:
                return

            rows: List[Dict[str, Any]] = []
            for row_idx, complex_id in enumerate(complex_ids):
                denorm = denormalize_params(candidate_batch[row_idx])
                row: Dict[str, Any] = {
                    'timestamp': datetime.now().isoformat(),
                    'run_id': run_id,
                    'metric_name': metric_name,
                    'shuffle_iteration': shuffle_iteration,
                    'evolution_id': evolution_step,
                    'complex_id': complex_id,
                    'stage': stage,
                    'score': float(candidate_fitness[row_idx]),
                }
                for name in param_names:
                    row[name] = _to_csv_scalar(denorm.get(name))
                rows.append(row)

            tracking_rows_buffer.extend(rows)

        def flush_tracking_rows() -> None:
            if tracking_csv_path is None or not tracking_rows_buffer:
                return

            with tracking_csv_path.open('a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=tracking_headers)
                writer.writerows(tracking_rows_buffer)

            tracking_rows_buffer.clear()

        if log_initial_population:
            log_initial_population(self.name, pop_size, best_fit)

        # ---- Precompute the static complex partitioning pattern ----
        # Complex c owns population rows c, c+n_complexes, c+2*n_complexes, …
        # Because the population is re-sorted after every shuffle iteration the
        # specific parameter vectors at these positions change, but the index
        # pattern itself is constant and can be computed once.
        complex_members_list: List[List[int]] = [
            list(range(c, pop_size, n_complexes)) for c in range(n_complexes)
        ]
        complex_sizes: List[int] = [len(m) for m in complex_members_list]

        # ---- Phase 2: Shuffle iterations ----
        stagnation_count = 0
        prev_best_fit = best_fit

        # max_iterations comes from base class (optimization.iterations in config)
        for iteration in range(1, self.max_iterations + 1):

            # -- 2a. Extract independent sub-complexes from the ranked population --
            # Each sub-complex is a working copy; we evolve it without touching the
            # main population array so that complexes remain fully independent.
            sub_complexes: List[np.ndarray] = [
                population[members].copy() for members in complex_members_list
            ]
            sub_fitnesses: List[np.ndarray] = [
                fitness[members].copy() for members in complex_members_list
            ]

            # -- 2b. CCE: n_evolution_steps parallel steps across all complexes --
            # At each step we generate one candidate per complex (reflection) and
            # batch-evaluate all of them simultaneously.  Complexes that fail
            # reflection proceed to contraction, then random replacement — these
            # fallback stages are also batched across the complexes that need them.
            for _ in range(n_evolution_steps):
                evolution_id += 1

                # Generate reflection candidates for all complexes (CPU-only, cheap)
                reflection_batch, per_step_state = self._generate_reflection_candidates(
                    sub_complexes, sub_fitnesses, complex_sizes, n_params
                )

                # Batch-evaluate all reflections in parallel
                reflection_fitness = evaluate_population(reflection_batch, iteration)

                # Accept reflections / fall back to contraction / random (all batched)
                self._apply_cce_step(
                    sub_complexes, sub_fitnesses,
                    per_step_state,
                    reflection_batch, reflection_fitness,
                    n_params, iteration, evaluate_population,
                    log_evolution_batch=partial(
                        log_evolution_batch,
                        shuffle_iteration=iteration,
                        evolution_step=evolution_id,
                    ),
                )

            # -- 2c. Merge evolved sub-complexes back into the population array --
            for c, members in enumerate(complex_members_list):
                population[members] = sub_complexes[c]
                fitness[members] = sub_fitnesses[c]

            # -- 2d. Shuffle: re-sort the merged population by fitness --
            # This is the "shuffle" that gives the algorithm its name.  It breaks
            # up any complex that has converged prematurely by redistributing the
            # best individuals across all complexes in the next iteration.
            sorted_idx = np.argsort(-fitness)
            population = population[sorted_idx]
            fitness = fitness[sorted_idx]

            # Update global best from the newly shuffled population
            if fitness[0] > best_fit:
                best_pos = population[0].copy()
                best_fit = fitness[0]

            # ---- Record and log iteration results ----
            params_dict = denormalize_params(best_pos)
            record_iteration(iteration, best_fit, params_dict)
            update_best(best_fit, params_dict, iteration)
            log_progress(self.name, iteration, best_fit)

            # Persist evolution tracking rows once per shuffle iteration to keep
            # per-batch overhead low while still providing near-real-time output.
            flush_tracking_rows()

            # ---- Early stopping: check for stagnation ----
            # Relative change from the previous best; use absolute change when
            # prev_best_fit is zero to avoid a division-by-zero.
            if prev_best_fit != 0:
                pct_change = abs(best_fit - prev_best_fit) / abs(prev_best_fit)
            else:
                pct_change = abs(best_fit - prev_best_fit)

            if pct_change < pct_change_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0

            if stagnation_count >= stagnation_limit:
                self.logger.info(
                    f"SCE-UA early stopping at iteration {iteration}: "
                    f"no improvement > {pct_change_threshold:.4f} for "
                    f"{stagnation_limit} consecutive shuffles"
                )
                break

            prev_best_fit = best_fit

        # Flush any remaining buffered rows (e.g., if loop exits early).
        flush_tracking_rows()

        return {
            'best_solution': best_pos,
            'best_score': best_fit,
            'best_params': denormalize_params(best_pos)
        }
