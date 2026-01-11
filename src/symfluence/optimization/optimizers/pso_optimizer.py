"""
Particle Swarm Optimization (PSO) implementation.

PSO is a population-based optimizer inspired by social behavior,
using particle velocities and swarm intelligence for global search.
"""

import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
from symfluence.optimization.optimizers.base_optimizer import BaseOptimizer

class PSOOptimizer(BaseOptimizer):
    """Particle Swarm Optimization (PSO) Optimizer for SYMFLUENCE.

    PSO is a population-based metaheuristic inspired by the social behavior of
    bird flocking and fish schooling. Each particle maintains a position (solution)
    and velocity, and is influenced by its own best-known position (cognitive)
    and the swarm's best-known position (social).

    Algorithm Overview:
        PSO solves: maximize score
        1. Initialize swarm: N particles with random positions and velocities
        2. Evaluate all particles
        3. For each iteration:
           a. Update velocity: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
              where:
              - w = inertia weight (decreases over time)
              - c1, c2 = cognitive and social acceleration coefficients
              - pbest = particle's personal best position
              - gbest = swarm's global best position
              - r1, r2 = random values in [0, 1]
           b. Update position: x = x + v
           c. Apply reflective boundary conditions if x < 0 or x > 1
           d. Evaluate all particles
           e. Update personal bests and global best
        4. Return best solution found

    Key Features:
        - Population-based (explores multiple solutions in parallel)
        - Balances exploration (early iterations, high inertia) and exploitation (later iterations, low inertia)
        - Adaptive inertia weight schedule reduces velocity over time
        - Reflecting boundary conditions prevent particles from escaping [0,1]
        - Two components drive search:
          * Cognitive (pbest): Each particle remembers its own best solution
          * Social (gbest): All particles are attracted to swarm's best solution
        - Velocity clamping prevents explosive behavior

    Configuration Parameters:
        SWRMSIZE: Number of particles in swarm (default: 20)
                  Larger swarms explore more but are slower
                  Typical range: 10-50
        PSO_COGNITIVE_PARAM (c1): Weight of personal best (default: 1.5)
                  Controls how much particle trusts its own experience
        PSO_SOCIAL_PARAM (c2): Weight of global best (default: 1.5)
                  Controls how much particle trusts swarm's collective knowledge
        PSO_INERTIA_WEIGHT (w_initial): Starting momentum (default: 0.7)
                  High w = more exploration, low w = more exploitation
        PSO_INERTIA_REDUCTION_RATE: Multiplier for w each iteration (default: 0.99)
                  w(t) = w(t-1) * reduction_rate
        INERTIA_SCHEDULE: Schedule for inertia reduction
                  - MULTIPLICATIVE: w *= reduction_rate each iteration
                  - LINEAR: w decreases linearly
        NUMBER_OF_ITERATIONS: Maximum iterations (= max function evaluations / swarm_size)

    Why PSO Works:
        - Particles share information (global best) → collective intelligence
        - Each particle remembers own best → no loss of promising regions
        - Velocity-based movement → smooth exploration
        - Inertia decay → automatic transition from exploration to exploitation
        - Often finds good solutions quickly compared to pure random search

    Comparison with Other Algorithms:
        vs DDS: PSO uses multiple particles, DDS uses single solution
                PSO better for multi-modal problems, DDS better for local refinement
        vs DE: PSO uses velocity vectors, DE uses difference vectors
               PSO often converges faster, DE more robust on noisy objectives
        vs NSGA2: PSO for single-objective, NSGA2 for multi-objective

    Attributes:
        swarm_size: Number of particles in the swarm
        c1, c2: Cognitive and social acceleration coefficients
        w_initial: Starting inertia weight
        current_inertia: Current inertia (decreases over iterations)
        swarm_positions: Current particle positions in normalized space
        swarm_velocities: Current particle velocities
        personal_best_positions: Best position found by each particle
        personal_best_scores: Best score achieved by each particle
        global_best_position: Best position found by any particle
        global_best_score: Best score achieved by any particle

    References:
        Kennedy, J., & Eberhart, R. C. (1995). Particle swarm optimization.
        In Proceedings of IEEE International Conference on Neural Networks,
        Vol. IV (pp. 1942-1948). Piscataway, NJ: IEEE.

        Eberhart, R. C., & Kennedy, J. (2001). A new optimizer using particle
        swarm theory. In Proceedings of the sixth international symposium on
        micro machine and human science (Vol. 1, pp. 39-43). New York: IEEE.

        Shi, Y., & Eberhart, R. C. (1998). A modified particle swarm optimizer.
        In Proceedings of the 1998 IEEE Congress on Evolutionary Computation
        (pp. 69-73). IEEE.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize PSO optimizer.

        Args:
            config: Configuration dict with PSO parameters
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.swarm_size = self._cfg('SWRMSIZE', default=20)
        self.c1 = self._cfg('PSO_COGNITIVE_PARAM', default=1.5)
        self.c2 = self._cfg('PSO_SOCIAL_PARAM', default=1.5)
        self.w_initial = self._cfg('PSO_INERTIA_WEIGHT', default=0.7)
        self.w_reduction_rate = self._cfg('PSO_INERTIA_REDUCTION_RATE', default=0.99)
        self.swarm_positions = None
        self.swarm_velocities = None
        self.swarm_scores = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self.current_inertia = self.w_initial

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "PSO"

    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Run PSO algorithm.

        Returns:
            Tuple of (best_params, best_score, iteration_history)
        """
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_swarm(initial_params)
        return self._run_pso_algorithm()
    
    def _initialize_swarm(self, initial_params: Dict[str, np.ndarray]) -> None:
        """Initialize PSO swarm with random particles and seed initial particle.

        Workflow:
        1. Create N particles with random normalized positions [0,1]
        2. If initial parameters provided, place them at particle 0 (warm start)
        3. Initialize velocities randomly in [-0.1, 0.1] range
        4. Set personal bests to initial positions (first evaluation will update)
        5. Evaluate all particles
        6. Update personal and global best records
        7. Record generation 0 for history tracking

        This warm-start strategy (placing good initial guess at particle 0) can
        significantly accelerate convergence by seeding the swarm in a
        promising region of parameter space.

        Args:
            initial_params: Initial parameter dictionary to seed particle 0,
                           or None for random initialization.
        """
        self._ensure_reproducible_initialization()
        param_count = len(self.parameter_manager.all_param_names)
        self.swarm_positions = np.random.random((self.swarm_size, param_count))
        self.swarm_scores = np.full(self.swarm_size, np.nan)
        if initial_params:
            self.swarm_positions[0] = np.clip(self.parameter_manager.normalize_parameters(initial_params), 0, 1)
        self.swarm_velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, param_count))
        self.personal_best_positions = self.swarm_positions.copy()
        self.personal_best_scores = np.full(self.swarm_size, float('-inf'))
        self._evaluate_swarm()
        self._update_personal_bests()
        self._update_global_best()
        self._record_generation(0)
    
    def _run_pso_algorithm(self) -> Tuple[Dict, float, List]:
        """Execute main PSO optimization loop.

        Main Iteration Loop (iterations 1 to max_iterations):
        1. Update inertia weight according to schedule (MULTIPLICATIVE or LINEAR)
           - MULTIPLICATIVE: w = w * decay_rate each iteration
           - LINEAR: w decreases linearly from w_initial to near 0
           - This decay enables automatic transition from exploration to exploitation
        2. Update velocities using cognitive and social components
        3. Update positions and apply reflecting boundary conditions
        4. Evaluate all particles in parallel (if configured) or sequentially
        5. Update personal best for each particle if current > personal_best
        6. Update global best if any particle exceeds previous global_best
        7. Record iteration history for convergence analysis

        Inertia Schedule Options:
            - MULTIPLICATIVE (default): Smooth exponential decay; w(t) = w(t-1) * decay_rate
              Recommended for smooth convergence, typical decay_rate=0.99
            - LINEAR: w decreases at constant rate; more aggressive early switching

        Returns:
            Tuple of (best_params, best_score, iteration_history) where:
            - best_params: Dictionary of parameter names to best found values
            - best_score: Highest objective value (fitness) found
            - iteration_history: List of generation records with convergence data

        Note:
            All particles share the same global best, creating strong pressure
            toward convergence. This can be effective for unimodal problems but
            may reduce exploration on multi-modal landscapes.
        """
        schedule = str(self._cfg("INERTIA_SCHEDULE", default="MULTIPLICATIVE")).upper()
        decay = float(self._cfg("INERTIA_DECAY_RATE", default=0.99))
        for iteration in range(1, self.max_iterations + 1):
            if schedule == "MULTIPLICATIVE": self.current_inertia *= decay
            self._update_velocities()
            self._update_positions()
            self._evaluate_swarm()
            self._update_personal_bests()
            if self._update_global_best():
                pass # New global best found
            self._record_generation(iteration)
        self.best_score = self.global_best_score
        self.best_params = self.parameter_manager.denormalize_parameters(self.global_best_position)
        return self.best_params, self.best_score, self.iteration_history

    def _update_velocities(self) -> None:
        """Update particle velocities based on cognitive and social components.

        Each particle's velocity is updated using three components:
        1. Inertia: w * v (momentum from previous velocity)
           - Maintains particle's current direction
           - High w = more exploration, low w = more exploitation
        2. Cognitive: c1 * r1 * (pbest - x) (attraction to personal best)
           - Pulls particle toward its own best-found position
           - r1 = random [0, 1] adds stochasticity
           - c1 controls strength of personal experience
        3. Social: c2 * r2 * (gbest - x) (attraction to global best)
           - Pulls all particles toward swarm's best-found position
           - Implements collective intelligence
           - r2 = random [0, 1] adds stochasticity
           - c2 controls strength of social influence

        Mathematical Update:
            v_i = w * v_i + c1 * r1 * (pbest_i - x_i) + c2 * r2 * (gbest - x_i)

        Velocity Clamping:
            Velocities clamped to [-0.2, 0.2] to prevent:
            - Explosive growth (particles flying off search space)
            - Loss of convergence (particles never settle)
            - Instability in high-dimensional spaces

        This is the "constriction factor" approach to PSO stability.
        """
        param_count = len(self.parameter_manager.all_param_names)
        for i in range(self.swarm_size):
            r1, r2 = np.random.random(param_count), np.random.random(param_count)
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.swarm_positions[i])
            social = self.c2 * r2 * (self.global_best_position - self.swarm_positions[i])
            self.swarm_velocities[i] = np.clip(self.current_inertia * self.swarm_velocities[i] + cognitive + social, -0.2, 0.2)
    
    def _update_positions(self) -> None:
        """Update particle positions and apply reflecting boundary conditions.

        Position Update:
            x(t+1) = x(t) + v(t)
            Simple addition of velocity to position.

        Reflecting Boundary Handling:
            When particles move outside [0,1] bounds in normalized space,
            we use reflective boundary conditions rather than hard clipping.

            Why Reflecting Boundaries?
            - Prevents particles from "sticking" at hard boundaries (absorbing barrier)
            - Bounces particles back into valid region while preserving directional info
            - Reduced velocity on bounce creates natural deceleration near boundaries
            - Biologically inspired: similar to animals bouncing off walls

            Algorithm per Dimension:
                if x < 0:  x' = -x,  v' = -v    (reflect about 0)
                if x > 1:  x' = 2-x, v' = -v    (reflect about 1)

            Example: If particle crosses at x=-0.2 with v=-0.3:
                Reflected position: x' = -(-0.2) = 0.2
                Reflected velocity: v' = -(-0.3) = 0.3 (now moving back into domain)

            This is superior to hard boundary clipping which would:
            - Accumulate particles at boundaries
            - Lose gradient information (would repeat same position)
            - Reduce exploration effectiveness

        Final Clipping:
            After reflecting boundaries, apply hard clip to [0,1] to ensure
            numerical stability handles any floating-point errors.
        """
        self.swarm_positions += self.swarm_velocities
        for i in range(self.swarm_size):
            for j in range(len(self.parameter_manager.all_param_names)):
                if self.swarm_positions[i, j] < 0:
                    self.swarm_positions[i, j] = -self.swarm_positions[i, j]
                    self.swarm_velocities[i, j] = -self.swarm_velocities[i, j]
                elif self.swarm_positions[i, j] > 1:
                    self.swarm_positions[i, j] = 2.0 - self.swarm_positions[i, j]
                    self.swarm_velocities[i, j] = -self.swarm_velocities[i, j]
        self.swarm_positions = np.clip(self.swarm_positions, 0, 1)
    
    def _evaluate_swarm(self) -> None:
        """Evaluate fitness of all particles in swarm.

        Dispatches to parallel or sequential evaluation based on configuration.
        All particles are evaluated at each iteration, which is computationally
        expensive but necessary for PSO (unlike genetic algorithms that can use
        multi-generation populations).

        See Also:
            _evaluate_swarm_sequential(): Sequential single-threaded evaluation
            _evaluate_swarm_parallel(): Parallel multi-process evaluation
        """
        if self.use_parallel: self._evaluate_swarm_parallel()
        else: self._evaluate_swarm_sequential()

    def _evaluate_swarm_sequential(self) -> None:
        """Evaluate each particle sequentially.

        Simple loop evaluating each particle position using the shared
        parameter manager and objective function from the parent BaseOptimizer.

        This is slower but more straightforward and useful for debugging.
        Typical use: when parallelization overhead exceeds benefits (small
        swarm sizes, fast objective functions).
        """
        for i in range(self.swarm_size):
            self.swarm_scores[i] = self._evaluate_individual(self.swarm_positions[i])

    def _evaluate_swarm_parallel(self) -> None:
        """Evaluate all particles in parallel using multi-processing.

        Workflow:
        1. Create evaluation task dict for each particle with:
           - individual_id: particle index [0, swarm_size)
           - params: denormalized parameter dictionary
           - proc_id: process index (round-robin assignment)
           - evaluation_id: unique identifier for logging "pso_0", "pso_1", etc.
        2. Submit all tasks to process pool via _run_parallel_evaluations()
        3. Collect results and store scores, replacing NaN with -inf for invalid evals

        Benefits:
            - Evaluates swarm size particles in parallel (N-fold speedup)
            - Useful when objective function (model run) is computationally expensive
            - Each process gets independent copy of model instance

        Typical speedup: Near linear for large swarms with expensive objectives.
        """
        tasks = [{'individual_id': i, 'params': self.parameter_manager.denormalize_parameters(self.swarm_positions[i]), 'proc_id': i % self.num_processes, 'evaluation_id': f"pso_{i}"} for i in range(self.swarm_size)]
        results = self._run_parallel_evaluations(tasks)
        for res in results: self.swarm_scores[res['individual_id']] = res['score'] if res['score'] is not None else float('-inf')
    
    def _update_personal_bests(self) -> int:
        """Update each particle's personal best position and score.

        For each particle, if current score exceeds its personal best score,
        update the personal best to current position/score.

        This implements the "cognitive" component of PSO: each particle remembers
        its own best-found solution and uses it to guide future search.

        Logic:
            if score(i) > pbest_score(i):
                pbest_score(i) = score(i)
                pbest_position(i) = position(i)

        Why Separate personal_best from swarm_scores?
            - Particles explore around pbest, not just current position
            - pbest is "nostalgic" memory, preserves good regions found
            - Creates multiple attractors preventing premature convergence
            - Without pbest, algorithm becomes simple hill climber

        NaN Handling:
            Invalid evaluations (NaN) are skipped, preserving previous pbest.
            This handles cases where model crash leaves score as NaN.

        Args:
            None

        Returns:
            int: Number of particles that found improvements in this iteration.
                Useful for convergence monitoring.
        """
        improvements = 0
        for i in range(self.swarm_size):
            if not np.isnan(self.swarm_scores[i]) and self.swarm_scores[i] > self.personal_best_scores[i]:
                self.personal_best_scores[i] = self.swarm_scores[i]
                self.personal_best_positions[i] = self.swarm_positions[i].copy()
                improvements += 1
        return improvements
    
    def _update_global_best(self) -> bool:
        """Update swarm's global best position and score if improved.

        Checks all particles' personal best scores and updates the global best
        if any particle has found a better solution than current gbest.

        This implements the "social" component of PSO: all particles are attracted
        toward the swarm's best-found solution, creating collective intelligence.

        Algorithm:
        1. Filter out NaN scores (invalid/crashed evaluations)
        2. Return False if no valid scores (all particles crashed)
        3. Find index of maximum personal best score
        4. If max > current global_best_score:
           - Update global_best_score to new maximum
           - Update global_best_position (copy particle's best position)
           - Return True (new best found)
        5. Else return False (no improvement)

        Why Global Best Matters:
            - Provides single attractor for all particles (strong convergence pressure)
            - All particles converge toward proven good region
            - Speeds convergence but can cause premature convergence on multi-modal
            - Balance with cognitive (pbest) which maintains exploration

        Returns:
            bool: True if global best was improved this iteration,
                  False otherwise (no improvement or all evals invalid).
        """
        valid_scores = self.personal_best_scores[~np.isnan(self.personal_best_scores)]
        if len(valid_scores) == 0: return False
        best_idx = np.nanargmax(self.personal_best_scores)
        if self.personal_best_scores[best_idx] > self.global_best_score:
            self.global_best_score = self.personal_best_scores[best_idx]
            self.global_best_position = self.personal_best_positions[best_idx].copy()
            return True
        return False

    def _record_generation(self, generation: int) -> None:
        """Record convergence metrics for this generation to iteration history.

        Stores swarm statistics including:
            - generation: Iteration number (0 to max_iterations)
            - algorithm: Always "PSO"
            - best_score: Current global best fitness found
            - best_params: Denormalized parameters of global best (for logging/checkpoint)
            - mean_score: Average fitness of valid particles this generation
            - valid_individuals: Number of particles with valid scores (not NaN)

        History records enable:
            - Convergence visualization (plot best_score over generations)
            - Stagnation detection (mean_score not improving)
            - Population diversity monitoring (std of scores)
            - Failure detection (valid_individuals drops near 0)
            - Performance post-analysis and parameter tuning

        Args:
            generation: Current iteration number in algorithm

        Note:
            NaN scores (from crashed model runs) are excluded from mean calculation.
            If all particles have NaN scores, mean_score is recorded as None.
        """
        valid_scores = self.swarm_scores[~np.isnan(self.swarm_scores)]
        self.iteration_history.append({
            'generation': generation, 'algorithm': 'PSO', 'best_score': self.global_best_score,
            'best_params': self.parameter_manager.denormalize_parameters(self.global_best_position) if self.global_best_position is not None else None,
            'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'valid_individuals': len(valid_scores)
        })
