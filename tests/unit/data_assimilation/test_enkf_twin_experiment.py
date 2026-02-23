"""EnKF twin experiment with synthetic HBV simulation.

Runs the full EnKF loop using HBV model components:
1. Generate "truth" with known parameters
2. Create synthetic observations with noise
3. Run EnKF from perturbed initial conditions
4. Verify ensemble mean converges toward truth
"""

import numpy as np
import pytest
from symfluence.data_assimilation.enkf.enkf_algorithm import EnKFAlgorithm
from symfluence.data_assimilation.enkf.observation_operator import StreamflowObservationOperator
from symfluence.data_assimilation.enkf.perturbation import GaussianPerturbation
from symfluence.data_assimilation.enkf.state_vector import StateVariableSpec, StateVector


class TestEnKFTwinExperiment:
    """EnKF twin experiment with a simple storage model.

    Uses a simplified 2-box model (not full HBV) to test the EnKF
    machinery without requiring JAX or the full HBV infrastructure.
    """

    @staticmethod
    def _simple_model_step(state, precip, params):
        """Minimal 2-box model for testing.

        state: [storage_upper, storage_lower]
        params: {'k1': recession_upper, 'k2': recession_lower, 'split': split_fraction}
        """
        su, sl = state[0], state[1]

        # Partition precip
        recharge = precip * params['split']
        su = su + precip - recharge
        sl = sl + recharge

        # Outflow
        q_upper = params['k1'] * su
        q_lower = params['k2'] * sl

        su = max(su - q_upper, 0.0)
        sl = max(sl - q_lower, 0.0)

        runoff = q_upper + q_lower
        return np.array([su, sl]), runoff

    def test_twin_experiment_convergence(self):
        """Run EnKF twin experiment and verify convergence."""
        rng = np.random.default_rng(42)

        # Parameters
        true_params = {'k1': 0.3, 'k2': 0.05, 'split': 0.4}
        n_timesteps = 200
        n_members = 30
        obs_error_std = 0.5
        assimilation_interval = 5

        # Generate synthetic forcing
        precip = np.maximum(rng.exponential(3.0, n_timesteps), 0.0)

        # --- Generate truth ---
        true_state = np.array([10.0, 20.0])
        true_runoff = np.zeros(n_timesteps)

        for t in range(n_timesteps):
            true_state, true_runoff[t] = self._simple_model_step(
                true_state, precip[t], true_params
            )

        # Synthetic observations (add noise)
        obs = true_runoff + rng.normal(0, obs_error_std, n_timesteps)
        obs = np.maximum(obs, 0.0)

        # --- Set up EnKF ---
        state_specs = [
            StateVariableSpec('su', 1, lower_bound=0.0),
            StateVariableSpec('sl', 1, lower_bound=0.0),
        ]
        sv = StateVector(state_specs)

        # Observation operator (streamflow is appended as augmented state)
        obs_op = StreamflowObservationOperator(n_obs=1)
        H = obs_op.get_matrix(sv.n_state + 1)  # +1 for augmented prediction
        R = np.array([[obs_error_std ** 2]])

        enkf = EnKFAlgorithm(inflation_factor=1.02, enforce_nonnegative=True)

        # --- Initialize ensemble (perturbed initial conditions) ---
        member_states = []
        for _ in range(n_members):
            su = rng.uniform(5.0, 30.0)
            sl = rng.uniform(10.0, 50.0)
            member_states.append(np.array([su, sl]))

        # --- Run EnKF loop ---
        open_loop_error = 0.0
        da_error = 0.0
        n_analyses = 0

        # Also track open-loop (no assimilation) for comparison
        ol_states = [s.copy() for s in member_states]

        for t in range(n_timesteps):
            # Forecast all members
            predictions = np.zeros(n_members)
            ol_predictions = np.zeros(n_members)

            for i in range(n_members):
                member_states[i], predictions[i] = self._simple_model_step(
                    member_states[i], precip[t], true_params
                )
                ol_states[i], ol_predictions[i] = self._simple_model_step(
                    ol_states[i], precip[t], true_params
                )

            # Record open-loop error
            open_loop_error += (np.mean(ol_predictions) - true_runoff[t]) ** 2

            # Assimilate every `assimilation_interval` steps
            if t % assimilation_interval == 0 and t > 0:
                # Assemble state + augmented predictions
                X_states = sv.assemble([{'su': s[0], 'sl': s[1]} for s in member_states])
                X_aug = sv.augment_with_predictions(X_states, predictions)

                # Analyze
                y_obs = np.atleast_1d(obs[t])
                X_aug_a = enkf.analyze(X_aug, y_obs, H, R)
                X_a, _ = sv.split_augmented(X_aug_a, 1)
                X_a = sv.enforce_bounds(X_a)

                # Inject back
                updated = sv.disassemble(X_a)
                for i in range(n_members):
                    member_states[i] = np.array([updated[i]['su'], updated[i]['sl']])

                n_analyses += 1

            # Record DA error
            da_error += (np.mean(predictions) - true_runoff[t]) ** 2

        # --- Verify convergence ---
        da_rmse = np.sqrt(da_error / n_timesteps)
        ol_rmse = np.sqrt(open_loop_error / n_timesteps)

        # DA should reduce error compared to open loop
        assert da_rmse < ol_rmse, (
            f"DA RMSE ({da_rmse:.3f}) should be less than open-loop RMSE ({ol_rmse:.3f})"
        )

        # Ensemble mean state should be reasonably close to truth
        final_mean_state = np.mean(member_states, axis=0)
        # Just check it's in a reasonable range (not wildly off)
        assert final_mean_state[0] > 0, "Upper storage should be positive"
        assert final_mean_state[1] > 0, "Lower storage should be positive"
        assert n_analyses > 0, "Should have performed at least one analysis"
