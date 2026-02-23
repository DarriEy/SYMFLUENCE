"""
Koopman operator analysis for multi-model hydrological ensembles.

Approximates the Koopman operator of a watershed system using Extended Dynamic
Mode Decomposition (EDMD) with structurally diverse model outputs as lifting
functions (dictionary). Extracts dominant dynamical modes, timescales, and
mode loadings to reveal how different models capture different aspects of
watershed dynamics.

Architecture:
    This module follows the same pattern as SensitivityAnalyzer and Benchmarker:
    - Standalone class with __init__(config, logger, reporting_manager)
    - Main entry point: run_koopman_analysis() -> Dict[str, Any]
    - Registered with AnalysisRegistry for discovery by AnalysisManager

Registration:
    @AnalysisRegistry.register_koopman_analyzer()
    class KoopmanAnalyzer: ...

Example:
    >>> analyzer = KoopmanAnalyzer(config, logger)
    >>> results = analyzer.run_koopman_analysis(ensemble_df, obs_streamflow)
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.mixins import ConfigMixin
from symfluence.evaluation.analysis_registry import AnalysisRegistry


@AnalysisRegistry.register_koopman_analyzer()
class KoopmanAnalyzer(ConfigMixin):
    """Koopman operator analysis of multi-model hydrological ensembles.

    Uses Extended Dynamic Mode Decomposition (EDMD) to approximate the
    Koopman operator of a watershed, with model ensemble outputs serving
    as the dictionary (lifting functions). Extracts eigenvalues, modes,
    timescales, and mode loadings for physical interpretation.

    The key insight is that structurally different hydrological models
    act as nonlinear observables of the same underlying dynamical system.
    EDMD finds the best-fit linear operator in this lifted space, revealing
    dominant modes that no single model captures alone.

    Attributes:
        config: SymfluenceConfig or dict
        logger: Logger instance
        reporting_manager: Optional ReportingManager
        delay_lags: Number of delay-embedding lags for observed streamflow
        svd_threshold: Cumulative energy threshold for SVD rank selection
        output_dir: Directory for saving results
    """

    def __init__(
        self,
        config: Any,
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None,
        delay_lags: int = 7,
        svd_threshold: float = 0.99,
    ):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.reporting_manager = reporting_manager
        self.delay_lags = delay_lags
        self.svd_threshold = svd_threshold

        self.data_dir = Path(self._get_config_value(
            lambda: self.config.paths.symfluence_data_dir,
            default=tempfile.gettempdir(),
            dict_key="SYMFLUENCE_DATA_DIR",
        ))
        self.domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            default="domain",
            dict_key="DOMAIN_NAME",
        )
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.output_dir = self.project_dir / "reporting" / "koopman_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core DMD algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def delay_embed(series: np.ndarray, n_lags: int) -> np.ndarray:
        """Build Hankel (delay-embedding) matrix from 1-D series."""
        T = len(series)
        indices = np.arange(n_lags)[None, :] + np.arange(T - n_lags + 1)[:, None]
        return series[indices]

    @staticmethod
    def _select_rank(sigma: np.ndarray, threshold: float) -> int:
        """Select SVD truncation rank by cumulative energy threshold."""
        energy = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
        return int(np.searchsorted(energy, threshold) + 1)

    def dmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Exact Dynamic Mode Decomposition.

        Parameters
        ----------
        X : (n, m) snapshot matrix
        Y : (n, m) shifted snapshots
        rank : explicit rank override (None = auto from SVD threshold)

        Returns
        -------
        dict with K_tilde, eigenvalues, modes, amplitudes, U, Sigma, rank
        """
        U, sigma, Vh = np.linalg.svd(X, full_matrices=False)
        r = rank if rank is not None else self._select_rank(sigma, self.svd_threshold)
        r = min(r, len(sigma))

        Ur = U[:, :r]
        Sr = sigma[:r]
        Vr = Vh[:r, :].conj().T
        Sr_inv = np.diag(1.0 / Sr)

        K_tilde = Ur.conj().T @ Y @ Vr @ Sr_inv
        eigenvalues, W = np.linalg.eig(K_tilde)
        modes = Y @ Vr @ Sr_inv @ W
        amplitudes = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]

        return {
            "K_tilde": K_tilde,
            "eigenvalues": eigenvalues,
            "modes": modes,
            "amplitudes": amplitudes,
            "U": Ur,
            "Sigma": Sr,
            "rank": r,
        }

    # ------------------------------------------------------------------
    # Dictionary construction
    # ------------------------------------------------------------------

    def build_dictionary(
        self,
        obs_Q: np.ndarray,
        ensemble: np.ndarray,
        model_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Build EDMD dictionary from delay-embedded obs + model outputs.

        Returns (dictionary array, feature_names list).
        """
        Q_embedded = self.delay_embed(obs_Q, self.delay_lags)
        ens_truncated = ensemble[self.delay_lags - 1:, :]
        dictionary = np.hstack([Q_embedded, ens_truncated])

        feature_names = [f"obs_Q(t-{self.delay_lags - 1 - j})" for j in range(self.delay_lags)]
        feature_names += list(model_names)
        return dictionary, feature_names

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def eigenvalues_to_continuous(lam: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Discrete eigenvalues -> continuous-time omega = log(lambda)/dt."""
        lam_safe = np.where(np.abs(lam) > 1e-15, lam, 1e-15)
        return np.log(lam_safe) / dt

    @staticmethod
    def extract_timescales(omega: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract decay_days and period_days from continuous eigenvalues."""
        with np.errstate(divide="ignore", invalid="ignore"):
            decay_days = np.where(
                np.abs(omega.real) > 1e-15, -1.0 / omega.real, np.inf
            )
            period_days = np.where(
                np.abs(omega.imag) > 1e-10, 2 * np.pi / np.abs(omega.imag), np.inf
            )
        return {"decay_days": decay_days, "period_days": period_days}

    @staticmethod
    def mode_importance(
        eigenvalues: np.ndarray, amplitudes: np.ndarray, n_steps: int
    ) -> np.ndarray:
        """Rank modes by time-averaged amplitude x persistence."""
        abs_lam = np.abs(eigenvalues)
        abs_b = np.abs(amplitudes)
        with np.errstate(divide="ignore", invalid="ignore"):
            geo_sum = np.where(
                np.abs(abs_lam - 1.0) > 1e-10,
                (1.0 - abs_lam ** n_steps) / (1.0 - abs_lam),
                float(n_steps),
            )
        return abs_b * geo_sum / n_steps

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def one_step_ahead(dmd_result: Dict, dictionary: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction: K_full @ x(k) for each state."""
        Ur = dmd_result["U"]
        K_tilde = dmd_result["K_tilde"]
        X = dictionary[:-1, :].T
        X_red = Ur.conj().T @ X
        Y_red = K_tilde @ X_red
        Y_full = Ur @ Y_red
        return Y_full.real.T

    @staticmethod
    def compute_kge(sim: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
        """KGE, NSE, r, alpha, beta from aligned arrays."""
        mask = np.isfinite(sim) & np.isfinite(obs)
        s, o = sim[mask], obs[mask]
        if len(s) < 10:
            return {"KGE": np.nan, "NSE": np.nan, "r": np.nan,
                    "alpha": np.nan, "beta": np.nan}
        r = np.corrcoef(s, o)[0, 1]
        alpha = np.std(s) / np.std(o) if np.std(o) > 0 else np.nan
        beta = np.mean(s) / np.mean(o) if np.mean(o) != 0 else np.nan
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        ss_res = np.sum((o - s) ** 2)
        ss_tot = np.sum((o - np.mean(o)) ** 2)
        nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"KGE": kge, "NSE": nse, "r": r, "alpha": alpha, "beta": beta}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_koopman_analysis(
        self,
        ensemble_df: pd.DataFrame,
        obs_streamflow: pd.Series,
        train_end: str = "2007-12-31",
        eval_start: str = "2008-01-01",
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the complete Koopman operator analysis pipeline.

        Parameters
        ----------
        ensemble_df : (T, N) DataFrame of model outputs (m^3/s), aligned daily
        obs_streamflow : (T,) Series of observed streamflow (m^3/s)
        train_end : last training date
        eval_start : first evaluation date
        rank : explicit DMD rank (None = auto)

        Returns
        -------
        dict with eigenvalues, timescales, metrics, mode_loadings, etc.
        """
        self.logger.info("Starting Koopman operator analysis")

        model_names = list(ensemble_df.columns)
        n_models = len(model_names)

        # Build dictionary
        dictionary, feature_names = self.build_dictionary(
            obs_streamflow.values, ensemble_df.values, model_names
        )
        dates = obs_streamflow.index[self.delay_lags - 1:]
        self.logger.info(
            f"Dictionary: {dictionary.shape} "
            f"({self.delay_lags} lags + {n_models} models)"
        )

        # Split train/eval
        train_mask = dates <= pd.Timestamp(train_end)
        eval_mask = dates >= pd.Timestamp(eval_start)
        train_raw = dictionary[train_mask]
        eval_raw = dictionary[eval_mask]

        # Normalize from training data only
        means = train_raw.mean(axis=0)
        stds = train_raw.std(axis=0)
        stds[stds < 1e-12] = 1.0
        train = (train_raw - means) / stds
        eval_ = (eval_raw - means) / stds

        # DMD on training data
        X_train = train[:-1, :].T
        Y_train = train[1:, :].T
        result = self.dmd(X_train, Y_train, rank=rank)
        self.logger.info(f"DMD rank: {result['rank']}")

        # Eigenstructure analysis
        omega = self.eigenvalues_to_continuous(result["eigenvalues"])
        timescales = self.extract_timescales(omega)
        importance = self.mode_importance(
            result["eigenvalues"], result["amplitudes"], train.shape[0]
        )

        # Mode loadings
        abs_modes = np.abs(result["modes"])
        col_max = abs_modes.max(axis=0)
        col_max[col_max < 1e-15] = 1.0
        loadings = abs_modes / col_max

        # One-step-ahead validation
        obs_col_idx = feature_names.index("obs_Q(t-0)")
        pred_eval = self.one_step_ahead(result, eval_)
        Q_pred = pred_eval[:, obs_col_idx] * stds[obs_col_idx] + means[obs_col_idx]
        Q_obs = eval_[1:, obs_col_idx] * stds[obs_col_idx] + means[obs_col_idx]
        eval_metrics = self.compute_kge(Q_pred, Q_obs)

        # Ensemble mean on eval for comparison
        ens_eval = (eval_[1:] * stds + means)[:, self.delay_lags:]
        ens_mean_eval = ens_eval.mean(axis=1)
        ens_metrics = self.compute_kge(ens_mean_eval, Q_obs)

        self.logger.info(
            f"Koopman eval KGE={eval_metrics['KGE']:.3f}, "
            f"NSE={eval_metrics['NSE']:.3f}"
        )
        self.logger.info(
            f"Ensemble mean eval KGE={ens_metrics['KGE']:.3f}, "
            f"NSE={ens_metrics['NSE']:.3f}"
        )

        # Save results
        results = {
            "eigenvalues": result["eigenvalues"],
            "modes": result["modes"],
            "amplitudes": result["amplitudes"],
            "rank": result["rank"],
            "singular_values": result["Sigma"],
            "decay_days": timescales["decay_days"],
            "period_days": timescales["period_days"],
            "importance": importance,
            "loadings": pd.DataFrame(
                loadings, index=feature_names,
                columns=[f"Mode_{j+1}" for j in range(result["rank"])]
            ),
            "feature_names": feature_names,
            "eval_metrics": eval_metrics,
            "ens_mean_metrics": ens_metrics,
            "normalization": {"means": means, "stds": stds},
        }

        # Save to disk
        self._save_results(results)

        # Visualize
        if self.reporting_manager:
            try:
                self.reporting_manager.visualize_koopman_analysis(results, self.output_dir)
            except Exception as e:  # noqa: BLE001 — must-not-raise contract
                self.logger.warning(f"Koopman visualization failed: {e}")

        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Persist results to disk."""
        np.savez(
            self.output_dir / "koopman_results.npz",
            eigenvalues=results["eigenvalues"],
            modes=results["modes"],
            amplitudes=results["amplitudes"],
            rank=results["rank"],
            singular_values=results["singular_values"],
            decay_days=results["decay_days"],
            period_days=results["period_days"],
            importance=results["importance"],
        )
        results["loadings"].to_csv(self.output_dir / "mode_loadings.csv")

        mode_table = pd.DataFrame({
            "eigenvalue_abs": np.abs(results["eigenvalues"]),
            "decay_days": results["decay_days"],
            "period_days": results["period_days"],
            "importance": results["importance"],
        })
        mode_table.to_csv(self.output_dir / "mode_summary.csv", index=False)
        self.logger.info(f"Koopman results saved to {self.output_dir}")
