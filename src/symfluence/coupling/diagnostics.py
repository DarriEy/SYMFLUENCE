"""Conservation diagnostics for dCoupler CouplingGraph.

Extracts conservation logs from the graph's ConservationChecker and formats
them for SYMFLUENCE's reporting system (logging + optional bar-chart plot).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CouplingDiagnostics:
    """Extracts and formats conservation diagnostics from dCoupler CouplingGraph."""

    @staticmethod
    def extract_conservation_report(graph) -> Dict[str, Any]:
        """Extract conservation log from graph's ConservationChecker.

        Args:
            graph: A dCoupler CouplingGraph instance.

        Returns:
            Dict with keys: mode, tolerance, connections, max_error, n_violations.
        """
        checker = getattr(graph, '_conservation', None)
        if checker is None:
            return {
                "mode": "disabled",
                "tolerance": 0.0,
                "connections": [],
                "max_error": 0.0,
                "n_violations": 0,
            }

        log = getattr(checker, 'conservation_log', [])
        tolerance = getattr(checker, 'tolerance', 1e-6)

        errors = [entry.get("relative_error", 0.0) for entry in log]

        return {
            "mode": getattr(checker, 'mode', 'unknown'),
            "tolerance": tolerance,
            "connections": list(log),
            "max_error": max(errors) if errors else 0.0,
            "n_violations": sum(1 for e in errors if e > tolerance),
        }

    @staticmethod
    def format_conservation_table(report: Dict[str, Any]) -> str:
        """Format conservation report as a human-readable text table.

        Args:
            report: Output of extract_conservation_report().

        Returns:
            Multi-line string suitable for logging.
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("COUPLING CONSERVATION DIAGNOSTICS")
        lines.append("=" * 60)
        lines.append(f"  Mode:       {report['mode']}")
        lines.append(f"  Tolerance:  {report['tolerance']:.2e}")
        lines.append(f"  Max error:  {report['max_error']:.2e}")
        lines.append(f"  Violations: {report['n_violations']}/{len(report['connections'])}")
        lines.append("-" * 60)

        if not report["connections"]:
            lines.append("  No connections logged.")
        else:
            lines.append(f"  {'Connection':<35} {'Rel. Error':>12} {'Status':>8}")
            lines.append(f"  {'-'*35} {'-'*12} {'-'*8}")
            for entry in report["connections"]:
                conn = entry.get("connection", "unknown")
                err = entry.get("relative_error", 0.0)
                status = "FAIL" if err > report["tolerance"] else "OK"
                lines.append(f"  {conn:<35} {err:>12.2e} {status:>8}")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def plot_conservation_errors(
        report: Dict[str, Any],
        output_path: Path,
    ) -> Optional[Path]:
        """Generate bar chart of relative conservation errors per connection.

        Args:
            report: Output of extract_conservation_report().
            output_path: Path to save the PNG figure.

        Returns:
            Path to the saved plot, or None if plotting failed.
        """
        connections = report.get("connections", [])
        if not connections:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [c.get("connection", f"conn_{i}") for i, c in enumerate(connections)]
            errors = [c.get("relative_error", 0.0) for c in connections]
            tolerance = report.get("tolerance", 1e-6)

            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
            colors = ["#e74c3c" if e > tolerance else "#2ecc71" for e in errors]
            ax.bar(range(len(labels)), errors, color=colors, edgecolor="grey")

            # Tolerance line
            ax.axhline(y=tolerance, color="orange", linestyle="--", linewidth=1.5,
                        label=f"Tolerance ({tolerance:.1e})")

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Relative Conservation Error")
            ax.set_title("Coupling Conservation Diagnostics")
            ax.legend(loc="upper right")
            ax.set_yscale("log")

            fig.tight_layout()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=150)
            plt.close(fig)
            return output_path

        except Exception as e:
            logger.warning(f"Failed to generate conservation plot: {e}")
            return None
