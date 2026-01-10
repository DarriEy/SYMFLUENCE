"""
Model Decisions Updater

Handles updates to SUMMA modelDecisions.txt for final evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional


class ModelDecisionsUpdater:
    """
    Updates SUMMA model decisions for final evaluation.

    Handles:
    - Updating model decisions for accurate solver
    - Backing up original optimization settings
    - Restoring optimization settings after final evaluation
    """

    # Default model decisions for final evaluation
    # Uses dictionary instead of elif chain for maintainability
    FINAL_EVALUATION_DECISIONS: Dict[str, str] = {
        'soilCatTbl': 'ROSETTA              ! soil-category dateset',
        'vegeParTbl': 'USGS                 ! vegetation category dataset',
        'soilStress': 'NoahType             ! choice of function for the soil moisture control on stomatal resistance',
        'stomResist': 'BallBerry            ! choice of function for stomatal resistance',
        'num_method': 'itertive             ! choice of numerical method',
        'fDerivMeth': 'analytic             ! choice of method to calculate flux derivatives',
        'LAI_method': 'monTable             ! choice of method to determine LAI and SAI',
        'cIntercept': 'sparseCanopy         ! choice of parameterization for canopy interception',
        'f_Richards': 'mixdform             ! choice of form of Richards\' equation',
        'groundwatr': 'qTopmodl             ! choice of groundwater parameterization',
        'hc_profile': 'pow_prof             ! choice of hydraulic conductivity profile',
        'bcUpprTdyn': 'nrg flux             ! type of upper boundary condition for thermodynamics',
        'bcLowrTdyn': 'zeroFlux             ! type of lower boundary condition for thermodynamics',
        'bcUpprSoiH': 'liq_flux             ! type of upper boundary condition for soil hydrology',
        'bcLowrSoiH': 'drainage             ! type of lower boundary condition for soil hydrology',
        'veg_traits': 'CM_QJRMS1988         ! choice of parameterization for vegetation roughness length and displacement height',
        'canopyEmis': 'difTrans             ! choice of parameterization for canopy emissivity',
        'snowIncept': 'lightSnow            ! choice of parameterization for snow interception',
        'windPrfile': 'logBelowCanopy       ! choice of wind profile through the canopy',
        'astability': 'louisinv             ! choice of stability function',
        'canopySrad': 'CLM_2stream          ! choice of method for canopy shortwave radiation',
        'alb_method': 'varDecay             ! choice of albedo representation',
        'compaction': 'anderson             ! choice of compaction routine',
        'snowLayers': 'CLM_2010             ! choice of method to combine and sub-divide snow layers',
        'thCondSnow': 'jrdn1991             ! choice of thermal conductivity representation for snow',
        'thCondSoil': 'funcSoilWet          ! choice of thermal conductivity representation for soil',
        'spatial_gw': 'localColumn          ! choice of spatial representation of groundwater',
        'subRouting': 'timeDlay             ! choice of method for sub-grid routing',
    }

    BACKUP_FILENAME = 'modelDecisions_optimization_backup.txt'

    def __init__(
        self,
        settings_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize model decisions updater.

        Args:
            settings_dir: Settings directory path
            logger: Optional logger instance
        """
        self.settings_dir = settings_dir
        self.model_decisions_path = settings_dir / 'modelDecisions.txt'
        self.backup_path = settings_dir / self.BACKUP_FILENAME
        self.logger = logger or logging.getLogger(__name__)

    def update_for_final_evaluation(self) -> None:
        """Update modelDecisions.txt to use more accurate solver for final evaluation."""
        if not self.model_decisions_path.exists():
            return

        try:
            with open(self.model_decisions_path, 'r') as f:
                lines = f.readlines()

            # Backup original if not already done
            if not self.backup_path.exists():
                with open(self.backup_path, 'w') as f:
                    f.writelines(lines)

            updated_lines = []
            for line in lines:
                updated = False

                for decision_key, decision_value in self.FINAL_EVALUATION_DECISIONS.items():
                    if decision_key in line and not line.strip().startswith('!'):
                        # Format: decision_key followed by spaces to column 25, then value
                        updated_lines.append(f"{decision_key:<20} {decision_value}\n")
                        updated = True
                        break

                if not updated:
                    updated_lines.append(line)

            with open(self.model_decisions_path, 'w') as f:
                f.writelines(updated_lines)

            self.logger.debug("Updated model decisions for final evaluation")

        except Exception as e:
            self.logger.error(f"Error updating model decisions for final run: {e}")

    def restore_for_optimization(self) -> None:
        """Restore model decisions to optimization settings."""
        if not self.backup_path.exists():
            return

        try:
            with open(self.backup_path, 'r') as f:
                lines = f.readlines()

            with open(self.model_decisions_path, 'w') as f:
                f.writelines(lines)

            self.logger.debug("Restored model decisions to optimization settings")

        except Exception as e:
            self.logger.error(f"Error restoring model decisions: {e}")

    def get_decision(self, key: str) -> Optional[str]:
        """
        Get a specific decision value from the file.

        Args:
            key: Decision key name

        Returns:
            Decision value or None if not found
        """
        if not self.model_decisions_path.exists():
            return None

        try:
            with open(self.model_decisions_path, 'r') as f:
                for line in f:
                    if key in line and not line.strip().startswith('!'):
                        parts = line.split()
                        if len(parts) >= 2:
                            return parts[1]
        except Exception:
            pass

        return None

    def set_decision(self, key: str, value: str, comment: str = "") -> bool:
        """
        Set a specific decision value.

        Args:
            key: Decision key name
            value: New value
            comment: Optional comment

        Returns:
            True if successful
        """
        if not self.model_decisions_path.exists():
            return False

        try:
            with open(self.model_decisions_path, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            found = False
            for line in lines:
                if key in line and not line.strip().startswith('!'):
                    if comment:
                        updated_lines.append(f"{key:<20} {value:<20} ! {comment}\n")
                    else:
                        updated_lines.append(f"{key:<20} {value}\n")
                    found = True
                else:
                    updated_lines.append(line)

            if found:
                with open(self.model_decisions_path, 'w') as f:
                    f.writelines(updated_lines)
                return True

        except Exception as e:
            self.logger.error(f"Error setting decision {key}: {e}")

        return False
