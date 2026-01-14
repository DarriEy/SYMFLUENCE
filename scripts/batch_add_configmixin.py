#!/usr/bin/env python3
"""
Batch script to add ConfigMixin to standalone classes.

This script automatically:
1. Adds ConfigMixin import
2. Adds ConfigMixin to class inheritance
3. Updates __init__ to set self._config with SymfluenceConfig conversion
"""

import re
from pathlib import Path
from typing import Tuple

# Files that need ConfigMixin added
FILES_TO_MIGRATE = [
    # data/acquisition
    'src/symfluence/data/acquisition/cleanup_processor.py',
    'src/symfluence/data/acquisition/cloud_downloader.py',
    'src/symfluence/data/acquisition/maf_pipeline.py',
    'src/symfluence/data/acquisition/maf_processor.py',
    'src/symfluence/data/acquisition/observed_processor.py',
    'src/symfluence/data/acquisition/zonal_stats_processor.py',
    'src/symfluence/data/acquisition/handlers/modis_et.py',
    'src/symfluence/data/acquisition/handlers/modis_sca.py',
    # data/preprocessing
    'src/symfluence/data/preprocessing/attribute_processor.py',
    'src/symfluence/data/preprocessing/em_earth_integrator.py',
    'src/symfluence/data/preprocessing/remapping_weights.py',
    'src/symfluence/data/preprocessing/resampling/file_processor.py',
    'src/symfluence/data/preprocessing/resampling/point_scale_extractor.py',
    'src/symfluence/data/preprocessing/resampling/shapefile_processor.py',
    'src/symfluence/data/preprocessing/resampling/weight_applier.py',
    'src/symfluence/data/preprocessing/resampling/weight_generator.py',
    'src/symfluence/data/preprocessing/shapefile_manager.py',
    # evaluation
    'src/symfluence/evaluation/benchmarking.py',
    'src/symfluence/evaluation/sensitivity_analysis.py',
    # geospatial methods
    'src/symfluence/geospatial/geofabric/methods/curvature.py',
    'src/symfluence/geospatial/geofabric/methods/drop_analysis.py',
    'src/symfluence/geospatial/geofabric/methods/multi_scale.py',
    'src/symfluence/geospatial/geofabric/methods/slope_area.py',
    'src/symfluence/geospatial/geofabric/methods/stream_threshold.py',
    # models
    'src/symfluence/models/fuse/calibration/parameter_manager.py',
    'src/symfluence/models/fuse/elevation_band_manager.py',
    'src/symfluence/models/fuse/forcing_adapter.py',
    'src/symfluence/models/fuse/forcing_processor.py',
    'src/symfluence/models/gnn/calibration/parameter_manager.py',
    'src/symfluence/models/gr/calibration/parameter_manager.py',
    'src/symfluence/models/hype/config_manager.py',
    'src/symfluence/models/hype/forcing_adapter.py',
    'src/symfluence/models/lstm/calibration/parameter_manager.py',
    'src/symfluence/models/mesh/preprocessing/config_generator.py',
    'src/symfluence/models/mesh/preprocessing/drainage_database.py',
    'src/symfluence/models/mesh/preprocessing/parameter_fixer.py',
    'src/symfluence/models/ngen/calibration/parameter_manager.py',
    'src/symfluence/models/ngen/calibration/targets.py',
    'src/symfluence/models/ngen/config_generator.py',
    'src/symfluence/models/ngen/forcing_adapter.py',
    'src/symfluence/models/summa/calibration/parameter_manager.py',
    'src/symfluence/models/summa/glacier_manager.py',
    # optimization
    'src/symfluence/optimization/core/model_executor.py',
    'src/symfluence/optimization/core/results_manager.py',
    'src/symfluence/optimization/core/transformers.py',
    'src/symfluence/optimization/mixins/gradient_optimization.py',
    'src/symfluence/optimization/mixins/parallel/config_updater.py',
    'src/symfluence/optimization/mixins/parallel/local_scratch_manager.py',
    'src/symfluence/optimization/mixins/parallel_execution.py',
    'src/symfluence/optimization/mixins/results_tracking.py',
    'src/symfluence/optimization/mixins/retry_execution.py',
    'src/symfluence/optimization/objectives/multivariate.py',
    'src/symfluence/optimization/optimizers/evaluators/task_builder.py',
    'src/symfluence/optimization/optimizers/final_evaluation/file_manager_updater.py',
    # project
    'src/symfluence/project/logging_manager.py',
    'src/symfluence/project/workflow_orchestrator.py',
    # reporting
    'src/symfluence/reporting/core/base_plotter.py',
    'src/symfluence/reporting/core/shapefile_helper.py',
    'src/symfluence/reporting/processors/data_processor.py',
    'src/symfluence/reporting/processors/spatial_processor.py',
    'src/symfluence/reporting/reporting_manager.py',
]


def add_configmixin_import(content: str) -> str:
    """Add ConfigMixin import if not already present."""
    if 'ConfigMixin' in content:
        return content

    # Find the best place to add the import
    # Try to add after other symfluence imports
    if 'from symfluence.' in content or 'import symfluence' in content:
        # Find last symfluence import line
        lines = content.split('\n')
        last_symfluence_idx = -1
        for i, line in enumerate(lines):
            if 'from symfluence.' in line or 'import symfluence' in line:
                last_symfluence_idx = i

        if last_symfluence_idx >= 0:
            lines.insert(last_symfluence_idx + 1, 'from symfluence.core.mixins import ConfigMixin')
            return '\n'.join(lines)

    # Otherwise add after standard imports
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i + 1

    lines.insert(insert_idx, '')
    lines.insert(insert_idx + 1, 'from symfluence.core.mixins import ConfigMixin')
    return '\n'.join(lines)


def add_configmixin_to_class(content: str) -> str:
    """Add ConfigMixin to all class definitions that don't have it."""
    # Find classes that don't inherit from anything or don't have ConfigMixin

    # Pattern 1: class ClassName:
    pattern1 = r'(class\s+\w+)\s*:'

    def replace1(match):
        class_def = match.group(1)
        return f'{class_def}(ConfigMixin):'

    # Only replace if not already inheriting from something
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('class ') and stripped.endswith(':') and '(' not in stripped:
            # Simple class definition without inheritance
            line = re.sub(pattern1, replace1, line)
        elif stripped.startswith('class ') and '(' in stripped and 'ConfigMixin' not in stripped:
            # Has inheritance but no ConfigMixin - add it
            line = re.sub(r'(class\s+\w+)\s*\(', r'\1(ConfigMixin, ', line)
        new_lines.append(line)

    return '\n'.join(new_lines)


def update_init_method(content: str) -> str:
    """Update __init__ to set self._config with proper conversion."""
    # Find self.config = config and replace with proper conversion

    old_pattern = r'(\s+)(self\.config\s*=\s*config)\s*\n'

    new_code = r'''\1# Import here to avoid circular imports
\1from symfluence.core.config.models import SymfluenceConfig
\1
\1# Auto-convert dict to typed config for backward compatibility
\1if isinstance(config, dict):
\1    try:
\1        self._config = SymfluenceConfig(**config)
\1    except Exception:
\1        # Fallback for partial configs (e.g., in tests)
\1        self._config = config
\1else:
\1    self._config = config
'''

    # Only replace the first occurrence
    content = re.sub(old_pattern, new_code, content, count=1)

    return content


def process_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Process a single file to add ConfigMixin."""
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    content = file_path.read_text()
    original = content

    # Skip if already has ConfigMixin properly set up
    if 'self._config = ' in content and 'ConfigMixin' in content:
        return False, "Already migrated"

    # Step 1: Add import
    content = add_configmixin_import(content)

    # Step 2: Add ConfigMixin to class
    content = add_configmixin_to_class(content)

    # Step 3: Update __init__
    content = update_init_method(content)

    if content != original:
        if not dry_run:
            file_path.write_text(content)
        return True, "Updated"

    return False, "No changes needed"


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch add ConfigMixin to standalone classes')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--file', type=str, help='Process single file')
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        files = [Path(f) for f in FILES_TO_MIGRATE]

    updated = 0
    skipped = 0
    errors = 0

    for file_path in files:
        try:
            changed, msg = process_file(file_path, dry_run=args.dry_run)
            if changed:
                updated += 1
                print(f"{'Would update' if args.dry_run else 'Updated'}: {file_path}")
            else:
                skipped += 1
                if 'not found' in msg.lower():
                    print(f"Skipped (not found): {file_path}")
        except Exception as e:
            errors += 1
            print(f"Error processing {file_path}: {e}")

    print(f"\n{'Would update' if args.dry_run else 'Updated'}: {updated} files")
    print(f"Skipped: {skipped} files")
    if errors:
        print(f"Errors: {errors} files")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
