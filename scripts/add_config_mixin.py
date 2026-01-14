#!/usr/bin/env python3
"""
Script to add ConfigMixin to standalone classes.

This script:
1. Adds ConfigMixin import
2. Adds ConfigMixin to class inheritance
3. Updates __init__ to properly set self._config

Usage:
    python scripts/add_config_mixin.py [--dry-run] [--file PATH]
"""

import re
from pathlib import Path
from typing import Tuple

# Files to process
FILES_TO_MIGRATE = [
    # data/acquisition
    'src/symfluence/data/acquisition/cleanup_processor.py',
    'src/symfluence/data/acquisition/cloud_downloader.py',
    'src/symfluence/data/acquisition/maf_pipeline.py',
    'src/symfluence/data/acquisition/maf_processor.py',
    'src/symfluence/data/acquisition/observed_processor.py',
    'src/symfluence/data/acquisition/zonal_stats_processor.py',
    # data/preprocessing
    'src/symfluence/data/preprocessing/attribute_processor.py',
    'src/symfluence/data/preprocessing/attribute_processors/base.py',
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
    # geospatial
    'src/symfluence/geospatial/geofabric/delineators/coastal_delineator.py',
    'src/symfluence/geospatial/geofabric/delineators/distributed_delineator.py',
    'src/symfluence/geospatial/geofabric/delineators/grid_delineator.py',
    'src/symfluence/geospatial/geofabric/delineators/lumped_delineator.py',
    'src/symfluence/geospatial/geofabric/delineators/point_delineator.py',
    'src/symfluence/geospatial/geofabric/delineators/subsetter.py',
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

# Algorithm files that inherit from OptimizationAlgorithm - need special handling
ALGORITHM_FILES = [
    'src/symfluence/optimization/optimizers/algorithms/adam.py',
    'src/symfluence/optimization/optimizers/algorithms/async_dds.py',
    'src/symfluence/optimization/optimizers/algorithms/dds.py',
    'src/symfluence/optimization/optimizers/algorithms/de.py',
    'src/symfluence/optimization/optimizers/algorithms/lbfgs.py',
    'src/symfluence/optimization/optimizers/algorithms/pso.py',
]

# BaseWorker file - needs special handling
BASE_WORKER_FILE = 'src/symfluence/optimization/workers/base_worker.py'

# Attribute processor files that inherit from BaseAttributeProcessor
ATTRIBUTE_PROCESSOR_FILES = [
    'src/symfluence/data/preprocessing/attribute_processors/climate.py',
    'src/symfluence/data/preprocessing/attribute_processors/elevation.py',
    'src/symfluence/data/preprocessing/attribute_processors/geology.py',
    'src/symfluence/data/preprocessing/attribute_processors/landcover.py',
    'src/symfluence/data/preprocessing/attribute_processors/soil.py',
]

# MODIS handlers that inherit from a base handler
MODIS_HANDLER_FILES = [
    'src/symfluence/data/acquisition/handlers/modis_et.py',
    'src/symfluence/data/acquisition/handlers/modis_sca.py',
]


def add_configmixin_import(content: str) -> str:
    """Add ConfigMixin import if not present."""
    if 'from symfluence.core.mixins import' in content:
        # Already has mixin import, check if ConfigMixin is included
        if 'ConfigMixin' not in content:
            # Add ConfigMixin to existing import
            content = re.sub(
                r'(from symfluence\.core\.mixins import )([^\n]+)',
                r'\1ConfigMixin, \2',
                content
            )
    elif 'from symfluence.core import' in content:
        # Has core import, add ConfigMixin
        if 'ConfigMixin' not in content:
            content = re.sub(
                r'(from symfluence\.core import )([^\n]+)',
                r'\1ConfigMixin, \2',
                content
            )
    else:
        # Need to add import - find a good place
        # Try to add after other symfluence imports
        if 'from symfluence' in content:
            # Add after first symfluence import
            content = re.sub(
                r'(from symfluence[^\n]+\n)',
                r'\1from symfluence.core.mixins import ConfigMixin\n',
                content,
                count=1
            )
        else:
            # Add after standard imports
            if 'import ' in content:
                # Find last import line
                lines = content.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        insert_idx = i + 1
                lines.insert(insert_idx, 'from symfluence.core.mixins import ConfigMixin')
                content = '\n'.join(lines)
    return content


def add_configmixin_to_class(content: str, class_name: str) -> str:
    """Add ConfigMixin to class inheritance."""
    # Pattern: class ClassName: or class ClassName(BaseClass):
    # Transform to: class ClassName(ConfigMixin): or class ClassName(ConfigMixin, BaseClass):

    # Check if class already has ConfigMixin
    if re.search(rf'class\s+{class_name}\s*\([^)]*ConfigMixin[^)]*\)', content):
        return content

    # Case 1: class ClassName: (no inheritance)
    pattern1 = rf'(class\s+{class_name})\s*:'
    if re.search(pattern1, content):
        content = re.sub(pattern1, r'\1(ConfigMixin):', content)
        return content

    # Case 2: class ClassName(BaseClass): or class ClassName(Base1, Base2):
    pattern2 = rf'(class\s+{class_name})\s*\(([^)]+)\):'
    match = re.search(pattern2, content)
    if match:
        bases = match.group(2)
        # Add ConfigMixin as first base class
        new_bases = f'ConfigMixin, {bases}'
        content = re.sub(pattern2, rf'\1({new_bases}):', content)

    return content


def update_init_for_config(content: str, class_name: str) -> str:
    """Update __init__ to properly handle config and set self._config."""
    # Find the __init__ method for this class
    # This is complex because we need to find the right __init__

    # Pattern to find __init__ after class definition
    class_pattern = rf'class\s+{class_name}\s*\([^)]*\):[^\n]*\n((?:\s+[^\n]*\n)*?)\s+def\s+__init__'

    match = re.search(class_pattern, content)
    if not match:
        return content

    # Find the self.config = config line and update it
    # Look for pattern: self.config = config
    init_pattern = r'(\s+)(self\.config\s*=\s*config)\b'

    def replacement(m):
        indent = m.group(1)
        return f'''{indent}# Auto-convert dict to typed config for backward compatibility
{indent}from symfluence.core.config.models import SymfluenceConfig
{indent}if isinstance(config, dict):
{indent}    self._config = SymfluenceConfig(**config)
{indent}else:
{indent}    self._config = config'''

    # Only replace within the class scope (rough approximation)
    content = re.sub(init_pattern, replacement, content, count=1)

    return content


def process_file(file_path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Process a single file to add ConfigMixin."""
    content = file_path.read_text()
    original = content

    # Find all class definitions in the file
    class_names = re.findall(r'class\s+(\w+)\s*[:\(]', content)

    if not class_names:
        return False, "No classes found"

    # Add import
    content = add_configmixin_import(content)

    # Add ConfigMixin to each class
    for class_name in class_names:
        content = add_configmixin_to_class(content, class_name)
        content = update_init_for_config(content, class_name)

    if content != original:
        if not dry_run:
            file_path.write_text(content)
        return True, f"Updated {len(class_names)} class(es)"

    return False, "No changes needed"


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add ConfigMixin to standalone classes')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--file', type=Path, help='Process single file')
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = [Path(f) for f in FILES_TO_MIGRATE]

    updated = 0
    for file_path in files:
        if not file_path.exists():
            print(f"Not found: {file_path}")
            continue

        changed, msg = process_file(file_path, dry_run=args.dry_run)
        if changed:
            updated += 1
            print(f"{'Would update' if args.dry_run else 'Updated'}: {file_path} - {msg}")

    print(f"\n{'Would update' if args.dry_run else 'Updated'} {updated} files")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
