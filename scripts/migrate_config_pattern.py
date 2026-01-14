#!/usr/bin/env python3
"""
Migration script to convert self.config.get('KEY') to _get_config_value pattern.

This script converts legacy config access patterns to the new unified pattern:
- FROM: self.config.get('DOMAIN_NAME', 'default')
- TO:   self._get_config_value(lambda: self.config.domain.name, default='default', dict_key='DOMAIN_NAME')

Usage:
    python scripts/migrate_config_pattern.py [--dry-run] [--file PATH]
"""

import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List

# Import the mapping from the codebase
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from symfluence.core.config.transformers import FLAT_TO_NESTED_MAP


def path_tuple_to_accessor(path_tuple: Tuple[str, ...]) -> str:
    """Convert path tuple to config accessor string.

    Examples:
        ('domain', 'name') -> 'self.config.domain.name'
        ('model', 'summa', 'install_path') -> 'self.config.model.summa.install_path'
    """
    return 'self.config.' + '.'.join(path_tuple)


def convert_config_get(match: re.Match, mapping: Dict[str, Tuple[str, ...]]) -> str:
    """Convert a self.config.get() call to _get_config_value pattern."""
    key = match.group(1)
    default = match.group(2)  # May be None if no default provided

    # Look up the key in the mapping
    path_tuple = mapping.get(key)

    if not path_tuple:
        # Key not in mapping, convert to config_dict.get() instead
        if default:
            return f"self.config_dict.get('{key}', {default})"
        else:
            return f"self.config_dict.get('{key}')"

    # Build the typed accessor
    accessor = path_tuple_to_accessor(path_tuple)

    # Build the _get_config_value call
    if default:
        # Clean up the default value (remove leading comma and space)
        default_clean = default.strip()
        return f"self._get_config_value(lambda: {accessor}, default={default_clean}, dict_key='{key}')"
    else:
        return f"self._get_config_value(lambda: {accessor}, dict_key='{key}')"


def has_config_mixin(content: str) -> bool:
    """Check if file contains classes that have access to ConfigMixin methods.

    Checks for:
    1. Direct inheritance from ConfigMixin or ConfigurableMixin
    2. Inheritance from common base classes that have ConfigMixin:
       - BaseAcquisitionHandler, BaseModelPreProcessor, BaseModelRunner, BaseModelPostProcessor
       - BaseManager, PathResolverMixin, ShapefileAccessMixin
       - Any class with 'Base' prefix that likely has ConfigMixin
    3. Import of ConfigMixin or ConfigurableMixin
    """
    # Direct inheritance patterns
    if re.search(r'class\s+\w+\s*\([^)]*(?:ConfigMixin|ConfigurableMixin)[^)]*\)', content):
        return True

    # Common base classes that have ConfigMixin
    base_classes_with_mixin = [
        'BaseAcquisitionHandler',
        'BaseModelPreProcessor',
        'BaseModelRunner',
        'BaseModelPostProcessor',
        'BaseManager',
        'PathResolverMixin',
        'ShapefileAccessMixin',
        'BaseObservationHandler',
        'BaseOptimizer',
        'BaseModelOptimizer',
        'BasePlotter',
        'BaseDatasetHandler',
        'BaseGeofabricDelineator',  # Has PathResolverMixin
        'BaseAttributeProcessor',  # Added ConfigMixin
        'OptimizationAlgorithm',  # Added ConfigMixin
        # Note: BaseWorker intentionally excluded - it uses dict configs directly
    ]

    for base in base_classes_with_mixin:
        if re.search(rf'class\s+\w+\s*\([^)]*{base}[^)]*\)', content):
            return True

    # Import of ConfigMixin or ConfigurableMixin (suggests file uses mixin pattern)
    if re.search(r'from\s+symfluence\.core\.mixins\s+import.*(?:ConfigMixin|ConfigurableMixin)', content):
        return True
    if re.search(r'from\s+symfluence\.core\s+import.*(?:ConfigMixin|ConfigurableMixin)', content):
        return True

    return False


def migrate_file(file_path: Path, mapping: Dict[str, Tuple[str, ...]], dry_run: bool = False,
                 force_all: bool = False) -> Tuple[int, List[str], bool]:
    """Migrate a single file to the new config pattern.

    Args:
        file_path: Path to the file
        mapping: Flat-to-nested key mapping
        dry_run: If True, don't write changes
        force_all: If True, migrate even without ConfigMixin (uses config_dict.get)

    Returns:
        Tuple of (number of replacements, list of changes made, has_mixin)
    """
    content = file_path.read_text()
    original_content = content
    changes = []

    has_mixin = has_config_mixin(content)

    # Pattern to match self.config.get('KEY') or self.config.get('KEY', default)
    # Captures: KEY, and optionally the default value
    pattern = r"self\.config\.get\(['\"]([A-Z_]+)['\"](?:,\s*([^)]+))?\)"

    def replacement(match):
        old = match.group(0)
        if has_mixin:
            new = convert_config_get(match, mapping)
        else:
            # For files without ConfigMixin, just convert to config_dict.get
            # which maintains dict-like access but is explicit
            key = match.group(1)
            default = match.group(2)
            if default:
                new = f"self.config.get('{key}', {default.strip()})"  # Keep as-is, it works
            else:
                new = f"self.config.get('{key}')"  # Keep as-is
            return old  # Don't change files without ConfigMixin unless force_all
        if old != new:
            changes.append(f"  {old}\n  -> {new}")
        return new

    new_content = re.sub(pattern, replacement, content)

    if new_content != original_content:
        if not dry_run:
            file_path.write_text(new_content)
        return len(changes), changes, has_mixin

    return 0, [], has_mixin


def find_python_files(directory: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """Find all Python files in directory, excluding specified patterns."""
    exclude_patterns = exclude_patterns or ['__pycache__', '.git', 'venv', 'scripts/migrate']

    files = []
    for f in directory.rglob('*.py'):
        if not any(excl in str(f) for excl in exclude_patterns):
            files.append(f)
    return files


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Migrate config.get() to _get_config_value pattern')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without applying')
    parser.add_argument('--file', type=Path, help='Migrate single file instead of all')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed changes')
    args = parser.parse_args()

    src_dir = Path(__file__).parent.parent / 'src' / 'symfluence'

    if args.file:
        files = [args.file.resolve()]
    else:
        files = find_python_files(src_dir)

    total_changes = 0
    files_changed = 0
    files_without_mixin = []

    for file_path in sorted(files):
        # Ensure file_path is absolute
        file_path = file_path.resolve()
        count, changes, has_mixin = migrate_file(file_path, FLAT_TO_NESTED_MAP, dry_run=args.dry_run)

        # Track files that have config.get but no ConfigMixin
        if not has_mixin:
            content = file_path.read_text()
            if re.search(r"self\.config\.get\(['\"][A-Z_]+['\"]", content):
                files_without_mixin.append(file_path)

        if count > 0:
            files_changed += 1
            total_changes += count
            try:
                rel_path = file_path.relative_to(src_dir.parent.parent)
            except ValueError:
                rel_path = file_path
            print(f"\n{rel_path}: {count} changes")
            if args.verbose:
                for change in changes:
                    print(change)

    action = "Would make" if args.dry_run else "Made"
    print(f"\n{action} {total_changes} changes in {files_changed} files")

    if files_without_mixin:
        print(f"\n⚠️  {len(files_without_mixin)} files use self.config.get() but don't inherit from ConfigMixin:")
        for f in files_without_mixin[:10]:
            try:
                rel_path = f.relative_to(src_dir.parent.parent)
            except ValueError:
                rel_path = f
            print(f"  - {rel_path}")
        if len(files_without_mixin) > 10:
            print(f"  ... and {len(files_without_mixin) - 10} more")
        print("\nThese files were NOT migrated. Consider adding ConfigMixin inheritance.")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
