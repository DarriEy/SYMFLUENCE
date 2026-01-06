#!/usr/bin/env python3
"""
Manual script to fix mizuroute control files in existing parallel process directories.
This applies the same fix that will be automatically applied in future runs.
"""

import sys
from pathlib import Path

# Add symfluence to path
sys.path.insert(0, '/Users/darrieythorsson/compHydro/code/SYMFLUENCE/src')

def fix_mizuroute_control(control_file_path: Path, proc_id: int, experiment_id: str):
    """Fix a single mizuroute control file."""

    if not control_file_path.exists():
        print(f"  ✗ Control file not found: {control_file_path}")
        return False

    try:
        # Read existing control file
        with open(control_file_path, 'r') as f:
            lines = f.readlines()

        # Get process root directory (go up 3 levels from control file)
        proc_root = control_file_path.parent.parent.parent

        # Construct process-specific paths
        # SUMMA output is in: .../process_N/simulations/run_1/SUMMA/
        proc_summa_dir = proc_root / 'simulations' / experiment_id / 'SUMMA'
        # mizuRoute output should be sibling: .../process_N/simulations/run_1/mizuRoute/
        proc_mizu_dir = proc_root / 'simulations' / experiment_id / 'mizuRoute'
        # Ancil dir should use process-specific settings
        proc_ancil_dir = control_file_path.parent

        # Ensure mizuRoute simulation directory exists
        proc_mizu_dir.mkdir(parents=True, exist_ok=True)

        # Normalize paths (forward slashes, trailing slash)
        def normalize_path(path):
            return str(path).replace('\\', '/').rstrip('/') + '/'

        input_dir = normalize_path(proc_summa_dir)
        output_dir = normalize_path(proc_mizu_dir)
        ancil_dir = normalize_path(proc_ancil_dir)
        case_name = f'proc_{proc_id:02d}_{experiment_id}'
        fname_qsim = f'proc_{proc_id:02d}_{experiment_id}_timestep.nc'

        # Update relevant lines
        updated_lines = []
        changes = []

        for line in lines:
            if '<ancil_dir>' in line:
                if '!' in line:
                    comment = '!' + '!'.join(line.split('!')[1:])
                    new_line = f"<ancil_dir>             {ancil_dir}    {comment}"
                else:
                    new_line = f"<ancil_dir>             {ancil_dir}    ! Folder that contains ancillary data\n"
                updated_lines.append(new_line)
                if new_line != line:
                    changes.append(f"ancil_dir: {line.split()[1]} → {ancil_dir}")
            elif '<input_dir>' in line:
                if '!' in line:
                    comment = '!' + '!'.join(line.split('!')[1:])
                    new_line = f"<input_dir>             {input_dir}    {comment}"
                else:
                    new_line = f"<input_dir>             {input_dir}    ! Folder that contains runoff data from SUMMA\n"
                updated_lines.append(new_line)
                if new_line != line:
                    changes.append(f"input_dir: {line.split()[1]} → {input_dir}")
            elif '<output_dir>' in line:
                if '!' in line:
                    comment = '!' + '!'.join(line.split('!')[1:])
                    new_line = f"<output_dir>            {output_dir}    {comment}"
                else:
                    new_line = f"<output_dir>            {output_dir}    ! Folder that will contain mizuRoute simulations\n"
                updated_lines.append(new_line)
                if new_line != line:
                    changes.append(f"output_dir: {line.split()[1]} → {output_dir}")
            elif '<case_name>' in line:
                if '!' in line:
                    comment = '!' + '!'.join(line.split('!')[1:])
                    new_line = f"<case_name>             {case_name}    {comment}"
                else:
                    new_line = f"<case_name>             {case_name}    ! Simulation case name\n"
                updated_lines.append(new_line)
                if new_line != line:
                    changes.append(f"case_name: {line.split()[1]} → {case_name}")
            elif '<fname_qsim>' in line:
                if '!' in line:
                    comment = '!' + '!'.join(line.split('!')[1:])
                    new_line = f"<fname_qsim>            {fname_qsim}    {comment}"
                else:
                    new_line = f"<fname_qsim>            {fname_qsim}    ! netCDF name for SUMMA runoff\n"
                updated_lines.append(new_line)
                if new_line != line:
                    changes.append(f"fname_qsim: {line.split()[1]} → {fname_qsim}")
            else:
                updated_lines.append(line)

        # Write updated control file
        with open(control_file_path, 'w') as f:
            f.writelines(updated_lines)

        if changes:
            print(f"  ✓ Updated process_{proc_id}:")
            for change in changes:
                print(f"    - {change}")
        else:
            print(f"  ℹ Process_{proc_id} already correct")

        return True

    except Exception as e:
        print(f"  ✗ Failed to update process_{proc_id}: {e}")
        return False


def main():
    """Fix all mizuroute control files in the async-dds run directory."""

    base_dir = Path('/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Bow_at_Banff_lumped_em_earth/simulations/run_async-dds')

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return 1

    print(f"Fixing mizuroute control files in: {base_dir}\n")

    # Find all process directories
    process_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('process_')])

    if not process_dirs:
        print("No process directories found!")
        return 1

    print(f"Found {len(process_dirs)} process directories\n")

    success_count = 0
    for proc_dir in process_dirs:
        # Extract process ID from directory name
        proc_id = int(proc_dir.name.split('_')[1])

        # Find mizuroute control file
        control_file = proc_dir / 'settings' / 'mizuRoute' / 'mizuroute.control'

        if fix_mizuroute_control(control_file, proc_id, 'run_1'):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Summary: Updated {success_count}/{len(process_dirs)} control files")
    print(f"{'='*60}\n")

    if success_count == len(process_dirs):
        print("✓ All control files have been updated successfully!")
        print("\nEach process will now use its own directories:")
        print("  - Input:  .../run_async-dds/process_N/simulations/SUMMA/")
        print("  - Output: .../run_async-dds/process_N/simulations/mizuRoute/")
        return 0
    else:
        print("✗ Some control files failed to update")
        return 1


if __name__ == '__main__':
    sys.exit(main())
