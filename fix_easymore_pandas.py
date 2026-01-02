#!/usr/bin/env python3
"""
Fix for EASYMORE pandas compatibility issue.
Patches the spatial_overlays method to handle newer pandas versions.
"""

import sys
from pathlib import Path

def patch_easymore():
    """Patch EASYMORE to fix pandas 2.x compatibility."""
    
    # Find the EASYMORE remapper.py file
    easymore_path = Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "easymore" / "remapper.py"
    
    if not easymore_path.exists():
        print(f"EASYMORE not found at {easymore_path}")
        return False
    
    # Read the current file
    with open(easymore_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "# PANDAS_FIX_APPLIED" in content:
        print("EASYMORE already patched for pandas compatibility")
        return True
    
    # Find the problematic line and replace it
    old_line = "            pairs['Intersection'] = pairs.apply(lambda x: (x['geometry_1'].intersection(x['geometry_2'])).buffer(0), axis=1)"
    
    new_line = """            # PANDAS_FIX_APPLIED - Fix for pandas 2.x compatibility
            try:
                pairs['Intersection'] = pairs.apply(lambda x: (x['geometry_1'].intersection(x['geometry_2'])).buffer(0), axis=1)
            except ValueError as e:
                if "multiple columns" in str(e):
                    # Handle pandas 2.x issue with apply returning multiple columns
                    intersection_result = pairs.apply(lambda x: (x['geometry_1'].intersection(x['geometry_2'])).buffer(0), axis=1, result_type='reduce')
                    pairs['Intersection'] = intersection_result
                else:
                    raise"""
    
    if old_line in content:
        # Replace the problematic line
        content = content.replace(old_line, new_line)
        
        # Write back the patched file
        with open(easymore_path, 'w') as f:
            f.write(content)
        
        print("Successfully patched EASYMORE for pandas 2.x compatibility")
        return True
    else:
        print("Could not find the problematic line to patch")
        return False

if __name__ == "__main__":
    success = patch_easymore()
    sys.exit(0 if success else 1)
