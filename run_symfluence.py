#!/usr/bin/env python3
"""
SYMFLUENCE Development Wrapper

This script allows running SYMFLUENCE from the source tree without installation.
It delegates all execution to the actual package entry point in src/symfluence/cli.py.
"""
import sys
from pathlib import Path

# Add src to python path for development so 'symfluence' package is found
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from symfluence.cli import main
except ImportError as e:
    print(f"Error importing symfluence: {e}")
    print(f"PYTHONPATH: {sys.path}")
    sys.exit(1)

if __name__ == "__main__":
    main()