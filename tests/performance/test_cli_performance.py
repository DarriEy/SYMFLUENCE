"""
CLI Performance Tests

Measures import time and CLI command execution time for performance tracking.
"""

import time
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable  # Use current Python interpreter


def measure_import_time():
    """Measure time to import the CLI argument parser."""
    start = time.perf_counter()
    cmd = [
        PYTHON, "-c",
        "import sys; from pathlib import Path; "
        "sys.path.insert(0, str(Path.cwd() / 'src')); "
        "import symfluence.cli.argument_parser"
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    end = time.perf_counter()
    return end - start


def measure_cli_help_time():
    """Measure time for 'symfluence --help' command."""
    start = time.perf_counter()
    cmd = [PYTHON, "-m", "symfluence", "--help"]
    subprocess.run(cmd, check=True, capture_output=True)
    end = time.perf_counter()
    return end - start


def measure_cli_list_steps_time():
    """Measure time for 'symfluence workflow list-steps' command."""
    start = time.perf_counter()
    cmd = [PYTHON, "-m", "symfluence", "workflow", "list-steps"]
    subprocess.run(cmd, check=True, capture_output=True)
    end = time.perf_counter()
    return end - start


if __name__ == "__main__":
    print("Measuring CLI performance...")

    import_time = measure_import_time()
    print(f"  Import time: {import_time:.4f}s")

    help_time = measure_cli_help_time()
    print(f"  '--help' time: {help_time:.4f}s")

    list_steps_time = measure_cli_list_steps_time()
    print(f"  'workflow list-steps' time: {list_steps_time:.4f}s")

    # Save results to a file for comparison later
    results_file = Path("cli_performance_baseline.txt")
    with open(results_file, "w") as f:
        f.write(f"import_time: {import_time}\n")
        f.write(f"help_time: {help_time}\n")
        f.write(f"list_steps_time: {list_steps_time}\n")

    print(f"\nResults saved to {results_file}")
