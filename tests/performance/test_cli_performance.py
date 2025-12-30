import time
import subprocess
import sys
import os
from pathlib import Path

PYTHON = "python3.11"

def measure_import_time():
    start = time.perf_counter()
    # Run in a separate process to avoid caching
    cmd = [PYTHON, "-c", "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd() / 'src')); import symfluence.utils.cli.cli_argument_manager"]
    subprocess.run(cmd, check=True)
    end = time.perf_counter()
    return end - start

def measure_cli_help_time():
    start = time.perf_counter()
    cmd = [PYTHON, "run_symfluence.py", "--help"]
    subprocess.run(cmd, check=True, capture_output=True)
    end = time.perf_counter()
    return end - start

def measure_cli_list_steps_time():
    start = time.perf_counter()
    cmd = [PYTHON, "run_symfluence.py", "--list_steps"]
    subprocess.run(cmd, check=True, capture_output=True)
    end = time.perf_counter()
    return end - start

if __name__ == "__main__":
    print("🚀 Measuring CLI performance...")
    
    import_time = measure_import_time()
    print(f"⏱️  Import time: {import_time:.4f}s")
    
    help_time = measure_cli_help_time()
    print(f"⏱️  '--help' time: {help_time:.4f}s")
    
    list_steps_time = measure_cli_list_steps_time()
    print(f"⏱️  '--list_steps' time: {list_steps_time:.4f}s")
    
    # Save results to a file for comparison later
    with open("cli_performance_baseline.txt", "w") as f:
        f.write(f"import_time: {import_time}\n")
        f.write(f"help_time: {help_time}\n")
        f.write(f"list_steps_time: {list_steps_time}\n")