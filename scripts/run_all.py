"""Full pipeline: build, inputs, environment, benchmarks, validation, analysis."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for step in ["build.py", "gen_inputs.py", "env_info.py", "run_benchmarks.py", "validate.py", "analyze.py"]:
    print(f"=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / step)], check=True)
