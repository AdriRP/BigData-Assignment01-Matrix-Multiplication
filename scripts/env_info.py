"""Record the hardware and software environment to results/environment.json."""
import json
import platform
import subprocess
import sys

import numpy
import psutil

from common import CONFIG, RESULTS


def cpu_name():
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def tool_output(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return (p.stdout + p.stderr).strip()
    except OSError as e:
        return f"unavailable: {e}"


def main():
    RESULTS.mkdir(exist_ok=True)
    env = {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "ram_bytes": psutil.virtual_memory().total,
        "os": platform.platform(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "numpy (inputs/validation only)": numpy.__version__,
        "psutil (memory measurement)": psutil.__version__,
        "gcc": tool_output([CONFIG["c_compiler"], "--version"]).splitlines()[0],
        "c_flags": " ".join(CONFIG["c_flags"]),
        "java": tool_output(["java", "-version"]),
        "javac": tool_output(["javac", "-version"]),
        "jvm_flags": " ".join(CONFIG["jvm_flags"]),
        "jvm_ergonomic_flags": tool_output(["java", *CONFIG["jvm_flags"], "-XX:+PrintCommandLineFlags", "-version"]).splitlines()[0],
        "kernel_threads": "1 in every implementation (JVM also runs JIT/GC service threads)",
    }
    with open(RESULTS / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
