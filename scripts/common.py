"""Shared paths, configuration and per-language commands."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
OUTPUTS = DATA / "outputs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

with open(ROOT / "config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)

EXE = ".exe" if os.name == "nt" else ""
C_BINARY = BUILD / f"matmul_c{EXE}"
JAVA_CLASSES = BUILD / "java"
PY_SOURCE = ROOT / "src" / "python" / "matmul.py"


def input_paths(n):
    return INPUTS / f"A_{n}.bin", INPUTS / f"B_{n}.bin"


def output_path(lang, n):
    return OUTPUTS / f"C_{lang}_{n}.bin"


def command(lang):
    """Command prefix that runs the implementation for `lang`."""
    if lang == "c":
        return [str(C_BINARY)]
    if lang == "java":
        return ["java", *CONFIG["jvm_flags"], "-cp", str(JAVA_CLASSES), "MatMul"]
    if lang == "python":
        return [sys.executable, str(PY_SOURCE)]
    raise ValueError(lang)


# Bytes per stored matrix element, used only for the memory-budget estimate.
# Python stores an 8-byte pointer per element plus a 24-byte float object.
BYTES_PER_ELEMENT = {"c": 8, "java": 8, "python": 32}
