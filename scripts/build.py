"""Compile the C and Java implementations into build/."""
import subprocess

from common import BUILD, C_BINARY, CONFIG, JAVA_CLASSES, RESULTS, ROOT


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    BUILD.mkdir(exist_ok=True)
    JAVA_CLASSES.mkdir(exist_ok=True)
    run([CONFIG["c_compiler"], *CONFIG["c_flags"], "-o", str(C_BINARY), str(ROOT / "src" / "c" / "matmul.c")])
    # Report which loops the compiler vectorised (expected: none in multiply()).
    RESULTS.mkdir(exist_ok=True)
    run([CONFIG["c_compiler"], *CONFIG["c_flags"], "-fopt-info-vec-all=" + str(RESULTS / "c_vectorization.txt"),
         "-c", "-o", str(BUILD / "matmul_check.o"), str(ROOT / "src" / "c" / "matmul.c")])
    run(["javac", "--release", "21", "-d", str(JAVA_CLASSES), str(ROOT / "src" / "java" / "MatMul.java")])
    print("build OK")


if __name__ == "__main__":
    main()
