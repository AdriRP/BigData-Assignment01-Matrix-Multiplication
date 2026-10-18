"""Basic dense matrix multiplication, C = A * B, in pure Python.

Storage: list of n row lists, each holding n Python float objects
(the lists hold pointers; every float is a separate heap object).
Algorithm: textbook i-j-k triple loop, scalar accumulator, O(n^3).
No NumPy or other library is used in the kernel.

Usage:
    python matmul.py --selftest
    python matmul.py <n> <A.bin> <B.bin> <warmups> <reps> <out.bin|-> [--hold]

Output (stdout), one line per run:  W|M,<index>,<nanoseconds>,<checksum>
then "DONE". With --hold the process waits for a line on stdin before
exiting, so the benchmark driver can read its peak working set.
"""
import sys
import time
from array import array


def multiply(a, b, c, n):
    for i in range(n):
        ai = a[i]
        ci = c[i]
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += ai[k] * b[k][j]
            ci[j] = s


def checksum(c):
    s = 0.0
    for row in c:
        for v in row:
            s += v
    return s


def read_matrix(path, n):
    m = []
    with open(path, "rb") as f:
        for _ in range(n):
            row = array("d")
            row.frombytes(f.read(8 * n))
            if sys.byteorder != "little":
                row.byteswap()
            m.append(row.tolist())
    return m


def write_matrix(path, m):
    with open(path, "wb") as f:
        for row in m:
            r = array("d", row)
            if sys.byteorder != "little":
                r.byteswap()
            f.write(r.tobytes())


# ---------- self-test: known answers, exact comparison ----------

def expect_equal(name, got, want):
    for i, row in enumerate(want):
        for j, w in enumerate(row):
            if got[i][j] != w:
                print(f"FAIL {name} at ({i},{j}): got {got[i][j]!r} want {w!r}")
                return 1
    print(f"PASS {name}")
    return 0


def selftest():
    fails = 0
    c2 = [[0.0] * 2 for _ in range(2)]
    multiply([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], c2, 2)
    fails += expect_equal("2x2_known", c2, [[19.0, 22.0], [43.0, 50.0]])

    c3 = [[0.0] * 3 for _ in range(3)]
    multiply([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
             [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], c3, 3)
    fails += expect_equal("3x3_known", c3, [[30.0, 24.0, 18.0], [84.0, 69.0, 54.0], [138.0, 114.0, 90.0]])

    n = 5
    a = [[float((i * n + j) * 7 % 11) - 5.0 for j in range(n)] for i in range(n)]
    ident = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    zero = [[0.0] * n for _ in range(n)]
    c = [[0.0] * n for _ in range(n)]
    multiply(a, ident, c, n)
    fails += expect_equal("A_times_identity", c, a)
    multiply(ident, a, c, n)
    fails += expect_equal("identity_times_A", c, a)
    multiply(a, zero, c, n)
    fails += expect_equal("A_times_zero", c, zero)
    print("SELFTEST OK" if fails == 0 else "SELFTEST FAILED")
    return 0 if fails == 0 else 1


def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) < 7:
        print("usage: python matmul.py <n> <A.bin> <B.bin> <warmups> <reps> <out.bin|-> [--hold]", file=sys.stderr)
        return 1
    n = int(argv[1])
    warmups, reps = int(argv[4]), int(argv[5])
    out_path = argv[6]
    hold = len(argv) >= 8 and argv[7] == "--hold"

    a = read_matrix(argv[2], n)
    b = read_matrix(argv[3], n)
    c = [[0.0] * n for _ in range(n)]

    for r in range(warmups + reps):
        t0 = time.perf_counter_ns()
        multiply(a, b, c, n)
        t1 = time.perf_counter_ns()
        warm = r < warmups
        print(f"{'W' if warm else 'M'},{r if warm else r - warmups},{t1 - t0},{checksum(c):.17g}", flush=True)

    if out_path != "-":
        write_matrix(out_path, c)
    print("DONE", flush=True)
    if hold:
        sys.stdin.readline()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
