"""Correctness checks.

1. Runs each implementation's --selftest (2x2 and 3x3 hand-computed products,
   A*I, I*A and A*0, compared exactly).
2. Compares every benchmark output matrix with a NumPy reference (A @ B).
   Tolerance per element: |C - C_ref| <= 2 * gamma_n * (|A| |B|)_ij with
   gamma_n = n*u / (1 - n*u), u = 2^-53. This is the standard forward error
   bound of a length-n dot product, doubled because the reference has its
   own rounding error in a different summation order.
3. Checks that the three languages produce bit-identical results (same loop
   order and IEEE-754 double arithmetic) and that the checksum of every
   repetition is identical.
Writes results/validation.csv.
"""
import csv
import subprocess
from collections import defaultdict

import numpy as np

from common import CONFIG, RESULTS, command, input_paths, output_path

U = 2.0 ** -53


def load(path, n):
    return np.fromfile(path, dtype="<f8").reshape(n, n)


def main():
    rows = []
    for lang in CONFIG["languages"]:
        p = subprocess.run(command(lang) + ["--selftest"], capture_output=True, text=True)
        ok = p.returncode == 0 and "SELFTEST OK" in p.stdout
        print(f"[{lang}] selftest:", "OK" if ok else "FAILED")
        print("   " + p.stdout.strip().replace("\n", "\n   "))
        rows.append({"check": "selftest (2x2, 3x3, A*I, I*A, A*0; exact)", "language": lang, "n": "",
                     "max_abs_error": 0.0 if ok else "", "max_error_over_bound": "", "result": "PASS" if ok else "FAIL"})

    for n in CONFIG["sizes"]:
        pa, pb = input_paths(n)
        a, b = load(pa, n), load(pb, n)
        ref = a @ b
        gamma = n * U / (1 - n * U)
        bound = 2 * gamma * (np.abs(a) @ np.abs(b))
        outputs = {}
        for lang in CONFIG["languages"]:
            path = output_path(lang, n)
            if not path.exists():
                continue
            c = load(path, n)
            outputs[lang] = c
            err = np.abs(c - ref)
            ratio = float((err / bound).max())
            ok = bool((err <= bound).all())
            rows.append({"check": "vs NumPy A@B, error-bound tolerance", "language": lang, "n": n,
                         "max_abs_error": f"{err.max():.3e}", "max_error_over_bound": f"{ratio:.3e}",
                         "result": "PASS" if ok else "FAIL"})
            print(f"n={n:5d} {lang:6s} max|err|={err.max():.3e}  max err/bound={ratio:.2e}  {'PASS' if ok else 'FAIL'}")
        if "c" in outputs:
            for lang, c in outputs.items():
                if lang == "c":
                    continue
                same = np.array_equal(c.view(np.uint64), outputs["c"].view(np.uint64))
                rows.append({"check": "bit-identical to C output", "language": lang, "n": n,
                             "max_abs_error": f"{np.abs(c - outputs['c']).max():.3e}", "max_error_over_bound": "",
                             "result": "PASS" if same else "DIFFERENT"})

    checksums = defaultdict(set)
    with open(RESULTS / "raw_times.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            checksums[(r["language"], r["n"])].add(r["checksum"])
    for (lang, n), values in sorted(checksums.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        rows.append({"check": "identical checksum in every run", "language": lang, "n": n,
                     "max_abs_error": "", "max_error_over_bound": "", "result": "PASS" if len(values) == 1 else "FAIL"})

    with open(RESULTS / "validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    failed = [r for r in rows if r["result"] not in ("PASS",)]
    print(f"{len(rows)} checks, {len(failed)} not PASS")


if __name__ == "__main__":
    main()
