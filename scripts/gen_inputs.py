"""Generate the shared input matrices read by all three implementations.

For each n, A and B are drawn from NumPy's PCG64 generator seeded with the
sequence [seed, n], uniform in [low, high), float64, and stored as raw
little-endian row-major binary files (8 * n * n bytes each).

If results/inputs_manifest.csv does not exist it is written with the SHA-256
of every file. If it exists, it is not overwritten: the regenerated files
are checked against it, so anyone can confirm they obtained identical inputs.
"""
import csv
import hashlib

import numpy as np

from common import CONFIG, INPUTS, RESULTS, input_paths


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    INPUTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    low, high = CONFIG["value_range"]
    rows = []
    for n in CONFIG["sizes"]:
        rng = np.random.default_rng([CONFIG["seed"], n])
        a = rng.uniform(low, high, size=(n, n)).astype("<f8")
        b = rng.uniform(low, high, size=(n, n)).astype("<f8")
        pa, pb = input_paths(n)
        a.tofile(pa)
        b.tofile(pb)
        rows.append({"n": n, "generator": "numpy.random.PCG64", "seed": f"[{CONFIG['seed']}, {n}]",
                     "low": low, "high": high, "dtype": "float64 little-endian row-major",
                     "sha256_A": sha256(pa), "sha256_B": sha256(pb)})
        print(f"n={n}: wrote {pa.name}, {pb.name}")

    manifest = RESULTS / "inputs_manifest.csv"
    if not manifest.exists():
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"manifest written: {manifest}")
        return
    with open(manifest, encoding="utf-8") as f:
        expected = {int(r["n"]): r for r in csv.DictReader(f)}
    mismatches = 0
    for r in rows:
        e = expected.get(r["n"])
        ok = e is not None and e["sha256_A"] == r["sha256_A"] and e["sha256_B"] == r["sha256_B"]
        mismatches += not ok
        print(f"n={r['n']}: {'identical to manifest' if ok else 'DIFFERENT from manifest'}")
    print("all inputs identical to the manifest" if mismatches == 0 else f"{mismatches} size(s) differ")


if __name__ == "__main__":
    main()
