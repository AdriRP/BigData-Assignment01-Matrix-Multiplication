"""Supplementary layout diagnostic (separate from the controlled comparison).

Builds src/probe/{probe.c, Probe.java} and measures the time per multiply-add
of the i-j-k kernel restricted to the first ROWS rows of C, for three storage
layouts of B: C contiguous array, Java double[][] and Java flat double[].
Writes results/probe_times.csv and figures/probe_layout.png.
"""
import csv
import subprocess

import matplotlib.pyplot as plt
import pandas as pd

from common import BUILD, CONFIG, EXE, FIGURES, RESULTS, ROOT

SIZES = [500, 1000, 1200, 1400, 1500, 1600, 1800, 2000, 2400, 2800, 3200, 4000]
ROWS, WARMUPS, REPS = 16, 3, 5


def main():
    probe_c = BUILD / f"probe_c{EXE}"
    subprocess.run([CONFIG["c_compiler"], *CONFIG["c_flags"], "-o", str(probe_c), str(ROOT / "src" / "probe" / "probe.c")], check=True)
    (BUILD / "probe").mkdir(parents=True, exist_ok=True)
    subprocess.run(["javac", "--release", "21", "-d", str(BUILD / "probe"), str(ROOT / "src" / "probe" / "Probe.java")], check=True)
    variants = {
        "c_flat": lambda n: [str(probe_c), str(n), str(ROWS), str(WARMUPS), str(REPS)],
        "java_2d": lambda n: ["java", *CONFIG["jvm_flags"], "-cp", str(BUILD / "probe"), "Probe", "2d", str(n), str(ROWS), str(WARMUPS), str(REPS)],
        "java_flat": lambda n: ["java", *CONFIG["jvm_flags"], "-cp", str(BUILD / "probe"), "Probe", "flat", str(n), str(ROWS), str(WARMUPS), str(REPS)],
    }
    rows = []
    for n in SIZES:
        for name, cmd in variants.items():
            out = subprocess.run(cmd(n), capture_output=True, text=True, check=True).stdout
            for line in out.split():
                v, nn, r, rep, ns, chk = line.split(",")
                rows.append({"variant": v, "n": int(nn), "rows": int(r), "rep": int(rep), "time_ns": int(ns), "checksum": chk})
            last = [x for x in rows if x["variant"] == name and x["n"] == n]
            med = sorted(x["time_ns"] for x in last)[len(last) // 2] / (ROWS * n * n)
            print(f"{name:10s} n={n:5d} {med:6.2f} ns/madd", flush=True)
    with open(RESULTS / "probe_times.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot()


def plot():
    df = pd.read_csv(RESULTS / "probe_times.csv")
    df["ns_per_madd"] = df.time_ns / (df.rows * df.n.astype(float) ** 2)
    g = df.groupby(["variant", "n"]).ns_per_madd
    s = pd.DataFrame({"median": g.median(), "q1": g.quantile(0.25), "q3": g.quantile(0.75)}).reset_index()
    s.to_csv(RESULTS / "probe_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    styles = {"c_flat": ("#2a78d6", "o", "C, contiguous array"),
              "java_2d": ("#eb6834", "s", "Java, double[n][n] (as in main study)"),
              "java_flat": ("#4a3aa7", "D", "Java, flat double[n·n]")}
    for v, (color, marker, label) in styles.items():
        d = s[s.variant == v]
        ax.errorbar(d.n, d["median"], yerr=[d["median"] - d.q1, d.q3 - d["median"]], color=color, marker=marker,
                    markersize=6, linewidth=2, capsize=3, label=label)
    ax.set_xlabel("Matrix size n (first 16 rows of C computed)")
    ax.set_ylabel("Median ns per multiply-add")
    ax.set_ylim(bottom=0)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "probe_layout.png", dpi=200)


if __name__ == "__main__":
    import sys
    plot() if "--plot-only" in sys.argv else main()
