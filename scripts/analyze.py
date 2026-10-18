"""Summarise raw measurements and draw the figures.

Reads results/raw_times.csv and results/runs.csv (measured runs only; warm-up
runs are summarised separately) and writes:
  results/summary.csv      median, quartiles, IQR, min, max per configuration
  results/summary.md       the same tables in Markdown, used by the report
  figures/*.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, RESULTS

LANGS = ["c", "java", "python"]
LABEL = {"c": "C (gcc -O2)", "java": "Java (HotSpot JIT)", "python": "Python (CPython)"}
COLOR = {"c": "#2a78d6", "java": "#eb6834", "python": "#1baf7a"}
MARKER = {"c": "o", "java": "s", "python": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def style(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, which="both")
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def summarise(raw):
    meas = raw[raw.phase == "measured"]
    g = meas.groupby(["language", "n"])["time_s"]
    s = pd.DataFrame({
        "runs": g.size(),
        "median_s": g.median(),
        "q1_s": g.quantile(0.25),
        "q3_s": g.quantile(0.75),
        "min_s": g.min(),
        "max_s": g.max(),
    }).reset_index()
    s["iqr_s"] = s.q3_s - s.q1_s
    s["rel_iqr_pct"] = 100 * s.iqr_s / s.median_s
    s["ns_per_madd"] = s.median_s * 1e9 / s.n.astype(float) ** 3
    warm = raw[(raw.phase == "warmup") & (raw.rep == 0)].set_index(["language", "n"])["time_s"]
    s["first_warmup_s"] = [warm.get((l, n), np.nan) for l, n in zip(s.language, s.n)]
    return s


def fit_exponent(s, lang, n_min=300):
    d = s[(s.language == lang) & (s.n >= n_min)]
    if len(d) < 2:
        return np.nan
    return np.polyfit(np.log(d.n), np.log(d.median_s), 1)[0]


def fig_time(s, runs):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for lang in LANGS:
        d = s[s.language == lang]
        ax.errorbar(d.n, d.median_s, yerr=[d.median_s - d.q1_s, d.q3_s - d.median_s], color=COLOR[lang],
                    marker=MARKER[lang], markersize=6, linewidth=2, capsize=3, label=LABEL[lang])
        # cubic model c*n^3 fitted to the measured points with exponent fixed at 3
        c = np.exp(np.mean(np.log(d.median_s) - 3 * np.log(d.n)))
        xs = np.geomspace(d.n.min(), d.n.max(), 50)
        ax.plot(xs, c * xs ** 3, ":", color=COLOR[lang], linewidth=1.2, alpha=0.9)
        t = runs[(runs.language == lang) & runs.status.str.startswith("timeout")]
        for n in t.n:
            ax.scatter([n], [60], marker="x", s=70, linewidths=2, color=COLOR[lang], zorder=5)
            ax.annotate(f"{LABEL[lang].split()[0]}\ntimeout", (n, 60), textcoords="offset points",
                        xytext=(0, -26), ha="center", fontsize=8, color=MUTED)
        last = d.iloc[-1]
        offset, ha = ((-8, 6), "right") if lang == "python" else ((7, -3), "left")
        ax.annotate(LABEL[lang].split()[0], (last.n, last.median_s), textcoords="offset points",
                    xytext=offset, ha=ha, fontsize=9, color=INK)
    ax.axhline(60, color="#898781", linewidth=1, linestyle="--")
    ax.text(95, 64, "per-run budget 60 s", fontsize=8, color=MUTED, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(85, 3400)
    ax.set_ylim(5e-4, 150)
    ax.set_xlabel("Matrix size n (n × n)")
    ax.set_ylabel("Kernel time per multiplication (s), median ± IQR")
    ax.plot([], [], ":", color=MUTED, label="fitted c·n³ (dotted)")
    ax.scatter([], [], marker="x", color=MUTED, label="run exceeded budget")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "time_vs_n.png", dpi=200)
    plt.close(fig)


def fig_normalised(s):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1]})
    for lang in LANGS:
        d = s[s.language == lang]
        target = ax[1] if lang == "python" else ax[0]
        target.errorbar(d.n, d.ns_per_madd, yerr=[(d.median_s - d.q1_s) * 1e9 / d.n ** 3,
                                                  (d.q3_s - d.median_s) * 1e9 / d.n ** 3],
                        color=COLOR[lang], marker=MARKER[lang], markersize=6, linewidth=2, capsize=3,
                        label=LABEL[lang])
    for a, title in zip(ax, ["Compiled / JIT-compiled", "Interpreted"]):
        a.set_xscale("log")
        a.set_ylim(bottom=0)
        a.set_xlabel("Matrix size n")
        a.set_title(title, fontsize=10, color=INK)
        a.legend(frameon=False, fontsize=9, loc="upper left")
        style(a)
    ax[0].set_ylabel("Median time / n³ (ns per multiply-add)")
    fig.tight_layout()
    fig.savefig(FIGURES / "ns_per_madd.png", dpi=200)
    plt.close(fig)


def fig_memory(runs):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ok = runs[runs.status == "ok"]
    for lang in LANGS:
        d = ok[ok.language == lang]
        ax.plot(d.n, d.peak_working_set_bytes / 1e6, color=COLOR[lang], marker=MARKER[lang], markersize=6,
                linewidth=2, label=LABEL[lang])
    xs = np.geomspace(100, ok.n.max(), 50)
    ax.plot(xs, 24 * xs ** 2 / 1e6, "--", color="#898781", linewidth=1.2, label="3 × 8n² B (raw float64 storage)")
    ax.plot(xs[xs <= 1000], 96 * xs[xs <= 1000] ** 2 / 1e6, ":", color="#898781", linewidth=1.2,
            label="3 × 32n² B (list pointer + float object)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Peak working set (MB)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "memory_vs_n.png", dpi=200)
    plt.close(fig)


def fig_warmup(raw):
    """Per-run times of Java, showing JIT warm-up relative to the measured median."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    d = raw[raw.language == "java"].copy()
    d["order"] = np.where(d.phase == "warmup", d.rep, d.rep + 3)
    for n, g in d.groupby("n"):
        med = g[g.phase == "measured"].time_s.median()
        ax.plot(g.order + 1, g.time_s / med, marker="o", markersize=4, linewidth=1.2, label=f"n = {n}",
                color=plt.cm.Blues(0.35 + 0.65 * (np.log(n) - np.log(100)) / (np.log(2800) - np.log(100))))
    ax.axvspan(0.5, 3.5, color="#f0efec", zorder=0)
    ax.text(2, ax.get_ylim()[1] * 0.97, "warm-up (excluded)", ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_xlabel("Run number inside one JVM process")
    ax.set_ylabel("Time / median of measured runs")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "java_warmup.png", dpi=200)
    plt.close(fig)


def md_table(df):
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else ("" if pd.isna(v) else str(v))
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def markdown(s, runs, env):
    lines = []
    piv = s.pivot(index="n", columns="language")
    lines.append("### Median kernel time (s) [IQR] , 5 measured runs\n")
    lines.append("| n | C | Java | Python | Java / C | Python / C |")
    lines.append("|---|---|---|---|---|---|")
    status = runs.set_index(["language", "n"]).status
    for n in sorted(runs.n.unique()):
        cells = []
        for lang in LANGS:
            if (lang, n) in s.set_index(["language", "n"]).index:
                r = s[(s.language == lang) & (s.n == n)].iloc[0]
                cells.append(f"{r.median_s:.4g} [{r.iqr_s:.2g}]")
            else:
                st = status.get((lang, n), "")
                cells.append("timeout" if st.startswith("timeout") else ("n/a" if st else ""))
        ratio = []
        for lang in ("java", "python"):
            try:
                v = piv.median_s[lang][n] / piv.median_s["c"][n]
                ratio.append("" if np.isnan(v) else f"{v:.1f}×")
            except KeyError:
                ratio.append("")
        lines.append(f"| {n} | " + " | ".join(cells + ratio) + " |")
    lines.append("")
    lines.append("### Exponent of fitted power law t ∝ n^b (n ≥ 300)\n")
    for lang in LANGS:
        lines.append(f"- {lang}: b = {fit_exponent(s, lang):.2f}")
    lines.append("")
    lines.append("### Runs table\n")
    lines.append(md_table(runs))
    lines.append("")
    lines.append("### Full summary\n")
    lines.append(md_table(s))
    with open(RESULTS / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    FIGURES.mkdir(exist_ok=True)
    raw = pd.read_csv(RESULTS / "raw_times.csv")
    raw["time_s"] = raw.time_ns / 1e9
    runs = pd.read_csv(RESULTS / "runs.csv")
    env = json.load(open(RESULTS / "environment.json", encoding="utf-8"))
    s = summarise(raw)
    s.to_csv(RESULTS / "summary.csv", index=False)
    fig_time(s, runs)
    fig_normalised(s)
    fig_memory(runs)
    fig_warmup(raw)
    markdown(s, runs, env)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    for lang in LANGS:
        print(f"exponent {lang}: {fit_exponent(s, lang):.3f}")


if __name__ == "__main__":
    main()
