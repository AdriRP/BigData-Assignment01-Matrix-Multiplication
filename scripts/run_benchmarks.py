"""Benchmark driver: runs every (size, language) configuration in its own process.

Each process loads the shared inputs, performs the configured warm-up and
measured runs, prints one line per run and then "DONE". It is started with
--hold, so it stays alive after "DONE" while the driver reads its peak
working set (Windows PeakWorkingSetSize via psutil). The same metric and tool
are used for all three languages.

Budgets (config.json): a run that produces no result within
time_budget_per_run_s is killed and recorded as a timeout; larger sizes of
that language are then recorded as not attempted. A configuration whose
estimated matrix storage exceeds memory_budget_bytes is not started.

Outputs:
  results/raw_times.csv  one row per run (warm-up and measured)
  results/runs.csv       one row per (language, n) process
Usage: python run_benchmarks.py [--sizes 100 200] [--langs c java]
"""
import argparse
import csv
import queue
import subprocess
import threading
import time

import psutil

from common import (BYTES_PER_ELEMENT, CONFIG, OUTPUTS, RESULTS, command,
                    input_paths, output_path)

LOAD_SLACK_S = 30  # extra time allowed before the first run, for loading inputs


def reader(stream, q):
    for line in stream:
        q.put(line.strip())
    q.put(None)


def run_one(lang, n, warmups, reps, budget):
    a, b = input_paths(n)
    cmd = command(lang) + [str(n), str(a), str(b), str(warmups), str(reps), str(output_path(lang, n)), "--hold"]
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    q = queue.Queue()
    threading.Thread(target=reader, args=(proc.stdout, q), daemon=True).start()
    runs, peak, status = [], None, "ok"
    first = True
    while True:
        try:
            line = q.get(timeout=budget + (LOAD_SLACK_S if first else 0))
        except queue.Empty:
            proc.kill()
            status = f"timeout (> {budget} s per run)"
            break
        first = False
        if line is None:
            status = f"crashed (exit {proc.wait()})"
            break
        if line == "DONE":
            peak = psutil.Process(proc.pid).memory_info().peak_wset
            proc.stdin.write("\n")
            proc.stdin.flush()
            proc.wait()
            break
        phase, idx, ns, chk = line.split(",")
        runs.append({"phase": "warmup" if phase == "W" else "measured", "rep": int(idx),
                     "time_ns": int(ns), "checksum": chk})
        print(f"  {lang:6s} n={n:5d} {runs[-1]['phase']:8s} #{idx}: {int(ns) / 1e9:10.4f} s", flush=True)
    proc.wait()
    return runs, peak, status, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="*", default=CONFIG["sizes"])
    parser.add_argument("--langs", nargs="*", default=CONFIG["languages"])
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    budget = CONFIG["time_budget_per_run_s"]
    mem_budget = CONFIG["memory_budget_bytes"]
    reps = CONFIG["repetitions"]
    stopped = set()

    with open(RESULTS / "raw_times.csv", "w", newline="", encoding="utf-8") as ft, \
         open(RESULTS / "runs.csv", "w", newline="", encoding="utf-8") as fr:
        times = csv.writer(ft)
        times.writerow(["language", "n", "dtype", "threads", "phase", "rep", "time_ns", "checksum"])
        runs_w = csv.writer(fr)
        runs_w.writerow(["language", "n", "status", "warmups", "measured_runs", "peak_working_set_bytes",
                         "estimated_matrix_bytes", "process_wall_s"])
        for n in args.sizes:
            for lang in args.langs:
                warmups = CONFIG["warmups"][lang]
                estimate = 3 * n * n * BYTES_PER_ELEMENT[lang]
                if lang in stopped:
                    runs_w.writerow([lang, n, "not attempted (smaller size timed out)", warmups, 0, "", estimate, ""])
                    continue
                if estimate > mem_budget:
                    runs_w.writerow([lang, n, "not attempted (memory budget)", warmups, 0, "", estimate, ""])
                    continue
                print(f"{lang} n={n}", flush=True)
                runs, peak, status, wall = run_one(lang, n, warmups, reps, budget)
                for r in runs:
                    times.writerow([lang, n, "float64", 1, r["phase"], r["rep"], r["time_ns"], r["checksum"]])
                measured = sum(r["phase"] == "measured" for r in runs)
                runs_w.writerow([lang, n, status, warmups, measured, peak if peak else "", estimate, f"{wall:.2f}"])
                ft.flush()
                fr.flush()
                if status != "ok":
                    print(f"  {lang} n={n}: {status}", flush=True)
                    stopped.add(lang)


if __name__ == "__main__":
    main()
