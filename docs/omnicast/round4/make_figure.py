#!/usr/bin/env python3
"""--flush_subnormals on three CPUs: statistics and figure of docs/omnicast/round4/REPORT.md.

    python3 docs/omnicast/round4/make_figure.py        (needs matplotlib)

Inputs (data/cpu_<host>.txt): interleaved timing files written by ab.sh, lines
    <case> flush=<0|1> r=<round>: <t1> <t2> <t3> | <tile mix>
Each process runs three factorizations; the first is a warm-up and is
dropped, the other two are averaged (data/tlr_*.txt: one factorization per
process). Speedups are the median of the per-round
ratios t(flush=0) / t(flush=1), with a bootstrap 95 % interval of that median.
"""
import collections
import os
import random
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

HOSTS = [  # file, label, colour (fixed order)
    ("caffeine", "Broadwell (Xeon E5-2660 v4, 28 cores)", BLUE),
    ("methane", "Skylake-SP (Xeon Gold 6140, 16 cores)", ORANGE),
    ("histamine3", "Zen 2 (EPYC 7302, 32 cores)", AQUA),
]
LABELS = {
    "dense_fp64": "FP64 dense",
    "k3_band": "kind 3, FP32 band",
    "k5_1e-8": "kind 5, adaptive, 1e-8",
    "k5_1e-4": "kind 5, adaptive, 1e-4",
    "k5_sr": "kind 5, 1e-4, stoch. rounding",
    "k10_runtime": "runtime decisions",
    "tlr_dp": "TLR, FP64",
    "tlr_mp_adaptive": "TLR, adaptive precision",
}
SHORT = {  # figure tick labels
    "dense_fp64": "FP64\ndense", "k3_band": "kind 3\nFP32 band", "k5_1e-8": "kind 5\n1e-8",
    "k5_1e-4": "kind 5\n1e-4", "k5_sr": "kind 5, 1e-4\nstoch. rounding", "k10_runtime": "runtime\ndecisions",
}
LINE = re.compile(r"(\S+) flush=([01]) r=(\d+): ([\d. ]+)")


def load(path):
    """{case: {flush: {round: steady-state time}}}, cases in file order"""
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    order = []
    for line in open(path):
        m = LINE.match(line)
        if not m:
            continue
        ts = [float(x) for x in m.group(4).split()]
        # TLR processes run one factorization (--nruns needs the dense layout): no warm-up to drop
        d[m.group(1)][int(m.group(2))][int(m.group(3))] = st.mean(ts[1:]) if len(ts) > 1 else ts[0]
        if m.group(1) not in order:
            order.append(m.group(1))
    return d, order


def boot_ci(ratios, n=20000, seed=1):
    rng = random.Random(seed)
    meds = sorted(st.median(rng.choices(ratios, k=len(ratios))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def paired(d, case):
    off, on = d[case][0], d[case][1]
    rounds = sorted(set(off) & set(on))
    ratios = [off[r] / on[r] for r in rounds]
    lo, hi = boot_ci(ratios) if len(ratios) > 1 else (ratios[0], ratios[0])
    return (st.median(off[r] for r in rounds), st.median(on[r] for r in rounds),
            st.median(ratios), lo, hi, sum(x > 1 for x in ratios), len(rounds))


def main():
    results = {}
    for host, label, _ in HOSTS:
        for kind in ("cpu", "tlr"):
            p = os.path.join(DATA, f"{kind}_{host}.txt")
            if not os.path.exists(p):
                continue
            d, order = load(p)
            for case in order:
                results[(host, case)] = paired(d, case)
    print("| CPU | configuration | flush 0 (s) | flush 1 (s) | speedup [95 % CI] | faster in |")
    print("|---|---|---|---|---|---|")
    for host, label, _ in HOSTS:
        for case in LABELS:
            if (host, case) in results:
                off, on, s, lo, hi, w, n = results[(host, case)]
                print(f"| {label.split(' (')[0]} | {LABELS[case]} | {off:.3f} | {on:.3f} | "
                      f"{s:.2f} [{lo:.2f}, {hi:.2f}] | {w}/{n} |")

    cases = [c for c in LABELS if c.startswith(("dense", "k"))]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    width = 0.26
    for i, (host, label, colour) in enumerate(HOSTS):
        xs, ys, errs = [], [], [[], []]
        for j, case in enumerate(cases):
            if (host, case) not in results:
                continue
            _, _, s, lo, hi, _, _ = results[(host, case)]
            xs.append(j + (i - 1) * width)
            ys.append(s)
            errs[0].append(s - lo)
            errs[1].append(hi - s)
        ax.bar(xs, ys, width - 0.03, color=colour, label=label, yerr=errs, capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": INK2})
    ax.axhline(1.0, color=INK2, linewidth=0.8)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([SHORT[c] for c in cases], fontsize=8)
    ax.set_ylabel("speedup from --flush_subnormals 1")
    ax.set_ylim(0, max(3.0, ax.get_ylim()[1]))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("CPU-only factorization, N = 16384, NB = 512: median of per-round ratios, 95 % CI\n(Skylake-SP: 4 of the 6 configurations)",
                 fontsize=9, loc="left")
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "flush_subnormals_cpu.png"))


if __name__ == "__main__":
    main()
