#!/usr/bin/env python3
"""Figures of docs/omnicast/round3/REPORT.md from the raw logs in data/.

    python3 docs/omnicast/round3/make_figures.py      (needs matplotlib)
"""
import collections
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")

# reference categorical palette (fixed order) and text inks
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BASE_GRAY = "#a3a29c"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

LABELS = {
    "k5_1e-8": "kind 5, adaptive, 1e-8 (default)",
    "k5_1e-4": "kind 5, adaptive, 1e-4",
    "k5_sr": "kind 5, 1e-4, stochastic rounding",
    "k4_band": "kind 4, FP32/FP16 band",
    "k8_1e-1": "kind 8, FP8, 1e-1 (120 FP8 tiles)",
    "k8_1e-3": "kind 8, FP8, 1e-3",
    "k9_1e-3": "kind 9, FP8 + FP32, 1e-3",
    "k10_1e-3": "kind 10, runtime FP8, 1e-3",
    "runtime": "runtime decisions (kind 5)",
}


def timings(path):
    """{(config, variant): [steady-state times]}: the first run of each process is a warm-up"""
    d = collections.defaultdict(list)
    for line in open(path):
        m = re.match(r"(\S+) (\S+) r=\d+(?: rc=0)?: (.*)", line)
        if m:
            t = [float(x) for x in m.group(3).split()]
            d[(m.group(1), m.group(2))] += t[1:]
    return d


def speedups(d, cfgs, variant, base="base"):
    out = []
    for c in cfgs:
        b, v = d[(c, base)], d[(c, variant)]
        out.append((st.median(b) / st.median(v), min(b) / max(v), max(b) / min(v)))
    return out


def barh_speedup(ax, cfgs, series, title, labels=None, xlabel="speedup over the baseline (median; whiskers: min-max)"):
    """series: [(label, color, [(median, lo, hi)])]"""
    labels = labels or LABELS
    n = len(series)
    h = 0.8 / n
    for s, (lab, col, vals) in enumerate(series):
        y = [i + (s - (n - 1) / 2) * h for i in range(len(cfgs))]
        med = [v[0] for v in vals]
        ax.barh(y, [m - 1 for m in med], left=1, height=h * 0.9, color=col, label=lab)
        ax.errorbar(med, y, xerr=[[m - v[1] for m, v in zip(med, vals)], [v[2] - m for m, v in zip(med, vals)]],
                    fmt="none", ecolor=INK2, elinewidth=0.7, capsize=2, alpha=0.7)
        for yy, m, v in zip(y, med, vals):
            ax.text(max(m, v[2]) + 0.004, yy, f"{m:.3f}x", va="center", fontsize=7.5, color=INK)
    ax.axvline(1.0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([labels.get(c, c) for c in cfgs])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)


def fig_end_to_end():
    d = timings(os.path.join(DATA, "h100_end_to_end_N45056.txt"))
    cfgs = ["k8_1e-1", "k8_1e-3", "k9_1e-3", "k10_1e-3", "k5_sr", "k4_band", "k5_1e-4", "runtime", "k5_1e-8"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    barh_speedup(ax, cfgs, [
        ("operand cache + OmniCast round 3", BLUE, speedups(d, cfgs, "cache")),
        ("OmniCast round 3 only (--operand_cache 0)", BASE_GRAY, speedups(d, cfgs, "nocache")),
    ], "HiCMA-X Cholesky on one H100, N = 45056, NB = 2048 (6 runs per bar)")
    ax.set_xlim(0.94, 1.13)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2)
    fig.savefig(os.path.join(FIG, "end_to_end_h100.png"))
    plt.close(fig)


def fig_v100():
    path = os.path.join(DATA, "v100_end_to_end_N45056.txt")
    if not os.path.exists(path):
        return
    d = timings(path)
    cfgs = [c for c in ["k5_sr", "k4_band", "k5_1e-4", "runtime", "k5_1e-8"] if (c, "cache") in d]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    barh_speedup(ax, cfgs, [
        ("operand cache + OmniCast round 3", BLUE, speedups(d, cfgs, "cache")),
        ("OmniCast round 3 only (--operand_cache 0)", BASE_GRAY, speedups(d, cfgs, "nocache")),
    ], "HiCMA-X Cholesky on one V100 (no FP8), N = 45056, NB = 2048")
    ax.set_xlim(0.94, 1.10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2)
    fig.savefig(os.path.join(FIG, "end_to_end_v100.png"))
    plt.close(fig)


def kernel_share():
    """{run: {class: ms}} from h100_kernel_share_N32768.txt"""
    runs, cur = {}, None
    for line in open(os.path.join(DATA, "h100_kernel_share_N32768.txt")):
        m = re.match(r"(\S+): GPU kernel time", line)
        if m:
            cur = m.group(1)
            runs[cur] = collections.Counter()
            continue
        m = re.match(r"\s+(.+?)\s+([\d.]+) ms\s+[\d.]+ %\s+(\d+) launches", line)
        if m and cur:
            k = m.group(1)
            k = "conversions (OmniCast)" if k.startswith("omc") else k
            runs[cur][k] += float(m.group(2))
    return runs


def fig_kernel_share():
    runs = kernel_share()
    cases = [("k8_acc1e-1", "kind 8, FP8, 1e-1"), ("k8_acc1e-3", "kind 8, FP8, 1e-3"),
             ("k5_acc1e-4", "kind 5, 1e-4"), ("k5_sr", "kind 5, 1e-4, SR")]
    classes = [("GEMM", BLUE), ("TRSM/SYRK/POTRF", AQUA), ("conversions (OmniCast)", ORANGE), ("other", BASE_GRAY)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ylab, y = [], 0
    for key, lab in cases:
        for tag, name in (("base", "baseline"), ("final", "this round")):
            r = runs[f"{tag}_{key}"]
            left = 0.0
            for cls, col in classes:
                v = r.get(cls, 0.0)
                ax.barh(y, v, left=left, height=0.72, color=col, edgecolor="white", linewidth=1.0,
                        label=cls if y == 0 else None)
                left += v
            conv = r.get("conversions (OmniCast)", 0.0)
            ax.text(left + 4, y, f"{left:.0f} ms, conversions {conv:.0f} ms ({100 * conv / left:.1f} %)",
                    va="center", fontsize=7.5)
            ylab.append(f"{lab} - {name}")
            y += 1
        y += 0.5
    ys = [i + 0.5 * (i // 2) for i in range(len(ylab))]
    ax.set_yticks(ys)
    ax.set_yticklabels(ylab)
    ax.invert_yaxis()
    ax.set_xlim(0, 640)
    ax.set_xlabel("GPU kernel time of one factorization (ms, Nsight Systems)")
    ax.set_title("Where the GPU time goes: H100, N = 32768, NB = 2048", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.15), ncol=4, fontsize=8)
    fig.savefig(os.path.join(FIG, "kernel_share_h100.png"))
    plt.close(fig)


def fig_conversions():
    d = collections.defaultdict(dict)
    for line in open(os.path.join(DATA, "h100_correctness_N16384.txt")):
        m = re.match(r"(\S+) (\S+) rc=\d+ res= \S+ \| Conversions= (\d+)", line)
        if m:
            d[m.group(1)][m.group(2)] = int(m.group(3))
    cases = [("k8_1e-3", "kind 8, 1e-3"), ("k9_1e-3", "kind 9, 1e-3"), ("k10_1e-3", "kind 10, 1e-3"),
             ("k5_bf16", "kind 5, BF16 band"), ("k5_adapt4", "kind 5, 1e-4"), ("k5_sr", "kind 5, SR"),
             ("k4_band", "kind 4 band"), ("runtime", "runtime"), ("k5_tf32", "kind 5, TF32 band"),
             ("k5_adapt", "kind 5, 1e-8")]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = range(len(cases))
    b = [d[c]["base"] for c, _ in cases]
    n = [d[c]["new"] for c, _ in cases]
    ax.bar([i - 0.2 for i in x], b, width=0.38, color=BASE_GRAY, label="baseline: per GEMM")
    ax.bar([i + 0.2 for i in x], n, width=0.38, color=BLUE, label="operand cache")
    for i, (bb, nn) in enumerate(zip(b, n)):
        ax.text(i + 0.2, nn + 15, f"{bb / nn:.1f}x", ha="center", fontsize=7.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([l for _, l in cases], rotation=30, ha="right")
    ax.set_ylabel("data-type conversions")
    ax.set_title("Conversions per factorization, N = 16384, NB = 1024 (all GPU conversions; label: reduction)",
                 loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(FIG, "conversions.png"))
    plt.close(fig)


def fig_fp8_scaling():
    pts = {}
    for n, f in ((32768, "h100_fp8_cache_stage1_N32768.txt"), (45056, "h100_fp8_cache_stage1_N45056.txt")):
        d = parse_stage1(os.path.join(DATA, f))
        for acc in ("1e-1", "1e-3"):
            cache_key = "cache_2k" if n == 32768 else "cache"
            b = d.get((f"acc={acc}", "base"))
            c = d.get((f"acc={acc}", cache_key))
            if b and c:
                pts[(acc, n)] = st.median(b) / st.median(c)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for acc, col, lab in (("1e-1", BLUE, "fixedacc 1e-1 (FP8 everywhere)"), ("1e-3", ORANGE, "fixedacc 1e-3")):
        xs = [n for n in (32768, 45056) if (acc, n) in pts]
        ys = [pts[(acc, n)] for n in xs]
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=2, color=col, label=lab)
        for xx, yy in zip(xs, ys):
            ax.text(xx, yy + 0.004, f"{yy:.3f}x", ha="center", fontsize=7.5)
    ax.set_xticks([32768, 45056])
    ax.set_xlim(30000, 48000)
    ax.set_ylim(0.99, 1.12)
    ax.set_xlabel("N (NB = 2048)")
    ax.set_ylabel("speedup, kind 8")
    ax.set_title("The FP8 operand cache gains more as N grows", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(FIG, "fp8_cache_vs_n.png"))
    plt.close(fig)


# HIP has no FP8 GEMMs: kinds 8 and 9 put those tiles in FP16, kind 10 scales its FP16 operands
LABELS_HIP = {
    "k5_1e-8": "kind 5, adaptive, 1e-8 (default)",
    "k5_1e-4": "kind 5, adaptive, 1e-4 (FP16)",
    "k5_1e-4_bf16": "kind 5, adaptive, 1e-4, BF16 (6x slower here)",
    "k5_sr": "kind 5, 1e-4, stochastic rounding",
    "k8_1e-3": "kind 8, 1e-3 (FP16 in place of FP8)",
    "k9_1e-3": "kind 9, 1e-3 (FP16 in place of FP8)",
    "k10_1e-3": "kind 10, runtime, 1e-3 (scaled FP16)",
    "runtime": "runtime decisions (kind 5)",
}


def fig_radeonvii():
    path = os.path.join(DATA, "radeonvii_end_to_end_N30720.txt")
    if not os.path.exists(path):
        return
    d = timings(path)
    cfgs = ["k5_1e-4", "k5_sr", "k9_1e-3", "k10_1e-3", "k8_1e-3", "k5_1e-8", "k5_1e-4_bf16", "runtime"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    barh_speedup(ax, cfgs, [("operand cache", BLUE, speedups(d, cfgs, "cache", base="nocache"))],
                 "HiCMA-X Cholesky on one Radeon VII (gfx906, HIP), N = 30720, NB = 2048 (6 runs per bar)",
                 labels=LABELS_HIP, xlabel="speedup over --operand_cache 0 (median; whiskers: min-max)")
    ax.set_xlim(0.94, 1.10)
    fig.savefig(os.path.join(FIG, "end_to_end_radeonvii.png"))
    plt.close(fig)


def parse_stage1(path):
    """stage-1 logs: lines '<variant> acc=<acc> r=<r>: times'"""
    d = collections.defaultdict(list)
    for line in open(path):
        m = re.match(r"(\S+) acc=(\S+) r=\d+: (.*)", line)
        if m:
            t = [float(x) for x in m.group(3).split()]
            d[(f"acc={m.group(2)}", m.group(1))] += t[1:]
    return d


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_end_to_end()
    fig_v100()
    fig_radeonvii()
    fig_kernel_share()
    fig_conversions()
    fig_fp8_scaling()
    print("figures in", FIG)


if __name__ == "__main__":
    main()
