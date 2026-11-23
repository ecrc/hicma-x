#!/usr/bin/env python3
"""HiCMA-X with and without OmniCast: statistics and figure of docs/omnicast/final/README.md.

    python3 docs/omnicast/final/make_figure.py        (needs matplotlib)

Inputs (data/): interleaved timing files written by gpu.sh / cpu.sh, lines
    <case> <variant> r=<round>: <t1> <t2> ... | <tile mix> [| temperature]
Each process runs several factorizations; the first is a warm-up and is
dropped, the others are averaged. Ratios are paired by round against the
build without OmniCast (variant "base").
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

# reference categorical palette (fixed order) and text inks, as in round3/make_figures.py
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

LABELS = {
    "dense_fp64": "FP64 dense (no conversions)",
    "k3_hp_band": "kind 3, FP32/FP16 band",
    "k3_fp16acc": "kind 3, band, FP16 accumulation",
    "k3_band": "kind 3, FP32 band",
    "k5_1e-8": "kind 5, adaptive, 1e-8",
    "k5_1e-4": "kind 5, adaptive, 1e-4",
    "k5_adapt": "kind 5, adaptive, 1e-4",
    "k5_sr": "kind 5, 1e-4, stochastic rounding",
    "k5_dtconv2": "kind 5, 1e-4, datatype_convert 2",
    "k8_1e-3": "kind 8, 1e-3 (FP8 on the H100, FP16 on the A100)",
    "k10_1e-3": "kind 10, runtime, 1e-3",
    "runtime": "runtime decisions, 1e-8 (kind 10)",
    "k10_runtime": "runtime decisions, 1e-8 (kind 10)",
}
LINE = re.compile(r"(\S+) (\S+) r=(\d+): ([\d. ]+)\|")


def load(path):
    """{case: {variant: {round: steady-state time}}}, cases in file order"""
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    order = []
    for line in open(path):
        m = LINE.match(line)
        if not m:
            continue
        ts = [float(x) for x in m.group(4).split()]
        if len(ts) < 2:
            continue  # a failed process
        d[m.group(1)][m.group(2)][int(m.group(3))] = st.mean(ts[1:])
        if m.group(1) not in order:
            order.append(m.group(1))
    return d, order


def boot_ci(ratios, n=20000, seed=1):
    rng = random.Random(seed)
    meds = sorted(st.median(rng.choices(ratios, k=len(ratios))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def paired(d, case, variant):
    """median time of base and variant, speedup base/variant (median of per-round ratios), 95 % CI, wins, rounds"""
    b, v = d[case]["base"], d[case].get(variant, {})
    rounds = sorted(set(b) & set(v))
    if not rounds:
        return None
    ratios = [b[r] / v[r] for r in rounds]
    lo, hi = boot_ci(ratios)
    return (st.median(b[r] for r in rounds), st.median(v[r] for r in rounds), st.median(ratios), lo, hi,
            sum(x > 1 for x in ratios), len(rounds))


def table(path, variants):
    d, order = load(path)
    out = [f"| configuration | without OmniCast (s) |" + "".join(f" {v} (s) | speedup [95 % CI] | faster |" for v in variants),
           "|---|---|" + "---|---|---|" * len(variants)]
    for c in order:
        row = f"| {LABELS.get(c, c)} |"
        first = True
        for v in variants:
            p = paired(d, c, v)
            if p is None:
                row += (" – |" if first else "") + " – | – | – |"
                first = False
                continue
            if first:
                row += f" {p[0]:.3f} |"
                first = False
            row += f" {p[1]:.3f} | {p[2]:.3f} [{p[3]:.3f}, {p[4]:.3f}] | {p[5]}/{p[6]} |"
        out.append(row)
    return "\n".join(out), d, order


def panel(ax, d, order, series, title):
    n = len(series)
    h = 0.8 / n
    for s, (variant, label, col) in enumerate(series):
        for i, c in enumerate(order):
            p = paired(d, c, variant)
            if p is None:
                continue
            y = i + (s - (n - 1) / 2) * h
            ax.barh(y, p[2] - 1.0, left=1.0, height=h * 0.85, color=col, label=label if i == 0 else None)
            ax.plot([p[3], p[4]], [y, y], color=INK2, linewidth=0.8)
            # values in a fixed column right of the plot: speedup and rounds where it was faster
            ax.text(1.02, y, f"{p[2]:.3f}  {p[5]}/{p[6]}", va="center", fontsize=7, color=INK2,
                    transform=ax.get_yaxis_transform())
    ax.text(1.02, 1.0, "speedup faster", va="bottom", fontsize=7, color=INK2, transform=ax.transAxes)
    ax.axvline(1.0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS.get(c, c) for c in order])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10)


def main():
    gpu, cpu = os.path.join(DATA, "a100_N45056.txt"), os.path.join(DATA, "cpu_broadwell_N16384.txt")
    h100 = os.path.join(DATA, "h100_N45056.txt")
    th, dh, oh = table(h100, ["nocache", "cache"])
    tg, dg, og = table(gpu, ["nocache", "cache"])
    tc, dc, oc = table(cpu, ["omc"])
    print("H100 (hexane), N=45056, NB=2048; nocache = OmniCast, cache = OmniCast + operand cache\n" + th)
    print("\nA100 (methane), N=45056, NB=2048; nocache = OmniCast, cache = OmniCast + operand cache\n" + tg)
    print("\nCPU (caffeine, Xeon E5-2660 v4, 28 cores), N=16384, NB=512; omc = OmniCast\n" + tc)
    # one x scale for all panels, so that the platforms compare by eye
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 10.2), sharex=True,
                            gridspec_kw={"height_ratios": [len(oh), len(og), len(oc)]})
    series = [("nocache", "OmniCast", BLUE), ("cache", "OmniCast + GPU operand cache", ORANGE)]
    panel(axs[0], dh, oh, series, "H100, N = 45056, NB = 2048 (5 rounds)")
    panel(axs[1], dg, og, series, "A100, N = 45056, NB = 2048 (4 rounds)")
    panel(axs[2], dc, oc, [("omc", "OmniCast", BLUE)], "CPU, 28-core Broadwell, N = 16384, NB = 512 (10 rounds)")
    axs[2].set_xlabel("speedup over the build without OmniCast (whiskers: 95 % interval)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "with_without_omnicast.png"))
    print("\nfigure:", os.path.join(HERE, "figures", "with_without_omnicast.png"))


if __name__ == "__main__":
    main()
