#!/usr/bin/env python3
"""Median, spread and paired statistics of interleaved A/B timing files.

Lines: <case> <base|omc> <rep> <time_s> [...]
"""
import collections
import random
import statistics as st
import sys


def load(path):
    d = collections.defaultdict(lambda: {"base": {}, "omc": {}})
    for line in open(path):
        p = line.split()
        if len(p) >= 4 and p[1] in ("base", "omc"):
            try:
                d[p[0]][p[1]][int(p[2])] = float(p[3])
            except ValueError:
                pass
    return d


def boot_ci(ratios, n=20000, seed=1):
    rng = random.Random(seed)
    meds = sorted(st.median(rng.choices(ratios, k=len(ratios))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


for path in sys.argv[1:]:
    print(path)
    print(f"  {'case':18s} {'n':>2s} {'base med':>8s} {'omc med':>8s} {'change':>7s} "
          f"{'paired med [95% CI]':>24s} {'omc faster':>10s} {'base IQR':>8s} {'omc IQR':>8s}")
    for case, v in load(path).items():
        b, o = v["base"], v["omc"]
        reps = sorted(set(b) & set(o))
        if not reps:
            continue
        bm, om = st.median(b[r] for r in reps), st.median(o[r] for r in reps)
        ratios = [o[r] / b[r] for r in reps]
        pm = st.median(ratios)
        lo, hi = boot_ci(ratios)
        wins = sum(x < 1 for x in ratios)

        def iqr(x):
            q = st.quantiles(x, n=4)
            return 100 * (q[2] - q[0]) / st.median(x)

        print(f"  {case:18s} {len(reps):2d} {bm:8.3f} {om:8.3f} {100*(om/bm-1):+6.1f}% "
              f"{100*(pm-1):+6.1f}% [{100*(lo-1):+5.1f}, {100*(hi-1):+5.1f}]% "
              f"{wins:>4d}/{len(reps):<5d} {iqr([b[r] for r in reps]):7.1f}% {iqr([o[r] for r in reps]):7.1f}%")
