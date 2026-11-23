"""Statistics of an ab.sh log (lines '<cfg> <arm> r<round>: t1 t2 t3 | ...').
Per process the first factorization is dropped and the others averaged; speedups
are the median of the per-round ratios X/Y with a bootstrap 95 % interval of that
median, and the number of rounds in which Y was faster."""
import collections, random, re, statistics as st, sys
t = collections.defaultdict(dict)
order = []
for line in open(sys.argv[1]):
    m = re.match(r'(\S+) (\S+) r(\d+): ([0-9. ]+)\|', line)
    if m:
        v = [float(x) for x in m[4].split()]
        t[(m[1], m[2])][int(m[3])] = sum(v[1:]) / len(v[1:])
        if m[1] not in order: order.append(m[1])
pairs = [p.split('/') for p in (sys.argv[2:] or ['off/on', 'base/on', 'base/off'])]
rng = random.Random(1)
for c in order:
    arms = [a for cc, a in t if cc == c]
    print(f'{c}: ' + '  '.join(f'{a} {st.mean(t[(c, a)].values()):.3f}s' for a in arms))
    for x, y in pairs:
        if (c, x) in t and (c, y) in t:
            rounds = sorted(set(t[(c, x)]) & set(t[(c, y)]))
            rs = [t[(c, x)][r] / t[(c, y)][r] for r in rounds]
            boots = sorted(st.median(rng.choices(rs, k=len(rs))) for _ in range(10000))
            print(f'    {x}/{y}: {st.median(rs):.3f} [{boots[249]:.3f}, {boots[9749]:.3f}]  {y} faster in {sum(r > 1 for r in rs)}/{len(rs)}  rounds ' + ' '.join(f'{r:.3f}' for r in rs))
