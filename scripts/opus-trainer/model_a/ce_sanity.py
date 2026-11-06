#!/usr/bin/env python3
"""CE-filter sanity: does the backward model prefer the teacher's own English over another sentence's English?
usage: ce_sanity.py prep  KM EN OUT_PREFIX [STEP=127]   -> OUT.km OUT.en (sampled rows) and OUT.en.shuf (English rotated by 1013 rows of the sample)
       ce_sanity.py judge SCORES_ALL STEP SCORES_SHUF  -> prints win rate (real score > shuffled score) and means; exit 1 if win rate < 0.9
Score = marian-scorer --normalize (log-prob per token of the Khmer given the English): higher = better."""
import sys
if sys.argv[1] == "prep":
    km, en, out = sys.argv[2:5]; step = int(sys.argv[5]) if len(sys.argv) > 5 else 127
    K = open(km, encoding="utf-8", newline="\n").read().split("\n")[:-1]; E = open(en, encoding="utf-8", newline="\n").read().split("\n")[:-1]
    ix = list(range(0, len(K), step)); k = [K[i] for i in ix]; e = [E[i] for i in ix]; r = 1013 % len(e); es = e[r:] + e[:r]
    for suf, L in (("km", k), ("en", e), ("en.shuf", es)): open(f"{out}.{suf}", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print(len(ix), "sampled")
else:
    allp, step, shuf = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    A = [float(l.split()[0]) for l in open(allp)]; real = A[::step]; S = [float(l.split()[0]) for l in open(shuf)]
    assert len(real) == len(S), (len(real), len(S))
    win = sum(r > s for r, s in zip(real, S)) / len(real)
    print(f"n={len(real)} win_rate={win:.4f} mean_real={sum(real)/len(real):.3f} mean_shuffled={sum(S)/len(S):.3f}")
    sys.exit(0 if win >= 0.9 else 1)
