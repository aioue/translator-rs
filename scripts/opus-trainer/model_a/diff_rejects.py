#!/usr/bin/env python3
"""Write the rows that exclude_eval.py removed (IN minus OUT, as a multiset in file order) to REJECTS, so nothing is lost.
usage: diff_rejects.py IN_TSV OUT_TSV REJECTS_TSV   (exclude_eval keeps the relative order of kept rows)"""
import sys
a, b, r = sys.argv[1:4]
n = k = 0
with open(a, encoding="utf-8", newline="\n") as fa, open(b, encoding="utf-8", newline="\n") as fb, open(r, "w", encoding="utf-8", newline="\n") as fr:
    cur = fb.readline()
    for l in fa:
        n += 1
        if l == cur: cur = fb.readline()
        else: fr.write(l); k += 1
    assert cur == "", "OUT has rows not found in order in IN"
print(f"{n} in, {n-k} kept, {k} written to {r}")
