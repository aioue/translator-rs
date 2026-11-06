#!/usr/bin/env python3
"""Split the teacher draw by the rule screen (kd/flags.tsv): rows with ANY flag go to kd/rejects.rule.tsv (idx, register, flags, km, en);
the rest go to kd/pass.tsv (km \t en, 2 columns, original order) with kd/pass.idx (original 0-based row index) and kd/pass.reg.
Nothing is deleted; every input row lands in exactly one of the two outputs (counts asserted)."""
import collections
W = "/home/km/work/model-a/kd"
op = lambda p, m="r": open(p, m, encoding="utf-8", newline="\n")
n = p = r = 0; keep = collections.Counter(); rej = collections.Counter()
with op(f"{W}/km.txt") as fk, op(f"{W}/en.txt") as fe, op(f"{W}/flags.tsv") as ff, op(f"{W}/pass.tsv", "w") as fo, \
     op(f"{W}/pass.idx", "w") as fi, op(f"{W}/pass.reg", "w") as fg, op(f"{W}/rejects.rule.tsv", "w") as fr:
    for k, e, f in zip(fk, fe, ff):
        k = k.rstrip("\n"); e = e.rstrip("\n"); i, reg, flags, *_ = f.rstrip("\n").split("\t")
        n += 1
        if flags == "-":
            fo.write(f"{k}\t{e}\n"); fi.write(i + "\n"); fg.write(reg + "\n"); p += 1; keep[reg] += 1
        else:
            fr.write(f"{i}\t{reg}\t{flags}\t{k}\t{e}\n"); r += 1; rej[reg] += 1
assert n == p + r
print("rows", n, "pass", p, "rejected", r); print("pass", dict(keep)); print("rej", dict(rej))
