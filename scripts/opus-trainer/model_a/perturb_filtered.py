#!/usr/bin/env python3
"""Number augmentation for Khmer->English rows, restricted to rows where it is safe, then merged back in place (row count and order unchanged).

Why a filter: perturb_numbers.py was written for Georgian signage and prices. Run unfiltered on news-like Khmer pairs (first test, 20k KD rows,
seed km-kd1) it turned "COVID-19" into "COVID-98", "SARS-CoV-2" into "SARS-CoV-6", "HCoV-229E" into "HCoV-660E", "April 5" into "April 0" and a
year "2020" into "940147". The run-1 student's errors are large amounts (330,000 -> "3,3000") and fractions, not identifiers or dates.
A row is eligible only if EVERY digit-bearing English token is a plain amount (digits with optional comma/point groups, optional
currency sign or percent sign, optional trailing punctuation), there is at least one such token, no token looks like a year (1900-2099 written
without a separator), and the English has no month name. Everything else passes through untouched, so identifiers, dates, times, ordinals, years
and versions are never rewritten. Fractions (vulgar fraction characters, a b/c) are NOT covered: perturb_numbers.py has no fraction support.
usage: perturb_filtered.py IN OUT --seed S --share F --report JSON [--perturb /path/perturb_numbers.py]"""
import argparse, json, re, subprocess, sys, tempfile, pathlib
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--seed", required=True)
ap.add_argument("--share", type=float, required=True); ap.add_argument("--report", required=True)
ap.add_argument("--perturb", default=str(pathlib.Path(__file__).with_name("perturb_numbers.py"))); a = ap.parse_args()
DIG = re.compile(r"\S*\d\S*"); PLAIN = re.compile(r"^[\(\[\"'$€£¥]?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?[\)\]\"'.,;:!?]*$|^[\(\[\"'$€£¥]?\d+(?:\.\d+)?%?[\)\]\"'.,;:!?]*$")
YEAR = re.compile(r"^[^\d]*(?:19|20)\d\d[^\d]*$")
MONTH = re.compile(r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", re.I)
rows = [l.rstrip("\n").split("\t") for l in open(a.inp, encoding="utf-8", newline="\n")]
elig = []
for i, r in enumerate(rows):
    toks = DIG.findall(r[1])
    if toks and all(PLAIN.match(t) and not YEAR.match(t) for t in toks) and not MONTH.search(r[1]): elig.append(i)
with tempfile.TemporaryDirectory() as d:
    sub = pathlib.Path(d) / "sub.tsv"; outp = pathlib.Path(d) / "sub.out.tsv"; rep = pathlib.Path(d) / "rep.json"
    sub.write_text("".join("\t".join(rows[i]) + "\n" for i in elig), encoding="utf-8")
    subprocess.run([sys.executable, a.perturb, "--in", str(sub), "--out", str(outp), "--pair", "km-en", "--seed", a.seed, "--mode", "replace",
                    "--share", str(a.share), "--report", str(rep), "--samples", "0"], check=True, stdout=subprocess.DEVNULL)
    new = [l.rstrip("\n").split("\t") for l in open(outp, encoding="utf-8", newline="\n")]
    prep = json.load(open(rep))
assert len(new) == len(elig), (len(new), len(elig))
changed = 0
for i, n in zip(elig, new):
    o = rows[i]
    assert len(n) == len(o) and [len(x.split()) for x in n[:2]] == [len(x.split()) for x in o[:2]], ("token counts moved", i)
    if n != o: changed += 1; rows[i] = n
with open(a.out, "w", encoding="utf-8", newline="\n") as f:
    for r in rows: f.write("\t".join(r) + "\n")
json.dump({"rows": len(rows), "eligible": len(elig), "changed": changed, "share": a.share, "seed": a.seed, "perturb_numbers_report": prep}, open(a.report, "w"), indent=1)
print(f"rows={len(rows)} eligible={len(elig)} ({100*len(elig)/len(rows):.1f}%) changed={changed}")
