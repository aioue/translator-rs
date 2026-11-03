#!/usr/bin/env python3
"""Attribute every pair prep_data.py drops to the FIRST filter that rejects it.

prep_data.py reports only "kept N/M" per corpus. For a new language that hides
which rule is doing the damage, and the rule is usually a whitespace or Latin
assumption rather than bad data: on en-km the token-ratio filter alone dropped
68% of TED2020 and tico-19 because Khmer does not put spaces between words.
Run this on the raw OPUS zips before trusting a pool, and again after changing a
filter, so retention is measured per filter and per corpus rather than guessed.

It mirrors clean_pair() step by step, then the register extras, then fastText
LID with the same short-line bypass, so the first failing check is the one
prep_data would have dropped the pair on. It does not modify anything.

    prep_drop_report.py --raw <dir of OPUS moses zips> --lang km --lid lid.176.ftz

Zips are named <Corpus>.en-<lang>.zip, as prep_data downloads them (or fetch
them yourself from the URLs prep_data prints).
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import zipfile

import fasttext

import prep_data as P
from registers import REGISTER, apply_extra


def first_failure(s: str, t: str, a: argparse.Namespace) -> str | None:
    s = P.deescape(s.replace("\t", " "))
    t = P.deescape(t.replace("\t", " "))
    if hasattr(P, "is_unspaced"):  # the Khmer branch strips ZWSP on unspaced lines
        s = s.replace("​", "") if P.is_unspaced(s) else s
        t = t.replace("​", "") if P.is_unspaced(t) else t
    if P.WIKI_CODE.search(s) or P.WIKI_CODE.search(t):
        return "wiki_code"
    s, t = P.strip_artifacts(P.fix_wiki(s)), P.strip_artifacts(P.fix_wiki(t))
    if not s or not t or s == t:
        return "empty_or_identical"
    if P.degenerate(s) or P.degenerate(t):
        return "degenerate"
    if P.corrupt_encoding(s) or P.corrupt_encoding(t):
        return "corrupt_encoding"
    if P.mixed_script(s):
        return "mixed_script_src"
    if P.mixed_script(t):
        return "mixed_script_tgt"
    if not P.max_word_ok(s, t, a.max_word_len):
        return "max_word_len"
    wc = getattr(P, "word_count", lambda x: len(x.split()))
    sw, tw = wc(s), wc(t)
    if not (a.min_tokens <= sw <= a.max_tokens):
        return "token_count_src"
    if not (a.min_tokens <= tw <= a.max_tokens):
        return "token_count_tgt"
    if max(sw, tw) / min(sw, tw) > a.ratio:
        return "token_ratio"
    if max(len(s), len(t)) > a.max_chars:
        return "max_chars"
    if not P.alpha_ratio_ok(s, a.word_ratio, a.alpha_ratio):
        return "alpha_ratio_src"
    if not P.alpha_ratio_ok(t, a.word_ratio, a.alpha_ratio):
        return "alpha_ratio_tgt"
    if not P.num_mismatch_ok(s, t, a.num_ratio):
        return "num_mismatch"
    if not P.url_mismatch_ok(s, t):
        return "url_mismatch"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=pathlib.Path, help="directory of OPUS moses zips")
    ap.add_argument("--lang", required=True, help="non-English side, as in prep_data --lang")
    ap.add_argument("--src", default="en")
    ap.add_argument("--lid", default="lid.176.ftz")
    ap.add_argument("--examples", type=int, default=1, help="example rows to print per reason")
    # Defaults match prep_data.py's.
    ap.add_argument("--min-tokens", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--max-word-len", type=int, default=150)
    ap.add_argument("--word-ratio", type=float, default=0.6)
    ap.add_argument("--alpha-ratio", type=float, default=0.4)
    ap.add_argument("--num-ratio", type=float, default=1.0)
    ap.add_argument("--min-lid", type=float, default=0.5)
    ap.add_argument("--lid-bypass-tokens", type=int, default=2)
    a = ap.parse_args()

    lid = fasttext.load_model(a.lid)
    want_s, want_t = f"__label__{a.src}", f"__label__{a.lang}"
    total: collections.Counter = collections.Counter()
    why: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    examples: dict[str, list] = collections.defaultdict(list)

    for zp in sorted(a.raw.glob(f"*.{a.src}-{a.lang}.zip")):
        name = zp.name.split(".")[0]
        if name not in REGISTER:
            print(f"skip {name}: not in the registers.py allowlist")
            continue
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
            S = z.read(next(n for n in names if n.endswith(f".{a.src}"))).decode("utf-8", "replace").splitlines()
            T = z.read(next(n for n in names if n.endswith(f".{a.lang}"))).decode("utf-8", "replace").splitlines()
        for s, t in zip(S, T):
            total[name] += 1
            reason = first_failure(s, t, a)
            if reason is None:
                pair = apply_extra(REGISTER[name], s.strip(), t.strip())
                if not pair:
                    reason = "register_extra"
                else:
                    (sl,), (sp,) = lid.predict(pair[0])
                    (tl,), (tp,) = lid.predict(pair[1])
                    short = (a.lid_bypass_tokens > 0 and len(pair[0].split()) <= a.lid_bypass_tokens
                             and len(pair[1].split()) <= a.lid_bypass_tokens)
                    if not short and not (sl == want_s and tl == want_t and sp >= a.min_lid and tp >= a.min_lid):
                        reason = "lid_wrong_language" if (sl != want_s or tl != want_t) else "lid_low_confidence"
            why[name][reason or "KEPT"] += 1
            if reason and len(examples[reason]) < a.examples:
                examples[reason].append((s[:100], t[:100]))

    overall: collections.Counter = collections.Counter()
    for name in sorted(total):
        print(f"\n{name} ({REGISTER[name]}): {total[name]} raw")
        for r, n in why[name].most_common():
            print(f"  {r:22s} {n:8d}  {100 * n / total[name]:5.1f}%")
        overall.update(why[name])
    grand = sum(total.values())
    if grand:
        print(f"\nALL: {grand} raw")
        for r, n in overall.most_common():
            print(f"  {r:22s} {n:8d}  {100 * n / grand:5.1f}%")
    for r, rows in examples.items():
        print(f"\n[{r}]")
        for s, t in rows:
            print(f"  {a.src}: {s}\n  {a.lang}: {t}")


if __name__ == "__main__":
    main()
