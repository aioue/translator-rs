#!/usr/bin/env python3
"""Parallel drop report: same output as prep_drop_report.py, LF-only zip reads, worker pool."""

from __future__ import annotations

import argparse
import collections
import pathlib
import signal
import sys
import zipfile
from multiprocessing import Pool
from typing import Any

import fasttext

import prep_data as P
from registers import REGISTER, apply_extra

BATCH_SIZE = 20_000
_MAX_INFLIGHT = 3

_LID = None
_WORKER_ARGS = None
_WANT_S = None
_WANT_T = None


def lf_lines(text: str) -> list[str]:
    """Split on LF only, as prep_data does (byte lines from the zip member).

    str.splitlines() also breaks on U+2028, U+2029, U+0085, VT, FF and FS/GS/RS.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def first_failure(s: str, t: str, a: argparse.Namespace) -> str | None:
    s = P.deescape(s.replace("\t", " "))
    t = P.deescape(t.replace("\t", " "))
    if hasattr(P, "is_unspaced"):
        s = s.replace("\u200b", "") if P.is_unspaced(s) else s
        t = t.replace("\u200b", "") if P.is_unspaced(t) else t
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


def lid_reason(pair: tuple[str, str], a: argparse.Namespace, lid, want_s: str, want_t: str) -> str | None:
    (sl,), (sp,) = lid.predict(pair[0])
    (tl,), (tp,) = lid.predict(pair[1])
    short = (
        a.lid_bypass_tokens > 0
        and len(pair[0].split()) <= a.lid_bypass_tokens
        and len(pair[1].split()) <= a.lid_bypass_tokens
    )
    if not short and not (sl == want_s and tl == want_t and sp >= a.min_lid and tp >= a.min_lid):
        return "lid_wrong_language" if (sl != want_s or tl != want_t) else "lid_low_confidence"
    return None


def classify_pair(
    s: str, t: str, a: argparse.Namespace, lid, want_s: str, want_t: str, register_name: str
) -> str:
    reason = first_failure(s, t, a)
    if reason is None:
        pair = apply_extra(REGISTER[register_name], s.strip(), t.strip())
        if not pair:
            reason = "register_extra"
        else:
            reason = lid_reason(pair, a, lid, want_s, want_t)
    return reason or "KEPT"


def _init_worker(lid_model: str, worker_args: tuple) -> None:
    global _LID, _WORKER_ARGS, _WANT_S, _WANT_T
    _LID = fasttext.load_model(lid_model)
    _WORKER_ARGS = worker_args[0]
    _WANT_S = worker_args[1]
    _WANT_T = worker_args[2]


def _process_batch(task: tuple[str, int, list[tuple[str, str]]]) -> tuple[collections.Counter, list[tuple[int, str, str, str]]]:
    name, start_idx, pairs = task
    a = _WORKER_ARGS
    assert _LID is not None and _WANT_S is not None and _WANT_T is not None
    why: collections.Counter = collections.Counter()
    drops: list[tuple[int, str, str, str]] = []
    for i, (s, t) in enumerate(pairs):
        gidx = start_idx + i
        reason = classify_pair(s, t, a, _LID, _WANT_S, _WANT_T, name)
        why[reason] += 1
        if reason != "KEPT":
            drops.append((gidx, reason, s[:100], t[:100]))
    return why, drops


def read_corpus_lines(zp: pathlib.Path, src: str, lang: str) -> tuple[list[str], list[str]]:
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        S = lf_lines(z.read(next(n for n in names if n.endswith(f".{src}"))).decode("utf-8", "replace"))
        T = lf_lines(z.read(next(n for n in names if n.endswith(f".{lang}"))).decode("utf-8", "replace"))
    return S, T


def process_corpus_sequential(
    name: str,
    S: list[str],
    T: list[str],
    a: argparse.Namespace,
    lid,
    want_s: str,
    want_t: str,
    total: collections.Counter,
    why: dict[str, collections.Counter],
    examples: dict[str, list],
) -> None:
    for s, t in zip(S, T):
        total[name] += 1
        reason = classify_pair(s, t, a, lid, want_s, want_t, name)
        why[name][reason] += 1
        if reason != "KEPT" and len(examples[reason]) < a.examples:
            examples[reason].append((s[:100], t[:100]))


def merge_drops_into_examples(
    examples: dict[str, list],
    drops: list[tuple[int, str, str, str]],
    max_examples: int,
) -> None:
    drops.sort(key=lambda x: x[0])
    for _gidx, reason, s, t in drops:
        if len(examples[reason]) < max_examples:
            examples[reason].append((s, t))


def process_corpus_parallel(
    name: str,
    S: list[str],
    T: list[str],
    pool: Pool,
    a: argparse.Namespace,
    total: collections.Counter,
    why: dict[str, collections.Counter],
    examples: dict[str, list],
) -> None:
    n = min(len(S), len(T))
    total[name] = n
    tasks: list[tuple[str, int, list[tuple[str, str]]]] = []
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        chunk = list(zip(S[start:end], T[start:end]))
        tasks.append((name, start, chunk))

    all_drops: list[tuple[int, str, str, str]] = []
    try:
        for batch_why, batch_drops in pool.imap(_process_batch, tasks, chunksize=1):
            why[name].update(batch_why)
            all_drops.extend(batch_drops)
    except Exception as exc:
        pool.terminate()
        pool.join()
        print(f"worker failed while processing {name}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    merge_drops_into_examples(examples, all_drops, a.examples)


def print_report(
    a: argparse.Namespace,
    total: collections.Counter,
    why: dict[str, collections.Counter],
    examples: dict[str, list],
) -> None:
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


def run(a: argparse.Namespace) -> None:
    total: collections.Counter = collections.Counter()
    why: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    examples: dict[str, list] = collections.defaultdict(list)
    want_s, want_t = f"__label__{a.src}", f"__label__{a.lang}"

    zips = sorted(a.raw.glob(f"*.{a.src}-{a.lang}.zip"))
    pool: Pool | None = None
    lid = None

    def _sigint(_signum, _frame):
        if pool is not None:
            pool.terminate()
            pool.join()
        raise SystemExit(130)

    old = signal.signal(signal.SIGINT, _sigint)

    try:
        if a.workers <= 1:
            lid = fasttext.load_model(a.lid)
            for zp in zips:
                name = zp.name.split(".")[0]
                if name not in REGISTER:
                    print(f"skip {name}: not in the registers.py allowlist")
                    continue
                S, T = read_corpus_lines(zp, a.src, a.lang)
                process_corpus_sequential(name, S, T, a, lid, want_s, want_t, total, why, examples)
        else:
            worker_args = (a, want_s, want_t)
            pool = Pool(a.workers, initializer=_init_worker, initargs=(a.lid, worker_args))
            try:
                for zp in zips:
                    name = zp.name.split(".")[0]
                    if name not in REGISTER:
                        print(f"skip {name}: not in the registers.py allowlist")
                        continue
                    S, T = read_corpus_lines(zp, a.src, a.lang)
                    process_corpus_parallel(name, S, T, pool, a, total, why, examples)
            finally:
                pool.close()
                pool.join()
    finally:
        signal.signal(signal.SIGINT, old)

    print_report(a, total, why, examples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=pathlib.Path, help="directory of OPUS moses zips")
    ap.add_argument("--lang", required=True, help="non-English side, as in prep_data --lang")
    ap.add_argument("--src", default="en")
    ap.add_argument("--lid", default="lid.176.ftz")
    ap.add_argument("--examples", type=int, default=1, help="example rows to print per reason")
    ap.add_argument("--workers", type=int, default=4, help="worker processes (1 = single-process path)")
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
    if a.workers < 1:
        ap.error("--workers must be >= 1")
    run(a)


if __name__ == "__main__":
    main()
