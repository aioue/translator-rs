#!/usr/bin/env python3
"""Check a ratings file against its sheet: same ids in the same order, score 1-3, known flag, four fields.
Usage: check_ratings.py SHEET.tsv RATINGS.tsv   (stdlib only; exits 1 if anything is wrong)"""
import csv, sys, collections

FLAGS = {"", "STRAY_LATIN", "MISALIGNED", "WRONG_TERM", "UNTRANSLATED", "OTHER"}


def rows(path):
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        sys.exit(f"PROBLEM: cannot read {path}: {e}")
    if raw.startswith(b"\xef\xbb\xbf"):
        sys.exit(f"PROBLEM: {path} starts with a byte-order mark; save it as plain UTF-8 without BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        sys.exit(f"PROBLEM: {path} is not valid UTF-8: {e}")
    # blank lines (a stray trailing newline is common) are ignored
    return [r for r in csv.reader(text.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE) if r]


def main(sheet_path, ratings_path):
    sheet, ratings = rows(sheet_path), rows(ratings_path)
    problems = []
    if not ratings or ratings[0] != ["id", "score", "flag", "note"]:
        problems.append("header must be: id, score, flag, note (tab separated)")
    want = [r[0] for r in sheet[1:]]
    got = [r[0] for r in ratings[1:]]
    if got != want:
        missing, extra = set(want) - set(got), set(got) - set(want)
        problems.append(f"ids differ from the sheet: {len(want)} expected, {len(got)} found, "
                        f"{len(missing)} missing, {len(extra)} unexpected, "
                        f"{'same ids in the wrong order' if sorted(got) == sorted(want) else 'order not comparable'}")
        dup = [i for i, c in collections.Counter(got).items() if c > 1]
        if dup:
            problems.append(f"duplicate ids: {dup[:5]}")
    scores = collections.Counter()
    for n, r in enumerate(ratings[1:], 2):
        if len(r) != 4:
            problems.append(f"line {n}: {len(r)} fields, need 4"); continue
        if r[1] not in {"1", "2", "3"}:
            problems.append(f"line {n}: score {r[1]!r} is not 1, 2 or 3")
        else:
            scores[r[1]] += 1
        if r[2] not in FLAGS:
            problems.append(f"line {n}: unknown flag {r[2]!r}")
        if len(r[3].split()) > 15:
            problems.append(f"line {n}: note has {len(r[3].split())} words, limit is 15")
    for p in problems[:30]:
        print("PROBLEM:", p)
    print(f"rows {max(len(ratings) - 1, 0)} of {len(want)}; scores 1:{scores['1']} 2:{scores['2']} 3:{scores['3']}")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
