#!/usr/bin/env python3
"""Word-segment Khmer lines (EXPERIMENTAL; for alignment only).

Khmer puts spaces between phrases, not words, and align.sh aligns on whitespace:
on 17,966 en-km pairs the median Khmer line had 3 whitespace tokens against 13
English, 36% of Khmer lines were a single token, and fast_align could tie each
English word only to a whole phrase (3 distinct Khmer tokens aligned per line).
Segmented with khmer-nltk (Apache-2.0, CRF) the same lines have a median of 16
tokens and 15 aligned, at ~930 lines/s on one M1 core.

How the segmented alignment reaches training is still open (question Q4 to the
maintainer): Marian maps whitespace words in the TRAINING TEXT to subwords, so
either the student trains on segmented Khmer (and its output needs the
inserted spaces removed) or the alignment is remapped onto the unsegmented text.
Until that is settled, use this to measure and to prototype, not in a run.

    pip install khmer-nltk
    segment_khmer.py < kd_src.km > kd_src.km.seg        # one line in, one line out
    segment_khmer.py --jobs 32 < big.km > big.km.seg

Non-Khmer text in a line passes through as the tokenizer splits it; empty lines
stay empty, so line counts always match.
"""

from __future__ import annotations

import argparse
import logging
import sys
from multiprocessing import Pool

# khmer-nltk logs its model path on import, in every worker; keep stderr clean.
logging.disable(logging.INFO)


def _segment(line: str) -> str:
    from khmernltk import word_tokenize

    return " ".join(tok for tok in word_tokenize(line) if tok.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    lines = [l.rstrip("\n") for l in sys.stdin]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            out = pool.map(_segment, lines, chunksize=1000)
    else:
        out = [_segment(l) for l in lines]
    sys.stdout.write("".join(o + "\n" for o in out))


if __name__ == "__main__":
    main()
