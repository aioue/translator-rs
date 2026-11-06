#!/usr/bin/env python3
"""Proportion checks on a built Khmer short-block pairs.jsonl (no model calls)."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

KHMER_DIGIT = re.compile(r"[០-៩]")
ASCII_DIGIT = re.compile(r"[0-9]")
ANY_DIGIT = re.compile(r"[0-9០-៩]")
ZWSP = "\u200b"
CURRENCY = re.compile(
    r"[$៛]|រៀល|ដុល្លារ|\bKHR\b|\briel\b",
    re.IGNORECASE,
)
DOLLAR = re.compile(r"[$]|ដុល្លារ|\bUSD\b", re.IGNORECASE)
RIEL = re.compile(r"[៛]|រៀល|\bKHR\b|\briel\b", re.IGNORECASE)
SPELL_COUNT = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
    re.IGNORECASE,
)


def _digit_kind(km: str) -> str | None:
    if not ANY_DIGIT.search(km):
        return None
    has_k = bool(KHMER_DIGIT.search(km))
    has_a = bool(ASCII_DIGIT.search(km))
    if has_k and not has_a:
        return "khmer"
    if has_a and not has_k:
        return "western"
    if has_k and has_a:
        return "mixed"
    return None


def _price_class(km: str, en: str) -> str | None:
    text = f"{km}\t{en}"
    if not CURRENCY.search(text):
        return None
    d = bool(DOLLAR.search(text))
    r = bool(RIEL.search(text))
    if d and r:
        return "both"
    if d:
        return "dollars"
    if r:
        return "riel"
    return "other"


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def _band_ok(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def check_rows(rows: list[dict]) -> dict:
    report: dict = {"bands": {}, "failures": []}

    zwsp = sum(1 for r in rows if ZWSP in r.get("km", ""))
    report["zwsp_in_km"] = zwsp
    if zwsp:
        report["failures"].append("zwsp_in_km")

    by_reg: dict[str, list[dict]] = {}
    for r in rows:
        by_reg.setdefault(r.get("register", ""), []).append(r)

    # menus
    menus = by_reg.get("menus", [])
    menu_digit = [r for r in menus if _digit_kind(r.get("km", ""))]
    if menu_digit:
        western = sum(1 for r in menu_digit if _digit_kind(r["km"]) == "western")
        khmer = sum(1 for r in menu_digit if _digit_kind(r["km"]) == "khmer")
        n = len(menu_digit)
        wp, kp = _pct(western, n), _pct(khmer, n)
        report["bands"]["menus_digit_western_pct"] = wp
        report["bands"]["menus_digit_khmer_pct"] = kp
        if not _band_ok(wp, 60, 85):
            report["failures"].append("menus_western_digits")
        if not _band_ok(kp, 15, 40):
            report["failures"].append("menus_khmer_digits")

    if menus:
        with_cur = sum(1 for r in menus if CURRENCY.search(f"{r.get('km','')}\t{r.get('en','')}"))
        cp = _pct(with_cur, len(menus))
        report["bands"]["menus_currency_pct"] = cp
        if cp < 35:
            report["failures"].append("menus_currency")

    priced = [r for r in menus if _price_class(r.get("km", ""), r.get("en", ""))]
    if priced:
        kinds = [_price_class(r.get("km", ""), r.get("en", "")) for r in priced]
        n = len(priced)
        d_pct = _pct(kinds.count("dollars"), n)
        r_pct = _pct(kinds.count("riel"), n)
        b_pct = _pct(kinds.count("both"), n)
        report["bands"]["menus_priced_dollars_pct"] = d_pct
        report["bands"]["menus_priced_riel_pct"] = r_pct
        report["bands"]["menus_priced_both_pct"] = b_pct
        if not _band_ok(d_pct, 45, 65):
            report["failures"].append("menus_priced_dollars")
        if not _band_ok(r_pct, 25, 45):
            report["failures"].append("menus_priced_riel")
        if not _band_ok(b_pct, 5, 15):
            report["failures"].append("menus_priced_both")

    # labels
    labels = by_reg.get("labels", [])
    label_digit = [r for r in labels if _digit_kind(r.get("km", ""))]
    if label_digit:
        western = sum(1 for r in label_digit if _digit_kind(r["km"]) == "western")
        khmer = sum(1 for r in label_digit if _digit_kind(r["km"]) == "khmer")
        n = len(label_digit)
        wp, kp = _pct(western, n), _pct(khmer, n)
        report["bands"]["labels_digit_western_pct"] = wp
        report["bands"]["labels_digit_khmer_pct"] = kp
        if not _band_ok(wp, 30, 60):
            report["failures"].append("labels_western_digits")
        if not _band_ok(kp, 30, 60):
            report["failures"].append("labels_khmer_digits")

    for reg_name in ("signage", "notices"):
        reg_rows = by_reg.get(reg_name, [])
        digit_rows = [r for r in reg_rows if _digit_kind(r.get("km", ""))]
        if digit_rows:
            khmer = sum(1 for r in digit_rows if _digit_kind(r["km"]) == "khmer")
            kp = _pct(khmer, len(digit_rows))
            report["bands"][f"{reg_name}_digit_khmer_pct"] = kp
            if not _band_ok(kp, 50, 80):
                report["failures"].append(f"{reg_name}_khmer_digits")

    li = [r for r in rows if r.get("form") in ("label", "instruction")]
    if li:
        spelled = sum(
            1 for r in li
            if not ANY_DIGIT.search(r.get("en", "") + r.get("km", ""))
            and SPELL_COUNT.search(r.get("en", ""))
        )
        rate = _pct(spelled, len(li))
        report["bands"]["label_instruction_spelled_pct"] = rate
        if rate > 20:
            report["failures"].append("spelled_count_high")

    report["ok"] = not report["failures"]
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_jsonl", type=pathlib.Path)
    ap.add_argument("--report", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.pairs_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    report = check_rows(rows)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
