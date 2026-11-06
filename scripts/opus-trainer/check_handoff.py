#!/usr/bin/env python3
"""Validate handoff model responses before `gen_pairs.py import-responses`."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from gen_sft import strip_fences
from pair_spec import load_spec

KHMER = re.compile(r"[\u1780-\u17FF\u19E0-\u19FF]")
FENCE = re.compile(r"```")


def load_manifest(batch_dir: pathlib.Path) -> dict:
    path = batch_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def response_text_errors(raw: str, spec_code: str) -> list[str]:
    errs: list[str] = []
    if FENCE.search(raw):
        errs.append("markdown fences present")
    text = raw.strip()
    if not text:
        errs.append("empty file")
        return errs
    try:
        payload = json.loads(strip_fences(text))
    except json.JSONDecodeError as e:
        errs.append(f"invalid JSON: {e}")
        return errs
    if not isinstance(payload, list):
        errs.append(f"expected JSON array, got {type(payload).__name__}")
        return errs
    if not payload:
        errs.append("zero rows")
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            errs.append(f"row {i}: not an object")
            continue
        km = item.get(spec_code)
        if not isinstance(km, str) or not KHMER.search(km):
            errs.append(f"row {i}: missing Khmer script in {spec_code!r}")
    return errs


def validate_response_file(path: pathlib.Path, spec_code: str,
                           manifest_keys: set[str]) -> tuple[list[str], int | None]:
    key = path.stem
    if key not in manifest_keys:
        return [f"key {key!r} not in manifest"], None
    raw = path.read_text(encoding="utf-8")
    errs = response_text_errors(raw, spec_code)
    row_count: int | None = None
    try:
        payload = json.loads(strip_fences(raw.strip()))
        if isinstance(payload, list):
            row_count = len(payload)
    except json.JSONDecodeError:
        pass
    return errs, row_count


def summarize_batch(batch_dir: pathlib.Path, spec_code: str) -> dict:
    manifest = load_manifest(batch_dir)
    keys = set(manifest.get("keys") or [])
    responses_dir = batch_dir / "responses"
    files = sorted(responses_dir.glob("*.json")) if responses_dir.is_dir() else []
    per_file: dict[str, dict] = {}
    ok = bad = unknown = 0
    for path in files:
        errs, n = validate_response_file(path, spec_code, keys)
        entry = {"rows": n, "errors": errs}
        per_file[path.stem] = entry
        if errs:
            if any("not in manifest" in e for e in errs):
                unknown += 1
            else:
                bad += 1
        else:
            ok += 1
    missing = sorted(keys - {p.stem for p in files})
    return {
        "manifest_keys": len(keys),
        "response_files": len(files),
        "ok": ok,
        "bad": bad,
        "unknown_key": unknown,
        "missing_keys": missing,
        "files": per_file,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", required=True, type=pathlib.Path,
                    help="handoff directory with manifest.json and responses/")
    ap.add_argument("--spec", type=pathlib.Path,
                    help="spec file (default: read code from manifest if present)")
    ap.add_argument("--code", default="",
                    help="target language code column (overrides spec)")
    args = ap.parse_args()
    manifest = load_manifest(args.batch_dir)
    if args.code:
        code = args.code
    elif args.spec:
        code = load_spec(args.spec).code
    elif manifest.get("code"):
        code = str(manifest["code"])
    else:
        sys.exit("pass --spec or --code, or add code to manifest.json")
    report = summarize_batch(args.batch_dir, code)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["bad"] or report["unknown_key"]:
        sys.exit(1)
    if report["missing_keys"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
