"""Handoff export/import for gen_pairs (no model backends)."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from check_handoff import load_manifest, response_text_errors, summarize_batch
from gen_pairs import (
    BACKENDS,
    HANDOFF_SYSTEM_MARKER,
    HANDOFF_USER_MARKER,
    cmd_export_prompts,
    cmd_import_responses,
    enforce_km_gen_guard,
    format_gen_messages,
)
from pair_spec import build_jobs, load_spec

KM_SPEC = pathlib.Path(__file__).parent / "configs" / "gen_pairs.km.v1.json"
KA_SPEC = pathlib.Path(__file__).parent / "configs" / "gen_pairs.ka.json"


@pytest.fixture(scope="module")
def km_spec():
    return load_spec(KM_SPEC)


def _grid_args(**overrides):
    class Args:
        registers = ""
        forms = ""
        bands = ""
        per_cell = 0
        limit = 0
        shard = "0/1"
        shuffle_seed = 0
        rounds = 1

    for k, v in overrides.items():
        setattr(Args, k, v)
    return Args()


def test_format_gen_messages_matches_job_prompt(km_spec):
    job = build_jobs(km_spec, 1, {}, registers=["menus"], forms=["label"],
                     bands=["w01_03"])[0]
    text = format_gen_messages(km_spec, job.prompt)
    assert text.startswith(HANDOFF_SYSTEM_MARKER)
    assert HANDOFF_USER_MARKER in text
    assert km_spec.system in text
    assert job.prompt in text


def test_export_prompts_writes_manifest_and_files(km_spec, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gen_pairs.enforce_km_gen_guard",
        lambda *a, **k: None,
    )
    out = tmp_path / "out"
    batch = tmp_path / "batch"
    args = _grid_args()
    args.spec = KM_SPEC
    args.out = out
    args.batch_dir = batch
    args.limit = 1
    cmd_export_prompts(args)
    manifest = load_manifest(batch)
    assert manifest["spec_sha"] == km_spec.sha
    assert manifest["prompt_version"] == km_spec.prompt_version
    assert len(manifest["keys"]) == 1
    key = manifest["keys"][0]
    prompt_path = batch / "prompts" / f"{key}.txt"
    assert prompt_path.is_file()
    assert HANDOFF_SYSTEM_MARKER in prompt_path.read_text(encoding="utf-8")


def test_export_km_guard_blocks_like_gen(km_spec, tmp_path, monkeypatch):
    out = tmp_path / "out"
    batch = tmp_path / "batch"
    args = _grid_args()
    args.spec = KM_SPEC
    args.out = out
    args.batch_dir = batch
    args.rounds = 99
    with pytest.raises(SystemExit) as exc:
        cmd_export_prompts(args)
    assert exc.value.code == 2


def test_import_writes_job_with_provenance(km_spec, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gen_pairs.enforce_km_gen_guard",
        lambda *a, **k: None,
    )
    out = tmp_path / "out"
    batch = tmp_path / "batch"
    args = _grid_args()
    args.spec = KM_SPEC
    args.out = out
    args.batch_dir = batch
    args.limit = 1
    cmd_export_prompts(args)
    key = load_manifest(batch)["keys"][0]
    payload = [{"en": "Exit", "km": "ចេញ"}]
    resp_dir = batch / "responses"
    resp_dir.mkdir()
    (resp_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
    imp = _grid_args()
    imp.spec = KM_SPEC
    imp.out = out
    imp.batch_dir = batch
    imp.model_label = "gemini-3.8-flash"
    cmd_import_responses(imp)
    job_path = out / "jobs" / f"{key}.json"
    assert job_path.is_file()
    doc = json.loads(job_path.read_text(encoding="utf-8"))
    assert doc["source_model"] == "gemini-3.8-flash"
    assert doc["source_backend"] == "handoff"
    assert doc["imported_at"]
    assert doc["backend"] == "handoff"
    assert doc["rows"][0]["km"] == "ចេញ"


def test_import_rejects_unknown_key_and_existing_job(km_spec, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gen_pairs.enforce_km_gen_guard",
        lambda *a, **k: None,
    )
    out = tmp_path / "out"
    batch = tmp_path / "batch"
    args = _grid_args()
    args.spec = KM_SPEC
    args.out = out
    args.batch_dir = batch
    args.limit = 1
    cmd_export_prompts(args)
    key = load_manifest(batch)["keys"][0]
    resp_dir = batch / "responses"
    resp_dir.mkdir()
    good = [{"en": "Open", "km": "បើក"}]
    (resp_dir / f"{key}.json").write_text(json.dumps(good), encoding="utf-8")
    imp = _grid_args()
    imp.spec = KM_SPEC
    imp.out = out
    imp.batch_dir = batch
    imp.model_label = "test-model"
    cmd_import_responses(imp)
    (resp_dir / "not.in.manifest.json").write_text(json.dumps(good), encoding="utf-8")
    cmd_import_responses(imp)


def test_import_does_not_call_backends(km_spec, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gen_pairs.enforce_km_gen_guard",
        lambda *a, **k: None,
    )

    def boom(*_a, **_k):
        raise AssertionError("BACKENDS must not run")

    monkeypatch.setitem(BACKENDS, "codex", boom)
    out = tmp_path / "out"
    batch = tmp_path / "batch"
    args = _grid_args()
    args.spec = KM_SPEC
    args.out = out
    args.batch_dir = batch
    args.limit = 1
    cmd_export_prompts(args)
    key = load_manifest(batch)["keys"][0]
    resp_dir = batch / "responses"
    resp_dir.mkdir()
    (resp_dir / f"{key}.json").write_text(
        json.dumps([{"en": "Stop", "km": "ឈប់"}]), encoding="utf-8")
    imp = _grid_args()
    imp.spec = KM_SPEC
    imp.out = out
    imp.batch_dir = batch
    imp.model_label = "delegate"
    cmd_import_responses(imp)


def test_check_handoff_script_summary(km_spec, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gen_pairs.enforce_km_gen_guard",
        lambda *a, **k: None,
    )
    batch = tmp_path / "batch"
    out = tmp_path / "out"
    args = _grid_args()
    args.spec = KM_SPEC
    args.out = out
    args.batch_dir = batch
    args.limit = 1
    cmd_export_prompts(args)
    key = load_manifest(batch)["keys"][0]
    resp_dir = batch / "responses"
    resp_dir.mkdir()
    (resp_dir / f"{key}.json").write_text(
        json.dumps([{"en": "Help", "km": "ជំនួយ"}]), encoding="utf-8")
    report = summarize_batch(batch, km_spec.code)
    assert report["ok"] == 1
    assert report["bad"] == 0
    proc = subprocess.run(
        [sys.executable, "check_handoff.py", "--batch-dir", str(batch),
         "--spec", str(KM_SPEC)],
        cwd=pathlib.Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0


def test_response_text_errors_fence_and_khmer():
    assert response_text_errors('```json\n[]\n```', "km")
    assert any("fence" in e for e in response_text_errors('```json\n[]\n```', "km"))
    assert any("Khmer" in e for e in response_text_errors(
        json.dumps([{"en": "x", "km": "english only"}]), "km"))


def test_enforce_km_gen_guard_noop_for_ka():
    ka = load_spec(KA_SPEC)
    enforce_km_gen_guard(ka, KA_SPEC, 99)
