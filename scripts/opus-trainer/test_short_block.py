"""Khmer short-block v1: spec grid, probe gate, gen guard, proportion checker."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from check_short_block import check_rows
from gen_pairs import BACKENDS, cmd_gen, select_judge_rows
from pair_spec import build_jobs, load_spec, numbers_agree, probe_hit

KM_SPEC = pathlib.Path(__file__).parent / "configs" / "gen_pairs.km.v1.json"
KM_SWITCH = pathlib.Path(__file__).parent / "configs" / "km_short_block.switch.json"


@pytest.fixture(scope="module")
def km_spec():
    return load_spec(KM_SPEC)


def test_km_v1_spec_loads(km_spec):
    assert km_spec.prompt_version == "km-pairs-v1"
    assert km_spec.sha == "8e7046c915b3"


def test_km_v1_job_grid_counts(km_spec):
    j1 = build_jobs(km_spec, 1, {})
    j2 = build_jobs(km_spec, 2, {})
    assert len(j1) == 1279
    assert len(j2) == 2558
    assert sum(j.n for j in j2) == 46044


def test_km_v1_instruction_prohibition_bands(km_spec):
    for key in ("instruction", "prohibition"):
        assert km_spec.forms[key].bands == ("w01_03", "w04_08")


def test_km_v1_form_rows(km_spec):
    for form in km_spec.forms.values():
        assert form.rows == 18


def test_km_v1_menu_categories(km_spec):
    menus = next(r for r in km_spec.registers if r.name == "menus")
    for cat in (
        "menu condiments, dipping sauces and accompaniments",
        "menu seafood and freshwater shrimp dishes",
        "menu fried snacks, spring rolls and street fritters",
        "menu coffee, tea and condensed-milk drinks",
    ):
        assert cat in menus.categories


def test_km_v1_latin_allow_cellcard(km_spec):
    assert "cellcard" in km_spec.latin_allow


def test_probe_hit_doses_daily():
    assert probe_hit(
        "Do not exceed 4 doses daily",
        "ignored",
        ["Do not exceed 4 doses"],
        [],
    )


def test_probe_hit_net_weight():
    assert probe_hit(
        "Net weight 250 g",
        "x",
        ["Net weight 250 g"],
        [],
    )


def test_probe_hit_long_line_low_jaccard_false():
    en = " ".join(f"word{i}" for i in range(12))
    probe = " ".join(f"word{i}" for i in range(12) if i % 3)
    assert not probe_hit(en, "y", [probe], [])


def test_khmer_digits_numbers_agree():
    assert numbers_agree("១២", "12")


def test_km_gen_blocked_when_switch_false(km_spec, tmp_path, monkeypatch):
    called = []

    def boom(*_a, **_k):
        called.append(1)
        raise AssertionError("BACKENDS must not run")

    monkeypatch.setitem(BACKENDS, "codex", boom)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    class Args:
        pass

    args = Args()
    args.spec = KM_SPEC
    args.out = out_dir
    args.model = "x"
    args.effort = "low"
    args.backend = "codex"
    args.workers = 1
    args.rounds = 1
    args.per_cell = 0
    args.registers = ""
    args.forms = ""
    args.bands = ""
    args.limit = 1
    args.shard = "0/1"
    args.shuffle_seed = 0

    with pytest.raises(SystemExit) as exc:
        cmd_gen(args)
    assert exc.value.code == 2
    assert not called


def test_select_judge_rows_seed_stable():
    rows = [
        {"register": "menus", "form": "label", "en": "a"},
        {"register": "signage", "form": "prohibition", "en": "b"},
        {"register": "labels", "form": "numeric", "en": "c"},
        {"register": "labels", "form": "label", "en": "d1"},
        {"register": "ui", "form": "statement", "en": "d2"},
        {"register": "ui", "form": "statement", "en": "d3"},
        {"register": "ui", "form": "statement", "en": "d4"},
        {"register": "ui", "form": "statement", "en": "d5"},
        {"register": "ui", "form": "statement", "en": "d6"},
        {"register": "ui", "form": "statement", "en": "d7"},
    ]
    a = select_judge_rows(rows, 42)
    b = select_judge_rows(rows, 42)
    assert a == b
    for r in rows:
        if r["register"] == "menus" or r["form"] in ("prohibition", "numeric"):
            assert r in a
    assert len(a) == 3 + round(7 * 0.1)


def test_check_short_block_fails_no_currency(tmp_path):
    rows = [
        {"register": "menus", "form": "numeric", "en": f"Item {i} 500g", "km": f"ទំនិញ{i} ៥០០"}
        for i in range(20)
    ]
    report = check_rows(rows)
    assert not report["ok"]
    jl = tmp_path / "pairs.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rep = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "check_short_block.py", str(jl), "--report", str(rep)],
        cwd=pathlib.Path(__file__).parent,
        capture_output=True,
    )
    assert proc.returncode == 1


def test_check_short_block_passes_in_band_fixture(tmp_path):
    rows = []
    khmer_n = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")
    for i in range(12):
        rows.append({
            "register": "menus", "form": "numeric",
            "en": f"Dish ${2 + i % 3}.00",
            "km": f"ម្ហូប {2 + i % 3}.00$",
        })
    for i in range(6):
        amt = str(1000 + i * 500).translate(khmer_n)
        rows.append({
            "register": "menus", "form": "numeric",
            "en": f"Snack {1000 + i * 500} riel",
            "km": f"អាហារ {amt} រៀល",
        })
    for i in range(2):
        rows.append({
            "register": "menus", "form": "numeric",
            "en": f"Combo ${3 + i}.00 / {5000 + i * 500} riel",
            "km": f"កombo {3 + i}.00$ / {5000 + i * 500}រៀល",
        })
    report = check_rows(rows)
    assert report["ok"], report["failures"]
    jl = tmp_path / "pairs.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rep = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, "check_short_block.py", str(jl), "--report", str(rep)],
        cwd=pathlib.Path(__file__).parent,
        capture_output=True,
    )
    assert proc.returncode == 0
