"""guards.py（モデルごとの入力・出力の扱い）の単体テスト。モデルは読み込まない。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guards import (  # noqa: E402
    KINDS,
    messages_for,
    model_slug,
    parse_llamaguard,
    parse_nemotron,
    parse_polyguard,
    parse_qwen3guard,
    parse_shieldstral,
    spec_for,
)


@pytest.mark.parametrize("raw,expected", [
    ("Safety: Safe\nCategories: None", ("Safe", 0, 0, 0)),
    ("Safety: Unsafe\nCategories: Violent", ("Unsafe", 1, 1, 0)),
    ("Safety: Controversial\nCategories: Politically Sensitive Topics", ("Controversial", 1, 0, 0)),
    ("???", (None, None, None, 1)),
])
def test_parse_qwen3guard(raw, expected):
    assert parse_qwen3guard(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("safe", ("Safe", 0, 0, 0)),
    ("\n\nunsafe\nS1", ("Unsafe", 1, 1, 0)),
    ("unsafe\nS2,S10", ("Unsafe", 1, 1, 0)),
    ("I cannot", (None, None, None, 1)),
])
def test_parse_llamaguard(raw, expected):
    assert parse_llamaguard(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ('{"User Safety": "unsafe", "Safety Categories": "Violence"}', ("Unsafe", 1, 1, 0)),
    ('{\n   "User Safety": "safe"\n}', ("Safe", 0, 0, 0)),
    ('{"User Safety": "unsafe", "Safety Categories": "Viol', ("Unsafe", 1, 1, 0)),  # 途中で切れても読める
    ("Output: safe", (None, None, None, 1)),
])
def test_parse_nemotron(raw, expected):
    assert parse_nemotron(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Harmful request: yes\nRequest safety violations: S2\nResponse refusal: None", ("Unsafe", 1, 1, 0)),
    ("Harmful request: no\nResponse refusal: None\nHarmful response: None", ("Safe", 0, 0, 0)),
    ("Response refusal: yes", (None, None, None, 1)),
])
def test_parse_polyguard(raw, expected):
    assert parse_polyguard(raw) == expected


def test_parse_shieldstral_threshold():
    assert parse_shieldstral(0.9) == ("Unsafe", 1, 1, 0)
    assert parse_shieldstral(0.1) == ("Safe", 0, 0, 0)
    assert parse_shieldstral(0.5) == ("Safe", 0, 0, 0)  # 0.5 より大きいときだけ unsafe


@pytest.mark.parametrize("model,kind,dtype", [
    ("Qwen/Qwen3Guard-Gen-0.6B", "qwen3guard", "float32"),
    ("Qwen/Qwen3Guard-Gen-8B", "qwen3guard", "bfloat16"),
    ("meta-llama/Llama-Guard-3-8B", "llamaguard3", "bfloat16"),
    ("ricoh-ai/Llama-Ricoh-SafeGuard-20260520", "ricoh", "bfloat16"),
    ("nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3", "nemotron", "bfloat16"),
    ("ToxicityPrompts/PolyGuard-Ministral", "polyguard", "bfloat16"),
    ("mistralai/Shieldstral-1.0-3B", "shieldstral", "bfloat16"),
])
def test_spec_for(model, kind, dtype):
    s = spec_for(model)
    assert s.kind == kind and s.dtype == dtype


def test_unknown_model_needs_kind():
    with pytest.raises(SystemExit):
        spec_for("someone/unknown-guard")
    assert spec_for("someone/unknown-guard", "llamaguard3").kind == "llamaguard3"


def test_model_slug_keeps_v3_names():
    assert model_slug("Qwen/Qwen3Guard-Gen-0.6B") == "qwen3guard-0.6b"
    assert model_slug("mistralai/Shieldstral-1.0-3B") == "shieldstral-1.0-3b"


@pytest.mark.parametrize("kind", KINDS)
def test_messages_contain_text_once(kind):
    text = "明日の会議に参加します"
    msgs = messages_for(kind, text)
    assert msgs[-1]["role"] == "user"
    assert sum(m["content"].count(text) for m in msgs) == 1
