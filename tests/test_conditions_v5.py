"""v5（割合スイープ・組み合わせ）の変換関数の単体テスト（無害な例文のみを使用）。
あわせて、v5 を追加しても v4 のデータが変わらないこと（data/perturbed_v4.jsonl との一致）を確かめる。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perturb.conditions import (  # noqa: E402
    CONDITIONS_V4,
    CONDITIONS_V5,
    SPLIT_CHAR,
    SPLIT_CHAR_HANKAKU,
    V5_COMBOS,
    V5_METHODS,
    V5_RATES,
    V5_SINGLE,
    conditions_for,
    n_partial_targets,
    n_targets,
    parse_condition,
    perturb,
    perturb_all,
)
from perturb.homophone import HomophoneDict  # noqa: E402
from perturb.kana import has_kanji, is_ascii_alpha  # noqa: E402
from test_conditions import S1, S2, S3, S4, SENTENCES, TOY_DICT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LONG = S3 + S1 + "。" + S2 + "。" + S4  # 内容語の多い文（割合の違いが出やすい）


def _changed(r):
    return {c.start_orig: c.new for c in r.changes if c.changed}


# ---------------------------------------------------------------------------
# 条件の一覧・条件名
# ---------------------------------------------------------------------------
def test_condition_lists():
    assert len(CONDITIONS_V4) == 14
    assert len(CONDITIONS_V5) == 1 + (len(V5_SINGLE) + len(V5_COMBOS)) * len(V5_RATES) == 66
    assert CONDITIONS_V5[0] == "original"
    assert conditions_for("v4") is CONDITIONS_V4 and conditions_for("v5") is CONDITIONS_V5
    with pytest.raises(ValueError):
        conditions_for("v9")


def test_parse_condition():
    assert parse_condition("hira@25") == ("hira", 25)
    assert parse_condition("romaji+replacement@100") == ("romaji+replacement", 100)
    for bad in ("hira", "hira@0", "hira@101", "foo@10", "hira@x"):
        with pytest.raises(ValueError):
            parse_condition(bad)


@pytest.mark.parametrize("n", range(0, 40))
def test_n_targets_10_matches_v4(n):
    assert n_targets(n, 10) == n_partial_targets(n)


def test_n_targets():
    assert [n_targets(14, r) for r in V5_RATES] == [1, 4, 7, 11, 14]
    assert n_targets(0, 50) == 0 and n_targets(1, 10) == 1


# ---------------------------------------------------------------------------
# 割合：入れ子・単調・同じ語は同じ変換
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", V5_METHODS)
def test_rates_are_nested(method):
    """25% で変えた語は 50% でも同じ表記に変わる（誤字も同じ）。"""
    prev = {}
    for r in V5_RATES:
        res = perturb(LONG, f"{method}@{r}", item_id=3, homophone_dict=TOY_DICT)
        cur = _changed(res)
        assert prev.items() <= cur.items(), (method, r)
        assert res.n_target <= n_targets(res.n_content, r)
        prev = cur


def test_scripts_share_word_order():
    """手法が違っても、同じ割合ならなるべく同じ語を変える（どれも変換できる語なら完全に同じ語）。"""
    pos = {m: set(_changed(perturb(LONG, f"{m}@25", item_id=4))) for m in ("kata", "hankata", "romaji")}
    assert pos["kata"] == pos["hankata"] == pos["romaji"]


def test_homophone_combo_uses_same_kanji_as_homophone():
    a = {c.start_orig: c.new for c in perturb(LONG, "homophone@100", item_id=2, homophone_dict=TOY_DICT).changes
         if c.changed}
    b = {c.start_orig: c.new for c in perturb(LONG, "homophone+hira@100", item_id=2, homophone_dict=TOY_DICT).changes
         if c.changed and c.info.get("method") == "homophone"}
    assert b and b.items() <= a.items()


# ---------------------------------------------------------------------------
# 各手法の中身
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", SENTENCES)
def test_romaji_typos_stay_ascii(text):
    plain = {c.start_orig: c.new for c in perturb(text, "romaji@100", item_id=1).changes}
    for method in ("romaji+replacement", "romaji+transposition"):
        r = perturb(text, f"{method}@100", item_id=1)
        assert r.n_changed >= 1
        for c in r.changes:
            assert is_ascii_alpha(c.new), (method, c.new)
            base = plain[c.start_orig]
            assert len(c.new) == len(base) and c.new != base
            diff = sum(a != b for a, b in zip(c.new, base))
            assert diff == (1 if method.endswith("replacement") else 2)
            if method.endswith("transposition"):
                assert sorted(c.new) == sorted(base)


def test_kata_split():
    r = perturb("コーヒーを飲んで本を読みたい", "kata+split@100", item_id=1)
    assert "コ・ー・ヒ・ー" in r.text_perturbed and "ヨ・ミ" in r.text_perturbed
    for c in r.changes:
        assert c.new.replace(SPLIT_CHAR, "") == perturb(c.orig, "kata@100").text_perturbed


def test_hankata_split_keeps_dakuten():
    r = perturb("散歩してください", "hankata+split@100", item_id=1)
    assert "ｻ･ﾝ･ﾎﾟ" in r.text_perturbed and "ｸ･ﾀﾞ･ｻ･ｲ" in r.text_perturbed
    assert not any(s.startswith(("ﾞ", "ﾟ")) for s in r.text_perturbed.split(SPLIT_CHAR_HANKAKU))


def test_homophone_plus_hira():
    r = perturb(S2, "homophone+hira@100", item_id=1, homophone_dict=TOY_DICT)
    methods = {c.info["method"] for c in r.changes if c.changed}
    assert methods == {"homophone", "hira"}
    for c in r.changes:
        if c.info["method"] == "hira":
            assert not has_kanji(c.new)
        else:
            assert c.new != c.orig and has_kanji(c.new)


def test_homophone_requires_dict():
    with pytest.raises(ValueError):
        perturb(S2, "homophone@25")
    with pytest.raises(ValueError):
        perturb(S2, "homophone+hira@25")


@pytest.mark.parametrize("cond", CONDITIONS_V5)
def test_offsets_and_reproducible(cond):
    a = perturb(LONG, cond, seed=42, item_id=3, homophone_dict=TOY_DICT)
    b = perturb(LONG, cond, seed=42, item_id=3, homophone_dict=TOY_DICT)
    assert a.text_perturbed == b.text_perturbed and a.changes_detail() == b.changes_detail()
    for c in a.changes:
        assert a.text_perturbed[c.start_new: c.start_new + len(c.new)] == c.new
        assert LONG[c.start_orig: c.start_orig + len(c.orig)] == c.orig
        assert c.new.strip() or not c.orig.strip()  # 語を消さない


def test_perturb_all_v5():
    res = perturb_all(S3, seed=42, item_id=2, homophone_dict=TOY_DICT, conditions=CONDITIONS_V5)
    assert [r.condition for r in res] == CONDITIONS_V5


def test_unreadable_word_is_kept():
    for m in ("hira", "kata", "hankata", "romaji", "kata+split", "hankata+split", "romaji+replacement"):
        assert "扬" in perturb("宣扬する", f"{m}@100", item_id=1).text_perturbed, m


def test_unreadable_word_is_not_split():
    """読みが取れずカタカナにできない語は kata+split / hankata+split の対象にしない（区切りだけ入れない）。"""
    for m in ("kata+split", "hankata+split"):
        r = perturb("伪装して", f"{m}@100", item_id=1)
        assert "伪装" in r.text_perturbed, m


# ---------------------------------------------------------------------------
# v4 が変わっていないこと（実データ・実辞書がある場合のみ。先頭 60 件 × 14 条件）
# ---------------------------------------------------------------------------
V4_JSONL = ROOT / "data" / "perturbed_v4.jsonl"
JM = sorted((ROOT / "data" / "dict").glob("jmdict-eng-common-*.json"))


@pytest.mark.skipif(not (V4_JSONL.exists() and JM), reason="data/perturbed_v4.jsonl または JMdict がない")
def test_v4_data_unchanged():
    hd = HomophoneDict.load(JM[-1])
    rows = []
    with open(V4_JSONL, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= 60 * len(CONDITIONS_V4):
                break
    for rec in rows:
        r = perturb(rec["text_original"], rec["condition"], seed=42, item_id=rec["id"], homophone_dict=hd)
        assert r.text_perturbed == rec["text_perturbed"], (rec["id"], rec["condition"])
        assert r.changes_detail() == json.loads(rec["changes_detail"]), (rec["id"], rec["condition"])
