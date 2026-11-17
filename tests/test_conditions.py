"""14条件（v4）の変換関数の単体テスト（無害な例文のみを使用）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb.conditions import (  # noqa: E402
    CONDITIONS,
    FULL_SCRIPT,
    PARTIAL_SCRIPT,
    SPLIT_CHAR,
    UNCOUNTED_KINDS,
    n_partial_targets,
    perturb,
    perturb_all,
)
from perturb.homophone import HomophoneDict  # noqa: E402
from perturb.kana import (  # noqa: E402
    has_japanese,
    has_kanji,
    is_ascii_alpha,
    kana_to_romaji,
    kana_to_romaji_display,
    romaji_to_kana,
    to_hankaku_katakana,
)
from perturb.morph import content_indices, tokenize  # noqa: E402

S1 = "東京駅への行き方を教えてください"
S2 = "明日の会議に参加します"
S3 = "コーヒーを飲んで、新しい本を読みたいです。\n週末は公園で散歩しました。"
S4 = "この方法は簡単で便利です。何か不明な点はありますか？"  # 形状詞・接頭辞・代名詞を含む
SENTENCES = [S1, S2, S3, S4]

# 小さなテスト用の同音異字辞書（本番は jmdict-simplified から構築）
TOY_DICT = HomophoneDict(
    {
        "かいぎ": [("会議", "1"), ("懐疑", "2")],
        "さんか": [("参加", "3"), ("酸化", "4"), ("産科", "5")],
        "えき": [("駅", "6"), ("液", "7"), ("益", "8")],
        "こうえん": [("公園", "9"), ("講演", "10"), ("公演", "11")],
        "てん": [("点", "12"), ("天", "13"), ("店", "14")],
    }
)


def _content_words(text):
    toks = tokenize(text)
    return [toks[i] for i in content_indices(toks)]


# ---------------------------------------------------------------------------
# 形態素解析・かな変換の基礎
# ---------------------------------------------------------------------------
def test_content_words_are_noun_verb_adjective():
    words = _content_words(S1)
    assert [w.surface for w in words][:3] == ["東京", "駅", "行き"]
    assert all(w.pos1 in ("名詞", "動詞", "形容詞", "形状詞") for w in words)
    assert all(w.reading for w in words)


def test_keijoushi_is_content_word():
    """形状詞（形容動詞の語幹：簡単・便利・不明）も内容語に含める（v2）。"""
    surfaces = {w.surface: w.pos1 for w in _content_words(S4)}
    assert surfaces.get("簡単") == "形状詞"
    # 「便利」「不明」は Sudachi では 名詞（普通名詞-形状詞可能）なので v1 でも内容語
    assert surfaces.get("便利") == "名詞"


@pytest.mark.parametrize(
    "kana",
    ["とうきょう", "えき", "かいぎ", "さんか", "こーひー", "きんえん", "ほんや", "こんにちは",
     "がっこう", "ちょっと", "しゃしん", "ふぁいる", "てぃっしゅ", "ゔぁいおりん", "かんい", "っ", "ん"],
)
def test_romaji_roundtrip(kana):
    """摂動なしなら かな→ローマ字→かな で元に戻る（transposition / replacement の前提）。"""
    assert romaji_to_kana(kana_to_romaji(kana)) == kana


def test_romaji_to_kana_keeps_unconvertible_letters():
    assert romaji_to_kana("kgiai") == "kぎあい"
    assert romaji_to_kana("toukyou") == "とうきょう"


def test_n_partial_targets():
    assert n_partial_targets(0) == 0
    assert n_partial_targets(3) == 1  # 最低1語
    assert n_partial_targets(14) == 1
    assert n_partial_targets(15) == 2
    assert n_partial_targets(30) == 3


# ---------------------------------------------------------------------------
# 7条件
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", SENTENCES)
def test_original(text):
    r = perturb(text, "original")
    assert r.text_perturbed == text
    assert r.n_target == 0 and r.n_changed == 0 and r.changes == []


def test_hira100_examples():
    # 「方」は接尾辞だが漢字を含むので変換対象（v2）
    assert perturb(S1, "hira100").text_perturbed == "とうきょうえきへのいきかたをおしえてください"
    assert perturb(S4, "hira100").text_perturbed == "このほうほうはかんたんでべんりです。なんかふめいなてんはありますか？"
    assert perturb(S2, "hira100").text_perturbed == "あすのかいぎにさんかします"
    # カタカナ語もひらがなに。「私」は わたし（v4）
    assert perturb("私はコーヒーを飲む", "hira100").text_perturbed == "わたしはこーひーをのむ"


@pytest.mark.parametrize("text", SENTENCES)
def test_hira100(text):
    r = perturb(text, "hira100")
    assert r.n_target >= len(_content_words(text))  # 日本語を含むすべての語
    for c in r.changes:
        assert not has_kanji(c.new)
    assert not has_kanji(r.text_perturbed)  # 文全体に漢字が残らない
    assert not any("ァ" <= ch <= "ヺ" for ch in r.text_perturbed)  # カタカナも残らない
    assert r.text_perturbed != text


def test_kata100_examples():
    assert perturb(S2, "kata100").text_perturbed == "アスノカイギニサンカシマス"
    assert perturb(S1, "kata100").text_perturbed == "トウキョウエキヘノイキカタヲオシエテクダサイ"


def test_hankata100_examples():
    assert perturb(S2, "hankata100").text_perturbed == "ｱｽﾉｶｲｷﾞﾆｻﾝｶｼﾏｽ"
    # 句読点・記号はそのまま
    assert perturb("コーヒーを飲んで、休む。", "hankata100").text_perturbed == "ｺｰﾋｰｦﾉﾝﾃﾞ、ﾔｽﾑ。"


@pytest.mark.parametrize("kata,han", [("ガギグ", "ｶﾞｷﾞｸﾞ"), ("パピプ", "ﾊﾟﾋﾟﾌﾟ"), ("ヴァイオリン", "ｳﾞｧｲｵﾘﾝ"),
                                      ("ッャュョー", "ｯｬｭｮｰ"), ("ヶ所", "ｹ所"), ("「テスト」。", "「ﾃｽﾄ」。")])
def test_to_hankaku_katakana(kata, han):
    assert to_hankaku_katakana(kata) == han


def test_romaji100_examples():
    # v4：助詞・助動詞も含めて文全体をローマ字に（v3 は asuのkaigiにsankashiます）
    assert perturb(S2, "romaji100").text_perturbed == "asunokaiginisankashimasu"
    assert perturb(S2, "romaji100_space").text_perturbed == "asu no kaigi ni sanka shimasu"
    # 助詞 は→wa・へ→e・を→o、語末の っ（行っ/て）、句読点は半角
    assert perturb("私は東京へ行って、本を読みたいです。", "romaji100_space").text_perturbed == \
        "watashi wa toukyou e itte, hon o yomitai desu."
    assert perturb("私は東京へ行って、本を読みたいです。", "romaji100").text_perturbed == \
        "watashiwatoukyoueitte,honoyomitaidesu."
    # 英数字の語の前後にもスペース。長単位（著作権）の中には入れない
    assert perturb("著作権でPythonを学ぶ", "romaji100_space").text_perturbed == "chosakuken de Python o manabu"


@pytest.mark.parametrize("kana,rom", [("とうきょう", "toukyou"), ("こーひー", "koohii"), ("たとぅー", "tatuu"),
                                      ("まっちゃ", "matcha"), ("がっこう", "gakkou"), ("しんぶん", "shinbun"),
                                      ("ぢ", "ji"), ("づ", "zu"), ("ふぁいる", "fairu"), ("てぃっしゅ", "tisshu")])
def test_romaji_display(kana, rom):
    assert kana_to_romaji_display(kana) == rom


def test_romaji_display_word_final_sokuon():
    assert kana_to_romaji_display("いっ", "て") == "it"
    assert kana_to_romaji_display("まっ", "ちゃ") == "mat"
    assert kana_to_romaji_display("あっ") == "a"  # 次がなければ表記しない


@pytest.mark.parametrize("cond", ["romaji100", "romaji100_space"])
@pytest.mark.parametrize("text", SENTENCES)
def test_romaji100(text, cond):
    r = perturb(text, cond)
    assert r.n_target >= len(_content_words(text))
    assert r.n_target == r.n_changed
    for c in r.changes:
        if c.info.get("kind") not in UNCOUNTED_KINDS:
            assert is_ascii_alpha(c.new.strip()), c
    assert not has_japanese(r.text_perturbed)  # 日本語の文字が1字も残らない


@pytest.mark.parametrize("cond", list(PARTIAL_SCRIPT))
@pytest.mark.parametrize("text", SENTENCES)
def test_partial_script(text, cond):
    """10%版：内容語の10%（最低1語）だけ、その文字種に変える。"""
    r = perturb(text, cond, item_id=1)
    assert r.n_target == n_partial_targets(r.n_content)
    assert r.n_changed == r.n_target >= 1
    for c in r.changes:
        assert c.changed and c.new != c.orig
        if PARTIAL_SCRIPT[cond] == "romaji":
            assert is_ascii_alpha(c.new)
        else:
            assert not has_kanji(c.new)
    full = perturb(text, cond.replace("10", "100"))
    assert r.text_perturbed != full.text_perturbed


def test_partial_script_skips_words_already_in_script():
    # 「これ」はひらがなだけの語なので hira10 の対象外 → 漢字の語が選ばれる
    for item_id in range(10):
        r = perturb("これを食べる", "hira10", item_id=item_id)
        assert [c.orig for c in r.changes] == ["食べる"]


@pytest.mark.parametrize("text", SENTENCES)
def test_transposition(text):
    r = perturb(text, "transposition", item_id=1)
    assert r.n_target == n_partial_targets(r.n_content)
    assert r.n_changed == r.n_target >= 1
    for c in r.changes:
        rom, pert = c.info["romaji"], c.info["romaji_perturbed"]
        i, j = c.info["swap"]
        assert sorted(rom) == sorted(pert) and rom != pert  # 2文字の入れ替え
        assert rom[i] == pert[j] and rom[j] == pert[i]
        assert c.new == romaji_to_kana(pert)
        assert not has_kanji(c.new)  # 漢字への再変換はしない
    assert r.text_perturbed != text


@pytest.mark.parametrize("text", SENTENCES)
def test_replacement(text):
    r = perturb(text, "replacement", item_id=1)
    assert r.n_changed == r.n_target >= 1
    for c in r.changes:
        rom, pert = c.info["romaji"], c.info["romaji_perturbed"]
        pos, old, new = c.info["replace"]
        diff = [k for k in range(len(rom)) if rom[k] != pert[k]]
        assert diff == [pos] and len(rom) == len(pert)  # 1文字だけ置換
        assert (old in "aeiou") == (new in "aeiou")  # 母音↔母音、子音↔子音
        assert c.new == romaji_to_kana(pert)
        assert not has_kanji(c.new)
    assert r.text_perturbed != text


@pytest.mark.parametrize("text", SENTENCES)
def test_homophone(text):
    r = perturb(text, "homophone", item_id=1, homophone_dict=TOY_DICT)
    assert r.n_changed == r.n_target >= 1
    for c in r.changes:
        assert c.pos == "名詞"
        assert c.new != c.orig
        assert c.reading in TOY_DICT.readings_of(c.new)  # 読みが一致
    assert r.text_perturbed != text


def test_homophone_reselects_noun_with_candidates():
    # S1 の名詞のうち候補があるのは「駅」だけ → 必ず駅が選ばれる
    for item_id in range(10):
        r = perturb(S1, "homophone", item_id=item_id, homophone_dict=TOY_DICT)
        assert [c.orig for c in r.changes] == ["駅"]
        assert r.changes[0].new in ("液", "益")


def test_homophone_skips_kana_only_nouns():
    """かなだけの名詞（ため・こと・ページ等）は同じ語の漢字表記になりうるので対象外（v3）。"""
    hd = HomophoneDict({"ため": [("為", "1"), ("溜め", "2")], "ぺーじ": [("頁", "3")]})
    assert hd.candidates("ため", "ため") == []
    assert hd.candidates("ページ", "ぺーじ") == []
    r = perturb("勉強のためのページです", "homophone", homophone_dict=hd)
    assert r.n_changed == 0


def test_homophone_excludes_noncommon_variant_of_same_word():
    hd = HomophoneDict({"ふぞく": [("付属", "1"), ("付則", "2")]}, surface_entries={"附属": {"1"}})
    assert hd.candidates("附属", "ふぞく") == ["付則"]


def test_homophone_no_candidate_leaves_text():
    r = perturb("東京に行きます", "homophone", homophone_dict=TOY_DICT)
    assert r.n_target == 0 and r.n_changed == 0 and r.text_perturbed == "東京に行きます"


def test_homophone_requires_dict():
    with pytest.raises(ValueError):
        perturb(S1, "homophone")


@pytest.mark.parametrize("text", SENTENCES)
def test_split(text):
    r = perturb(text, "split", item_id=1)
    assert r.n_changed == r.n_target >= 1
    for c in r.changes:
        assert c.new == SPLIT_CHAR.join(c.orig)
        assert c.new.replace(SPLIT_CHAR, "") == c.orig
    assert r.text_perturbed.replace(SPLIT_CHAR, "") == text


@pytest.mark.parametrize("cond", CONDITIONS)
def test_offsets_point_to_new_words(cond):
    r = perturb(S3, cond, item_id=3, homophone_dict=TOY_DICT)
    for c in r.changes:
        assert r.text_perturbed[c.start_new : c.start_new + len(c.new)] == c.new
        assert S3[c.start_orig : c.start_orig + len(c.orig)] == c.orig


# ---------------------------------------------------------------------------
# 再現性
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cond", CONDITIONS)
@pytest.mark.parametrize("text", SENTENCES)
def test_same_seed_same_result(cond, text):
    a = perturb(text, cond, seed=42, item_id=7, homophone_dict=TOY_DICT)
    b = perturb(text, cond, seed=42, item_id=7, homophone_dict=TOY_DICT)
    assert a.text_perturbed == b.text_perturbed
    assert a.changes_detail() == b.changes_detail()


def test_result_independent_of_call_order():
    """乱数は (seed, id, condition) ごとに独立 → 他の文・条件の処理順に影響されない。"""
    first = perturb(S3, "replacement", seed=42, item_id=5).text_perturbed
    for t in SENTENCES:
        for c in ("transposition", "split"):
            perturb(t, c, seed=42, item_id=99)
    assert perturb(S3, "replacement", seed=42, item_id=5).text_perturbed == first


def test_different_seed_can_differ():
    outs = {perturb(S3, "transposition", seed=s, item_id=1).text_perturbed for s in range(10)}
    assert len(outs) > 1


def test_perturb_all_matches_individual_calls():
    res = perturb_all(S3, seed=42, item_id=2, homophone_dict=TOY_DICT)
    assert [r.condition for r in res] == CONDITIONS
    for r in res:
        assert r.text_perturbed == perturb(S3, r.condition, 42, 2, TOY_DICT).text_perturbed


# 実辞書（data/dict に取得済みの場合のみ）
JM = sorted((Path(__file__).resolve().parents[1] / "data" / "dict").glob("jmdict-eng-common-*.json"))


@pytest.mark.skipif(not JM, reason="JMdict 未取得（build_dataset.py 実行後にテストされる）")
def test_homophone_with_real_jmdict():
    hd = HomophoneDict.load(JM[-1])
    assert "懐疑" in hd.candidates("会議", "かいぎ")
    r = perturb(S2, "homophone", item_id=1, homophone_dict=hd)
    assert r.n_changed >= 1
    for c in r.changes:
        assert c.reading in hd.readings_of(c.new) and c.new != c.orig
    for w, rd in [("ため", "ため"), ("こと", "こと"), ("ページ", "ぺーじ"), ("イタズラ", "いたずら")]:
        assert hd.candidates(w, rd) == []


def test_unreadable_word_is_kept():
    """読みが取れない語（簡体字など）は消さずに残す。"""
    for cond in [*FULL_SCRIPT, *PARTIAL_SCRIPT]:
        r = perturb("宣扬する", cond, item_id=1)
        assert "扬" in r.text_perturbed, cond
