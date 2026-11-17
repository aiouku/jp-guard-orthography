"""表記揺れ・誤字の14条件（v4）の変換関数。

条件は1つの系列として並べる（「表記揺れ」「誤字」は説明上の区別で、集計は分けない）:
  original
  文全体の文字種変換 : hira100 / kata100 / hankata100 / romaji100（スペースなし）/ romaji100_space
  内容語10%の文字種変換 : hira10 / kata10 / hankata10 / romaji10
  誤字など           : transposition / replacement / homophone / split

すべての変換は (text, condition, seed, item_id) が同じなら同じ結果になる。
乱数は「seed・item_id・condition」から導いた専用の random.Random を使うため、
処理順序や件数（--limit 等）に依存しない。
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Callable

from .homophone import HomophoneDict
from .kana import (
    has_japanese,
    hira2kata,
    is_kana_reading,
    kana_to_romaji,
    kana_to_romaji_display,
    kata2hira,
    romaji_to_kana,
    to_hankaku_katakana,
)
from .morph import Token, content_indices, tokenize, word_starts

# 文全体を変換する条件 → 文字種
FULL_SCRIPT = {
    "hira100": "hira",
    "kata100": "kata",
    "hankata100": "hankata",
    "romaji100": "romaji",
    "romaji100_space": "romaji",
}
# 内容語の10%だけ変換する条件 → 文字種（誤字系と同じ割合で比べるため）
PARTIAL_SCRIPT = {"hira10": "hira", "kata10": "kata", "hankata10": "hankata", "romaji10": "romaji"}
TYPO_CONDITIONS = ["transposition", "replacement", "homophone", "split"]
CONDITIONS = ["original", *FULL_SCRIPT, *PARTIAL_SCRIPT, *TYPO_CONDITIONS]
PARTIAL_CONDITIONS = [*PARTIAL_SCRIPT, *TYPO_CONDITIONS]  # 内容語の10%（最低1語）を変える条件
PARTIAL_RATIO = 0.10
SPLIT_CHAR = "・"
VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
MAX_TRIES = 50


@dataclass
class Change:
    orig: str
    new: str
    pos: str
    reading: str
    start_orig: int
    start_new: int = -1
    changed: bool = True
    info: dict = field(default_factory=dict)


@dataclass
class PerturbResult:
    condition: str
    text_original: str
    text_perturbed: str
    n_content: int
    n_target: int
    n_changed: int
    changes: list[Change]

    @property
    def change_strings(self) -> list[str]:
        return [f"{c.orig}→{c.new.strip()}" for c in self.changes if c.changed and c.info.get("kind") not in UNCOUNTED_KINDS]

    def changes_detail(self) -> list[dict]:
        return [asdict(c) for c in self.changes]


def make_rng(seed: int, item_id, condition: str) -> random.Random:
    h = hashlib.sha256(f"{seed}|{item_id}|{condition}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def n_partial_targets(n_content: int) -> int:
    """内容語の10%（四捨五入、最低1語）。内容語が0なら0。"""
    if n_content == 0:
        return 0
    return max(1, int(n_content * PARTIAL_RATIO + 0.5))


# ---------------------------------------------------------------------------
# 文字種変換（hira / kata / hankata / romaji）
# ---------------------------------------------------------------------------
_PARTICLE_ROMAJI = {"は": "wa", "へ": "e"}  # 助詞の は・へ（を は表で o）
_GREETINGS_WA = ("こんにちは", "こんばんは")


def convert_script(tok: Token, script: str, next_tok: Token | None = None) -> str:
    """語を指定の文字種に変換した表記。
    - かなだけの語（ひらがな・カタカナ）は表層の文字をそのまま変換する（Sudachi の読みの正規化に左右されない）
    - 漢字などを含む語は Sudachi の読みを使う
    - ローマ字は表示用ヘボン式（長音は母音を重ねる）。助詞の は→wa・へ→e、を→o
      next_tok：次の語。語末の「っ」を次の語の子音で表す（行っ/て → it/te → itte）"""
    hira = kata2hira(tok.surface) if is_kana_reading(tok.surface) else tok.reading
    if not hira:
        # 読みが取れない語（例：元データに混じった簡体字「扬」）は消さずにそのまま残す（validate で検出される）
        return tok.surface
    if script == "hira":
        return hira
    if script == "kata":
        return hira2kata(hira)
    if script == "hankata":
        return to_hankaku_katakana(hira2kata(hira))
    if script == "romaji":
        if tok.pos1 == "助詞" and hira in _PARTICLE_ROMAJI:
            return _PARTICLE_ROMAJI[hira]
        if tok.pos1 == "感動詞" and hira in _GREETINGS_WA:
            return kana_to_romaji_display(hira[:-1]) + "wa"
        nxt = ""
        if next_tok is not None and has_japanese(next_tok.surface):
            nxt = kata2hira(next_tok.surface) if is_kana_reading(next_tok.surface) else next_tok.reading
        return kana_to_romaji_display(hira, nxt)
    raise ValueError(script)


def _next(tokens: list[Token], i: int) -> Token | None:
    return tokens[i + 1] if i + 1 < len(tokens) else None


# romaji100 / romaji100_space では句読点も半角に（変換しない語の扱いは kind=punct。n_changed には数えない）
ROMAJI_PUNCT = {"、": ",", "，": ",", "。": ".", "．": ".", "！": "!", "？": "?"}
SPACE_AFTER_PUNCT = (",", ".", "!", "?")
UNCOUNTED_KINDS = ("space", "punct")  # 文字種変換に伴う付随的な変更（n_changed・change_strings に含めない）


# romaji100_space の分かち書き：Sudachi の長単位（C 単位）の切れ目にスペースを入れる。
# ただし次の語は前の語につなげる（例：sakusei shimasu / motozuite / dekimasen / chosakuken）
_NON_WORD_POS = ("補助記号", "記号", "空白")
_COPULA = ("です", "だ")


def _is_word(t: Token) -> bool:
    return t.pos1 not in _NON_WORD_POS and t.surface.strip() != ""


def _space_before(tokens: list[Token], i: int, starts: set[int]) -> bool:
    if i == 0:
        return False
    t, p = tokens[i], tokens[i - 1]
    if t.start not in starts or not _is_word(t) or not _is_word(p):
        return False
    if p.pos1 == "接頭辞" or t.pos1 == "接尾辞":
        return False
    if t.pos1 == "助動詞" and t.lemma not in _COPULA:  # ます・た・ない 等は前の語につなげる。です・だ は離す
        return False
    if t.pos1 == "助詞" and len(t.pos) > 1 and t.pos[1] == "接続助詞" and t.surface in ("て", "で", "ば"):
        return False
    return True


def _convert_punct(tokens: list[Token], changes: dict[int, "Change"]) -> None:
    for i, t in enumerate(tokens):
        if t.surface in ROMAJI_PUNCT and i not in changes:
            changes[i] = Change(t.surface, ROMAJI_PUNCT[t.surface], t.pos1, t.reading, t.start, changed=True,
                                info={"kind": "punct"})


def _add_spaces(text: str, tokens: list[Token], changes: dict[int, "Change"]) -> None:
    """語の前に半角スペースを入れる。変換しない語（英数字など）の前に入れた場合は kind=space として記録する。
    句読点（, . ! ?）の後ろに語が続く場合もスペースを入れる。"""
    starts = word_starts(text)
    for i, t in enumerate(tokens):
        after_punct = (
            i > 0 and _is_word(t)
            and (changes[i - 1].new if i - 1 in changes else tokens[i - 1].surface) in SPACE_AFTER_PUNCT
        )
        if not (after_punct or _space_before(tokens, i, starts)):
            continue
        if i in changes:
            changes[i].new = " " + changes[i].new
            changes[i].changed = True
            changes[i].info["space_before"] = True
        else:
            changes[i] = Change(t.surface, " " + t.surface, t.pos1, t.reading, t.start, changed=True,
                                info={"kind": "space", "space_before": True})


# ---------------------------------------------------------------------------
# テキスト再構成
# ---------------------------------------------------------------------------
def _rebuild(text: str, tokens: list[Token], changes: dict[int, Change]) -> str:
    parts: list[str] = []
    cur = 0
    for i, t in enumerate(tokens):
        if i in changes:
            ch = changes[i]
            ch.start_new = cur
            parts.append(ch.new)
            cur += len(ch.new)
        else:
            parts.append(t.surface)
            cur += len(t.surface)
    return "".join(parts)


# ---------------------------------------------------------------------------
# 単語単位の変換（成功すれば新しい表記、失敗すれば None）
# ---------------------------------------------------------------------------
def _perturb_transposition(tok: Token, rng: random.Random) -> tuple[str, dict] | None:
    rom = kana_to_romaji(tok.reading)
    idxs = [i for i, c in enumerate(rom) if c.isalpha()]
    if len({rom[i] for i in idxs}) < 2:
        return None
    for _ in range(MAX_TRIES):
        i, j = sorted(rng.sample(idxs, 2))
        if rom[i] == rom[j]:
            continue
        lst = list(rom)
        lst[i], lst[j] = lst[j], lst[i]
        pert = "".join(lst)
        new = romaji_to_kana(pert)
        if new != tok.reading and new != tok.surface:
            return new, {"romaji": rom, "romaji_perturbed": pert, "swap": [i, j]}
    return None


def _perturb_replacement(tok: Token, rng: random.Random) -> tuple[str, dict] | None:
    rom = kana_to_romaji(tok.reading)
    idxs = [i for i, c in enumerate(rom) if c.isalpha()]
    if not idxs:
        return None
    for _ in range(MAX_TRIES):
        i = rng.choice(idxs)
        c = rom[i]
        pool = VOWELS if c in VOWELS else CONSONANTS
        r = rng.choice([x for x in pool if x != c])
        pert = rom[:i] + r + rom[i + 1 :]
        new = romaji_to_kana(pert)
        if new != tok.reading and new != tok.surface:
            return new, {"romaji": rom, "romaji_perturbed": pert, "replace": [i, c, r]}
    return None


def _perturb_split(tok: Token, rng: random.Random) -> tuple[str, dict] | None:
    if len(tok.surface) < 2:
        return None
    return SPLIT_CHAR.join(tok.surface), {}


def _make_homophone(hdict: HomophoneDict):
    def _f(tok: Token, rng: random.Random) -> tuple[str, dict] | None:
        if tok.pos1 != "名詞":
            return None
        cands = hdict.candidates(tok.surface, tok.reading)
        if not cands:
            return None
        return rng.choice(cands), {"n_candidates": len(cands)}

    return _f


def _make_script(script: str):
    def _f(tok: Token, rng: random.Random) -> tuple[str, dict] | None:
        new = convert_script(tok, script)
        return (new, {"kind": "content"}) if new != tok.surface else None

    return _f


def _eligible(tok: Token, condition: str, hdict: HomophoneDict | None) -> bool:
    """その語が変換可能か（候補選定の段階で使う）。乱数を消費しない判定のみ。"""
    if condition in PARTIAL_SCRIPT:  # 変換して表記が変わる語（例：hira10 ではひらがなだけの語は対象外）
        return convert_script(tok, PARTIAL_SCRIPT[condition]) != tok.surface
    if condition == "transposition":
        rom = kana_to_romaji(tok.reading)
        return len({c for c in rom if c.isalpha()}) >= 2
    if condition == "replacement":
        return any(c.isalpha() for c in kana_to_romaji(tok.reading))
    if condition == "split":
        return len(tok.surface) >= 2
    if condition == "homophone":
        return tok.pos1 == "名詞" and hdict is not None and bool(hdict.candidates(tok.surface, tok.reading))
    raise ValueError(condition)


# ---------------------------------------------------------------------------
# 公開関数
# ---------------------------------------------------------------------------
def perturb(
    text: str,
    condition: str,
    seed: int = 42,
    item_id=0,
    homophone_dict: HomophoneDict | None = None,
    tokens: list[Token] | None = None,
) -> PerturbResult:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if condition == "homophone" and homophone_dict is None:
        raise ValueError("homophone condition requires homophone_dict")
    tokens = tokens if tokens is not None else tokenize(text)
    content = content_indices(tokens)
    rng = make_rng(seed, item_id, condition)
    changes: dict[int, Change] = {}

    if condition == "original":
        return PerturbResult(condition, text, text, len(content), 0, 0, [])

    if condition in FULL_SCRIPT:
        # 対象：日本語の文字を含むすべての語（内容語・助詞・助動詞・接辞など）
        # → 変換後の文はすべてその文字種になる（英数字・記号・句読点はそのまま）
        script = FULL_SCRIPT[condition]
        content_set = set(content)
        targets = [i for i, t in enumerate(tokens) if has_japanese(t.surface)]
        for i in targets:
            t = tokens[i]
            new = convert_script(t, script, _next(tokens, i))
            kind = "content" if i in content_set else "other"
            changes[i] = Change(t.surface, new, t.pos1, t.reading, t.start, changed=(new != t.surface), info={"kind": kind})
        if script == "romaji":
            _convert_punct(tokens, changes)
        if condition == "romaji100_space":
            _add_spaces(text, tokens, changes)
        n_target = len(targets)
    else:
        if condition in PARTIAL_SCRIPT:
            fn: Callable = _make_script(PARTIAL_SCRIPT[condition])
        else:
            fn = {
                "transposition": _perturb_transposition,
                "replacement": _perturb_replacement,
                "split": _perturb_split,
            }.get(condition) or _make_homophone(homophone_dict)
        k = n_partial_targets(len(content))
        # 変換できる語（例：homophone なら候補のある名詞）の中から選ぶ＝候補なしの語は選び直す
        pool = [i for i in content if _eligible(tokens[i], condition, homophone_dict)]
        selected = sorted(rng.sample(pool, min(k, len(pool))))
        for i in selected:
            t = tokens[i]
            if condition in PARTIAL_SCRIPT:
                new = convert_script(t, PARTIAL_SCRIPT[condition], _next(tokens, i))
                res = (new, {"kind": "content"}) if new != t.surface else None
            else:
                res = fn(t, rng)
            if res is None:
                changes[i] = Change(t.surface, t.surface, t.pos1, t.reading, t.start, changed=False)
            else:
                new, info = res
                changes[i] = Change(t.surface, new, t.pos1, t.reading, t.start, changed=(new != t.surface), info=info)
        n_target = len(selected)

    new_text = _rebuild(text, tokens, changes)
    ordered = [changes[i] for i in sorted(changes)]
    return PerturbResult(
        condition=condition,
        text_original=text,
        text_perturbed=new_text,
        n_content=len(content),
        n_target=n_target,
        n_changed=sum(c.changed and c.info.get("kind") not in UNCOUNTED_KINDS for c in ordered),
        changes=ordered,
    )


def perturb_all(text: str, seed: int = 42, item_id=0, homophone_dict: HomophoneDict | None = None) -> list[PerturbResult]:
    tokens = tokenize(text)
    return [perturb(text, c, seed, item_id, homophone_dict, tokens=tokens) for c in CONDITIONS]
