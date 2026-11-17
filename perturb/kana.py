"""かな・ローマ字まわりのユーティリティ。

条件4（transposition）・5（replacement）では「かな→ローマ字→（摂動）→かな」と往復させるため、
摂動を加えない場合に必ず元のかなへ戻る（可逆な）独自のヘボン式ベースの変換表を使う。
romaji100 条件の「見た目のローマ字化」は pykakasi を使う（conditions.py 参照）。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 文字種判定
# ---------------------------------------------------------------------------
_KANJI_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿々〇〻]")
_HIRA_RE = re.compile(r"[ぁ-ゖゝゞ]")
_KATA_RE = re.compile(r"[ァ-ヺヽヾㇰ-ㇿｦ-ﾝ]")
_JP_RE = re.compile(
    r"[ぁ-ゖゝゞァ-ヺーヽヾ㐀-䶿一-鿿豈-﫿々〇]"
)
_KANA_READING_RE = re.compile(r"^[ぁ-ゖァ-ヺーゝゞヽヾ]+$")


def has_kanji(s: str) -> bool:
    return bool(_KANJI_RE.search(s))


def has_japanese(s: str) -> bool:
    return bool(_JP_RE.search(s))


def is_kana_reading(s: str) -> bool:
    """ひらがな・カタカナ・長音記号のみからなる文字列か。"""
    return bool(s) and bool(_KANA_READING_RE.match(s))


def is_ascii_alpha(s: str) -> bool:
    return bool(s) and bool(re.fullmatch(r"[A-Za-z]+", s))


def kata2hira(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0x30A1 <= o <= 0x30F6 or o in (0x30FD, 0x30FE):
            out.append(chr(o - 0x60))
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# 可逆なかな→ローマ字表
# ---------------------------------------------------------------------------
_BASE = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "wyi", "ゑ": "wye", "を": "wo",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "di", "づ": "du", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ゔ": "vu",
    # 小書き文字（単独で現れた場合）
    "ぁ": "xa", "ぃ": "xi", "ぅ": "xu", "ぇ": "xe", "ぉ": "xo",
    "ゃ": "xya", "ゅ": "xyu", "ょ": "xyo", "ゎ": "xwa", "ゕ": "xka", "ゖ": "xke",
    "ー": "-",
}

_DIGRAPH = {
    # 拗音
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho", "しぇ": "she",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho", "ちぇ": "che",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo", "じぇ": "je",
    "ぢゃ": "dya", "ぢゅ": "dyu", "ぢょ": "dyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    # 外来音
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "てぃ": "thi", "でぃ": "dhi", "とぅ": "twu", "どぅ": "dwu",
    "うぃ": "wi", "うぇ": "we", "うぉ": "who",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
    "つぁ": "tsa", "つぃ": "tsi", "つぇ": "tse", "つぉ": "tso",
}

_VOWELS = set("aeiou")


def _romanize_units(hira: str) -> list[str]:
    """ひらがな列を（っ・ん を除いた）ローマ字単位の列に分解する。っ・ん はそのまま残す。"""
    units: list[str] = []
    i = 0
    while i < len(hira):
        two = hira[i : i + 2]
        if two in _DIGRAPH:
            units.append(_DIGRAPH[two])
            i += 2
            continue
        ch = hira[i]
        if ch in ("っ", "ん"):
            units.append(ch)
        elif ch in _BASE:
            units.append(_BASE[ch])
        else:
            units.append(ch)  # かな以外はそのまま
        i += 1
    return units


def kana_to_romaji(kana: str) -> str:
    """かな（ひらがな/カタカナ）→ローマ字。romaji_to_kana と組で可逆になるように設計している。"""
    units = _romanize_units(kata2hira(kana))
    out: list[str] = []
    for idx, u in enumerate(units):
        nxt = units[idx + 1] if idx + 1 < len(units) else ""
        nxt_first = nxt[0] if nxt and nxt not in ("っ", "ん") else ""
        if u == "ん":
            # 母音・y の前の ん は n' と書いて「な行」と区別する
            out.append("n'" if nxt_first and (nxt_first in _VOWELS or nxt_first == "y") else "n")
        elif u == "っ":
            # 子音の前は子音を重ねる。それ以外（母音・n・小書き・末尾など）は xtsu
            if nxt_first.isalpha() and nxt_first not in _VOWELS and nxt_first not in ("n", "x"):
                out.append(nxt_first)
            else:
                out.append("xtsu")
        else:
            out.append(u)
    return "".join(out)


# ---------------------------------------------------------------------------
# ローマ字→かな（パーサ）
# ---------------------------------------------------------------------------
_R2K: dict[str, str] = {}
for _k, _r in _BASE.items():
    _R2K.setdefault(_r, _k)
for _k, _r in _DIGRAPH.items():
    _R2K.setdefault(_r, _k)
# 訓令式・その他の別表記（摂動の結果として現れうるもの）
_EXTRA = {
    "si": "し", "ti": "ち", "tu": "つ", "hu": "ふ", "zi": "じ",
    "sya": "しゃ", "syu": "しゅ", "syo": "しょ", "tya": "ちゃ", "tyu": "ちゅ", "tyo": "ちょ",
    "zya": "じゃ", "zyu": "じゅ", "zyo": "じょ", "jya": "じゃ", "jyu": "じゅ", "jyo": "じょ",
    "cya": "ちゃ", "cyu": "ちゅ", "cyo": "ちょ",
    "ji": "じ", "je": "じぇ", "wu": "う", "ye": "いぇ", "yi": "い",
    "la": "ぁ", "li": "ぃ", "lu": "ぅ", "le": "ぇ", "lo": "ぉ",
    "lya": "ゃ", "lyu": "ゅ", "lyo": "ょ", "xtu": "っ", "ltu": "っ", "ltsu": "っ", "xtsu": "っ",
    "ca": "か", "ci": "し", "cu": "く", "ce": "せ", "co": "こ",
    "qa": "くぁ", "qi": "くぃ", "qe": "くぇ", "qo": "くぉ", "qu": "く",
    "fu": "ふ", "fyu": "ふゅ", "va": "ゔぁ", "vi": "ゔぃ", "ve": "ゔぇ", "vo": "ゔぉ", "vu": "ゔ",
    "kwa": "くゎ", "gwa": "ぐゎ", "tsa": "つぁ",
}
for _r, _k in _EXTRA.items():
    _R2K.setdefault(_r, _k)
_MAXLEN = max(len(r) for r in _R2K)


def romaji_to_kana(romaji: str) -> str:
    """ローマ字→ひらがな。かなにならない文字はアルファベット（記号）のまま残す。"""
    s = romaji
    low = s.lower()
    out: list[str] = []
    i = 0
    n = len(low)
    while i < n:
        c = low[i]
        nx = low[i + 1] if i + 1 < n else ""
        if c == "-":
            out.append("ー")
            i += 1
            continue
        if c == "n":
            if nx == "'":
                out.append("ん")
                i += 2
                continue
            if nx and (nx in _VOWELS or nx == "y"):
                m = _longest_match(low, i)
                if m:
                    out.append(_R2K[m])
                    i += len(m)
                    continue
            out.append("ん")
            i += 1
            continue
        # 促音（子音の重ね、tch）
        if c.isalpha() and c not in _VOWELS and nx == c:
            out.append("っ")
            i += 1
            continue
        if c == "t" and low[i + 1 : i + 3] == "ch":
            out.append("っ")
            i += 1
            continue
        m = _longest_match(low, i)
        if m:
            out.append(_R2K[m])
            i += len(m)
            continue
        out.append(s[i])  # 変換できない文字は元のまま
        i += 1
    return "".join(out)


def _longest_match(low: str, i: int) -> str | None:
    for L in range(min(_MAXLEN, len(low) - i), 0, -1):
        seg = low[i : i + L]
        if seg in _R2K:
            return seg
    return None


# ---------------------------------------------------------------------------
# 文字種変換（v4：kata100 / hankata100 / romaji100 などの「見た目」の変換）
# ---------------------------------------------------------------------------
def hira2kata(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0x3041 <= o <= 0x3096 or o in (0x309D, 0x309E):
            out.append(chr(o + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _build_hankaku() -> dict[str, str]:
    import unicodedata

    table: dict[str, str] = {}
    for cp in range(0xFF66, 0xFF9E):  # ｦ〜ﾝ（小書き・長音 ｰ を含む）
        table[unicodedata.normalize("NFKC", chr(cp))] = chr(cp)
    # 半角に対応する字がない小書き・古いかな → 近い字
    table.update({"ヮ": "ﾜ", "ヵ": "ｶ", "ヶ": "ｹ", "ヰ": "ｲ", "ヱ": "ｴ"})
    return table


_HANKAKU = _build_hankaku()
_HAN_MARK = {"゙": "ﾞ", "゚": "ﾟ"}


def to_hankaku_katakana(s: str) -> str:
    """全角カタカナ → 半角カタカナ（濁点・半濁点は ｶﾞ のように2文字）。カタカナ以外（句読点・記号を含む）はそのまま。"""
    import unicodedata

    out = []
    for ch in s:
        if ch in _HANKAKU:
            out.append(_HANKAKU[ch])
            continue
        d = unicodedata.normalize("NFD", ch)
        if len(d) == 2 and d[0] in _HANKAKU and d[1] in _HAN_MARK:
            out.append(_HANKAKU[d[0]] + _HAN_MARK[d[1]])
        else:
            out.append(ch)
    return "".join(out)


# 表示用のヘボン式（romaji100 / romaji10 用）。transposition 等の可逆表とは別。
#   長音は母音を重ねる（ソーシャル→soosharu、おう→ou）、ん→n、っ→子音の重ね（ち の前は t）
_DISP_OVERRIDE = {
    "ぢ": "ji", "づ": "zu", "を": "o", "ゐ": "i", "ゑ": "e",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo", "ゎ": "wa", "ゕ": "ka", "ゖ": "ke",
    "ぢゃ": "ja", "ぢゅ": "ju", "ぢょ": "jo",
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du", "うぉ": "wo",
    "てゅ": "tyu", "でゅ": "dyu", "ふゅ": "fyu", "いぇ": "ye", "くぁ": "kwa", "ぐぁ": "gwa",
}
_DISP1 = {k: v for k, v in _BASE.items() if k != "ー"}
_DISP1.update({k: v for k, v in _DISP_OVERRIDE.items() if len(k) == 1})
_DISP2 = dict(_DIGRAPH)
_DISP2.update({k: v for k, v in _DISP_OVERRIDE.items() if len(k) == 2})


def kana_to_romaji_display(kana: str, next_kana: str = "") -> str:
    """かな → 表示用ローマ字（小文字・記号なし）。かな以外の文字はそのまま残す。
    next_kana：次の語の読み。語末の「っ」（Sudachi は 行っ/て のように分ける）を次の子音で表すのに使う。"""
    hira = kata2hira(kana)
    units: list[str] = []
    i = 0
    while i < len(hira):
        two = hira[i : i + 2]
        if two in _DISP2:
            units.append(_DISP2[two])
            i += 2
            continue
        ch = hira[i]
        units.append(ch if ch in ("っ", "ん", "ー") else _DISP1.get(ch, ch))
        i += 1
    out: list[str] = []
    for idx, u in enumerate(units):
        nxt = units[idx + 1] if idx + 1 < len(units) else ""
        if not nxt and u == "っ" and next_kana:
            nxt = kana_to_romaji_display(next_kana)
        if u == "ん":
            out.append("n")
        elif u == "っ":
            # 子音の前は子音を重ねる（ch の前は t）。母音の前・語末などは表記しない
            if nxt and nxt not in ("っ", "ん", "ー") and nxt[0].isascii() and nxt[0].isalpha() and nxt[0] not in _VOWELS:
                out.append("t" if nxt.startswith("ch") else nxt[0])
        elif u == "ー":
            prev = "".join(out)
            v = next((c for c in reversed(prev) if c in _VOWELS), "")
            out.append(v)
        else:
            out.append(u)
    return "".join(out)
