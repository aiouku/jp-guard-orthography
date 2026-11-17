"""SudachiPy による形態素解析。内容語（名詞・動詞・形容詞・形状詞）の抽出と読みの取得を行う。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .kana import has_japanese, is_kana_reading, kata2hira

CONTENT_POS = ("名詞", "動詞", "形容詞", "形状詞")  # 形状詞＝形容動詞の語幹（静か・適切 など）

# Sudachi の読みを上書きする語（v4）。「私」は core 辞書の読みが「ワタクシ」になるため一般的な「わたし」にする
READING_OVERRIDES = {("私", "代名詞"): "わたし"}

# Sudachi の入力上限（バイト数）に余裕を持たせた1チャンクあたりの最大文字数
_MAX_CHUNK_CHARS = 4000


@dataclass(frozen=True)
class Token:
    surface: str
    start: int  # 元テキスト中の開始位置（文字オフセット）
    end: int
    pos: tuple[str, ...]
    reading: str  # ひらがなの読み（取得できなければ pykakasi で補完）
    lemma: str = ""  # 辞書形（romaji100_space の分かち書き判定で使用）

    @property
    def pos1(self) -> str:
        return self.pos[0] if self.pos else ""

    @property
    def is_content(self) -> bool:
        """内容語＝名詞・動詞・形容詞・形状詞。日本語文字を含まない語（英数字・記号）は対象外。"""
        return self.pos1 in CONTENT_POS and has_japanese(self.surface)


@lru_cache(maxsize=1)
def _tokenizer():
    from sudachipy import dictionary, tokenizer  # 遅延 import

    d = dictionary.Dictionary(dict="core")
    tok = d.tokenizer() if hasattr(d, "tokenizer") else d.create()
    return tok, tokenizer.Tokenizer.SplitMode.A, tokenizer.Tokenizer.SplitMode.C


@lru_cache(maxsize=1)
def _kakasi():
    import pykakasi

    return pykakasi.kakasi()


def reading_hira_fallback(text: str) -> str:
    """pykakasi で表層からひらがな読みを得る（Sudachi の読みが使えない未知語用）。"""
    return "".join(item["hira"] for item in _kakasi().convert(text))


def _chunks(text: str) -> list[tuple[int, str]]:
    """改行ごと（長すぎる行はさらに句点・固定長）に分割し、(オフセット, 部分文字列) を返す。"""
    pieces: list[tuple[int, str]] = []
    pos = 0
    for line in re.split(r"(?<=\n)", text):
        if not line:
            continue
        if len(line) <= _MAX_CHUNK_CHARS:
            pieces.append((pos, line))
        else:
            sub_pos = pos
            for sent in re.split(r"(?<=[。．！？!?])", line):
                while sent:
                    head, sent = sent[:_MAX_CHUNK_CHARS], sent[_MAX_CHUNK_CHARS:]
                    pieces.append((sub_pos, head))
                    sub_pos += len(head)
        pos += len(line)
    return pieces


def tokenize(text: str) -> list[Token]:
    tok, mode, _ = _tokenizer()
    tokens: list[Token] = []
    for offset, chunk in _chunks(text):
        for m in tok.tokenize(chunk, mode):
            surface = m.surface()
            pos = tuple(m.part_of_speech())
            reading = kata2hira(m.reading_form())
            if not is_kana_reading(reading):
                reading = kata2hira(reading_hira_fallback(surface))
            reading = READING_OVERRIDES.get((surface, pos[0]), reading)
            tokens.append(
                Token(
                    surface=surface,
                    start=offset + m.begin(),
                    end=offset + m.end(),
                    pos=pos,
                    reading=reading,
                    lemma=m.dictionary_form(),
                )
            )
    # 念のため：トークン表層の連結が元テキストと一致することを確認
    assert "".join(t.surface for t in tokens) == text, "tokenization does not reconstruct the text"
    return tokens


def content_indices(tokens: list[Token]) -> list[int]:
    return [i for i, t in enumerate(tokens) if t.is_content]


def word_starts(text: str) -> set[int]:
    """Sudachi の長単位（SplitMode.C）の語の開始位置（文字オフセット）の集合。
    C 単位は A 単位をまとめたものなので、A 単位のトークン境界のうち「語の切れ目」を判定するのに使う
    （例：著作/権・固定/観念 は C では1語 → romaji100_space で間にスペースを入れない）。"""
    tok, _, mode_c = _tokenizer()
    starts: set[int] = set()
    for offset, chunk in _chunks(text):
        for m in tok.tokenize(chunk, mode_c):
            starts.add(offset + m.begin())
    return starts
