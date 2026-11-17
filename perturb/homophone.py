"""jmdict-simplified (common 版) から「読み→漢字表記（名詞）」の対応を作る。"""
from __future__ import annotations

import io
import json
import re
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

from .kana import has_kanji, kata2hira

GITHUB_LATEST = "https://api.github.com/repos/scriptin/jmdict-simplified/releases/latest"
ASSET_RE = re.compile(r"^jmdict-eng-common-[\d.]+\+\d+\.json\.(zip|tgz)$")
NOUN_POS = {"n"}  # JMdict の品詞タグ。普通名詞のみ


class HomophoneDict:
    """reading(ひらがな) -> [(漢字表記, entry_id), ...]"""

    def __init__(
        self,
        mapping: dict[str, list[tuple[str, str]]],
        version: str = "custom",
        surface_entries: dict[str, set[str]] | None = None,
    ):
        self.mapping = {k: sorted(set(v)) for k, v in mapping.items()}
        self.version = version
        # 表記 -> その表記を持つ entry_id 集合（同じ語の異表記を除外するため）。
        # surface_entries には common 以外の漢字表記も含める（例：非 common の異表記からの置換を防ぐ）
        self._entries_of: dict[str, set[str]] = defaultdict(set)
        for s, eids in (surface_entries or {}).items():
            self._entries_of[s] |= set(eids)
        # 漢字表記 -> 読み集合（検証用）
        self._readings_of: dict[str, set[str]] = defaultdict(set)
        for reading, pairs in self.mapping.items():
            for kanji, eid in pairs:
                self._entries_of[kanji].add(eid)
                self._readings_of[kanji].add(reading)

    def candidates(self, surface: str, reading: str) -> list[str]:
        """同じ読みを持ち、元の語とは別の語（別エントリ）の漢字表記の候補（ソート済み）。

        元の語が漢字を含まない（ひらがな・カタカナのみ）場合は候補なし：
        かな表記だけではどの語か特定できず、「ため→為」「ページ→頁」のように
        同じ語の漢字表記に置き換わってしまうため（v3 で修正）。"""
        if not has_kanji(surface):
            return []
        reading = kata2hira(reading)
        own_entries = self._entries_of.get(surface, set())
        out = {
            kanji
            for kanji, eid in self.mapping.get(reading, [])
            if kanji != surface and eid not in own_entries and has_kanji(kanji)
        }
        return sorted(out)

    def readings_of(self, kanji: str) -> set[str]:
        return self._readings_of.get(kanji, set())

    # ------------------------------------------------------------------
    @classmethod
    def from_jmdict_json(cls, data: dict) -> "HomophoneDict":
        mapping: dict[str, list[tuple[str, str]]] = defaultdict(list)
        surface_entries: dict[str, set[str]] = defaultdict(set)
        for entry in data["words"]:
            for k in entry.get("kanji", []):  # 品詞・common を問わずすべての漢字表記
                surface_entries[k["text"]].add(str(entry["id"]))
            if not any(NOUN_POS & set(s.get("partOfSpeech", [])) for s in entry.get("sense", [])):
                continue
            kanji_forms = [k["text"] for k in entry.get("kanji", []) if k.get("common")]
            if not kanji_forms:
                continue
            for kana in entry.get("kana", []):
                if not kana.get("common"):
                    continue
                applies = kana.get("appliesToKanji", ["*"])
                reading = kata2hira(kana["text"])
                for kj in kanji_forms:
                    if "*" in applies or kj in applies:
                        mapping[reading].append((kj, str(entry["id"])))
        return cls(
            mapping,
            version=str(data.get("version", "unknown")) + "/" + str(data.get("dictDate", "")),
            surface_entries=surface_entries,
        )

    @classmethod
    def load(cls, path: str | Path) -> "HomophoneDict":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_jmdict_json(data)


def download_jmdict_common(dest_dir: str | Path) -> Path:
    """GitHub の最新リリースから jmdict-eng-common の JSON を取得して dest_dir に展開する。
    既に展開済みのファイルがあればそれを使う（再現性のため、初回取得後は固定される）。"""
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("jmdict-eng-common-*.json"))
    if existing:
        return existing[-1]

    rel = requests.get(GITHUB_LATEST, timeout=60, headers={"Accept": "application/vnd.github+json"})
    rel.raise_for_status()
    assets = [a for a in rel.json()["assets"] if ASSET_RE.match(a["name"])]
    if not assets:
        raise RuntimeError("jmdict-eng-common asset not found in the latest jmdict-simplified release")
    asset = sorted(assets, key=lambda a: a["name"].endswith(".zip"), reverse=True)[0]
    print(f"[jmdict] downloading {asset['name']} ...")
    blob = requests.get(asset["browser_download_url"], timeout=300)
    blob.raise_for_status()

    if asset["name"].endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob.content)) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".json"))
            raw = zf.read(name)
    else:
        with tarfile.open(fileobj=io.BytesIO(blob.content), mode="r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith(".json") and m.isfile())
            raw = tf.extractfile(member).read()

    out = dest_dir / asset["name"].replace(".zip", "").replace(".tgz", "")
    out.write_bytes(raw)
    (dest_dir / "SOURCE.txt").write_text(
        f"{asset['name']}\n{asset['browser_download_url']}\nrelease: {rel.json().get('tag_name')}\n", encoding="utf-8"
    )
    return out
