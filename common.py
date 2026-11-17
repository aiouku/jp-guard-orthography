"""共通設定（パス・seed・入出力ヘルパー）。"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DICT_DIR = DATA_DIR / "dict"

DATASET_NAME = "sbintuitions/WildGuardTestJP"
# 抽出データのバージョン
#   v1: adversarial==False から harmful 100件・unharmful 100件を seed=42 で抽出（データ v1〜v3 で使用）
#   v2: adversarial==False でラベルのある行を全件（harmful 413・unharmful 490 = 903件。データ v4 から）
SAMPLE_VERSION = "v2"
SAMPLE_CSV = DATA_DIR / f"sample_{SAMPLE_VERSION}.csv"
SAMPLE_JSONL = DATA_DIR / f"sample_{SAMPLE_VERSION}.jsonl"
# 変換データのバージョン（変換仕様を変えたら上げる）
#   v1: 内容語＝名詞・動詞・形容詞
#   v2: 内容語に形状詞を追加。hira100/romaji100 は内容語＋漢字を含む全語を変換（文中に漢字が残らない）
#   v3: homophone の対象を「漢字を含む名詞」に限定（v2 では ため→為 など同じ語の漢字化が 299件中80件）
#   v4: 抽出を全件に。文字種変換を14条件に拡張（kata100・hankata100・romaji100_space・10%版を追加、
#       romaji100 は助詞・助動詞・句読点も含めて文全体を変換）。「私」の読みを わたし に
DATA_VERSION = "v4"
PERTURBED_CSV = DATA_DIR / f"perturbed_{DATA_VERSION}.csv"
PERTURBED_JSONL = DATA_DIR / f"perturbed_{DATA_VERSION}.jsonl"

HARMFUL_VALUE = "harmful"
UNHARMFUL_VALUE = "unharmful"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def load_perturbed(path: Path = PERTURBED_JSONL) -> pd.DataFrame:
    """JSONL（型が保たれる）を優先して読む。"""
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype={"id": int})
    return pd.read_csv(path)
