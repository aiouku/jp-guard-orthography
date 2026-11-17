"""1. データ抽出と表記揺れ変換（判定はしない）。

出力:
  data/sample_v2.csv / .jsonl     抽出データ（既定は adversarial==False でラベルのある行を全件。元データの全列＋行番号 id）
  data/perturbed_v4.csv / .jsonl  抽出データ × 14条件（バージョンは common.DATA_VERSION）の変換済みデータセット

--n-per-class N を付けると、v1 と同じく harmful・unharmful を N 件ずつ seed 固定で抽出する（試し用）。
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from tqdm import tqdm

from common import (
    DATA_DIR,
    DATASET_NAME,
    DICT_DIR,
    HARMFUL_VALUE,
    PERTURBED_CSV,
    PERTURBED_JSONL,
    SAMPLE_CSV,
    SAMPLE_JSONL,
    SEED,
    UNHARMFUL_VALUE,
    set_seed,
    write_jsonl,
)
from perturb.conditions import CONDITIONS, perturb
from perturb.homophone import HomophoneDict, download_jmdict_common
from perturb.morph import tokenize

LABEL_COLS = ["adversarial", "prompt_harm_label", "response_harm_label", "response_refusal_label", "subcategory"]


def load_source() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME)
    split = "test" if "test" in ds else next(iter(ds))
    df = ds[split].to_pandas()
    df.insert(0, "id", range(len(df)))  # 元データ（split 内）の行番号
    print(f"[source] {DATASET_NAME} split={split} rows={len(df)}")
    print(f"[source] columns: {list(df.columns)}")
    for col in LABEL_COLS:
        if col in df.columns:
            print(f"\n[source] {col} の値と件数:")
            print(df[col].value_counts(dropna=False).to_string())
    if "adversarial" in df.columns:
        print("\n[source] adversarial × prompt_harm_label:")
        print(pd.crosstab(df["adversarial"], df["prompt_harm_label"].fillna("<NA>")).to_string())
    return df


def sample(df: pd.DataFrame, n_per_class: int | None, seed: int) -> pd.DataFrame:
    for col in ("adversarial", "prompt_harm_label", "prompt"):
        if col not in df.columns:
            raise KeyError(f"必要な列 {col} がありません: {list(df.columns)}")
    values = set(df["prompt_harm_label"].dropna().unique())
    if not {HARMFUL_VALUE, UNHARMFUL_VALUE} <= values:
        raise ValueError(f"prompt_harm_label の値が想定と異なります: {values}")

    adv = df["adversarial"].astype(str).str.lower().isin(["true", "1"])
    base = df[~adv]
    harm = base[base["prompt_harm_label"] == HARMFUL_VALUE]
    safe = base[base["prompt_harm_label"] == UNHARMFUL_VALUE]
    print(f"\n[sample] adversarial==False: {len(base)} 件（harmful {len(harm)}, unharmful {len(safe)}, ラベル欠損 {base['prompt_harm_label'].isna().sum()}）")
    if n_per_class is None:  # 全件（元データの並び順）
        out = pd.concat([harm, safe]).sort_values("id").reset_index(drop=True)
    else:
        out = pd.concat(
            [harm.sample(n=n_per_class, random_state=seed), safe.sample(n=n_per_class, random_state=seed)]
        ).reset_index(drop=True)
    out["label"] = (out["prompt_harm_label"] == HARMFUL_VALUE).astype(int)
    print(f"[sample] 抽出: {len(out)} 件（harmful {out['label'].sum()}, unharmful {(out['label'] == 0).sum()}）")
    return out


def build_perturbed(sample_df: pd.DataFrame, hdict: HomophoneDict, seed: int) -> pd.DataFrame:
    rows = []
    for rec in tqdm(sample_df.to_dict(orient="records"), desc="perturb"):
        text = rec["prompt"]
        tokens = tokenize(text)
        for cond in CONDITIONS:
            r = perturb(text, cond, seed=seed, item_id=rec["id"], homophone_dict=hdict, tokens=tokens)
            rows.append(
                {
                    "id": rec["id"],
                    "condition": cond,
                    "text_original": text,
                    "text_perturbed": r.text_perturbed,
                    "n_target": r.n_target,
                    "n_changed": r.n_changed,
                    "changes": json.dumps(r.change_strings, ensure_ascii=False),
                    "prompt_harm_label": rec["prompt_harm_label"],
                    "label": rec["label"],
                    "adversarial": rec["adversarial"],
                    "prompt_original": rec.get("prompt_original"),
                    "subcategory": rec.get("subcategory"),
                    # 以下は検証・目視確認用の補助列
                    "n_content": r.n_content,
                    "changes_detail": json.dumps(r.changes_detail(), ensure_ascii=False),
                }
            )
    out = pd.DataFrame(rows)
    if "subcategory" not in sample_df.columns:
        out = out.drop(columns=["subcategory"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-class", type=int, default=None, help="各クラスの抽出件数（省略時は全件）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--jmdict", type=str, default=None, help="jmdict-eng-common の JSON パス（省略時は data/dict に自動ダウンロード）")
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    src = load_source()
    smp = sample(src, args.n_per_class, args.seed)
    smp.to_csv(SAMPLE_CSV, index=False, encoding="utf-8")
    write_jsonl(smp, SAMPLE_JSONL)
    print(f"[save] {SAMPLE_CSV} / {SAMPLE_JSONL}")

    jm_path = args.jmdict or download_jmdict_common(DICT_DIR)
    hdict = HomophoneDict.load(jm_path)
    print(f"[jmdict] {jm_path}  version={hdict.version}  readings={len(hdict.mapping)}")

    pert = build_perturbed(smp, hdict, args.seed)
    pert.to_csv(PERTURBED_CSV, index=False, encoding="utf-8")
    write_jsonl(pert, PERTURBED_JSONL)
    print(f"[save] {PERTURBED_CSV} / {PERTURBED_JSONL}  ({len(pert)} 行)")

    summ = pert.groupby("condition", sort=False).agg(
        n=("id", "size"),
        mean_n_target=("n_target", "mean"),
        mean_n_changed=("n_changed", "mean"),
        rows_unchanged=("n_changed", lambda s: int((s == 0).sum())),
    )
    print("\n[summary]\n" + summ.round(2).to_string())
    print(f"\n[time] {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
