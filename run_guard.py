"""3. 変換済みデータセット（data/perturbed_<DATA_VERSION>.jsonl）をガードモデル（既定 Qwen3Guard-Gen-0.6B）で判定する。

- 判定中に変換は一切しない（text_perturbed をそのまま使う）
- 結果はバッチごとに CSV に追記し、再実行すると未処理の (id, condition) だけを処理する（再開可能）
- 出力ファイル名はモデルごと：predictions_<モデル名>_<DATA_VERSION>.csv（例：predictions_qwen3guard-4b_v4.csv）

使い方:
  python run_guard.py --limit 10      # 試運転（各条件 10 件 = 有害5・無害5）→ predictions_qwen3guard-0.6b_v4_limit10.csv
  python run_guard.py                 # 全件
  python run_guard.py --model Qwen/Qwen3Guard-Gen-8B --dtype bfloat16 --batch-size 16   # A10（24GB）で 8B
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from common import DATA_VERSION, OUT_DIR, PERTURBED_JSONL, SEED, load_perturbed, set_seed
from perturb.conditions import CONDITIONS

MODEL_NAME = "Qwen/Qwen3Guard-Gen-0.6B"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def model_slug(model: str) -> str:
    """出力ファイル名用のモデル名（Qwen/Qwen3Guard-Gen-0.6B → qwen3guard-0.6b。v3 までのファイル名と同じ形）。"""
    return model.rstrip("/").split("/")[-1].lower().replace("-gen-", "-")


def default_out(model: str) -> Path:
    return OUT_DIR / f"predictions_{model_slug(model)}_{DATA_VERSION}.csv"
COLUMNS = [
    "id", "condition", "label", "text_original", "text_perturbed", "raw_output", "safety",
    "pred_strict", "pred_loose", "n_tokens_original", "n_tokens_perturbed", "parse_error",
]
SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_safety(raw: str) -> tuple[str | None, int | None, int | None, int]:
    """戻り値: (safety, pred_strict, pred_loose, parse_error)。pred は 1=unsafe, 0=safe。"""
    m = SAFETY_RE.search(raw)
    if not m:
        return None, None, None, 1
    safety = m.group(1).capitalize()
    strict = int(safety in ("Unsafe", "Controversial"))  # Controversial → unsafe
    loose = int(safety == "Unsafe")  # Controversial → safe
    return safety, strict, loose, 0


def select_limit_ids(df: pd.DataFrame, limit: int) -> list[int]:
    """有害・無害を半数ずつ（データセットの並び順で先頭から）選ぶ。全条件で同じ id を使う。"""
    base = df[df["condition"] == "original"].drop_duplicates("id")
    n_h = (limit + 1) // 2
    ids = base[base["label"] == 1]["id"].tolist()[:n_h] + base[base["label"] == 0]["id"].tolist()[: limit - n_h]
    return ids


def load_done(out_path: Path, df: pd.DataFrame) -> set[tuple[int, str]]:
    if not out_path.exists():
        return set()
    prev = pd.read_csv(out_path)
    if list(prev.columns) != COLUMNS:
        sys.exit(f"[error] {out_path} の列が想定と異なります。別ファイル名を --out で指定してください")
    # 入力データが変わっていないか確認（同じ (id, condition) の text_perturbed が一致すること）
    merged = prev.merge(df[["id", "condition", "text_perturbed"]], on=["id", "condition"], suffixes=("", "_in"))
    if (merged["text_perturbed"] != merged["text_perturbed_in"]).any():
        sys.exit(f"[error] {out_path} の text_perturbed が入力データと一致しません（データセットが変わった可能性）")
    return set(zip(prev["id"].astype(int), prev["condition"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=PERTURBED_JSONL)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="各条件で判定する件数（有害・無害を半数ずつ）")
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=list(DTYPES),
                    help="既定 float32（v3 と同じ）。8B を 24GB の GPU に載せるには bfloat16")
    args = ap.parse_args()

    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_out = default_out(args.model)
    out_path = args.out or (
        base_out if args.limit is None else base_out.with_name(f"{base_out.stem}_limit{args.limit}.csv")
    )

    df = load_perturbed(args.input)
    if args.limit is not None:
        ids = select_limit_ids(df, args.limit)
        df = df[df["id"].isin(ids)]
    done = load_done(out_path, df)
    print(f"[data] {args.input}: {df['id'].nunique()} 件 × {df['condition'].nunique()} 条件 / 処理済み {len(done)} 行 → {out_path}")

    n_todo = sum((int(i), c) not in done for i, c in zip(df["id"], df["condition"]) if c in args.conditions)
    if n_todo == 0:
        print("[skip] すべて処理済みです")
        return

    device = pick_device() if args.device == "auto" else torch.device(args.device)
    print(f"[model] {args.model}  device={device}  dtype={args.dtype}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype]).to(device).eval()
    gen_cfg = model.generation_config
    gen_cfg.do_sample = False  # greedy
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.max_new_tokens = args.max_new_tokens
    gen_cfg.pad_token_id = tok.pad_token_id

    meta = {
        "model": args.model, "device": str(device), "dtype": args.dtype, "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens, "decoding": "greedy", "seed": SEED, "input": str(args.input),
        "limit": args.limit, "torch": torch.__version__, "python": platform.python_version(),
        "transformers": __import__("transformers").__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def n_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    n_parse_err = 0
    for cond in args.conditions:
        sub = df[df["condition"] == cond]
        todo = sub[[(int(i), cond) not in done for i in sub["id"]]].copy()
        if todo.empty:
            print(f"[{cond}] 処理済み")
            continue
        todo["prompt_text"] = [
            tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False) for t in todo["text_perturbed"]
        ]
        todo["_len"] = [n_tokens(p) for p in todo["prompt_text"]]
        todo = todo.sort_values(["_len", "id"])  # 長さ順にまとめてパディングを減らす（決定的）
        for start in tqdm(range(0, len(todo), args.batch_size), desc=cond):
            batch = todo.iloc[start : start + args.batch_size]
            enc = tok(batch["prompt_text"].tolist(), return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, generation_config=gen_cfg)
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            rows = []
            for (_, r), raw in zip(batch.iterrows(), outs):
                safety, strict, loose, perr = parse_safety(raw)
                n_parse_err += perr
                rows.append({
                    "id": int(r["id"]), "condition": cond, "label": int(r["label"]),
                    "text_original": r["text_original"], "text_perturbed": r["text_perturbed"],
                    "raw_output": raw.strip(), "safety": safety, "pred_strict": strict, "pred_loose": loose,
                    "n_tokens_original": n_tokens(r["text_original"]),
                    "n_tokens_perturbed": n_tokens(r["text_perturbed"]), "parse_error": perr,
                })
            pd.DataFrame(rows, columns=COLUMNS).to_csv(
                out_path, mode="a", header=not out_path.exists(), index=False, encoding="utf-8"
            )

    res = pd.read_csv(out_path)
    print(f"\n[done] {out_path}: {len(res)} 行 / 今回のパース失敗 {n_parse_err} 件 / 累計パース失敗 {int(res['parse_error'].sum())} 件")
    print(res.groupby("condition", sort=False)["safety"].value_counts(dropna=False).unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
