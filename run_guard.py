"""3. 変換済みデータセット（data/perturbed_<データのバージョン>.jsonl）をガードモデルで判定する。

- 判定中に変換は一切しない（text_perturbed をそのまま使う）
- 結果はバッチごとに CSV に追記し、再実行すると未処理の (id, condition) だけを処理する（再開可能）
- 出力ファイル名はモデルごと：outputs/predictions_<モデル名>_<データのバージョン>.csv
- --labels harmful で有害だけを判定できる。あとで --labels を外して同じコマンドを実行すると、無害の分だけ追加される
- モデルごとの入力の作り方・出力の読み取り方は guards.py

使い方:
  python run_guard.py --limit 10                                   # 試運転（各条件 10 件 = 有害5・無害5）
  python run_guard.py                                              # Qwen3Guard-Gen-0.6B で全件（データ v5）
  python run_guard.py --labels harmful                             # 有害だけ（v5 の割合スイープ向け）
  python run_guard.py --data-version v4                            # v4 の14条件
  python run_guard.py --model Qwen/Qwen3Guard-Gen-8B --batch-size 16
  python run_guard.py --model nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3
  python run_guard.py --model mistralai/Shieldstral-1.0-3B         # 要 pip install mistral-common
  python run_guard.py --model ToxicityPrompts/PolyGuard-Ministral
  python run_guard.py --model meta-llama/Llama-Guard-3-8B          # 要 利用規約への同意＋ hf auth login
  python run_guard.py --model ricoh-ai/Llama-Ricoh-SafeGuard-20260520   # 同上
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from common import DATA_VERSION, DATA_VERSIONS, OUT_DIR, SEED, load_perturbed, perturbed_paths, set_seed
from guards import ADD_GENERATION_PROMPT, KINDS, PARSERS, messages_for, model_slug, parse_shieldstral, spec_for
from perturb.conditions import conditions_for

MODEL_NAME = "Qwen/Qwen3Guard-Gen-0.6B"
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
COLUMNS = [
    "id", "condition", "label", "text_original", "text_perturbed", "raw_output", "safety",
    "pred_strict", "pred_loose", "score", "n_tokens_original", "n_tokens_perturbed", "parse_error",
]


LABELS = {"all": (0, 1), "harmful": (1,), "unharmful": (0,)}


def default_out(model: str, version: str = DATA_VERSION) -> Path:
    return OUT_DIR / f"predictions_{model_slug(model)}_{version}.csv"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        sys.exit(f"[error] {out_path} の列が想定と異なります（v3 以前の形式？）。別ファイル名を --out で指定してください")
    # 入力データが変わっていないか確認（同じ (id, condition) の text_perturbed が一致すること）
    merged = prev.merge(df[["id", "condition", "text_perturbed"]], on=["id", "condition"], suffixes=("", "_in"))
    if (merged["text_perturbed"] != merged["text_perturbed_in"]).any():
        sys.exit(f"[error] {out_path} の text_perturbed が入力データと一致しません（データセットが変わった可能性）")
    return set(zip(prev["id"].astype(int), prev["condition"]))


# ---------------------------------------------------------------------------
# 生成して出力文字列を読むモデル（Qwen3Guard・Llama Guard 3・リコー・Nemotron・PolyGuard）
# ---------------------------------------------------------------------------
class GenerativeGuard:
    def __init__(self, model_id: str, kind: str, dtype, device, max_new_tokens: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.kind = kind
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        cfg = self.model.generation_config
        cfg.do_sample = False  # greedy
        cfg.temperature = None
        cfg.top_p = None
        cfg.top_k = None
        cfg.max_new_tokens = max_new_tokens
        cfg.pad_token_id = self.tok.pad_token_id
        self.gen_cfg = cfg

    def prompt_text(self, text: str) -> str:
        return self.tok.apply_chat_template(messages_for(self.kind, text), tokenize=False,
                                            add_generation_prompt=ADD_GENERATION_PROMPT[self.kind])

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def judge(self, prompts: list[str]) -> list[tuple[str, float | None]]:
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        gen = self.model.generate(**enc, generation_config=self.gen_cfg)
        outs = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [(o.strip(), None) for o in outs]

    def parse(self, raw: str, score):
        return PARSERS[self.kind](raw)


# ---------------------------------------------------------------------------
# Shieldstral：生成せず、1回の forward で次トークンの yes/no の確率を比べる（モデルカードの方法）
# ---------------------------------------------------------------------------
class ShieldstralGuard:
    _YES = ("yes", "yes.", '"yes"', "'yes'")
    _NO = ("no", "no.", '"no"', "'no'")

    def __init__(self, model_id: str, dtype, device):
        try:
            from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        except ImportError as e:  # mistral-common が必要
            raise SystemExit(f"[error] Shieldstral には mistral-common が必要です: pip install 'mistral-common>=1.11.5' ({e})")
        self.kind = "shieldstral"
        self.device = device
        self.tok = MistralCommonBackend.from_pretrained(model_id)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()

    def prompt_text(self, text: str) -> str:
        return text  # 判定時にメッセージ列を組み立てる（長さの並べ替え用に本文を返す）

    def n_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    @torch.no_grad()
    def _score(self, text: str) -> float:
        enc = self.tok.apply_chat_template(messages_for("shieldstral", text), return_tensors="pt", return_dict=True)
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in enc.items()}
        logits = self.model(**inputs).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        z_yes, z_no = -1e9, -1e9
        values, indices = logprobs.topk(20)
        for logp, token_id in zip(values.tolist(), indices.tolist()):
            t = self.tok.decode([token_id]).strip().lower()
            if t in self._YES:
                z_yes = max(z_yes, logp)
            elif t in self._NO:
                z_no = max(z_no, logp)
        if z_yes == -1e9 and z_no == -1e9:
            return float("nan")  # 上位20トークンに yes/no がない → パース失敗扱い
        return math.exp(z_yes) / (math.exp(z_yes) + math.exp(z_no))

    def judge(self, prompts: list[str]) -> list[tuple[str, float | None]]:
        res = []
        for p in prompts:  # 1件ずつ（3B なので十分速い）
            s = self._score(p)
            res.append(("" if math.isnan(s) else f"p(yes)={s:.4f}", s))
        return res

    def parse(self, raw: str, score):
        if score is None or math.isnan(score):
            return None, None, None, 1
        return parse_shieldstral(score)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-version", default=DATA_VERSION, choices=DATA_VERSIONS)
    ap.add_argument("--input", type=Path, default=None, help="省略時は data/perturbed_<data-version>.jsonl")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--kind", default=None, choices=KINDS, help="モデルの種類（通常はモデル名から自動判定）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="各条件で判定する件数（有害・無害を半数ずつ）")
    ap.add_argument("--conditions", nargs="*", default=None, help="判定する条件（省略時はそのバージョンの全条件）")
    ap.add_argument("--labels", default="all", choices=list(LABELS), help="判定するラベル（harmful＝有害だけ）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="省略時はモデルごとの既定値（guards.py）")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--dtype", default=None, choices=list(DTYPES),
                    help="省略時はモデルごとの既定値（Qwen3Guard-Gen-0.6B は v3 と同じ float32、その他は bfloat16）")
    args = ap.parse_args()
    all_conds = conditions_for(args.data_version)
    if args.conditions is None:
        args.conditions = all_conds
    elif unknown := [c for c in args.conditions if c not in all_conds]:
        sys.exit(f"[error] データ {args.data_version} にない条件です: {unknown}")
    args.input = args.input or perturbed_paths(args.data_version)[1]

    spec = spec_for(args.model, args.kind)
    dtype_name = args.dtype or spec.dtype
    max_new_tokens = args.max_new_tokens or spec.max_new_tokens

    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_out = default_out(args.model, args.data_version)
    out_path = args.out or (
        base_out if args.limit is None else base_out.with_name(f"{base_out.stem}_limit{args.limit}.csv")
    )

    df = load_perturbed(args.input)
    if args.limit is not None:
        ids = select_limit_ids(df, args.limit)
        df = df[df["id"].isin(ids)]
    df = df[df["label"].isin(LABELS[args.labels])]
    done = load_done(out_path, df)
    print(f"[data] {args.input}（{args.labels}）: {df['id'].nunique()} 件 × {df['condition'].nunique()} 条件 / "
          f"処理済み {len(done)} 行 → {out_path}")

    n_todo = sum((int(i), c) not in done for i, c in zip(df["id"], df["condition"]) if c in args.conditions)
    if n_todo == 0:
        print("[skip] すべて処理済みです")
        return

    device = pick_device() if args.device == "auto" else torch.device(args.device)
    print(f"[model] {args.model}  kind={spec.kind}  device={device}  dtype={dtype_name}")
    if spec.kind == "shieldstral":
        guard = ShieldstralGuard(args.model, DTYPES[dtype_name], device)
    else:
        guard = GenerativeGuard(args.model, spec.kind, DTYPES[dtype_name], device, max_new_tokens)

    meta = {
        "model": args.model, "kind": spec.kind, "device": str(device), "dtype": dtype_name,
        "batch_size": args.batch_size, "max_new_tokens": max_new_tokens,
        "decoding": "single forward (p(yes) > 0.5)" if spec.kind == "shieldstral" else "greedy",
        "seed": SEED, "data_version": args.data_version, "input": str(args.input), "limit": args.limit,
        "labels": args.labels, "torch": torch.__version__,
        "python": platform.python_version(), "transformers": __import__("transformers").__version__,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    n_parse_err = 0
    t0 = time.time()
    for cond in args.conditions:
        sub = df[df["condition"] == cond]
        todo = sub[[(int(i), cond) not in done for i in sub["id"]]].copy()
        if todo.empty:
            print(f"[{cond}] 処理済み")
            continue
        todo["prompt_text"] = [guard.prompt_text(t) for t in todo["text_perturbed"]]
        todo["_len"] = [guard.n_tokens(p) for p in todo["prompt_text"]]
        todo = todo.sort_values(["_len", "id"])  # 長さ順にまとめてパディングを減らす（決定的）
        for start in tqdm(range(0, len(todo), args.batch_size), desc=cond):
            batch = todo.iloc[start: start + args.batch_size]
            outs = guard.judge(batch["prompt_text"].tolist())
            rows = []
            for (_, r), (raw, score) in zip(batch.iterrows(), outs):
                safety, strict, loose, perr = guard.parse(raw, score)
                n_parse_err += perr
                rows.append({
                    "id": int(r["id"]), "condition": cond, "label": int(r["label"]),
                    "text_original": r["text_original"], "text_perturbed": r["text_perturbed"],
                    "raw_output": raw, "safety": safety, "pred_strict": strict, "pred_loose": loose,
                    "score": score, "n_tokens_original": guard.n_tokens(r["text_original"]),
                    "n_tokens_perturbed": guard.n_tokens(r["text_perturbed"]), "parse_error": perr,
                })
            pd.DataFrame(rows, columns=COLUMNS).to_csv(
                out_path, mode="a", header=not out_path.exists(), index=False, encoding="utf-8"
            )

    res = pd.read_csv(out_path)
    print(f"\n[done] {out_path}: {len(res)} 行 / 今回 {time.time() - t0:.0f} 秒 / "
          f"今回のパース失敗 {n_parse_err} 件 / 累計パース失敗 {int(res['parse_error'].sum())} 件")
    print(res.groupby("condition", sort=False)["safety"].value_counts(dropna=False).unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
