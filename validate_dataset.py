"""2. 変換済みデータセットの自動チェックと、目視確認用ファイルの出力・集計。

使い方:
  python validate_dataset.py                       # 自動チェック＋ outputs/manual_check_v4.csv を出力
  python validate_dataset.py --score outputs/manual_check_v4.csv   # 記入済みファイルの ok 率を集計
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from common import DATA_VERSION, DICT_DIR, OUT_DIR, PERTURBED_JSONL, SEED, load_perturbed
from perturb.conditions import FULL_SCRIPT, PARTIAL_CONDITIONS, PARTIAL_SCRIPT, UNCOUNTED_KINDS
from perturb.homophone import HomophoneDict
from perturb.kana import has_japanese, has_kanji, is_ascii_alpha, kata2hira
from perturb.morph import tokenize

_HIRA = re.compile(r"[ぁ-ゖゝゞ]")
_KATA_FULL = re.compile(r"[ァ-ヺヽヾ]")
_KATA_HALF = re.compile(r"[ｦ-ﾝ]")


def script_violations(script: str, s: str) -> str:
    """文字列 s に、その文字種で残ってはいけない文字があれば種類を返す（なければ空文字）。"""
    bad = []
    if has_kanji(s):
        bad.append("kanji")
    if script == "romaji":
        if has_japanese(s) or _KATA_HALF.search(s):
            bad.append("japanese")
        return ",".join(bad)
    if script != "hira" and _HIRA.search(s):
        bad.append("hiragana")
    if script != "kata" and _KATA_FULL.search(s):
        bad.append("katakana")
    if script != "hankata" and _KATA_HALF.search(s):
        bad.append("hankaku")
    return ",".join(bad)

N_MANUAL = 20
MANUAL_CSV = OUT_DIR / f"manual_check_{DATA_VERSION}.csv"


def _detail(row) -> list[dict]:
    return json.loads(row["changes_detail"]) if isinstance(row["changes_detail"], str) else []


def find_issues(df: pd.DataFrame, hdict: HomophoneDict | None) -> pd.DataFrame:
    issues: list[dict] = []

    def add(row, kind, msg=""):
        issues.append({"id": row["id"], "condition": row["condition"], "issue": kind, "detail": msg,
                       "text_original": row["text_original"], "text_perturbed": row["text_perturbed"]})

    for _, row in df.iterrows():
        cond = row["condition"]
        det = _detail(row)
        if cond == "original":
            if row["text_perturbed"] != row["text_original"]:
                add(row, "original_modified")
            continue
        if row["text_perturbed"] == row["text_original"]:
            add(row, "unchanged_text")
        if cond in PARTIAL_CONDITIONS and row["n_changed"] == 0:
            add(row, "n_changed_zero", f"n_content={row.get('n_content')}")
        for c in det:
            if c["orig"].strip() and not c["new"].strip():
                add(row, "word_deleted", c["orig"])
        # 変換位置の整合性（changes_detail のオフセットが変換後テキストと一致するか）
        for c in det:
            seg = row["text_perturbed"][c["start_new"]: c["start_new"] + len(c["new"])]
            if seg != c["new"]:
                add(row, "offset_mismatch", f"{c['orig']}→{c['new']}")
        script = FULL_SCRIPT.get(cond) or PARTIAL_SCRIPT.get(cond)
        if cond in FULL_SCRIPT:
            # 文全体：その文字種以外の日本語の文字が1字も残っていないか
            bad = script_violations(script, row["text_perturbed"])
            if bad:
                add(row, f"{cond}_other_script_in_text", bad)
        if script:
            # 変換した語：その文字種だけになっているか（ローマ字は英字のみ）
            for c in det:
                if not c["changed"] or c["info"].get("kind") in UNCOUNTED_KINDS:
                    continue
                new = c["new"].strip()
                bad = ("not_alpha" if not is_ascii_alpha(new) else "") if script == "romaji" else script_violations(script, new)
                if bad:
                    add(row, f"{cond}_word_not_converted", f"{c['orig']}→{new} ({bad})")
        if cond == "homophone":
            for c in det:
                if not c["changed"]:
                    continue
                if c["new"] == c["orig"]:
                    add(row, "homophone_same_word", c["orig"])
                if not has_kanji(c["orig"]):
                    add(row, "homophone_kana_source", f"{c['orig']}→{c['new']}")
                if hdict is not None and kata2hira(c["reading"]) not in hdict.readings_of(c["new"]):
                    add(row, "homophone_reading_mismatch",
                        f"{c['orig']}({c['reading']})→{c['new']}{sorted(hdict.readings_of(c['new']))}")
    return pd.DataFrame(issues, columns=["id", "condition", "issue", "detail", "text_original", "text_perturbed"])


def homophone_sudachi_check(df: pd.DataFrame) -> pd.DataFrame:
    """参考：置換後の語を Sudachi で単独解析したときの読みが元の読みと一致するか（不一致は警告扱い）。"""
    rows = []
    for _, row in df[df["condition"] == "homophone"].iterrows():
        for c in _detail(row):
            if not c["changed"]:
                continue
            rd = "".join(t.reading for t in tokenize(c["new"]))
            rows.append({"id": row["id"], "orig": c["orig"], "new": c["new"], "reading": c["reading"],
                         "sudachi_reading_new": rd, "match": rd == c["reading"]})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    bad_rows = issues.groupby("condition")["id"].nunique() if len(issues) else pd.Series(dtype=int)
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        n = len(g)
        n_bad = int(bad_rows.get(cond, 0))
        rec = {
            "condition": cond,
            "n_rows": n,
            "n_rows_with_issue": n_bad,
            "success_rate": (n - n_bad) / n,
            "mean_n_content": g["n_content"].mean(),
            "mean_n_target": g["n_target"].mean(),
            "mean_n_changed": g["n_changed"].mean(),
            "word_change_rate": g["n_changed"].sum() / g["n_target"].sum() if g["n_target"].sum() else float("nan"),
        }
        if len(issues):
            for kind, cnt in issues[issues["condition"] == cond]["issue"].value_counts().items():
                rec[f"issue:{kind}"] = int(cnt)
        rows.append(rec)
    out = pd.DataFrame(rows)
    issue_cols = [c for c in out.columns if c.startswith("issue:")]
    out[issue_cols] = out[issue_cols].fillna(0).astype(int)
    return out


def mark_changes(row) -> str:
    """変換箇所を【】で囲んだ変換後テキスト。"""
    text = row["text_perturbed"]
    spans = sorted(((c["start_new"], c["start_new"] + len(c["new"])) for c in _detail(row) if c["changed"]), reverse=True)
    for s, e in spans:
        text = text[:s] + "【" + text[s:e] + "】" + text[e:]
    return text


def write_manual_check(df: pd.DataFrame, path: Path, seed: int, force: bool) -> None:
    if path.exists() and not force:
        old = pd.read_csv(path, encoding="utf-8-sig")
        if old["review"].notna().any():
            print(f"[manual] {path} には記入済みの review があるため上書きしません（--force で上書き）")
            return
    parts = []
    for cond, g in df[df["condition"] != "original"].groupby("condition", sort=False):
        s = g.sample(n=min(N_MANUAL, len(g)), random_state=seed).sort_values("id")
        parts.append(s)
    m = pd.concat(parts)
    out = pd.DataFrame({
        "condition": m["condition"],
        "id": m["id"],
        "label": m["label"],
        "changes": m["changes"],
        "text_perturbed_marked": m.apply(mark_changes, axis=1),
        "text_original": m["text_original"],
        "review": "",
        "memo": "",
    })
    out.to_csv(path, index=False, encoding="utf-8-sig")  # Excel で文字化けしないよう BOM 付き
    print(f"[manual] {path} に {len(out)} 行を出力（条件ごと {N_MANUAL} 件）。review 列に ok / ng を記入してください")


def score_manual(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path, encoding="utf-8-sig", dtype={"review": str})
    m["review_norm"] = m["review"].fillna("").str.strip().str.lower()
    rows = []
    for cond, g in m.groupby("condition", sort=False):
        ok = int((g["review_norm"] == "ok").sum())
        ng = int((g["review_norm"] == "ng").sum())
        other = int((~g["review_norm"].isin(["ok", "ng", ""])).sum())
        rows.append({"condition": cond, "n": len(g), "n_reviewed": ok + ng, "ok": ok, "ng": ng,
                     "invalid_entries": other, "unreviewed": int((g["review_norm"] == "").sum()),
                     "ok_rate": ok / (ok + ng) if ok + ng else float("nan")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=PERTURBED_JSONL)
    ap.add_argument("--score", type=Path, default=None, help="記入済みの manual_check.csv を集計する")
    ap.add_argument("--force", action="store_true", help="記入済みの manual_check.csv でも上書きする")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.score:
        res = score_manual(args.score)
        out = OUT_DIR / f"manual_check_score_{DATA_VERSION}.csv"
        res.to_csv(out, index=False)
        print(res.to_string(index=False))
        print(f"[save] {out}")
        return

    df = load_perturbed(args.input)
    jm = sorted(DICT_DIR.glob("jmdict-eng-common-*.json"))
    hdict = HomophoneDict.load(jm[-1]) if jm else None
    if hdict is None:
        print("[warn] JMdict が見つからないため homophone の読みチェックを省略します", file=sys.stderr)

    issues = find_issues(df, hdict)
    issues.to_csv(OUT_DIR / f"validation_issues_{DATA_VERSION}.csv", index=False, encoding="utf-8-sig")
    summ = summarize(df, issues)
    summ.to_csv(OUT_DIR / f"validation_summary_{DATA_VERSION}.csv", index=False)
    hs = homophone_sudachi_check(df)
    hs.to_csv(OUT_DIR / f"validation_homophone_sudachi_{DATA_VERSION}.csv", index=False, encoding="utf-8-sig")

    print("[validation] 条件ごとの変換成功率:")
    print(summ.round(3).to_string(index=False))
    if len(issues):
        print(f"\n[validation] 問題のある行: {issues[['id', 'condition']].drop_duplicates().shape[0]} 行 / 問題 {len(issues)} 件")
        print(issues[["id", "condition", "issue", "detail"]].to_string(index=False, max_colwidth=60))
    else:
        print("\n[validation] 問題は見つかりませんでした")
    if len(hs):
        print(f"\n[参考] homophone 置換語の Sudachi 読み一致率: {hs['match'].mean():.3f}（{len(hs)} 語）"
              f" → outputs/validation_homophone_sudachi_{DATA_VERSION}.csv")
    print(f"[save] {OUT_DIR / f'validation_summary_{DATA_VERSION}.csv'} / {OUT_DIR / f'validation_issues_{DATA_VERSION}.csv'}")

    write_manual_check(df, MANUAL_CSV, args.seed, args.force)


if __name__ == "__main__":
    main()
