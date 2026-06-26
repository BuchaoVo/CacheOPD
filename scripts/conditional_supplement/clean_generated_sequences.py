from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import json

from common import OUT_ROOT, clean_seq, collect_strings, ensure_dirs, infer_condition_from_name, is_valid_seq, read_json_any


def load_any(path: Path, default_method: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            obj = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            obj = read_json_any(path)
        seqs = []
        for s in collect_strings(obj):
            seq = clean_seq(s)
            if seq:
                seqs.append(seq)
        df = pd.DataFrame({"sequence": seqs})
    else:
        return pd.DataFrame()
    if "sequence" not in df.columns:
        for cand in ["seq", "protein_sequence", "text", "generated_text", "response"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "sequence"})
                break
    if "sequence" not in df.columns:
        return pd.DataFrame()
    df["method"] = df.get("method", default_method)
    if "condition_id" not in df.columns:
        df["condition_id"] = infer_condition_from_name(str(path))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=str(OUT_ROOT / "generations"))
    ap.add_argument("--out_dir", default=str(OUT_ROOT / "cleaned"))
    ap.add_argument("--min_len", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    ensure_dirs(OUT_ROOT)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() not in {".csv", ".json"}:
            continue
        if "status" in path.parts:
            continue
        if path.name in {"internal_matched_generations.csv"}:
            continue
        df = load_any(path, path.stem)
        if df.empty:
            continue
        df["sequence"] = df["sequence"].apply(clean_seq)
        df["is_valid"] = df["sequence"].apply(lambda s: is_valid_seq(s, args.min_len, args.max_len))
        df = df[df["is_valid"]].copy()
        if "sequence_id" not in df.columns:
            df["sequence_id"] = [f"{str(df['method'].iloc[0]).replace(' ', '_')}_{i:05d}" for i in range(len(df))]
        if "sample_id" not in df.columns:
            df["sample_id"] = df.groupby(["method", "condition_id"]).cumcount()
        df["length"] = df["sequence"].str.len()
        rows.append(df[["method", "condition_id", "sample_id", "sequence_id", "sequence", "length"]])
    cleaned = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["method", "condition_id", "sample_id", "sequence_id", "sequence", "length"])
    cleaned.to_csv(out_dir / "cleaned_sequences.csv", index=False)
    cleaned.groupby(["method", "condition_id"], dropna=False).size().reset_index(name="valid_n").to_csv(out_dir / "valid_n_by_method_condition.csv", index=False)
    print(f"[OK] cleaned -> {out_dir / 'cleaned_sequences.csv'}")


if __name__ == "__main__":
    main()
