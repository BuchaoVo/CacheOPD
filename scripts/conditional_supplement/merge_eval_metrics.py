from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import OUT_ROOT, ensure_dirs, load_metric_file, safe_read_csv


def merge_one(base: pd.DataFrame, path: Path, keep: dict[str, str]) -> pd.DataFrame:
    df = load_metric_file(path)
    if df.empty:
        return base
    metric_cols = []
    for src, dst in keep.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
            if dst not in metric_cols:
                metric_cols.append(dst)
    keys = [c for c in ["method", "condition_id", "sample_id", "sequence_id"] if c in base.columns and c in df.columns]
    if "method" in base.columns and "method" in df.columns and "sequence_id" in base.columns and "sequence_id" in df.columns:
        keys = ["method", "sequence_id"]
    cols = keys + [c for c in metric_cols if c in df.columns]
    if len(keys) < 2 or len(cols) == len(keys):
        return base
    return base.merge(df[cols].drop_duplicates(keys), on=keys, how="left")


def add_diversity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "sequence" not in df.columns:
        return df
    df = df.copy()
    df["duplicate_sequence"] = df.duplicated(["method", "sequence"], keep=False)
    df["duplicate_rate_unit"] = df["duplicate_sequence"].astype(float)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    base = safe_read_csv(out_root / "cleaned" / "cleaned_sequences.csv")
    if base.empty:
        base = pd.DataFrame(columns=["method", "condition_id", "sample_id", "sequence_id", "sequence", "length"])

    metric_specs = [
        (out_root / "eval/ppl/per_sequence_ppl.csv", {"ppl": "ppl", "ppl_loss": "ppl_loss", "nll": "ppl_loss"}),
        (out_root / "eval/structure/per_sequence_structure.csv", {"plddt": "plddt", "plddt_mean": "plddt", "pae": "pae", "pae_mean": "pae", "ptm": "ptm", "ptm_mean": "ptm"}),
        (out_root / "eval/solubility/per_sequence_solubility.csv", {"sol_score": "sol", "sol_score_mean": "sol", "scaled_sol": "sol"}),
        (out_root / "eval/thermostability/per_sequence_thermostability.csv", {"thermo_score": "thermo", "thermo_score_mean": "thermo"}),
        (out_root / "eval/condition_consistency/per_sequence_condition_consistency.csv", {"condition_correct": "condition_correct", "predicted_condition": "predicted_condition"}),
    ]
    merged = base
    for path, keep in metric_specs:
        merged = merge_one(merged, path, keep)
    if "plddt" in merged.columns:
        plddt = pd.to_numeric(merged["plddt"], errors="coerce")
        if plddt.notna().any() and plddt.max() <= 1.0:
            merged["plddt"] = plddt * 100.0
    merged = add_diversity(merged)
    merged.to_csv(out_root / "summaries" / "merged_per_sequence_metrics.csv", index=False)
    print(f"[OK] merged -> {out_root / 'summaries' / 'merged_per_sequence_metrics.csv'}")


if __name__ == "__main__":
    main()
