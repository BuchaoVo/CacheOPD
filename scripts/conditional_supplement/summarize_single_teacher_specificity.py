from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import OUT_ROOT, summarize_numeric


SINGLE = {"Fold-teacher KD", "Sol-teacher KD", "Thermo-teacher KD"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out_root)
    df = pd.read_csv(out_root / "summaries" / "merged_per_sequence_metrics.csv")
    df = df[df["method"].isin(SINGLE)].copy()
    table = summarize_numeric(df, ["method", "condition_id"], ["plddt", "pae", "ptm", "sol", "thermo", "condition_correct"])
    table.to_csv(out_root / "summaries" / "table_single_teacher_specificity.csv", index=False)
    print(f"[OK] single-teacher table -> {out_root / 'summaries' / 'table_single_teacher_specificity.csv'}")


if __name__ == "__main__":
    main()

