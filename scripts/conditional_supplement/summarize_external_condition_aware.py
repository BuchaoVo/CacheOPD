from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import OUT_ROOT, summarize_numeric


EXTERNAL = {"ProteinDT", "ProDVa", "ESM3"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out_root)
    df = pd.read_csv(out_root / "summaries" / "merged_per_sequence_metrics.csv")
    df = df[df["method"].isin(EXTERNAL)].copy()
    metrics = ["ppl", "plddt", "pae", "ptm", "sol", "thermo", "duplicate_rate_unit", "condition_correct", "length"]
    table = summarize_numeric(df, ["method"], metrics)
    table.to_csv(out_root / "summaries" / "table_external_condition_aware.csv", index=False)
    print(f"[OK] external table -> {out_root / 'summaries' / 'table_external_condition_aware.csv'}")


if __name__ == "__main__":
    main()
