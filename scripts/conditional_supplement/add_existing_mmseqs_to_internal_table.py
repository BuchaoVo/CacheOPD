from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NAME_MAP = {
    "Base-ProLLaMA": "ProLLaMA",
    "SFT-RefRollout": "SFT",
    "Exp3-Fusion-ProbAvg": "ProbAvg-KD",
    "Exp3-Fusion-LogitAvg": "LogitAvg-KD",
    "Conditional-OPD": "Online OPD",
    "Offline-OPDRollout-K64": "CacheOPD Full",
    "FinalUtility-ShiftEntropy-q04-High50": "CacheOPD Sparse",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="analysis_outputs/conditional_supplement")
    ap.add_argument("--novelty_summary", default="analysis_outputs/offline_proteinopd/eval_main_table/mmseqs_novelty/mmseqs_novelty_summary.csv")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    table_path = out_root / "summaries/table_internal_matched.csv"
    out_path = out_root / "summaries/table_internal_matched_with_existing_mmseqs.csv"
    table = pd.read_csv(table_path)

    novelty_path = Path(args.novelty_summary)
    if not novelty_path.exists():
        table.to_csv(out_path, index=False)
        print(f"[WARN] novelty summary missing; copied table to {out_path}")
        return

    novelty = pd.read_csv(novelty_path)
    novelty["method"] = novelty["method"].map(NAME_MAP).fillna(novelty["method"])
    novelty = novelty[[
        "method",
        "novelty_mmseqs_mean",
        "max_identity_mean",
        "pct_identity_ge_0.5",
        "pct_identity_ge_0.8",
    ]].drop_duplicates("method")
    merged = table.merge(novelty, on="method", how="left")
    merged.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
