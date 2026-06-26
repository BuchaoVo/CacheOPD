from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import OUT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out_root)
    df = pd.read_csv(out_root / "summaries" / "merged_per_sequence_metrics.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if {"condition_correct", "plddt", "method"}.issubset(df.columns):
        table = df.groupby("method").agg(condition_consistency=("condition_correct", "mean"), plddt=("plddt", "mean")).reset_index()
        ax.scatter(table["condition_consistency"], table["plddt"], s=48, color="#2f8f5b")
        for _, r in table.iterrows():
            ax.annotate(str(r["method"]), (r["condition_consistency"], r["plddt"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_title("Condition consistency vs. quality")
    ax.set_xlabel("Condition consistency")
    ax.set_ylabel("pLDDT")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = out_root / "figures" / "condition_consistency_vs_quality"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[OK] figure -> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()

