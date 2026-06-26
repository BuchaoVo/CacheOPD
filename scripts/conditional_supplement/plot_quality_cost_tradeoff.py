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
    table = pd.read_csv(out_root / "summaries" / "table_internal_matched.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if not table.empty and {"Teacher Calls_mean", "plddt_mean"}.issubset(table.columns):
        x = table["Teacher Calls_mean"] if "Teacher Calls_mean" in table.columns else table.get("teacher_calls_mean")
        y = table["plddt_mean"]
    elif not table.empty and {"ppl_mean", "plddt_mean"}.issubset(table.columns):
        x = table["ppl_mean"]
        y = table["plddt_mean"]
        ax.set_xlabel("PPL (lower is better)")
    else:
        x = []
        y = []
    if len(table) and len(x):
        ax.scatter(x, y, s=48, color="#2a6fbb")
        for _, r in table.iterrows():
            ax.annotate(str(r["method"]), (x.loc[r.name], y.loc[r.name]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_title("Quality-cost trade-off")
    if not ax.get_xlabel():
        ax.set_xlabel("Cost proxy")
    ax.set_ylabel("pLDDT")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = out_root / "figures" / "quality_cost_tradeoff"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[OK] figure -> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()

