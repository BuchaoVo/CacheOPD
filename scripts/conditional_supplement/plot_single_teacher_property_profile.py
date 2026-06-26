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
    table = pd.read_csv(out_root / "summaries" / "table_single_teacher_specificity.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    metrics = [m for m in ["plddt_mean", "sol_mean", "thermo_mean"] if m in table.columns]
    if not table.empty and metrics:
        agg = table.groupby("method")[metrics].mean()
        agg.plot(kind="bar", ax=ax)
    ax.set_title("Single-teacher property profile")
    ax.set_ylabel("Mean score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = out_root / "figures" / "single_teacher_property_profile"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[OK] figure -> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()

