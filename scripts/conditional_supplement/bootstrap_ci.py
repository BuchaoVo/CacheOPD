from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_ROOT


def boot_ci(vals: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, size=vals.size, replace=True).mean() for _ in range(n_boot)])
    return float(vals.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    out_root = Path(args.out_root)
    df = pd.read_csv(out_root / "summaries" / "merged_per_sequence_metrics.csv")
    metrics = ["ppl", "plddt", "pae", "ptm", "sol", "thermo", "condition_correct"]
    rows = []
    for (method, condition), sub in df.groupby(["method", "condition_id"], dropna=False):
        for metric in metrics:
            if metric not in sub.columns:
                continue
            mean, lo, hi = boot_ci(pd.to_numeric(sub[metric], errors="coerce").to_numpy(float), args.n_boot, args.seed)
            rows.append({"method": method, "condition_id": condition, "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(sub)})
    pd.DataFrame(rows).to_csv(out_root / "summaries" / "bootstrap_ci.csv", index=False)
    per = df.groupby(["method", "condition_id"], dropna=False).agg(valid_n=("sequence", "count")).reset_index()
    for metric in metrics:
        if metric in df.columns:
            tmp = df.groupby(["method", "condition_id"], dropna=False)[metric].mean().reset_index(name=f"{metric}_mean")
            per = per.merge(tmp, on=["method", "condition_id"], how="left")
    per.to_csv(out_root / "summaries" / "per_superfamily_breakdown.csv", index=False)
    print(f"[OK] bootstrap -> {out_root / 'summaries' / 'bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()
