from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


UNCOND_NAME_MAP = {
    "Base-ProtGPT2": "ProtGPT2",
    "ProteinOPD-Uncond": "ProteinOPD",
    "OfflinePoE-Uncond-Full": "CacheOPD-Uncond-Full",
    "CacheOPD-Uncond-q04-High50": "CacheOPD-Uncond-Sparse",
}

COND_NAME_MAP = {
    "Base-ProLLaMA": "ProLLaMA",
    "Conditional-OPD": "ProteinOPD",
}


def attach_novelty(summary: pd.DataFrame, novelty: pd.DataFrame, setting: str, name_map: dict[str, str]) -> pd.DataFrame:
    if summary.empty or novelty.empty:
        return summary
    out = summary.copy()
    nov = novelty.copy()
    nov["method"] = nov["method"].map(name_map).fillna(nov["method"])
    nov["setting"] = setting
    keep = ["setting", "method", "novelty_mmseqs_mean", "max_identity_mean", "pct_identity_ge_0.5", "pct_identity_ge_0.8"]
    nov = nov[[c for c in keep if c in nov.columns]].drop_duplicates(["setting", "method"])
    key_cols = ["setting", "method"]
    nov = nov.set_index(key_cols)
    idx = pd.MultiIndex.from_frame(out[key_cols])
    for col in nov.columns:
        if col not in out.columns:
            out[col] = pd.NA
        values = nov[col].reindex(idx).to_numpy()
        mask = pd.notna(values)
        if mask.any():
            out.loc[mask, col] = values[mask]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-root", default="analysis_outputs/proteinopd_reproduction")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    summary_path = out_root / "summaries/proteinopd_reproduction_method_summary.csv"
    summary = pd.read_csv(summary_path)

    uncond_path = repo / "analysis_outputs/unconditional_cacheopd/eval/mmseqs_novelty/mmseqs_novelty_summary.csv"
    cond_path = repo / "analysis_outputs/offline_proteinopd/eval_main_table/mmseqs_novelty/mmseqs_novelty_summary.csv"

    if uncond_path.exists():
        summary = attach_novelty(summary, pd.read_csv(uncond_path), "unconditional", UNCOND_NAME_MAP)
    if cond_path.exists():
        summary = attach_novelty(summary, pd.read_csv(cond_path), "conditional", COND_NAME_MAP)

    out_path = out_root / "summaries/proteinopd_reproduction_method_summary_with_existing_mmseqs.csv"
    summary.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
