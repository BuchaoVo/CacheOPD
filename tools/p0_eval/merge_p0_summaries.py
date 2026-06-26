from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/data/users/zbc/opd/ProteinOPD")
OUT = ROOT / "analysis_outputs/p0_current_eval/final/p0_current_methods_full_summary.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

paths = {
    "basic": ROOT / "analysis_outputs/p0_current_eval/raw/p0_basic_ppl_summary.csv",
    "structure_hf": ROOT / "analysis_outputs/p0_current_eval/structure_hf/p0_hf_esmfold_structure_summary.csv",
    "temberture": ROOT / "analysis_outputs/p0_current_eval/property/temberture/p0_temberture_summary.csv",
    "protein_sol": ROOT / "analysis_outputs/p0_current_eval/property/protein_sol/p0_protein_sol_summary.csv",
}

dfs = []

for name, path in paths.items():
    if not path.exists():
        print(f"[WARN] missing {name}: {path}")
        continue

    df = pd.read_csv(path)
    if "method" not in df.columns:
        print(f"[WARN] no method column in {path}, skip")
        continue

    rename = {}
    for c in df.columns:
        if c != "method" and c in [
            "n_total", "n_valid", "n_scored_ok", "n_failed",
            "mean_length", "median_length"
        ]:
            rename[c] = f"{name}_{c}"

    df = df.rename(columns=rename)
    dfs.append(df)

if not dfs:
    raise FileNotFoundError("No P0 summary files found.")

merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="method", how="outer")

preferred = [
    "method",

    "basic_n_total",
    "n_valid",
    "valid_rate",
    "duplicate_rate",
    "unique_valid",
    "mean_length",
    "median_length",

    "ppl_mean",
    "ppl_median",
    "ppl_loss_mean",

    "novelty_t_3mer_proxy_mean",
    "max_ref_3mer_jaccard_mean",
    "diversity_score_3mer",
    "mean_pairwise_3mer_jaccard",

    "structure_hf_n_scored_ok",
    "structure_hf_n_failed",
    "plddt_mean",
    "plddt_median",
    "pct_plddt_gt70",
    "pae_mean",
    "pae_median",
    "pct_pae_lt10",
    "ptm_mean",

    "thermo_score_mean",
    "thermo_score_median",
    "thermophilic_rate",
    "temberture_n_scored_ok",
    "temberture_n_failed",

    "sol_score_mean",
    "sol_score_median",
    "protein_sol_n_scored_ok",
    "protein_sol_n_failed",
]

cols = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
merged = merged[cols]

merged.to_csv(OUT, index=False)

print(merged.to_string(index=False))
print(f"\n[OK] saved: {OUT}")
