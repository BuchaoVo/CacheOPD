from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/data/users/zbc/opd/ProteinOPD")
BASE = ROOT / "analysis_outputs/offline_proteinopd/p0_eval"
OUT = BASE / "final/offline_p0_full_summary.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

paths = {
    "basic": BASE / "raw/p0_basic_ppl_summary.csv",
    "structure_hf": BASE / "structure_hf/p0_hf_esmfold_structure_summary.csv",
    "temberture": BASE / "property/temberture/p0_temberture_summary.csv",
    "protein_sol": BASE / "property/protein_sol/p0_protein_sol_summary.csv",
}

dfs = []

for name, path in paths.items():
    if not path.exists():
        print(f"[WARN] missing {name}: {path}")
        continue

    df = pd.read_csv(path)
    if "method" not in df.columns:
        print(f"[WARN] skip {name}, no method column: {path}")
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
    raise FileNotFoundError("No offline P0 summary files found.")

merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on="method", how="outer")

# Convert pLDDT from 0-1 to 0-100 if present.
if "plddt_mean" in merged.columns and "plddt_mean_100" not in merged.columns:
    merged["plddt_mean_100"] = merged["plddt_mean"] * 100.0
if "plddt_median" in merged.columns and "plddt_median_100" not in merged.columns:
    merged["plddt_median_100"] = merged["plddt_median"] * 100.0

preferred = [
    "method",

    "basic_n_total",
    "basic_n_valid",
    "valid_rate",
    "duplicate_rate",
    "unique_valid",
    "basic_mean_length",
    "basic_median_length",
    "min_length",
    "max_length",

    "ppl_mean",
    "ppl_median",
    "ppl_loss_mean",
    "novelty_t_3mer_proxy_mean",
    "max_ref_3mer_jaccard_mean",
    "diversity_score_3mer",
    "mean_pairwise_3mer_jaccard",

    "structure_hf_n_scored_ok",
    "structure_hf_n_failed",
    "plddt_mean_100",
    "plddt_median_100",
    "pct_plddt_gt70",
    "pae_mean",
    "pae_median",
    "pct_pae_lt10",
    "ptm_mean",

    "sol_score_mean",
    "sol_score_median",
    "protein_sol_n_scored_ok",
    "protein_sol_n_failed",

    "thermo_score_mean",
    "thermo_score_median",
    "thermophilic_rate",
    "temberture_n_scored_ok",
    "temberture_n_failed",
]

cols = [c for c in preferred if c in merged.columns] + [
    c for c in merged.columns if c not in preferred
]
merged = merged[cols]

merged.to_csv(OUT, index=False)

print(merged.to_string(index=False))
print(f"\n[OK] saved: {OUT}")
