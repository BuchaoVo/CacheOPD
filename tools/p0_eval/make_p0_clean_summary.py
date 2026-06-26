from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/mnt/data/users/zbc/opd/ProteinOPD")
IN = ROOT / "analysis_outputs/p0_current_eval/final/p0_current_methods_full_summary.csv"
OUT = ROOT / "analysis_outputs/p0_current_eval/final/p0_current_methods_clean_summary.csv"
OUT_MD = ROOT / "analysis_outputs/p0_current_eval/final/p0_current_methods_clean_summary.md"

df = pd.read_csv(IN)

clean = pd.DataFrame()
clean["method"] = df["method"]

clean["Valid Rate"] = df["valid_rate"]
clean["Duplicate Rate"] = df["duplicate_rate"]
clean["PPL↓"] = df["ppl_mean"]
clean["Novelty-T Proxy↑"] = df["novelty_t_3mer_proxy_mean"]
clean["3-mer Diversity↑"] = df["diversity_score_3mer"]

# HF ESMFold pLDDT is currently stored in 0-1 scale; convert to 0-100 for table.
clean["pLDDT↑"] = df["plddt_mean"] * 100.0
clean["pLDDT Median"] = df["plddt_median"] * 100.0
clean["% pLDDT > 70↑"] = df["pct_plddt_gt70"]

clean["pAE↓"] = df["pae_mean"]
clean["% pAE < 10↑"] = df["pct_pae_lt10"]
clean["pTM↑"] = df["ptm_mean"]

clean["Sol↑"] = df["sol_score_mean"]
clean["Thermo↑"] = df["thermo_score_mean"]
clean["Thermophilic Rate"] = df["thermophilic_rate"]

# Add relative delta vs Base-ProLLaMA for main metrics.
base = clean[clean["method"] == "Base-ProLLaMA"].iloc[0]

rows = []
for _, r in clean.iterrows():
    row = r.to_dict()
    row["ΔPPL vs Base (%)"] = 100.0 * (r["PPL↓"] - base["PPL↓"]) / base["PPL↓"]
    row["ΔpLDDT vs Base"] = r["pLDDT↑"] - base["pLDDT↑"]
    row["ΔpAE vs Base (%)"] = 100.0 * (r["pAE↓"] - base["pAE↓"]) / base["pAE↓"]
    row["ΔSol vs Base"] = r["Sol↑"] - base["Sol↑"]
    row["ΔThermo vs Base"] = r["Thermo↑"] - base["Thermo↑"]
    rows.append(row)

clean = pd.DataFrame(rows)

clean.to_csv(OUT, index=False)

view = clean.copy()
for c in view.columns:
    if c == "method":
        continue
    if "Rate" in c or c.startswith("%") or c.endswith("(%)"):
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    elif c == "PPL↓":
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    else:
        view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

OUT_MD.write_text(view.to_markdown(index=False), encoding="utf-8")

print(view.to_markdown(index=False))
print(f"\n[OK] saved CSV: {OUT}")
print(f"[OK] saved MD:  {OUT_MD}")
