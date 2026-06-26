from pathlib import Path
import pandas as pd

ROOT = Path("/mnt/data/users/zbc/opd/ProteinOPD")

table_path = ROOT / "analysis_outputs/offline_proteinopd/contribution1/contribution1_master_table.csv"
ppl_path = ROOT / "analysis_outputs/offline_proteinopd/ppl_diagnostic/prollama_ref_full_raw/ppl_diagnostic_summary.csv"

out_csv = ROOT / "analysis_outputs/offline_proteinopd/contribution1/contribution1_master_table_prollama_ppl.csv"
out_md = ROOT / "analysis_outputs/offline_proteinopd/contribution1/contribution1_master_table_prollama_ppl.md"

table = pd.read_csv(table_path)
ppl = pd.read_csv(ppl_path)

table.columns = [c.strip() for c in table.columns]
ppl.columns = [c.strip() for c in ppl.columns]

# Only use raw ProLLaMA PPL.
ppl = ppl[(ppl["model"] == "ProLLaMA") & (ppl["mode"] == "raw")].copy()

# Prefer ppl_from_mean_nll to match formula exp(mean NLL).
ppl_map = dict(zip(ppl["cohort"], ppl["ppl_from_mean_nll"]))

rename = {
    "Base-ProLLaMA": "Base-ProLLaMA",
    "Baseline": "Baseline",
    "Offline-OPD": "Offline-OPD",
}

for method in table["Method"].astype(str):
    cohort = rename.get(method)
    if cohort in ppl_map:
        table.loc[table["Method"].astype(str).eq(method), "PPL↓"] = ppl_map[cohort]

table.to_csv(out_csv, index=False)

view = table.copy()
for col in ["PPL↓", "pLDDT↑", "pAE↓", "Sol↑", "Thermo↑"]:
    view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")

for col in ["Train Time (wall-clock, min)", "Train Cost (GPU hours)"]:
    view[col] = view[col].map(lambda x: "-" if pd.isna(x) else f"{float(x):.3f}")

view["GPU Count"] = view["GPU Count"].map(lambda x: "-" if pd.isna(x) else f"{int(float(x))}")

out_md.write_text(view.to_markdown(index=False), encoding="utf-8")

print(view.to_markdown(index=False))
print(f"\n[OK] saved: {out_csv}")
print(f"[OK] saved: {out_md}")
