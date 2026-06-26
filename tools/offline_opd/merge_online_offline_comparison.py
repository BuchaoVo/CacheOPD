from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/mnt/data/users/zbc/opd/ProteinOPD")

ONLINE_CLEAN = ROOT / "analysis_outputs/p0_current_eval/final/p0_current_methods_clean_summary.csv"
ONLINE_FULL = ROOT / "analysis_outputs/p0_current_eval/final/p0_current_methods_full_summary.csv"
OFFLINE = ROOT / "analysis_outputs/offline_proteinopd/p0_eval/final/offline_p0_full_summary.csv"

OUT_DIR = ROOT / "analysis_outputs/offline_proteinopd/p0_eval/final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "online_offline_comparison.csv"
OUT_MD = OUT_DIR / "online_offline_comparison.md"

if not OFFLINE.exists():
    raise FileNotFoundError(f"Missing offline summary: {OFFLINE}")

rows = []

# Existing online/base/released results.
if ONLINE_CLEAN.exists():
    online = pd.read_csv(ONLINE_CLEAN)

    for _, r in online.iterrows():
        rows.append({
            "method": r.get("method"),
            "setting": "Existing",
            "teacher_online_during_training": "Yes" if str(r.get("method")) == "Conditional-OPD" else "No",
            "PPL↓": r.get("PPL↓", np.nan),
            "PPL_median↓": np.nan,
            "pLDDT↑": r.get("pLDDT↑", np.nan),
            "pAE↓": r.get("pAE↓", np.nan),
            "Sol↑": r.get("Sol↑", np.nan),
            "Thermo↑": r.get("Thermo↑", np.nan),
            "3-mer Diversity↑": r.get("3-mer Diversity↑", np.nan),
            "Valid Rate": r.get("Valid Rate", np.nan),
            "Duplicate Rate": r.get("Duplicate Rate", np.nan),
        })

elif ONLINE_FULL.exists():
    online = pd.read_csv(ONLINE_FULL)

    for _, r in online.iterrows():
        rows.append({
            "method": r.get("method"),
            "setting": "Existing",
            "teacher_online_during_training": "Yes" if str(r.get("method")) == "Conditional-OPD" else "No",
            "PPL↓": r.get("ppl_mean", np.nan),
            "PPL_median↓": r.get("ppl_median", np.nan),
            "pLDDT↑": r.get("plddt_mean", np.nan) * 100.0 if pd.notna(r.get("plddt_mean", np.nan)) else np.nan,
            "pAE↓": r.get("pae_mean", np.nan),
            "Sol↑": r.get("sol_score_mean", np.nan),
            "Thermo↑": r.get("thermo_score_mean", np.nan),
            "3-mer Diversity↑": r.get("diversity_score_3mer", np.nan),
            "Valid Rate": r.get("valid_rate", np.nan),
            "Duplicate Rate": r.get("duplicate_rate", np.nan),
        })
else:
    print("[WARN] missing online/current summary. Only offline rows will be written.")

# Offline results.
offline = pd.read_csv(OFFLINE)

for _, r in offline.iterrows():
    rows.append({
        "method": r.get("method"),
        "setting": "Offline",
        "teacher_online_during_training": "No",
        "PPL↓": r.get("ppl_mean", np.nan),
        "PPL_median↓": r.get("ppl_median", np.nan),
        "pLDDT↑": r.get("plddt_mean_100", np.nan),
        "pAE↓": r.get("pae_mean", np.nan),
        "Sol↑": r.get("sol_score_mean", np.nan),
        "Thermo↑": r.get("thermo_score_mean", np.nan),
        "3-mer Diversity↑": r.get("diversity_score_3mer", np.nan),
        "Valid Rate": r.get("valid_rate", np.nan),
        "Duplicate Rate": r.get("duplicate_rate", np.nan),
    })

df = pd.DataFrame(rows)

order = [
    "Base-ProLLaMA",
    "Conditional-OPD",
    "Offline-BaseRollout-K64",
    "Offline-OPDRollout-K64",
    "Released-Adapter",
]
df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
df = df.sort_values("method", na_position="last")

df.to_csv(OUT_CSV, index=False)

view = df.copy()
for c in view.columns:
    if c in ["method", "setting", "teacher_online_during_training"]:
        continue
    view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")

view.to_markdown(index=False)
OUT_MD.write_text(view.to_markdown(index=False), encoding="utf-8")

print(view.to_markdown(index=False))
print(f"\n[OK] saved CSV: {OUT_CSV}")
print(f"[OK] saved MD:  {OUT_MD}")
