import re
from pathlib import Path

import pandas as pd

ROOT = Path("/home/zbc/data/opd/ProteinOPD")
OUT_DIR = ROOT / "analysis_outputs/p0_current_eval/cohorts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AA = set("ACDEFGHIKLMNPQRSTVWY")

SOURCES = [
    {
        "method": "Base-ProLLaMA",
        "path": ROOT / "analysis_outputs/conditional_qc/base_4superfamily_qc.csv",
    },
    {
        "method": "Released-Adapter",
        "path": ROOT / "analysis_outputs/conditional_qc/merged_4superfamily_qc.csv",
    },
    {
        "method": "Conditional-OPD",
        "path": ROOT / "analysis_outputs/conditional_qc/opd_4superfamily_qc.csv",
    },
    {
        "method": "ProteinOPD-uncond",
        "path": ROOT / "analysis_outputs/current_method_eval/sequences/proteinopd_uncond.csv",
    },
]

def clean_seq(x):
    x = str(x)
    m = re.search(r"Seq=<([^>]*)>", x)
    if m:
        x = m.group(1)
    return "".join(x.split()).upper()

def is_valid(seq):
    return len(seq) > 0 and set(seq).issubset(AA)

def to_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])

def main():
    manifest = []

    for src in SOURCES:
        method = src["method"]
        path = src["path"]

        if not path.exists():
            print(f"[SKIP] missing: {method} -> {path}")
            manifest.append({
                "method": method,
                "source": str(path),
                "status": "missing",
                "n": 0,
            })
            continue

        df = pd.read_csv(path)
        if "sequence" not in df.columns:
            print(f"[SKIP] no sequence column: {method} -> {path}")
            manifest.append({
                "method": method,
                "source": str(path),
                "status": "no_sequence_column",
                "n": 0,
            })
            continue

        df["sequence"] = df["sequence"].apply(clean_seq)

        if "is_valid_aa" in df.columns:
            df = df[to_bool_series(df["is_valid_aa"])].copy()
        else:
            df = df[df["sequence"].apply(is_valid)].copy()

        df = df[df["sequence"].str.len() > 0].copy()
        df = df.drop_duplicates("sequence").copy()

        rows = []
        for i, seq in enumerate(df["sequence"].tolist()[:256]):
            rows.append({
                "method": method,
                "seq_id": f"{method}_{i:05d}",
                "sequence": seq,
            })

        out_path = OUT_DIR / f"{method}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)

        print(f"[OK] {method}: {len(rows)} sequences -> {out_path}")

        manifest.append({
            "method": method,
            "source": str(path),
            "status": "ok",
            "n": len(rows),
            "out_csv": str(out_path),
        })

    manifest_path = OUT_DIR / "p0_cohort_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"[OK] manifest -> {manifest_path}")

if __name__ == "__main__":
    main()
