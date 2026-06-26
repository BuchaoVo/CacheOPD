from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import OUT_ROOT, ensure_dirs


SPECS = [
    {
        "src_subdir": "raw_eval/ppl",
        "glob": "*_per_sequence_basic_ppl.csv",
        "dst": "eval/ppl/per_sequence_ppl.csv",
        "columns": ["method", "seq_id", "sequence", "length", "ppl", "ppl_loss"],
        "rename": {"seq_id": "sequence_id"},
    },
    {
        "src_subdir": "raw_eval/structure_hf",
        "glob": "*_hf_esmfold_structure_scores.csv",
        "dst": "eval/structure/per_sequence_structure.csv",
        "columns": ["method", "seq_id", "sequence", "length", "plddt", "pae", "ptm", "status"],
        "rename": {"seq_id": "sequence_id"},
    },
    {
        "src_subdir": "raw_eval/protein_sol",
        "glob": "*_protein_sol_scores.csv",
        "dst": "eval/solubility/per_sequence_solubility.csv",
        "columns": ["method", "seq_id", "sequence", "length", "sol_score", "status"],
        "rename": {"seq_id": "sequence_id"},
    },
    {
        "src_subdir": "raw_eval/temberture",
        "glob": "*_temberture_scores.csv",
        "dst": "eval/thermostability/per_sequence_thermostability.csv",
        "columns": ["method", "seq_id", "sequence", "length", "thermo_score", "thermo_label", "status"],
        "rename": {"seq_id": "sequence_id"},
    },
]


def collect_one(out_root: Path, spec: dict) -> tuple[Path, int]:
    src_dir = out_root / spec["src_subdir"]
    frames = []
    for path in sorted(src_dir.glob(spec["glob"])):
        df = pd.read_csv(path)
        if "method" not in df.columns:
            method = path.name
            for suffix in [
                "_per_sequence_basic_ppl.csv",
                "_hf_esmfold_structure_scores.csv",
                "_protein_sol_scores.csv",
                "_temberture_scores.csv",
            ]:
                method = method.removesuffix(suffix)
            df["method"] = method
        if "seq_id" not in df.columns and "sequence_id" in df.columns:
            df["seq_id"] = df["sequence_id"]
        keep = [c for c in spec["columns"] if c in df.columns]
        if keep:
            frames.append(df[keep].copy())

    out_path = out_root / spec["dst"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.rename(columns=spec["rename"])
        out.to_csv(out_path, index=False)
        return out_path, len(out)
    pd.DataFrame().to_csv(out_path, index=False)
    return out_path, 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect per-method evaluator outputs into CacheOPD supplement per-sequence metric files.")
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)
    rows = []
    for spec in SPECS:
        out_path, n = collect_one(out_root, spec)
        rows.append({"metric_file": str(out_path), "rows": n})
        print(f"[OK] {out_path} rows={n}")
    pd.DataFrame(rows).to_csv(out_root / "summaries" / "unified_eval_collection_status.csv", index=False)


if __name__ == "__main__":
    main()
