#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def read_one(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def first_row(df):
    return df.iloc[0].to_dict() if len(df) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zbc/data/opd/ProteinOPD")
    ap.add_argument("--method", default="FinalUtility-ShiftEntropy-q02-High50")
    ap.add_argument("--out_dir", default="analysis_outputs/offline_proteinopd/final_utility_eval/final")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    basic = first_row(read_one(root / "analysis_outputs/offline_proteinopd/supplemental_analysis/final_utility_basic_sequence_metrics_prollama_ppl.csv"))
    train = first_row(read_one(root / "analysis_outputs/offline_proteinopd/supplemental_analysis/final_utility_downstream_training_start.csv"))
    sol = first_row(read_one(root / "analysis_outputs/offline_proteinopd/final_utility_eval/property/protein_sol/p0_protein_sol_summary.csv"))
    thermo = first_row(read_one(root / "analysis_outputs/offline_proteinopd/final_utility_eval/property/temberture/p0_temberture_summary.csv"))
    structure = first_row(read_one(root / "analysis_outputs/offline_proteinopd/final_utility_eval/structure_hf/p0_hf_esmfold_structure_summary.csv"))

    row = {"method": args.method}
    for prefix, data in [
        ("basic", basic),
        ("train", train),
        ("protein_sol", sol),
        ("temberture", thermo),
        ("structure_hf", structure),
    ]:
        for key, value in data.items():
            if key == "method":
                continue
            row[f"{prefix}_{key}"] = value

    # Paper-facing aliases matching other offline summaries.
    aliases = {
        "valid_rate": row.get("basic_valid_rate"),
        "duplicate_rate": row.get("basic_duplicate_rate"),
        "n_valid": row.get("basic_n_valid"),
        "mean_length": row.get("basic_mean_length"),
        "median_length": row.get("basic_median_length"),
        "min_length": row.get("basic_min_length"),
        "max_length": row.get("basic_max_length"),
        "novelty_t_3mer_proxy_mean": row.get("basic_novelty_t_3mer_proxy_mean"),
        "diversity_score_3mer": row.get("basic_diversity_score_3mer"),
        "prollama_ppl_from_mean_nll": row.get("basic_prollama_ppl_from_mean_nll"),
        "prollama_mean_nll": row.get("basic_prollama_mean_nll"),
        "plddt_mean": row.get("structure_hf_plddt_mean"),
        "pae_mean": row.get("structure_hf_pae_mean"),
        "ptm_mean": row.get("structure_hf_ptm_mean"),
        "sol_score_mean": row.get("protein_sol_sol_score_mean"),
        "thermo_score_mean": row.get("temberture_thermo_score_mean"),
        "thermophilic_rate": row.get("temberture_thermophilic_rate"),
        "effective_tokens": row.get("train_effective_training_tokens"),
        "train_runtime_sec": row.get("train_total_elapsed_sec"),
        "teacher_calls_during_training": row.get("train_teacher_forward_calls_during_training"),
        "final_loss_poe": row.get("train_final_logged_loss_poe"),
        "final_loss_nll": row.get("train_final_logged_loss_nll"),
    }
    row.update(aliases)

    df = pd.DataFrame([row])
    csv_path = out_dir / "final_utility_full_summary.csv"
    md_path = out_dir / "final_utility_full_summary.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df[[
        "method",
        "prollama_ppl_from_mean_nll",
        "plddt_mean",
        "pae_mean",
        "ptm_mean",
        "sol_score_mean",
        "thermo_score_mean",
        "thermophilic_rate",
        "diversity_score_3mer",
        "novelty_t_3mer_proxy_mean",
        "effective_tokens",
        "train_runtime_sec",
    ]].to_string(index=False))
    print(f"[OK] {csv_path}")


if __name__ == "__main__":
    main()
