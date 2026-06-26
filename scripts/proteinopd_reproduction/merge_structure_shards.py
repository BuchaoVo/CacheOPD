import argparse
from pathlib import Path

import pandas as pd


def combine(method, cohort_csv, source_files, out_csv):
    order = pd.read_csv(cohort_csv)["seq_id"].astype(str).tolist()
    frames = []
    for source in source_files:
        path = Path(source)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No shard files found for {method}")

    df = pd.concat(frames, ignore_index=True)
    df["seq_id"] = df["seq_id"].astype(str)
    df = df.drop_duplicates("seq_id", keep="last")

    rank = {seq_id: i for i, seq_id in enumerate(order)}
    missing = [seq_id for seq_id in order if seq_id not in set(df["seq_id"])]
    df["_rank"] = df["seq_id"].map(rank)
    df = df[df["_rank"].notna()].sort_values("_rank").drop(columns=["_rank"])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    missing_plddt = int(pd.to_numeric(df.get("plddt"), errors="coerce").isna().sum())
    print(
        f"{method}: rows={len(df)} expected={len(order)} "
        f"missing_ids={len(missing)} missing_plddt={missing_plddt}"
    )
    if missing:
        print(f"{method}: missing sample={missing[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="analysis_outputs/proteinopd_reproduction")
    args = parser.parse_args()

    root = Path(args.out_root)
    combine(
        "CacheOPD-Uncond-Full",
        root / "eval_cohorts/unconditional/CacheOPD-Uncond-Full.csv",
        [
            root / "raw_eval/structure_unconditional/CacheOPD-Uncond-Full_hf_esmfold_structure_scores.csv",
            root / "raw_eval/structure_unconditional_full_tail_a/CacheOPD-Uncond-Full_hf_esmfold_structure_scores.csv",
            root / "raw_eval/structure_unconditional_full_tail_b/CacheOPD-Uncond-Full_hf_esmfold_structure_scores.csv",
        ],
        root / "raw_eval/structure_unconditional/CacheOPD-Uncond-Full_hf_esmfold_structure_scores.csv",
    )
    combine(
        "CacheOPD-Uncond-Sparse",
        root / "eval_cohorts/unconditional/CacheOPD-Uncond-Sparse.csv",
        [
            root / "raw_eval/structure_unconditional/CacheOPD-Uncond-Sparse_hf_esmfold_structure_scores.csv",
            root / "raw_eval/structure_unconditional_sparse_mid_shard/CacheOPD-Uncond-Sparse_hf_esmfold_structure_scores.csv",
            root / "raw_eval/structure_unconditional_sparse_tail_shard/CacheOPD-Uncond-Sparse_hf_esmfold_structure_scores.csv",
        ],
        root / "raw_eval/structure_unconditional/CacheOPD-Uncond-Sparse_hf_esmfold_structure_scores.csv",
    )


if __name__ == "__main__":
    main()
