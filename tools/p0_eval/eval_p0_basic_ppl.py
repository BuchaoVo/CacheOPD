import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_aa(seq):
    seq = str(seq).upper()
    return len(seq) > 0 and set(seq).issubset(AA)

def kmer_set(seq, k=3):
    seq = str(seq).upper()
    if len(seq) < k:
        return set()
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}

def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)

def mean_pairwise_jaccard(seqs, k=3, max_pairs=10000):
    seqs = [s for s in seqs if len(s) >= k]
    if len(seqs) < 2:
        return 0.0

    sets = [kmer_set(s, k) for s in seqs]
    vals = []
    count = 0

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            vals.append(jaccard(sets[i], sets[j]))
            count += 1
            if count >= max_pairs:
                return float(np.mean(vals))

    return float(np.mean(vals)) if vals else 0.0

def load_reference_sets(ref_csvs):
    refs = []
    for fp in ref_csvs:
        p = Path(fp)
        if not p.exists():
            print(f"[WARN] missing ref_csv, skip: {p}")
            continue

        df = pd.read_csv(p)
        if "sequence" not in df.columns:
            print(f"[WARN] no sequence column in ref_csv, skip: {p}")
            continue

        refs.extend(df["sequence"].dropna().astype(str).str.upper().tolist())

    return [kmer_set(s, 3) for s in refs if len(s) >= 3]

def max_ref_jaccard(seq, ref_sets):
    s = kmer_set(seq, 3)
    if not s or not ref_sets:
        return np.nan
    return max(jaccard(s, r) for r in ref_sets)

def score_ppl(model, tokenizer, seq, device, max_length):
    eos = tokenizer.eos_token_id
    ids = tokenizer(seq, add_special_tokens=False)["input_ids"]
    ids = [eos] + ids + [eos]
    ids = ids[:max_length]

    if len(ids) < 2:
        return np.nan, np.nan

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = input_ids.clone()

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)

    loss = float(out.loss.item())
    ppl = float(np.exp(min(loss, 20)))
    return loss, ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--protgpt2", default="/home/zbc/data/models/ProtGPT2")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--ref_csv", action="append", default=[])
    args = parser.parse_args()

    cohort_dir = Path(args.cohort_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([
        p for p in cohort_dir.glob("*.csv")
        if "manifest" not in p.stem.lower() and "summary" not in p.stem.lower()
    ])
    if not files:
        raise FileNotFoundError(f"No cohort CSV files found in {cohort_dir}")

    ref_sets = load_reference_sets(args.ref_csv)
    print(f"[INFO] loaded reference 3-mer sets: {len(ref_sets)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.protgpt2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.protgpt2)
    model.to(device)
    model.eval()

    all_summaries = []

    for fp in files:
        df = pd.read_csv(fp)
        if "method" not in df.columns:
            df["method"] = fp.stem
        if "seq_id" not in df.columns:
            df["seq_id"] = [f"{fp.stem}_{i:05d}" for i in range(len(df))]

        method = str(df["method"].iloc[0])
        print(f"\n[RUN] {method}: {len(df)} sequences")

        df["sequence"] = df["sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
        df["length"] = df["sequence"].str.len()
        df["is_valid_aa"] = df["sequence"].apply(is_valid_aa)

        valid = df[df["is_valid_aa"]].copy()
        valid["is_duplicate"] = valid.duplicated("sequence", keep=False)

        if len(valid) == 0:
            print(f"[WARN] no valid sequences for {method}")
            continue

        valid["max_ref_3mer_jaccard"] = [
            max_ref_jaccard(seq, ref_sets) for seq in valid["sequence"]
        ]
        valid["novelty_t_3mer_proxy"] = 1.0 - valid["max_ref_3mer_jaccard"]

        losses, ppls = [], []
        for seq in tqdm(valid["sequence"].tolist(), desc=f"PPL {method}"):
            loss, ppl = score_ppl(model, tokenizer, seq, device, args.max_length)
            losses.append(loss)
            ppls.append(ppl)

        valid["ppl_loss"] = losses
        valid["ppl"] = ppls

        diversity_jaccard = mean_pairwise_jaccard(valid["sequence"].tolist(), k=3)
        diversity_score = 1.0 - diversity_jaccard

        summary = {
            "method": method,
            "n_total": len(df),
            "n_valid": len(valid),
            "valid_rate": len(valid) / max(len(df), 1),
            "unique_valid": valid["sequence"].nunique(),
            "duplicate_rate": 1.0 - valid["sequence"].nunique() / max(len(valid), 1),
            "mean_length": valid["length"].mean(),
            "median_length": valid["length"].median(),
            "min_length": valid["length"].min(),
            "max_length": valid["length"].max(),
            "ppl_mean": valid["ppl"].mean(),
            "ppl_median": valid["ppl"].median(),
            "ppl_loss_mean": valid["ppl_loss"].mean(),
            "novelty_t_3mer_proxy_mean": valid["novelty_t_3mer_proxy"].mean(),
            "max_ref_3mer_jaccard_mean": valid["max_ref_3mer_jaccard"].mean(),
            "mean_pairwise_3mer_jaccard": diversity_jaccard,
            "diversity_score_3mer": diversity_score,
        }

        per_seq_path = out_dir / f"{method}_per_sequence_basic_ppl.csv"
        valid.to_csv(per_seq_path, index=False)
        print(f"[OK] per-sequence -> {per_seq_path}")

        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = out_dir / "p0_basic_ppl_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n[SUMMARY]")
    print(summary_df.to_string(index=False))
    print(f"[OK] summary -> {summary_path}")

if __name__ == "__main__":
    main()
