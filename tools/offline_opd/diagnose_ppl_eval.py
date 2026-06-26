import argparse
import math
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


AA_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")


def parse_kv(s):
    if "=" not in s:
        raise ValueError(f"Expected NAME=PATH, got: {s}")
    k, v = s.split("=", 1)
    return k.strip(), Path(v.strip())


def clean_sequence(x):
    """Extract clean canonical amino-acid sequence only."""
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()

    # Remove common wrappers.
    s = s.replace("\\n", "\n")
    s = re.sub(r">[^\n]*", "", s)
    s = re.sub(r"[^A-Z]", "", s)

    chunks = AA_RE.findall(s)
    if not chunks:
        return ""

    # Use the longest canonical AA run.
    return max(chunks, key=len)


def format_sequence(seq, mode):
    if mode == "raw":
        return seq
    if mode == "spaced":
        return " ".join(list(seq))
    raise ValueError(f"Unknown mode: {mode}")


def find_sequence_col(df):
    candidates = ["sequence", "seq", "protein_sequence", "generated_sequence", "text"]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]

    # fallback: choose first object column with long AA-like strings
    best_col = None
    best_score = -1
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(20).tolist()
        score = 0
        for v in vals:
            seq = clean_sequence(v)
            if len(seq) >= 30:
                score += 1
        if score > best_score:
            best_col = c
            best_score = score

    if best_col is None or best_score <= 0:
        raise ValueError(f"Cannot find sequence column. Columns: {list(df.columns)}")
    return best_col


@torch.no_grad()
def score_batch(model, tokenizer, texts, device, max_length=1024):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    # Causal LM: predict token t from tokens < t.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().bool()

    vocab = shift_logits.size(-1)
    losses = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())

    losses = losses * shift_mask
    token_counts = shift_mask.sum(dim=1).clamp(min=1)
    nll = losses.sum(dim=1) / token_counts

    return nll.detach().float().cpu().tolist(), token_counts.detach().cpu().tolist()


def safe_exp(x):
    if x > 80:
        return float("inf")
    return math.exp(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--model", action="append", required=True, help="TAG=PATH")
    ap.add_argument("--modes", nargs="+", default=["raw", "spaced"], choices=["raw", "spaced"])
    ap.add_argument("--max_seqs", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--root", default="/home/zbc/data/opd/ProteinOPD")
    ap.add_argument("--out_dir", default="analysis_outputs/offline_proteinopd/ppl_diagnostic")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cohorts = [parse_kv(x) for x in args.cohort]
    models = [parse_kv(x) for x in args.model]

    # Load cohorts once.
    cohort_data = {}
    for name, path in cohorts:
        path = path if path.is_absolute() else root / path
        if not path.exists():
            print(f"[WARN] Missing cohort: {name} -> {path}")
            continue

        df = pd.read_csv(path)
        seq_col = find_sequence_col(df)
        seqs = []
        for x in df[seq_col].tolist():
            seq = clean_sequence(x)
            if len(seq) >= 20:
                seqs.append(seq)

        seqs = seqs[: args.max_seqs]
        cohort_data[name] = {
            "path": str(path),
            "seq_col": seq_col,
            "seqs": seqs,
        }
        print(f"[COHORT] {name}: {len(seqs)} seqs, seq_col={seq_col}, path={path}")

    if not cohort_data:
        raise RuntimeError("No valid cohorts found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else None

    all_rows = []
    example_lines = []

    for model_tag, model_path in models:
        model_path = model_path if model_path.is_absolute() else root / model_path
        if not model_path.exists():
            print(f"[WARN] Missing model: {model_tag} -> {model_path}")
            continue

        print(f"\n[LOAD MODEL] {model_tag}: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {"trust_remote_code": True}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
        model.to(device)
        model.eval()

        for cohort_name, info in cohort_data.items():
            seqs = info["seqs"]

            for mode in args.modes:
                texts = [format_sequence(s, mode) for s in seqs]

                # Tokenization examples.
                example_lines.append(f"\n=== model={model_tag} cohort={cohort_name} mode={mode} ===")
                for i, text in enumerate(texts[:3]):
                    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
                    toks = tokenizer.convert_ids_to_tokens(ids[:80])
                    example_lines.append(f"[example {i}] AA_len={len(seqs[i])}, token_len={len(ids)}")
                    example_lines.append(f"seq: {seqs[i][:120]}")
                    example_lines.append(f"text: {text[:160]}")
                    example_lines.append(f"tokens[:80]: {toks}")

                detailed = []
                for st in range(0, len(texts), args.batch_size):
                    batch_texts = texts[st : st + args.batch_size]
                    batch_seqs = seqs[st : st + args.batch_size]
                    nlls, token_counts = score_batch(
                        model,
                        tokenizer,
                        batch_texts,
                        device=device,
                        max_length=args.max_length,
                    )

                    for j, (seq, nll, ntok) in enumerate(zip(batch_seqs, nlls, token_counts)):
                        detailed.append({
                            "model": model_tag,
                            "cohort": cohort_name,
                            "mode": mode,
                            "idx": st + j,
                            "aa_len": len(seq),
                            "pred_token_count": int(ntok),
                            "nll": float(nll),
                            "ppl": safe_exp(float(nll)),
                            "sequence": seq,
                        })

                det_df = pd.DataFrame(detailed)
                all_rows.append(det_df)

                mean_nll = det_df["nll"].mean()
                median_nll = det_df["nll"].median()

                print(
                    f"[RESULT] model={model_tag:12s} cohort={cohort_name:20s} mode={mode:6s} "
                    f"mean_nll={mean_nll:.4f} ppl_from_mean_nll={safe_exp(mean_nll):.2f} "
                    f"median_ppl={det_df['ppl'].median():.2f}"
                )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    detailed_all = pd.concat(all_rows, ignore_index=True)

    summary = (
        detailed_all
        .groupby(["model", "cohort", "mode"], as_index=False)
        .agg(
            n=("ppl", "size"),
            aa_len_mean=("aa_len", "mean"),
            pred_token_count_mean=("pred_token_count", "mean"),
            mean_nll=("nll", "mean"),
            median_nll=("nll", "median"),
            ppl_individual_mean=("ppl", "mean"),
            ppl_individual_median=("ppl", "median"),
        )
    )
    summary["ppl_from_mean_nll"] = summary["mean_nll"].map(safe_exp)
    summary["ppl_from_median_nll"] = summary["median_nll"].map(safe_exp)

    detail_path = out_dir / "ppl_diagnostic_detailed.csv"
    summary_path = out_dir / "ppl_diagnostic_summary.csv"
    example_path = out_dir / "tokenization_examples.txt"

    detailed_all.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    example_path.write_text("\n".join(example_lines), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    print(f"\n[OK] detailed: {detail_path}")
    print(f"[OK] summary:  {summary_path}")
    print(f"[OK] examples: {example_path}")


if __name__ == "__main__":
    main()
