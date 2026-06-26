import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(x):
    return "".join(str(x).split()).upper()


def is_valid_aa(seq):
    return len(seq) > 0 and set(seq).issubset(AA)


def parse_teacher_specs(specs):
    out = []
    for s in specs:
        if ":" not in s:
            raise ValueError(f"teacher spec must be name:path, got {s}")
        name, path = s.split(":", 1)
        out.append((name, path))
    return out


def get_dtype(name):
    name = str(name).lower()
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def load_tokenizer(base_model):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<pad>"})
    return tok


def load_model(base_model, adapter_path=None, dtype=torch.float16, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.to(device)
    return model


def tokenize_sequence(tokenizer, seq, max_length):
    enc = tokenizer(
        seq,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"][0]


def score_topk_for_model(
    model,
    tokenizer,
    sequences,
    seq_ids,
    top_k,
    max_length,
    device,
    out_dir,
    prefix,
):
    """
    Compute top-k next-token log-probs for each sequence.
    Distribution position t predicts input_ids[t+1] from prefix input_ids[:t+1].
    """
    tokenized = []
    max_t = 0

    for seq in sequences:
        ids = tokenize_sequence(tokenizer, seq, max_length=max_length)
        # Need at least two tokens for next-token target.
        if ids.numel() < 2:
            tokenized.append(ids)
            continue
        max_t = max(max_t, ids.numel() - 1)
        tokenized.append(ids)

    n = len(tokenized)
    topk_ids = np.full((n, max_t, top_k), -1, dtype=np.int32)
    topk_logprobs = np.full((n, max_t, top_k), -1e4, dtype=np.float16)
    target_ids = np.full((n, max_t), -1, dtype=np.int32)
    valid_mask = np.zeros((n, max_t), dtype=bool)
    lengths = np.zeros(n, dtype=np.int32)

    t0 = time.time()

    for i, ids in enumerate(tqdm(tokenized, desc=f"Scoring {prefix}")):
        if ids.numel() < 2:
            continue

        ids = ids.to(device)
        inp = ids[:-1].unsqueeze(0)
        tgt = ids[1:]

        with torch.no_grad():
            out = model(input_ids=inp)
            logits = out.logits[0]  # [T, V]
            logp = torch.log_softmax(logits.float(), dim=-1)
            vals, inds = torch.topk(logp, k=top_k, dim=-1)

        T = vals.shape[0]
        topk_ids[i, :T, :] = inds.detach().cpu().numpy().astype(np.int32)
        topk_logprobs[i, :T, :] = vals.detach().cpu().numpy().astype(np.float16)
        target_ids[i, :T] = tgt.detach().cpu().numpy().astype(np.int32)
        valid_mask[i, :T] = True
        lengths[i] = int(ids.numel())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{prefix}_topk_ids.npy", topk_ids)
    np.save(out_dir / f"{prefix}_topk_logprobs.npy", topk_logprobs)
    np.save(out_dir / f"{prefix}_target_ids.npy", target_ids)
    np.save(out_dir / f"{prefix}_valid_mask.npy", valid_mask)
    np.save(out_dir / f"{prefix}_lengths.npy", lengths)

    meta = {
        "prefix": prefix,
        "n": n,
        "max_t": int(max_t),
        "top_k": int(top_k),
        "runtime_sec": time.time() - t0,
    }
    (out_dir / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "topk_ids": topk_ids,
        "topk_logprobs": topk_logprobs,
        "target_ids": target_ids,
        "valid_mask": valid_mask,
        "lengths": lengths,
    }


def logsumexp_np(x):
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(x - m))))


def build_poe_and_diagnostics(cache, teacher_names, ref_name, top_k, out_dir, missing_logprob=-30.0):
    """
    Build approximate PoE target on union support of teacher top-k tokens.
    Save poe_topk_ids/logprobs and diagnostics.
    """
    first = cache[teacher_names[0]]
    n, max_t, _ = first["topk_ids"].shape

    poe_topk_ids = np.full((n, max_t, top_k), -1, dtype=np.int32)
    poe_topk_logprobs = np.full((n, max_t, top_k), -1e4, dtype=np.float16)

    rows = []

    for i in tqdm(range(n), desc="Building PoE diagnostics"):
        for t in range(max_t):
            if not first["valid_mask"][i, t]:
                continue

            teacher_sets = []
            teacher_lps = []

            for name in teacher_names:
                ids = cache[name]["topk_ids"][i, t]
                lps = cache[name]["topk_logprobs"][i, t].astype(np.float32)

                valid = ids >= 0
                ids = ids[valid]
                lps = lps[valid]

                d = {int(a): float(b) for a, b in zip(ids, lps)}
                teacher_lps.append(d)
                teacher_sets.append(set(d.keys()))

            support = sorted(set().union(*teacher_sets))
            if not support:
                continue

            # Teacher pairwise top-k overlap
            pair_ovs = []
            for a in range(len(teacher_sets)):
                for b in range(a + 1, len(teacher_sets)):
                    denom = max(min(len(teacher_sets[a]), len(teacher_sets[b]), top_k), 1)
                    pair_ovs.append(len(teacher_sets[a] & teacher_sets[b]) / denom)
            teacher_pairwise_overlap = float(np.mean(pair_ovs)) if pair_ovs else np.nan

            # Approximate PoE logit = sum teacher logprob on support.
            scores = []
            for tok in support:
                s = 0.0
                for d in teacher_lps:
                    s += d.get(tok, missing_logprob)
                scores.append(s)

            scores = np.array(scores, dtype=np.float32)
            z = logsumexp_np(scores)
            poe_lp = scores - z

            order = np.argsort(-poe_lp)
            support_arr = np.array(support, dtype=np.int32)

            k_eff = min(top_k, len(order))
            top_order = order[:k_eff]
            poe_ids = support_arr[top_order]
            poe_lps = poe_lp[top_order]

            poe_topk_ids[i, t, :k_eff] = poe_ids
            poe_topk_logprobs[i, t, :k_eff] = poe_lps.astype(np.float16)

            q = np.exp(poe_lp)
            poe_entropy = float(-(q * poe_lp).sum())

            # Reference-PoE overlap
            ref_overlap = np.nan
            if ref_name in cache:
                ref_ids = cache[ref_name]["topk_ids"][i, t]
                ref_ids = set([int(x) for x in ref_ids if x >= 0])
                poe_set = set([int(x) for x in poe_ids])
                ref_overlap = len(ref_ids & poe_set) / max(min(len(ref_ids), len(poe_set), top_k), 1)

            target_id = int(first["target_ids"][i, t])
            target_in_poe_topk = int(target_id in set([int(x) for x in poe_ids]))
            target_in_any_teacher = int(any(target_id in s for s in teacher_sets))

            rows.append({
                "sample_index": i,
                "position": t,
                "teacher_pairwise_overlap": teacher_pairwise_overlap,
                "ref_poe_overlap": ref_overlap,
                "poe_entropy": poe_entropy,
                "support_size": len(support),
                "target_in_poe_topk": target_in_poe_topk,
                "target_in_any_teacher_topk": target_in_any_teacher,
            })

    out_dir = Path(out_dir)
    np.save(out_dir / "poe_topk_ids.npy", poe_topk_ids)
    np.save(out_dir / "poe_topk_logprobs.npy", poe_topk_logprobs)

    diag = pd.DataFrame(rows)
    diag.to_csv(out_dir / "offline_cache_token_diagnostics.csv", index=False)

    summary = {
        "n_tokens": int(len(diag)),
        "teacher_pairwise_overlap_mean": float(diag["teacher_pairwise_overlap"].mean()),
        "ref_poe_overlap_mean": float(diag["ref_poe_overlap"].mean()),
        "poe_entropy_mean": float(diag["poe_entropy"].mean()),
        "support_size_mean": float(diag["support_size"].mean()),
        "target_in_poe_topk_rate": float(diag["target_in_poe_topk"].mean()),
        "target_in_any_teacher_topk_rate": float(diag["target_in_any_teacher_topk"].mean()),
    }

    pd.DataFrame([summary]).to_csv(out_dir / "offline_cache_summary.csv", index=False)
    (out_dir / "offline_cache_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n[Offline cache summary]")
    print(pd.DataFrame([summary]).to_string(index=False))

    return diag, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts_csv", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--teacher", action="append", required=True, help="name:adapter_path")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--include_reference", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teachers = parse_teacher_specs(args.teacher)
    dtype = get_dtype(args.dtype)

    df = pd.read_csv(args.rollouts_csv)
    if "sequence" not in df.columns:
        raise ValueError(f"rollouts_csv must contain sequence column. Columns={list(df.columns)}")

    if "seq_id" not in df.columns:
        df["seq_id"] = [f"sample_{i:05d}" for i in range(len(df))]

    df["sequence"] = df["sequence"].apply(clean_seq)
    df = df[df["sequence"].apply(is_valid_aa)].copy()
    df = df.head(args.max_samples).copy()

    rollouts_path = out_dir / "rollouts_used.csv"
    df.to_csv(rollouts_path, index=False)
    print(f"[INFO] rollouts used: {len(df)} -> {rollouts_path}")

    seqs = df["sequence"].tolist()
    seq_ids = df["seq_id"].astype(str).tolist()

    tokenizer = load_tokenizer(args.base_model)

    cache = {}

    if args.include_reference:
        print("\n[INFO] scoring reference/base model")
        model = load_model(args.base_model, adapter_path=None, dtype=dtype, device=args.device)
        cache["reference"] = score_topk_for_model(
            model=model,
            tokenizer=tokenizer,
            sequences=seqs,
            seq_ids=seq_ids,
            top_k=args.top_k,
            max_length=args.max_length,
            device=args.device,
            out_dir=out_dir,
            prefix="reference",
        )
        del model
        torch.cuda.empty_cache()

    teacher_names = []
    for name, adapter_path in teachers:
        print(f"\n[INFO] scoring teacher: {name}")
        print(f"[INFO] adapter: {adapter_path}")

        model = load_model(args.base_model, adapter_path=adapter_path, dtype=dtype, device=args.device)
        cache[name] = score_topk_for_model(
            model=model,
            tokenizer=tokenizer,
            sequences=seqs,
            seq_ids=seq_ids,
            top_k=args.top_k,
            max_length=args.max_length,
            device=args.device,
            out_dir=out_dir,
            prefix=name,
        )
        teacher_names.append(name)

        del model
        torch.cuda.empty_cache()

    build_poe_and_diagnostics(
        cache=cache,
        teacher_names=teacher_names,
        ref_name="reference",
        top_k=args.top_k,
        out_dir=out_dir,
    )

    manifest = {
        "rollouts_csv": args.rollouts_csv,
        "base_model": args.base_model,
        "teachers": [{"name": n, "adapter_path": p} for n, p in teachers],
        "top_k": args.top_k,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "include_reference": args.include_reference,
    }
    (out_dir / "offline_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print(f"\n[OK] offline cache diagnostic saved to: {out_dir}")


if __name__ == "__main__":
    main()
