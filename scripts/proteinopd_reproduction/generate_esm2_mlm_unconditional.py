from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


AA = "ACDEFGHIKLMNPQRSTVWY"
NATURAL_AA_FREQ = {
    "A": 0.0825,
    "C": 0.0137,
    "D": 0.0545,
    "E": 0.0675,
    "F": 0.0386,
    "G": 0.0707,
    "H": 0.0227,
    "I": 0.0596,
    "K": 0.0584,
    "L": 0.0966,
    "M": 0.0242,
    "N": 0.0406,
    "P": 0.0470,
    "Q": 0.0393,
    "R": 0.0553,
    "S": 0.0656,
    "T": 0.0534,
    "V": 0.0687,
    "W": 0.0108,
    "Y": 0.0292,
}


def sample_lengths(ref_csv: Path, n: int, seed: int, min_len: int, max_len: int) -> list[int]:
    rng = np.random.default_rng(seed)
    if ref_csv.exists():
        df = pd.read_csv(ref_csv)
        if "length" in df.columns:
            vals = pd.to_numeric(df["length"], errors="coerce").dropna().astype(int).to_numpy()
        elif "sequence" in df.columns:
            vals = df["sequence"].dropna().astype(str).str.len().to_numpy()
        else:
            vals = np.array([], dtype=int)
        vals = vals[(vals >= min_len) & (vals <= max_len)]
        if vals.size:
            return rng.choice(vals, size=n, replace=True).astype(int).tolist()
    return rng.integers(min_len, max_len + 1, size=n).astype(int).tolist()


def top_k_filter(probs: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= probs.shape[-1]:
        return probs
    vals, idx = torch.topk(probs, k=k, dim=-1)
    out = torch.zeros_like(probs)
    out.scatter_(-1, idx, vals)
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


def aa_prior_tensor(aa_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    vals = torch.tensor([NATURAL_AA_FREQ[aa] for aa in AA], dtype=torch.float32, device=device)
    vals = vals / vals.sum()
    return vals


@torch.no_grad()
def generate_batch(
    model,
    alphabet,
    lengths: list[int],
    aa_ids: torch.Tensor,
    device: torch.device,
    max_iter: int,
    temperature: float,
    top_k: int,
) -> list[str]:
    batch = len(lengths)
    max_len = max(lengths)
    tokens = torch.full((batch, max_len + 2), alphabet.padding_idx, dtype=torch.long, device=device)
    tokens[:, 0] = alphabet.cls_idx
    for i, length in enumerate(lengths):
        tokens[i, 1 : length + 1] = alphabet.mask_idx
        tokens[i, length + 1] = alphabet.eos_idx

    for step in range(max_iter):
        remaining = tokens.eq(alphabet.mask_idx)
        if not bool(remaining.any()):
            break

        out = model(tokens, repr_layers=[], need_head_weights=False)
        logits = out["logits"][..., aa_ids] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        probs = top_k_filter(probs, top_k)

        for i, length in enumerate(lengths):
            pos = torch.nonzero(tokens[i, 1 : length + 1].eq(alphabet.mask_idx), as_tuple=False).flatten() + 1
            if pos.numel() == 0:
                continue
            pos_probs = probs[i, pos]
            sampled_local = torch.multinomial(pos_probs, num_samples=1).squeeze(-1)
            sampled_ids = aa_ids[sampled_local]
            confidence = pos_probs.gather(1, sampled_local[:, None]).squeeze(-1)
            n_fill = max(1, math.ceil(pos.numel() / max(max_iter - step, 1)))
            chosen = torch.topk(confidence, k=min(n_fill, pos.numel())).indices
            tokens[i, pos[chosen]] = sampled_ids[chosen]

    # Fill any residual masks in one final pass.
    remaining = tokens.eq(alphabet.mask_idx)
    if bool(remaining.any()):
        out = model(tokens, repr_layers=[], need_head_weights=False)
        logits = out["logits"][..., aa_ids] / max(temperature, 1e-6)
        probs = top_k_filter(torch.softmax(logits, dim=-1), top_k)
        for i, length in enumerate(lengths):
            pos = torch.nonzero(tokens[i, 1 : length + 1].eq(alphabet.mask_idx), as_tuple=False).flatten() + 1
            if pos.numel() == 0:
                continue
            sampled_local = torch.multinomial(probs[i, pos], num_samples=1).squeeze(-1)
            tokens[i, pos] = aa_ids[sampled_local]

    idx_to_aa = {alphabet.get_idx(aa): aa for aa in AA}
    seqs = []
    for i, length in enumerate(lengths):
        ids = tokens[i, 1 : length + 1].detach().cpu().tolist()
        seqs.append("".join(idx_to_aa.get(x, "A") for x in ids))
    return seqs


@torch.no_grad()
def generate_gibbs_batch(
    model,
    alphabet,
    lengths: list[int],
    aa_ids: torch.Tensor,
    prior: torch.Tensor,
    device: torch.device,
    gibbs_steps: int,
    mask_rate: float,
    temperature: float,
    top_k: int,
    prior_mix: float,
) -> list[str]:
    batch = len(lengths)
    max_len = max(lengths)
    tokens = torch.full((batch, max_len + 2), alphabet.padding_idx, dtype=torch.long, device=device)
    tokens[:, 0] = alphabet.cls_idx
    for i, length in enumerate(lengths):
        init = torch.multinomial(prior, num_samples=length, replacement=True)
        tokens[i, 1 : length + 1] = aa_ids[init]
        tokens[i, length + 1] = alphabet.eos_idx

    for _ in range(gibbs_steps):
        masked_positions: list[torch.Tensor] = []
        original_rows = []
        for i, length in enumerate(lengths):
            n_mask = max(1, round(length * mask_rate))
            pos = torch.randperm(length, device=device)[:n_mask] + 1
            masked_positions.append(pos)
            original_rows.append(tokens[i, pos].clone())
            tokens[i, pos] = alphabet.mask_idx

        out = model(tokens, repr_layers=[], need_head_weights=False)
        logits = out["logits"][..., aa_ids] / max(temperature, 1e-6)
        probs = top_k_filter(torch.softmax(logits, dim=-1), top_k)
        if prior_mix > 0:
            probs = (1.0 - prior_mix) * probs + prior_mix * prior.view(1, 1, -1)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        for i, pos in enumerate(masked_positions):
            sampled_local = torch.multinomial(probs[i, pos], num_samples=1).squeeze(-1)
            tokens[i, pos] = aa_ids[sampled_local]

    idx_to_aa = {alphabet.get_idx(aa): aa for aa in AA}
    seqs = []
    for i, length in enumerate(lengths):
        ids = tokens[i, 1 : length + 1].detach().cpu().tolist()
        seqs.append("".join(idx_to_aa.get(x, "A") for x in ids))
    return seqs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", default="analysis_outputs/proteinopd_reproduction/eval_cohorts/unconditional_esm2_mlm/ESM2-35M-MLM.csv")
    parser.add_argument("--ref_length_csv", default="analysis_outputs/proteinopd_reproduction/eval_cohorts/unconditional/ProtGPT2.csv")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_len", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_iter", type=int, default=12)
    parser.add_argument("--decoding", choices=["confidence", "gibbs"], default="gibbs")
    parser.add_argument("--gibbs_steps", type=int, default=64)
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--prior_mix", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--torch_home", default="/home/zbc/data/models/torch_cache")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ["TORCH_HOME"] = args.torch_home
    import esm

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.eval().to(device)

    aa_ids = torch.tensor([alphabet.get_idx(aa) for aa in AA], dtype=torch.long, device=device)
    prior = aa_prior_tensor(aa_ids, device)
    lengths = sample_lengths(Path(args.ref_length_csv), args.n, args.seed, args.min_len, args.max_len)

    rows = []
    for start in tqdm(range(0, args.n, args.batch_size), desc="ESM2 MLM generation"):
        batch_lengths = lengths[start : start + args.batch_size]
        if args.decoding == "confidence":
            seqs = generate_batch(
                model=model,
                alphabet=alphabet,
                lengths=batch_lengths,
                aa_ids=aa_ids,
                device=device,
                max_iter=args.max_iter,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            source = "fair-esm esm2_t12_35M_UR50D iterative masked decoding"
        else:
            seqs = generate_gibbs_batch(
                model=model,
                alphabet=alphabet,
                lengths=batch_lengths,
                aa_ids=aa_ids,
                prior=prior,
                device=device,
                gibbs_steps=args.gibbs_steps,
                mask_rate=args.mask_rate,
                temperature=args.temperature,
                top_k=args.top_k,
                prior_mix=args.prior_mix,
            )
            source = "fair-esm esm2_t12_35M_UR50D pseudo-Gibbs MLM decoding"
        for j, seq in enumerate(seqs):
            idx = start + j
            rows.append(
                {
                    "setting": "unconditional",
                    "method": "ESM2-35M-MLM",
                    "seq_id": f"ESM2-35M-MLM_{idx:05d}",
                    "sequence": seq,
                    "length": len(seq),
                    "source": source,
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {len(rows)} sequences -> {out_csv}")


if __name__ == "__main__":
    main()
