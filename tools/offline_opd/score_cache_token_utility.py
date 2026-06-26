#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
MISSING_LOGPROB = -30.0


def load_optional(path):
    path = Path(path)
    if path.exists():
        return np.load(path, mmap_mode="r")
    return None


def normalize_logprobs(logp, mask=None):
    x = np.asarray(logp, dtype=np.float64)
    if mask is not None:
        x = np.where(mask, x, -1e9)
    m = np.max(x, axis=-1, keepdims=True)
    p = np.exp(x - m)
    if mask is not None:
        p = np.where(mask, p, 0.0)
    p = p / np.clip(p.sum(axis=-1, keepdims=True), EPS, None)
    return p


def entropy_from_logprobs(logp, mask=None):
    p = normalize_logprobs(logp, mask=mask)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis=-1)


def normalize_score(x, valid_mask, larger_is_better=True):
    y = np.asarray(x, dtype=np.float64).copy()
    vals = y[valid_mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        z = np.zeros_like(y, dtype=np.float64)
    else:
        lo, hi = np.percentile(vals, [5, 95])
        if hi <= lo:
            z = np.zeros_like(y, dtype=np.float64)
        else:
            z = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    if not larger_is_better:
        z = 1.0 - z
    z[~valid_mask] = 0.0
    return z


def token_js_and_support(
    poe_ids,
    poe_lps,
    ref_ids,
    ref_lps,
    valid_mask,
    missing_logprob=MISSING_LOGPROB,
):
    n, max_t, _ = poe_ids.shape
    js = np.zeros((n, max_t), dtype=np.float64)
    support_mass = np.zeros((n, max_t), dtype=np.float64)
    overlap = np.zeros((n, max_t), dtype=np.float64)

    for i, t in np.argwhere(valid_mask):
        poe_tok = poe_ids[i, t]
        poe_lp = poe_lps[i, t].astype(np.float64)
        ref_tok = ref_ids[i, t]
        ref_lp = ref_lps[i, t].astype(np.float64)

        poe_valid = poe_tok >= 0
        ref_valid = ref_tok >= 0
        if not poe_valid.any() or not ref_valid.any():
            continue

        poe_dict = {int(k): float(v) for k, v in zip(poe_tok[poe_valid], poe_lp[poe_valid])}
        ref_dict = {int(k): float(v) for k, v in zip(ref_tok[ref_valid], ref_lp[ref_valid])}
        support = sorted(set(poe_dict) | set(ref_dict))
        if not support:
            continue

        poe_vec = np.array([poe_dict.get(tok, missing_logprob) for tok in support], dtype=np.float64)
        ref_vec = np.array([ref_dict.get(tok, missing_logprob) for tok in support], dtype=np.float64)
        p = normalize_logprobs(poe_vec)
        q = normalize_logprobs(ref_vec)
        m = 0.5 * (p + q)
        js[i, t] = 0.5 * np.sum(p * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(m, EPS, None))))
        js[i, t] += 0.5 * np.sum(q * (np.log(np.clip(q, EPS, None)) - np.log(np.clip(m, EPS, None))))

        ref_set = set(ref_dict)
        poe_set = set(poe_dict)
        support_mass[i, t] = float(sum(prob for tok, prob in zip(support, p) if tok in ref_set))
        overlap[i, t] = len(ref_set & poe_set) / max(min(len(ref_set), len(poe_set)), 1)

    return js, support_mass, overlap


def read_diagnostics(cache_dir, valid_mask):
    diag_path = Path(cache_dir) / "offline_cache_token_diagnostics.csv"
    out = {
        "teacher_pairwise_overlap": np.ones_like(valid_mask, dtype=np.float64),
        "diag_ref_poe_overlap": np.zeros_like(valid_mask, dtype=np.float64),
        "diag_available": False,
    }
    if not diag_path.exists():
        return out

    diag = pd.read_csv(diag_path)
    if diag.empty:
        return out

    seq_col = None
    for candidate in ["sample_index", "sequence_id", "seq_id", "sample_id", "i"]:
        if candidate in diag.columns:
            seq_col = candidate
            break
    pos_col = None
    for candidate in ["position", "pos", "token_position", "t"]:
        if candidate in diag.columns:
            pos_col = candidate
            break
    if seq_col is None or pos_col is None:
        return out

    out["diag_available"] = True
    for _, row in diag.iterrows():
        try:
            i = int(row[seq_col])
            t = int(row[pos_col])
        except (TypeError, ValueError):
            continue
        if i < 0 or i >= valid_mask.shape[0] or t < 0 or t >= valid_mask.shape[1]:
            continue
        if "teacher_pairwise_overlap" in diag.columns and pd.notna(row["teacher_pairwise_overlap"]):
            out["teacher_pairwise_overlap"][i, t] = float(row["teacher_pairwise_overlap"])
        if "ref_poe_overlap" in diag.columns and pd.notna(row["ref_poe_overlap"]):
            out["diag_ref_poe_overlap"][i, t] = float(row["ref_poe_overlap"])
    return out


def mask_name(prefix, keep_ratio):
    pct = int(round(keep_ratio * 100))
    return f"mask_{prefix}{pct}.npy"


def select_masks(utility, valid_mask, keep_ratio, seed):
    valid_indices = np.argwhere(valid_mask)
    valid_scores = utility[valid_mask]
    n_valid = len(valid_scores)
    n_keep = int(round(keep_ratio * n_valid))
    if n_valid <= 0:
        raise ValueError("No valid tokens found.")
    if n_keep <= 0 or n_keep > n_valid:
        raise ValueError(f"keep_ratio selects invalid token count: keep_ratio={keep_ratio}, n_keep={n_keep}, n_valid={n_valid}")

    order = np.argsort(valid_scores)
    low_sel_flat = order[:n_keep]
    high_sel_flat = order[-n_keep:]

    rng = np.random.default_rng(seed)
    random_sel_flat = rng.choice(n_valid, size=n_keep, replace=False)

    masks = {}
    for name, flat in [("high", high_sel_flat), ("low", low_sel_flat), ("random", random_sel_flat)]:
        mask = np.zeros_like(valid_mask, dtype=bool)
        coords = valid_indices[flat]
        mask[coords[:, 0], coords[:, 1]] = True
        masks[name] = mask
    return masks, n_valid, n_keep


def summarize_group(name, mask, utility, metrics):
    row = {"selection": name, "n_tokens": int(mask.sum())}
    for key, value in metrics.items():
        vals = value[mask]
        row[f"{key}_mean"] = float(vals.mean()) if vals.size else math.nan
        row[f"{key}_median"] = float(np.median(vals)) if vals.size else math.nan
    utility_vals = utility[mask]
    utility_vals = utility_vals[np.isfinite(utility_vals) & (utility_vals > -1e8)]
    row["utility_mean"] = float(utility_vals.mean()) if utility_vals.size else math.nan
    row["utility_median"] = float(np.median(utility_vals)) if utility_vals.size else math.nan
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--keep_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--utility_mode", choices=["legacy", "shift_entropy"], default="legacy")
    parser.add_argument(
        "--support_gate_metric",
        choices=["ref_poe_overlap", "support_mass", "max_overlap_mass"],
        default="ref_poe_overlap",
    )
    parser.add_argument("--support_gate_quantile", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=1.0)

    parser.add_argument("--alpha_shift", type=float, default=1.0)
    parser.add_argument("--beta_support", type=float, default=0.8)
    parser.add_argument("--beta_target", type=float, default=0.4)
    parser.add_argument("--lambda_conflict", type=float, default=0.8)
    parser.add_argument("--mu_collapse", type=float, default=0.2)

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_mask = np.load(cache_dir / "reference_valid_mask.npy").astype(bool)
    target_ids = np.load(cache_dir / "reference_target_ids.npy")
    poe_ids = np.load(cache_dir / "poe_topk_ids.npy", mmap_mode="r")
    poe_logp = np.load(cache_dir / "poe_topk_logprobs.npy", mmap_mode="r")
    ref_ids = np.load(cache_dir / "reference_topk_ids.npy", mmap_mode="r")
    ref_logp = np.load(cache_dir / "reference_topk_logprobs.npy", mmap_mode="r")

    if poe_ids.shape[:2] != valid_mask.shape or ref_ids.shape[:2] != valid_mask.shape:
        raise ValueError("Cache arrays have incompatible leading dimensions.")

    target_in_poe = (poe_ids == target_ids[..., None]).any(axis=-1).astype(np.float64)
    target_in_poe[~valid_mask] = 0.0

    preference_shift, support_mass, computed_overlap = token_js_and_support(
        poe_ids=poe_ids,
        poe_lps=poe_logp,
        ref_ids=ref_ids,
        ref_lps=ref_logp,
        valid_mask=valid_mask,
    )

    poe_entropy = entropy_from_logprobs(poe_logp, mask=poe_ids >= 0)
    poe_entropy[~valid_mask] = 0.0
    valid_entropy = poe_entropy[valid_mask]
    low_ent_thr = np.percentile(valid_entropy, 10) if len(valid_entropy) else 0.0
    collapse_penalty = ((poe_entropy < low_ent_thr) & valid_mask).astype(np.float64)

    diag = read_diagnostics(cache_dir, valid_mask)
    teacher_overlap = np.clip(diag["teacher_pairwise_overlap"], 0.0, 1.0)
    teacher_conflict = 1.0 - teacher_overlap
    support_learnability = np.maximum(support_mass, computed_overlap)
    if diag["diag_available"]:
        support_learnability = np.maximum(support_learnability, np.clip(diag["diag_ref_poe_overlap"], 0.0, 1.0))
    support_learnability[~valid_mask] = 0.0

    shift_n = normalize_score(preference_shift, valid_mask, True)
    support_n = normalize_score(support_learnability, valid_mask, True)
    target_n = normalize_score(target_in_poe, valid_mask, True)
    conflict_n = normalize_score(teacher_conflict, valid_mask, True)
    collapse_n = normalize_score(collapse_penalty, valid_mask, True)

    selection_mask = valid_mask.copy()
    support_gate_threshold = math.nan
    if args.utility_mode == "shift_entropy":
        if args.support_gate_metric == "ref_poe_overlap":
            support_gate_score = computed_overlap
        elif args.support_gate_metric == "support_mass":
            support_gate_score = support_mass
        elif args.support_gate_metric == "max_overlap_mass":
            support_gate_score = np.maximum(computed_overlap, support_mass)
        else:
            raise ValueError(f"Unknown support_gate_metric: {args.support_gate_metric}")

        vals = support_gate_score[valid_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise ValueError("No finite support-gate scores found.")
        q = min(max(args.support_gate_quantile, 0.0), 1.0)
        support_gate_threshold = float(np.quantile(vals, q))
        selection_mask = valid_mask & (support_gate_score >= support_gate_threshold)

        shift_n = normalize_score(preference_shift, selection_mask, True)
        entropy_n = normalize_score(poe_entropy, selection_mask, True)
        utility = args.alpha_shift * shift_n + args.entropy_weight * entropy_n
        utility[~selection_mask] = -1e9
    else:
        support_gate_score = computed_overlap
        entropy_n = normalize_score(poe_entropy, valid_mask, True)
        utility = (
            args.alpha_shift * shift_n
            + args.beta_support * support_n
            + args.beta_target * target_n
            - args.lambda_conflict * conflict_n
            - args.mu_collapse * collapse_n
        )
        utility[~valid_mask] = -1e9

    masks, n_valid, n_keep = select_masks(
        utility=utility,
        valid_mask=selection_mask,
        keep_ratio=args.keep_ratio,
        seed=args.seed,
    )

    np.save(out_dir / "token_utility.npy", utility)
    np.save(out_dir / "mask_support_gate.npy", selection_mask)
    for name, mask in masks.items():
        np.save(out_dir / mask_name(name, args.keep_ratio), mask)

    # Stable aliases for training scripts.
    np.save(out_dir / "mask_high.npy", masks["high"])
    np.save(out_dir / "mask_low.npy", masks["low"])
    np.save(out_dir / "mask_random.npy", masks["random"])

    rows = []
    selected_lookup = {name: mask for name, mask in masks.items()}
    for i, t in np.argwhere(valid_mask):
        row = {
            "sample_index": int(i),
            "position": int(t),
            "utility": float(utility[i, t]),
            "preference_shift_js": float(preference_shift[i, t]),
            "support_learnability": float(support_learnability[i, t]),
            "support_mass_on_reference_topk": float(support_mass[i, t]),
            "ref_poe_overlap": float(computed_overlap[i, t]),
            "target_in_poe_topk": float(target_in_poe[i, t]),
            "teacher_pairwise_overlap": float(teacher_overlap[i, t]),
            "teacher_conflict": float(teacher_conflict[i, t]),
            "poe_entropy": float(poe_entropy[i, t]),
            "poe_entropy_normalized": float(entropy_n[i, t]),
            "support_gate_score": float(support_gate_score[i, t]),
            "support_gate_valid": bool(selection_mask[i, t]),
            "collapse_penalty": float(collapse_penalty[i, t]),
        }
        for name, mask in selected_lookup.items():
            row[f"selected_{name}"] = bool(mask[i, t])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "token_utility_scores.csv", index=False)

    metrics = {
        "preference_shift_js": preference_shift,
        "support_learnability": support_learnability,
        "support_mass_on_reference_topk": support_mass,
        "ref_poe_overlap": computed_overlap,
        "target_in_poe_topk": target_in_poe,
        "teacher_conflict": teacher_conflict,
        "teacher_pairwise_overlap": teacher_overlap,
        "poe_entropy": poe_entropy,
        "poe_entropy_normalized": entropy_n,
        "support_gate_score": support_gate_score,
        "support_gate_valid": selection_mask.astype(np.float64),
        "collapse_penalty": collapse_penalty,
    }
    summary_rows = [summarize_group("all_valid", valid_mask, utility, metrics)]
    if args.utility_mode == "shift_entropy":
        summary_rows.append(summarize_group("support_gate", selection_mask, utility, metrics))
    summary_rows.extend([
        summarize_group("high", masks["high"], utility, metrics),
        summarize_group("random", masks["random"], utility, metrics),
        summarize_group("low", masks["low"], utility, metrics),
    ])
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "selection_summary.csv", index=False)

    summary = {
        "cache_dir": str(cache_dir),
        "out_dir": str(out_dir),
        "keep_ratio": args.keep_ratio,
        "seed": args.seed,
        "utility_mode": args.utility_mode,
        "support_gate_metric": args.support_gate_metric,
        "support_gate_quantile": args.support_gate_quantile,
        "support_gate_threshold": support_gate_threshold,
        "n_reference_valid_tokens": int(valid_mask.sum()),
        "n_valid_tokens": int(n_valid),
        "n_keep_tokens": int(n_keep),
        "weights": {
            "alpha_shift": args.alpha_shift,
            "entropy_weight": args.entropy_weight,
            "beta_support": args.beta_support,
            "beta_target": args.beta_target,
            "lambda_conflict": args.lambda_conflict,
            "mu_collapse": args.mu_collapse,
        },
        "mask_files": {
            name: str(out_dir / mask_name(name, args.keep_ratio))
            for name in ["high", "random", "low"]
        },
        "mask_aliases": {
            name: str(out_dir / f"mask_{name}.npy")
            for name in ["high", "random", "low"]
        },
        "summary": summary_rows,
    }
    with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
