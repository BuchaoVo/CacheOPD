#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


MISSING_LOGPROB = -30.0


def logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.exp(x - m).sum()))


def infer_teacher_names(cache_dir):
    manifest_path = Path(cache_dir) / "offline_cache_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        names = [str(x["name"]) for x in manifest.get("teachers", []) if "name" in x]
        if names:
            return names

    names = []
    for path in sorted(Path(cache_dir).glob("*_topk_ids.npy")):
        name = path.name.removesuffix("_topk_ids.npy")
        if name not in {"reference", "poe"}:
            names.append(name)
    if not names:
        raise FileNotFoundError(f"No teacher arrays found in {cache_dir}")
    return names


def symlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        if src.suffix == ".npy":
            np.save(dst, np.load(src))
        else:
            dst.write_bytes(src.read_bytes())


def teacher_dict(ids, logp):
    valid = ids >= 0
    return {int(k): float(v) for k, v in zip(ids[valid], logp[valid].astype(np.float64))}


def softmax_from_log_scores(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return np.exp(scores - logsumexp(scores))


def pairwise_js_conflict(teacher_vecs):
    dists = [softmax_from_log_scores(v) for v in teacher_vecs]
    vals = []
    for a in range(len(dists)):
        for b in range(a + 1, len(dists)):
            m = 0.5 * (dists[a] + dists[b])
            ha = -float((dists[a] * np.log(np.clip(dists[a], 1e-300, None))).sum())
            hb = -float((dists[b] * np.log(np.clip(dists[b], 1e-300, None))).sum())
            hm = -float((m * np.log(np.clip(m, 1e-300, None))).sum())
            vals.append((hm - 0.5 * (ha + hb)) / np.log(2.0))
    return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0


def build_scores(teacher_vecs, mode, smoothing_alpha=0.5):
    mat = np.stack(teacher_vecs, axis=0)
    if mode == "prob_avg":
        probs = np.exp(mat)
        probs = probs.mean(axis=0)
        return np.log(np.clip(probs, 1e-300, None))
    if mode == "logit_avg":
        return mat.mean(axis=0)
    if mode == "raw_poe_sum":
        return mat.sum(axis=0)
    if mode == "conflict_smooth":
        poe = softmax_from_log_scores(mat.sum(axis=0))
        avg = softmax_from_log_scores(np.log(np.clip(np.exp(mat).mean(axis=0), 1e-300, None)))
        lam = float(np.clip(smoothing_alpha * pairwise_js_conflict(teacher_vecs), 0.0, 1.0))
        probs = (1.0 - lam) * poe + lam * avg
        return np.log(np.clip(probs, 1e-300, None))
    raise ValueError(f"Unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--mode",
        required=True,
        choices=["prob_avg", "logit_avg", "raw_poe_sum", "conflict_smooth"],
    )
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--missing_logprob", type=float, default=MISSING_LOGPROB)
    ap.add_argument("--smoothing_alpha", type=float, default=0.5)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_names = infer_teacher_names(cache_dir)
    teacher_arrays = {}
    for name in teacher_names:
        teacher_arrays[name] = {
            "ids": np.load(cache_dir / f"{name}_topk_ids.npy", mmap_mode="r"),
            "logp": np.load(cache_dir / f"{name}_topk_logprobs.npy", mmap_mode="r"),
            "valid": np.load(cache_dir / f"{name}_valid_mask.npy", mmap_mode="r"),
        }

    first = teacher_arrays[teacher_names[0]]
    n, max_t, _ = first["ids"].shape
    out_ids = np.full((n, max_t, args.top_k), -1, dtype=np.int32)
    out_lps = np.full((n, max_t, args.top_k), -1e4, dtype=np.float16)

    rows = []
    for i in tqdm(range(n), desc=f"Building {args.mode}"):
        for t in range(max_t):
            if not bool(first["valid"][i, t]):
                continue

            dicts = []
            sets = []
            for name in teacher_names:
                d = teacher_dict(
                    teacher_arrays[name]["ids"][i, t],
                    teacher_arrays[name]["logp"][i, t],
                )
                dicts.append(d)
                sets.append(set(d))

            support = sorted(set().union(*sets))
            if not support:
                continue

            teacher_vecs = [
                np.array([d.get(tok, args.missing_logprob) for tok in support], dtype=np.float64)
                for d in dicts
            ]
            conflict_js = pairwise_js_conflict(teacher_vecs)
            scores = build_scores(teacher_vecs, args.mode, smoothing_alpha=args.smoothing_alpha)
            lp = scores - logsumexp(scores)

            order = np.argsort(-lp)
            support_arr = np.array(support, dtype=np.int32)
            k_eff = min(args.top_k, len(order))
            top_order = order[:k_eff]
            ids = support_arr[top_order]
            lps = lp[top_order]
            out_ids[i, t, :k_eff] = ids
            out_lps[i, t, :k_eff] = lps.astype(np.float16)

            q = np.exp(lp)
            pair_ovs = []
            for a in range(len(sets)):
                for b in range(a + 1, len(sets)):
                    denom = max(min(len(sets[a]), len(sets[b]), args.top_k), 1)
                    pair_ovs.append(len(sets[a] & sets[b]) / denom)

            rows.append({
                "sample_index": i,
                "position": t,
                "mode": args.mode,
                "support_size": len(support),
                "target_entropy": float(-(q * lp).sum()),
                "teacher_pairwise_js_conflict": conflict_js,
                "smoothing_lambda": float(np.clip(args.smoothing_alpha * conflict_js, 0.0, 1.0)) if args.mode == "conflict_smooth" else 0.0,
                "teacher_pairwise_overlap": float(np.mean(pair_ovs)) if pair_ovs else np.nan,
            })

    np.save(out_dir / "poe_topk_ids.npy", out_ids)
    np.save(out_dir / "poe_topk_logprobs.npy", out_lps)

    for name in [
        "rollouts_used.csv",
        "reference_target_ids.npy",
        "reference_valid_mask.npy",
        "reference_lengths.npy",
        "offline_cache_manifest.json",
        "offline_cache_summary.csv",
        "offline_cache_summary.json",
    ]:
        src = cache_dir / name
        if src.exists():
            symlink_or_copy(src, out_dir / name)

    diag = pd.DataFrame(rows)
    diag.to_csv(out_dir / "fusion_target_diagnostics.csv", index=False)
    summary = {
        "source_cache": str(cache_dir),
        "mode": args.mode,
        "n_tokens": int(len(diag)),
        "top_k": int(args.top_k),
        "smoothing_alpha": float(args.smoothing_alpha),
        "teacher_names": ",".join(teacher_names),
        "target_entropy_mean": float(diag["target_entropy"].mean()) if len(diag) else np.nan,
        "target_entropy_median": float(diag["target_entropy"].median()) if len(diag) else np.nan,
        "teacher_pairwise_js_conflict_mean": float(diag["teacher_pairwise_js_conflict"].mean()) if len(diag) else np.nan,
        "smoothing_lambda_mean": float(diag["smoothing_lambda"].mean()) if len(diag) else np.nan,
        "support_size_mean": float(diag["support_size"].mean()) if len(diag) else np.nan,
        "teacher_pairwise_overlap_mean": float(diag["teacher_pairwise_overlap"].mean()) if len(diag) else np.nan,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "fusion_target_summary.csv", index=False)
    (out_dir / "fusion_target_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] wrote fusion cache:", out_dir)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
