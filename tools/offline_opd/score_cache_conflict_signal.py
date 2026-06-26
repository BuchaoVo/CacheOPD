#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
MISSING_LOGPROB = -30.0


def logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.exp(x - m).sum()))


def softmax_from_logp(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    p = np.exp(x - m)
    return p / np.clip(p.sum(), EPS, None)


def js_divergence(logp_a, logp_b):
    p = softmax_from_logp(logp_a)
    q = softmax_from_logp(logp_b)
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(m, EPS, None))))
        + 0.5 * np.sum(q * (np.log(np.clip(q, EPS, None)) - np.log(np.clip(m, EPS, None))))
    )


def normalize_score(x, valid_mask):
    y = np.asarray(x, dtype=np.float64)
    vals = y[valid_mask]
    vals = vals[np.isfinite(vals)]
    out = np.zeros_like(y, dtype=np.float64)
    if vals.size == 0:
        return out
    lo, hi = np.percentile(vals, [5, 95])
    if hi <= lo:
        return out
    out = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    out[~valid_mask] = 0.0
    return out


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
        raise FileNotFoundError(f"No teacher *_topk_ids.npy files found in {cache_dir}")
    return names


def load_teacher_arrays(cache_dir, teacher_names):
    out = {}
    cache_dir = Path(cache_dir)
    for name in teacher_names:
        ids_path = cache_dir / f"{name}_topk_ids.npy"
        lp_path = cache_dir / f"{name}_topk_logprobs.npy"
        if not ids_path.exists() or not lp_path.exists():
            raise FileNotFoundError(f"Missing cached teacher arrays for {name}")
        out[name] = {
            "ids": np.load(ids_path, mmap_mode="r"),
            "logp": np.load(lp_path, mmap_mode="r"),
        }
    return out


def teacher_dict(ids, logp):
    valid = ids >= 0
    return {int(k): float(v) for k, v in zip(ids[valid], logp[valid].astype(np.float64))}


def normalize_weights(raw_weight, valid_mask):
    w = np.asarray(raw_weight, dtype=np.float64)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    vals = w[valid_mask]
    mean = vals.mean() if vals.size else 0.0
    if mean > 0:
        w = w / mean
    w[~valid_mask] = 0.0
    return w.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--missing_logprob", type=float, default=MISSING_LOGPROB)
    parser.add_argument("--up_lambda", type=float, default=1.0)
    parser.add_argument("--down_lambda", type=float, default=0.8)
    parser.add_argument("--down_min", type=float, default=0.2)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_mask = np.load(cache_dir / "reference_valid_mask.npy").astype(bool)
    teacher_names = infer_teacher_names(cache_dir)
    teachers = load_teacher_arrays(cache_dir, teacher_names)

    n, max_t = valid_mask.shape
    m_teachers = len(teacher_names)

    logz_cross = np.zeros((n, max_t), dtype=np.float64)
    logz_self_mean = np.zeros((n, max_t), dtype=np.float64)
    normalizer_gap = np.zeros((n, max_t), dtype=np.float64)
    pairwise_js = np.zeros((n, max_t), dtype=np.float64)
    pairwise_overlap = np.zeros((n, max_t), dtype=np.float64)
    support_size = np.zeros((n, max_t), dtype=np.float64)

    rows = []
    coords = np.argwhere(valid_mask)

    for i, t in coords:
        dicts = []
        sets = []
        for name in teacher_names:
            d = teacher_dict(teachers[name]["ids"][i, t], teachers[name]["logp"][i, t])
            dicts.append(d)
            sets.append(set(d))

        support = sorted(set().union(*sets))
        if not support:
            continue

        support_size[i, t] = len(support)

        teacher_vecs = []
        for d in dicts:
            teacher_vecs.append(
                np.array([d.get(tok, args.missing_logprob) for tok in support], dtype=np.float64)
            )

        cross_scores = np.sum(np.stack(teacher_vecs, axis=0), axis=0)
        cross_z = logsumexp(cross_scores)
        self_zs = [logsumexp(m_teachers * v) for v in teacher_vecs]
        self_z = float(np.mean(self_zs))
        gap = max(0.0, self_z - cross_z)

        js_vals = []
        ov_vals = []
        for a in range(m_teachers):
            for b in range(a + 1, m_teachers):
                js_vals.append(js_divergence(teacher_vecs[a], teacher_vecs[b]))
                denom = max(min(len(sets[a]), len(sets[b])), 1)
                ov_vals.append(len(sets[a] & sets[b]) / denom)

        mean_js = float(np.mean(js_vals)) if js_vals else 0.0
        mean_ov = float(np.mean(ov_vals)) if ov_vals else 1.0

        logz_cross[i, t] = cross_z
        logz_self_mean[i, t] = self_z
        normalizer_gap[i, t] = gap
        pairwise_js[i, t] = mean_js
        pairwise_overlap[i, t] = mean_ov

        rows.append({
            "sample_index": int(i),
            "position": int(t),
            "logz_cross": float(cross_z),
            "logz_self_mean": float(self_z),
            "normalizer_gap": float(gap),
            "pairwise_teacher_js": mean_js,
            "pairwise_teacher_overlap": mean_ov,
            "teacher_conflict_overlap": 1.0 - mean_ov,
            "support_size": int(len(support)),
        })

    gap_n = normalize_score(normalizer_gap, valid_mask)
    js_n = normalize_score(pairwise_js, valid_mask)
    overlap_conflict_n = normalize_score(1.0 - pairwise_overlap, valid_mask)

    conflict_score = (gap_n + js_n + overlap_conflict_n) / 3.0
    conflict_score[~valid_mask] = 0.0

    weight_up = normalize_weights(1.0 + args.up_lambda * conflict_score, valid_mask)
    weight_down = normalize_weights(
        np.clip(1.0 - args.down_lambda * conflict_score, args.down_min, None),
        valid_mask,
    )

    np.save(out_dir / "poe_logz_cross.npy", logz_cross.astype(np.float32))
    np.save(out_dir / "poe_logz_self_mean.npy", logz_self_mean.astype(np.float32))
    np.save(out_dir / "poe_normalizer_gap.npy", normalizer_gap.astype(np.float32))
    np.save(out_dir / "teacher_pairwise_js.npy", pairwise_js.astype(np.float32))
    np.save(out_dir / "conflict_score.npy", conflict_score.astype(np.float32))
    np.save(out_dir / "token_weight_conflict_up.npy", weight_up)
    np.save(out_dir / "token_weight_conflict_down.npy", weight_down)

    token_df = pd.DataFrame(rows)
    if not token_df.empty:
        token_df["conflict_score"] = [
            float(conflict_score[int(r.sample_index), int(r.position)])
            for r in token_df.itertuples(index=False)
        ]
    token_df.to_csv(out_dir / "conflict_token_scores.csv", index=False)

    summary_rows = []
    metric_arrays = {
        "logz_cross": logz_cross,
        "logz_self_mean": logz_self_mean,
        "normalizer_gap": normalizer_gap,
        "pairwise_teacher_js": pairwise_js,
        "pairwise_teacher_overlap": pairwise_overlap,
        "support_size": support_size,
        "conflict_score": conflict_score,
        "weight_up": weight_up,
        "weight_down": weight_down,
    }
    row = {
        "cache_dir": str(cache_dir),
        "n_valid_tokens": int(valid_mask.sum()),
        "n_teachers": int(m_teachers),
        "teacher_names": ",".join(teacher_names),
    }
    for name, arr in metric_arrays.items():
        vals = arr[valid_mask]
        row[f"{name}_mean"] = float(vals.mean()) if vals.size else math.nan
        row[f"{name}_median"] = float(np.median(vals)) if vals.size else math.nan
        row[f"{name}_p90"] = float(np.percentile(vals, 90)) if vals.size else math.nan
    summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "conflict_signal_summary.csv", index=False)
    (out_dir / "conflict_signal_summary.json").write_text(
        json.dumps(summary_rows[0], indent=2),
        encoding="utf-8",
    )

    print("[OK] conflict signal saved:", out_dir)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
