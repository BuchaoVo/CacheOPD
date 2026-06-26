import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(x - m)))


def normalize_logprobs(lp):
    lp = np.asarray(lp, dtype=np.float64)
    return lp - logsumexp(lp)


def kl_from_logprobs(logq, logr):
    q = np.exp(logq)
    return float(np.sum(q * (logq - logr)))


def js_from_logprobs(logp, logq):
    p = np.exp(logp)
    q = np.exp(logq)
    m = 0.5 * (p + q)
    logm = np.log(np.clip(m, 1e-300, None))
    return 0.5 * float(np.sum(p * (logp - logm))) + 0.5 * float(np.sum(q * (logq - logm)))


def lookup_logprob(ids, lps, query_ids, missing=-30.0):
    d = {}
    for tid, lp in zip(ids, lps):
        if tid >= 0:
            d[int(tid)] = float(lp)
    return np.array([d.get(int(q), missing) for q in query_ids], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--missing_logprob", type=float, default=-30.0)
    args = ap.parse_args()

    root = Path(args.cache_dir)

    poe_ids = np.load(root / "poe_topk_ids.npy")
    poe_lps = np.load(root / "poe_topk_logprobs.npy")

    ref_ids = np.load(root / "reference_topk_ids.npy")
    ref_lps = np.load(root / "reference_topk_logprobs.npy")
    ref_targets = np.load(root / "reference_target_ids.npy")
    ref_mask = np.load(root / "reference_valid_mask.npy")

    teacher_names = ["foldability", "solubility", "thermostability"]
    teacher_ids = {}
    teacher_lps = {}
    for name in teacher_names:
        teacher_ids[name] = np.load(root / f"{name}_topk_ids.npy")
        teacher_lps[name] = np.load(root / f"{name}_topk_logprobs.npy")

    rows = []
    n, T, K = poe_ids.shape

    for i in range(n):
        for t in range(T):
            if not ref_mask[i, t]:
                continue

            ids = poe_ids[i, t]
            lps = poe_lps[i, t].astype(np.float64)
            valid = ids >= 0
            ids = ids[valid]
            lps = lps[valid]

            if len(ids) == 0:
                continue

            logq = normalize_logprobs(lps)

            ref_on_poe = lookup_logprob(
                ref_ids[i, t],
                ref_lps[i, t],
                ids,
                missing=args.missing_logprob,
            )
            logr = normalize_logprobs(ref_on_poe)

            kl_q_ref = kl_from_logprobs(logq, logr)
            js_q_ref = js_from_logprobs(logq, logr)

            teacher_matrix = []
            for name in teacher_names:
                teacher_on_poe = lookup_logprob(
                    teacher_ids[name][i, t],
                    teacher_lps[name][i, t],
                    ids,
                    missing=args.missing_logprob,
                )
                teacher_matrix.append(teacher_on_poe)
            teacher_matrix = np.vstack(teacher_matrix)
            teacher_logprob_std = float(np.mean(np.std(teacher_matrix, axis=0)))

            target_id = int(ref_targets[i, t])
            target_poe_prob = 0.0
            target_ref_prob = 0.0

            for tok, lp in zip(ids, logq):
                if int(tok) == target_id:
                    target_poe_prob = float(np.exp(lp))
                    break

            ref_ids_t = ref_ids[i, t]
            ref_lps_t = ref_lps[i, t].astype(np.float64)
            valid_ref = ref_ids_t >= 0
            ref_ids_valid = ref_ids_t[valid_ref]
            ref_lps_valid = normalize_logprobs(ref_lps_t[valid_ref])
            for tok, lp in zip(ref_ids_valid, ref_lps_valid):
                if int(tok) == target_id:
                    target_ref_prob = float(np.exp(lp))
                    break

            rows.append({
                "sample_index": i,
                "position": t,
                "kl_poe_ref": kl_q_ref,
                "js_poe_ref": js_q_ref,
                "teacher_logprob_std_on_poe": teacher_logprob_std,
                "target_poe_prob": target_poe_prob,
                "target_ref_prob": target_ref_prob,
                "target_prob_delta_poe_minus_ref": target_poe_prob - target_ref_prob,
            })

    df = pd.DataFrame(rows)

    summary = pd.DataFrame([{
        "n_tokens": len(df),
        "kl_poe_ref_mean": df["kl_poe_ref"].mean(),
        "kl_poe_ref_median": df["kl_poe_ref"].median(),
        "js_poe_ref_mean": df["js_poe_ref"].mean(),
        "js_poe_ref_median": df["js_poe_ref"].median(),
        "teacher_logprob_std_on_poe_mean": df["teacher_logprob_std_on_poe"].mean(),
        "target_poe_prob_mean": df["target_poe_prob"].mean(),
        "target_ref_prob_mean": df["target_ref_prob"].mean(),
        "target_prob_delta_poe_minus_ref_mean": df["target_prob_delta_poe_minus_ref"].mean(),
    }])

    out_csv = Path(args.out_csv) if args.out_csv else root / "offline_cache_distribution_diagnostics.csv"
    out_summary = out_csv.with_name(out_csv.stem + "_summary.csv")

    df.to_csv(out_csv, index=False)
    summary.to_csv(out_summary, index=False)

    print(summary.to_string(index=False))
    print(f"\n[OK] token diagnostics: {out_csv}")
    print(f"[OK] summary:           {out_summary}")


if __name__ == "__main__":
    main()
