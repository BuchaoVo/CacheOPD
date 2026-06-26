from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


TEACHERS = ["foldability", "solubility", "thermostability"]
REF = "reference"


def logsumexp(xs: np.ndarray) -> float:
    m = float(np.max(xs))
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.exp(xs - m).sum()))


def normalize_logps(logps: np.ndarray) -> np.ndarray:
    z = logsumexp(logps)
    return np.exp(logps - z)


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-300, 1.0)
    return float(-(p * np.log(p)).sum())


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-300, 1.0)
    q = np.clip(q, 1e-300, 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    xs = sorted(vals)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def load_cache(cache_dir: Path, names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    cache = {}
    for name in names:
        cache[name] = {
            "ids": np.load(cache_dir / f"{name}_topk_ids.npy"),
            "lps": np.load(cache_dir / f"{name}_topk_logprobs.npy").astype(np.float32),
            "target_ids": np.load(cache_dir / f"{name}_target_ids.npy"),
            "valid": np.load(cache_dir / f"{name}_valid_mask.npy"),
        }
    return cache


def dist_on_support(ids: np.ndarray, lps: np.ndarray, support: list[int], missing_logprob: float) -> np.ndarray:
    d = {int(tok): float(lp) for tok, lp in zip(ids, lps) if int(tok) >= 0}
    scores = np.array([d.get(tok, missing_logprob) for tok in support], dtype=np.float64)
    return normalize_logps(scores)


def token_dict(ids: np.ndarray, lps: np.ndarray) -> dict[int, float]:
    return {int(tok): float(lp) for tok, lp in zip(ids, lps) if int(tok) >= 0}


def summarize(vals: list[float]) -> dict[str, float]:
    return {
        "mean": mean(vals),
        "median": quantile(vals, 0.5),
        "p75": quantile(vals, 0.75),
        "p90": quantile(vals, 0.9),
        "p95": quantile(vals, 0.95),
        "max": max(vals) if vals else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="analysis_outputs/offline_proteinopd/cache_diag/base_rollout")
    ap.add_argument("--out_dir", default="analysis_outputs/offline_proteinopd/teacher_distribution_diagnostics")
    ap.add_argument("--missing_logprob", type=float, default=-30.0)
    args = ap.parse_args()

    root = Path.cwd()
    cache_dir = root / args.cache_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(cache_dir, TEACHERS + [REF])
    n, max_t, top_k = cache[TEACHERS[0]]["ids"].shape

    pair_data = {(a, b): {"js": [], "overlap": [], "top1": [], "entropy_diff": []} for i, a in enumerate(TEACHERS) for b in TEACHERS[i + 1:]}
    ref_data = {t: {"js": [], "overlap": [], "top1_agree": [], "entropy": [], "top1_prob": [], "target_prob_delta": []} for t in TEACHERS}
    poe_top1_teacher_top1 = {t: [] for t in TEACHERS}
    token_rows = []

    for i in range(n):
        for pos in range(max_t):
            if not all(bool(cache[name]["valid"][i, pos]) for name in TEACHERS + [REF]):
                continue
            sets = {}
            dicts = {}
            entropies = {}
            top1 = {}
            top1_prob = {}
            target_id = int(cache[REF]["target_ids"][i, pos])

            for name in TEACHERS + [REF]:
                ids = cache[name]["ids"][i, pos]
                lps = cache[name]["lps"][i, pos]
                valid = ids >= 0
                ids_v = ids[valid]
                lps_v = lps[valid]
                sets[name] = set(int(x) for x in ids_v)
                dicts[name] = token_dict(ids_v, lps_v)
                p_topk = normalize_logps(lps_v.astype(np.float64))
                entropies[name] = entropy(p_topk)
                top1[name] = int(ids_v[0]) if len(ids_v) else -1
                top1_prob[name] = float(math.exp(float(lps_v[0]))) if len(lps_v) else float("nan")

            pair_js_vals = []
            pair_ov_vals = []
            for a, b in pair_data:
                support = sorted(sets[a] | sets[b])
                pa = dist_on_support(cache[a]["ids"][i, pos], cache[a]["lps"][i, pos], support, args.missing_logprob)
                pb = dist_on_support(cache[b]["ids"][i, pos], cache[b]["lps"][i, pos], support, args.missing_logprob)
                js = js_div(pa, pb)
                ov = len(sets[a] & sets[b]) / max(min(len(sets[a]), len(sets[b]), top_k), 1)
                pair_data[(a, b)]["js"].append(js)
                pair_data[(a, b)]["overlap"].append(ov)
                pair_data[(a, b)]["top1"].append(float(top1[a] == top1[b]))
                pair_data[(a, b)]["entropy_diff"].append(abs(entropies[a] - entropies[b]))
                pair_js_vals.append(js)
                pair_ov_vals.append(ov)

            ref_js_vals = []
            for t in TEACHERS:
                support = sorted(sets[t] | sets[REF])
                pt = dist_on_support(cache[t]["ids"][i, pos], cache[t]["lps"][i, pos], support, args.missing_logprob)
                pr = dist_on_support(cache[REF]["ids"][i, pos], cache[REF]["lps"][i, pos], support, args.missing_logprob)
                js = js_div(pt, pr)
                ov = len(sets[t] & sets[REF]) / max(min(len(sets[t]), len(sets[REF]), top_k), 1)
                ref_data[t]["js"].append(js)
                ref_data[t]["overlap"].append(ov)
                ref_data[t]["top1_agree"].append(float(top1[t] == top1[REF]))
                ref_data[t]["entropy"].append(entropies[t])
                ref_data[t]["top1_prob"].append(top1_prob[t])
                ref_data[t]["target_prob_delta"].append(
                    math.exp(dicts[t].get(target_id, args.missing_logprob))
                    - math.exp(dicts[REF].get(target_id, args.missing_logprob))
                )
                ref_js_vals.append(js)

            # Approximate PoE top-1 on union teacher support.
            support = sorted(set().union(*(sets[t] for t in TEACHERS)))
            poe_scores = []
            for tok in support:
                poe_scores.append(sum(dicts[t].get(tok, args.missing_logprob) for t in TEACHERS))
            poe_top1 = support[int(np.argmax(np.array(poe_scores)))] if support else -1
            for t in TEACHERS:
                poe_top1_teacher_top1[t].append(float(top1[t] == poe_top1))

            token_rows.append({
                "sample_index": i,
                "position": pos,
                "pairwise_teacher_js_mean": mean(pair_js_vals),
                "pairwise_teacher_js_max": max(pair_js_vals),
                "pairwise_teacher_overlap_mean": mean(pair_ov_vals),
                "teacher_ref_js_mean": mean(ref_js_vals),
                "teacher_ref_js_max": max(ref_js_vals),
                "target_id": target_id,
            })

    pair_rows = []
    for (a, b), data in pair_data.items():
        js_s = summarize(data["js"])
        ov_s = summarize(data["overlap"])
        pair_rows.append({
            "teacher_pair": f"{a}_vs_{b}",
            "n_tokens": len(data["js"]),
            "js_mean": js_s["mean"],
            "js_median": js_s["median"],
            "js_p90": js_s["p90"],
            "js_p95": js_s["p95"],
            "topk_overlap_mean": ov_s["mean"],
            "topk_overlap_median": ov_s["median"],
            "topk_overlap_p10": quantile(data["overlap"], 0.1),
            "top1_agreement": mean(data["top1"]),
            "entropy_absdiff_mean": mean(data["entropy_diff"]),
        })

    ref_rows = []
    for t, data in ref_data.items():
        js_s = summarize(data["js"])
        ov_s = summarize(data["overlap"])
        ref_rows.append({
            "teacher": t,
            "n_tokens": len(data["js"]),
            "entropy_mean": mean(data["entropy"]),
            "entropy_median": quantile(data["entropy"], 0.5),
            "js_to_ref_mean": js_s["mean"],
            "js_to_ref_median": js_s["median"],
            "js_to_ref_p90": js_s["p90"],
            "js_to_ref_p95": js_s["p95"],
            "topk_overlap_with_ref_mean": ov_s["mean"],
            "top1_agreement_with_ref": mean(data["top1_agree"]),
            "top1_prob_mean": mean(data["top1_prob"]),
            "target_prob_delta_vs_ref_mean": mean(data["target_prob_delta"]),
            "poe_top1_equals_teacher_top1_rate": mean(poe_top1_teacher_top1[t]),
        })

    js_values = [float(r["pairwise_teacher_js_mean"]) for r in token_rows]
    ref_values = [float(r["teacher_ref_js_mean"]) for r in token_rows]
    hard_rows = []
    for q in [0.5, 0.75, 0.9, 0.95]:
        js_thr = quantile(js_values, q)
        ref_thr = quantile(ref_values, q)
        hard_js = [r for r in token_rows if float(r["pairwise_teacher_js_mean"]) >= js_thr]
        hard_ref = [r for r in token_rows if float(r["teacher_ref_js_mean"]) >= ref_thr]
        hard_both = [
            r for r in token_rows
            if float(r["pairwise_teacher_js_mean"]) >= js_thr and float(r["teacher_ref_js_mean"]) >= ref_thr
        ]
        hard_rows.append({
            "quantile": q,
            "pairwise_js_threshold": js_thr,
            "teacher_ref_js_threshold": ref_thr,
            "n_high_pairwise_js": len(hard_js),
            "frac_high_pairwise_js": len(hard_js) / max(len(token_rows), 1),
            "n_high_ref_shift": len(hard_ref),
            "frac_high_ref_shift": len(hard_ref) / max(len(token_rows), 1),
            "n_high_both": len(hard_both),
            "frac_high_both": len(hard_both) / max(len(token_rows), 1),
            "high_both_pairwise_js_mean": mean([float(r["pairwise_teacher_js_mean"]) for r in hard_both]),
            "high_both_ref_js_mean": mean([float(r["teacher_ref_js_mean"]) for r in hard_both]),
        })

    write_csv(out_dir / "pairwise_teacher_distribution_summary.csv", pair_rows)
    write_csv(out_dir / "per_teacher_reference_shift_summary.csv", ref_rows)
    write_csv(out_dir / "hard_subset_summary.csv", hard_rows)
    write_csv(out_dir / "teacher_distribution_token_scores.csv", token_rows)

    report_path = root / "paper_md_report/CacheOPD_teacher_distribution_signal_diagnosis.md"
    lines = [
        "# CacheOPD Teacher Distribution Signal Diagnosis",
        "",
        "## Purpose",
        "",
        "This report tests whether the current three-teacher ensemble provides enough token-level preference signal to separate CacheOPD variants. The diagnosis uses cached top-k teacher distributions from the base-rollout cache.",
        "",
        "## Pairwise Teacher Diagnostics",
        "",
        "| Teacher pair | mean JS | median JS | p90 JS | top-k overlap | top-1 agreement | entropy abs diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pair_rows:
        lines.append(
            f"| {r['teacher_pair']} | {float(r['js_mean']):.4f} | {float(r['js_median']):.4f} | "
            f"{float(r['js_p90']):.4f} | {float(r['topk_overlap_mean']):.4f} | "
            f"{float(r['top1_agreement']):.4f} | {float(r['entropy_absdiff_mean']):.4f} |"
        )
    lines += [
        "",
        "## Per-Teacher Shift From Reference",
        "",
        "| Teacher | entropy | JS to Ref | p90 JS to Ref | top-k overlap with Ref | top-1 agreement with Ref | PoE top1 equals teacher top1 | target prob delta vs Ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ref_rows:
        lines.append(
            f"| {r['teacher']} | {float(r['entropy_mean']):.4f} | {float(r['js_to_ref_mean']):.4f} | "
            f"{float(r['js_to_ref_p90']):.4f} | {float(r['topk_overlap_with_ref_mean']):.4f} | "
            f"{float(r['top1_agreement_with_ref']):.4f} | {float(r['poe_top1_equals_teacher_top1_rate']):.4f} | "
            f"{float(r['target_prob_delta_vs_ref_mean']):.5f} |"
        )
    lines += [
        "",
        "## Hard Subset Coverage",
        "",
        "| Quantile | pairwise JS thr | ref-shift JS thr | high pairwise JS | high ref shift | high both | mean JS in high both | mean ref JS in high both |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in hard_rows:
        lines.append(
            f"| {float(r['quantile']):.2f} | {float(r['pairwise_js_threshold']):.4f} | "
            f"{float(r['teacher_ref_js_threshold']):.4f} | {int(r['n_high_pairwise_js'])} "
            f"({float(r['frac_high_pairwise_js']):.3f}) | {int(r['n_high_ref_shift'])} "
            f"({float(r['frac_high_ref_shift']):.3f}) | {int(r['n_high_both'])} "
            f"({float(r['frac_high_both']):.3f}) | {float(r['high_both_pairwise_js_mean']):.4f} | "
            f"{float(r['high_both_ref_js_mean']):.4f} |"
        )

    mean_pair_js = mean([r["js_mean"] for r in pair_rows])
    mean_topk_overlap = mean([r["topk_overlap_mean"] for r in pair_rows])
    mean_ref_js = mean([r["js_to_ref_mean"] for r in ref_rows])
    q90_hard = next(r for r in hard_rows if abs(float(r["quantile"]) - 0.9) < 1e-9)
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Mean pairwise teacher JS is {mean_pair_js:.4f}, while mean top-k overlap is {mean_topk_overlap:.4f}. This confirms that the teachers are highly overlapping at the local token-support level.",
        f"- Mean teacher-to-reference JS is {mean_ref_js:.4f}. This indicates non-zero teacher shift, but the shift is modest relative to the high support overlap.",
        f"- At the 90th percentile hard-subset threshold, only {int(q90_hard['n_high_both'])} tokens ({float(q90_hard['frac_high_both']):.3f}) are simultaneously high-disagreement and high-reference-shift.",
        "- Therefore, the weak downstream separation among q_delta, smoothing, and utility variants is likely not primarily a student-training failure. The teacher ensemble itself provides limited differential signal over most cached positions.",
        "",
        "## Consequence",
        "",
        "The most defensible next step is not to keep sweeping q_delta or smoothing. Instead:",
        "",
        "1. Calibrate or retrain teachers to create stronger objective-specific shifts.",
        "2. Construct a hard subset from high teacher-disagreement and high teacher-reference-shift positions.",
        "3. Add stress tests with skewed teacher weights or single-objective teacher baselines.",
        "4. If teacher recalibration is out of scope, frame the paper as cached PoE efficiency plus sparse non-inferiority rather than a strong teacher-conflict method.",
        "",
        "## Artifacts",
        "",
        f"- `{(out_dir / 'pairwise_teacher_distribution_summary.csv').relative_to(root)}`",
        f"- `{(out_dir / 'per_teacher_reference_shift_summary.csv').relative_to(root)}`",
        f"- `{(out_dir / 'hard_subset_summary.csv').relative_to(root)}`",
        f"- `{(out_dir / 'teacher_distribution_token_scores.csv').relative_to(root)}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {report_path}")
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
