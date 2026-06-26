from __future__ import annotations

import csv
import glob
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "analysis_outputs/offline_proteinopd/supplemental_analysis/metric_discrimination_diagnostics.csv"
OUT_REPORT = ROOT / "paper_md_report/CacheOPD_metric_discrimination_diagnosis.md"


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fval(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sd(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def metric_summary(group: str, values_by_method: dict[str, list[float]], direction: str) -> list[dict[str, object]]:
    means = {m: mean(v) for m, v in values_by_method.items() if v}
    sds = {m: sd(v) for m, v in values_by_method.items() if v}
    if not means:
        return []
    between_range = max(means.values()) - min(means.values())
    avg_within_sd = mean(list(sds.values()))
    ratio = between_range / avg_within_sd if avg_within_sd > 0 else math.nan
    rows = []
    for m in sorted(means):
        rows.append({
            "group": group,
            "method": m,
            "direction": direction,
            "n": len(values_by_method[m]),
            "mean": means[m],
            "sd": sds[m],
            "between_method_range": between_range,
            "avg_within_method_sd": avg_within_sd,
            "range_to_within_sd_ratio": ratio,
        })
    return rows


def load_metric_from_files(pattern: str, column: str, scale: float = 1.0) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for fp in glob.glob(str(ROOT / pattern)):
        rows = read_rows(fp)
        if not rows:
            continue
        method = rows[0].get("method") or Path(fp).name.split("_")[0]
        vals = []
        for r in rows:
            if r.get("status", "ok") != "ok":
                continue
            x = fval(r.get(column, ""))
            if x is not None:
                vals.append(x * scale)
        out[method] = vals
    return out


def main() -> None:
    all_rows: list[dict[str, object]] = []

    # Main-table spread across conditional methods.
    table = read_rows(ROOT / "paper_figures_cacheopd/tables/table1_main_quality_cost_filled.csv")
    cond = [r for r in table if r["Block"] != "External Unconditional Reference Baselines"]
    for metric, direction in [
        ("PPL", "lower"),
        ("pLDDT", "higher"),
        ("pAE", "lower"),
        ("pTM", "higher"),
        ("Sol", "higher"),
        ("Thermo", "higher"),
    ]:
        vals = {}
        for r in cond:
            x = fval(r.get(metric, ""))
            if x is not None:
                vals[r["Model"]] = [x]
        all_rows.extend(metric_summary(f"main_table_{metric}", vals, direction))

    # Final validation q_delta/smoothing per-sequence spread.
    q_root = "analysis_outputs/offline_proteinopd/final_validation_eval_qdelta_smoothing"
    for metric, pattern, column, scale, direction in [
        ("pLDDT", f"{q_root}/structure_hf/*_hf_esmfold_structure_scores.csv", "plddt", 100.0, "higher"),
        ("pAE", f"{q_root}/structure_hf/*_hf_esmfold_structure_scores.csv", "pae", 1.0, "lower"),
        ("pTM", f"{q_root}/structure_hf/*_hf_esmfold_structure_scores.csv", "ptm", 1.0, "higher"),
        ("Sol", f"{q_root}/property/protein_sol/*_protein_sol_scores.csv", "sol_score", 1.0, "higher"),
        ("Thermo", f"{q_root}/property/temberture/*_temberture_scores.csv", "thermo_score", 1.0, "higher"),
    ]:
        all_rows.extend(metric_summary(f"qdelta_smoothing_{metric}", load_metric_from_files(pattern, column, scale), direction))

    # Utility controls per-sequence spread.
    u_root = "analysis_outputs/offline_proteinopd/final_validation_eval"
    for metric, pattern, column, scale, direction in [
        ("pLDDT", f"{u_root}/structure_hf/*_hf_esmfold_structure_scores.csv", "plddt", 100.0, "higher"),
        ("pAE", f"{u_root}/structure_hf/*_hf_esmfold_structure_scores.csv", "pae", 1.0, "lower"),
        ("pTM", f"{u_root}/structure_hf/*_hf_esmfold_structure_scores.csv", "ptm", 1.0, "higher"),
        ("Sol", f"{u_root}/property/protein_sol/*_protein_sol_scores.csv", "sol_score", 1.0, "higher"),
        ("Thermo", f"{u_root}/property/temberture/*_temberture_scores.csv", "thermo_score", 1.0, "higher"),
    ]:
        all_rows.extend(metric_summary(f"utility_controls_{metric}", load_metric_from_files(pattern, column, scale), direction))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Compact group-level diagnostics for the report.
    group_stats = {}
    for r in all_rows:
        g = str(r["group"])
        group_stats[g] = {
            "range": float(r["between_method_range"]),
            "avg_sd": float(r["avg_within_method_sd"]),
            "ratio": float(r["range_to_within_sd_ratio"]),
        }

    cache_summary = read_rows(ROOT / "analysis_outputs/offline_proteinopd/cache_diag/base_rollout/offline_cache_summary.csv")[0]
    cache_dist = read_rows(ROOT / "analysis_outputs/offline_proteinopd/cache_diag/base_rollout/offline_cache_distribution_diagnostics_summary.csv")[0]
    q04_sel = read_rows(ROOT / "analysis_outputs/offline_proteinopd/final_utility_sweeps/shift_entropy_q04_keep50/selection_summary.csv")
    q04_by_sel = {r["selection"]: r for r in q04_sel}

    lines = [
        "# CacheOPD Metric Discrimination Diagnosis",
        "",
        "## Verdict",
        "",
        "The current experiments show real coarse separation between weak baselines and PoE-based methods, but poor fine-grained separation among the strongest offline variants. This is mainly caused by weak teacher disagreement, high support overlap, small evaluation cohorts, metric saturation, and several summary/protocol inconsistencies.",
        "",
        "## Quantitative Symptoms",
        "",
        "### Final q_delta / smoothing variants",
        "",
        "| Metric | Between-method range | Avg within-method SD | Range / SD | Diagnosis |",
        "|---|---:|---:|---:|---|",
    ]
    for metric in ["pLDDT", "pAE", "pTM", "Sol", "Thermo"]:
        s = group_stats[f"qdelta_smoothing_{metric}"]
        diag = "weak separation" if s["ratio"] < 0.25 else ("moderate" if s["ratio"] < 0.75 else "clear")
        lines.append(f"| {metric} | {s['range']:.4f} | {s['avg_sd']:.4f} | {s['ratio']:.3f} | {diag} |")

    lines += [
        "",
        "### Final utility controls",
        "",
        "| Metric | Between-method range | Avg within-method SD | Range / SD | Diagnosis |",
        "|---|---:|---:|---:|---|",
    ]
    for metric in ["pLDDT", "pAE", "pTM", "Sol", "Thermo"]:
        s = group_stats[f"utility_controls_{metric}"]
        diag = "weak separation" if s["ratio"] < 0.25 else ("moderate" if s["ratio"] < 0.75 else "clear")
        lines.append(f"| {metric} | {s['range']:.4f} | {s['avg_sd']:.4f} | {s['ratio']:.3f} | {diag} |")

    lines += [
        "",
        "## Root Causes",
        "",
        "### 1. Teacher signals are highly similar",
        "",
        f"- Teacher pairwise top-K overlap: {float(cache_summary['teacher_pairwise_overlap_mean']):.3f}.",
        f"- Reference-PoE overlap: {float(cache_summary['ref_poe_overlap_mean']):.3f}.",
        f"- Mean teacher log-probability std on PoE support: {float(cache_dist['teacher_logprob_std_on_poe_mean']):.3f}.",
        f"- Mean target probability shift over reference: {float(cache_dist['target_prob_delta_poe_minus_ref_mean']):.4f}.",
        "",
        "This means the three teachers often rank very similar local token supports. The PoE target changes the distribution enough to beat average-KD, but not enough to create large separations among refined CacheOPD variants.",
        "",
        "### 2. Support gate is not strongly selective",
        "",
        f"- q_delta=0.4 support-gate tokens: {q04_by_sel['support_gate']['n_tokens']} / {q04_by_sel['all_valid']['n_tokens']}.",
        f"- q_delta=0.4 high selected tokens: {q04_by_sel['high']['n_tokens']}.",
        f"- q_delta=0.4 high preference-shift JS: {float(q04_by_sel['high']['preference_shift_js_mean']):.3f}; random: {float(q04_by_sel['random']['preference_shift_js_mean']):.3f}; low: {float(q04_by_sel['low']['preference_shift_js_mean']):.3f}.",
        f"- q_delta=0.4 high teacher conflict: {float(q04_by_sel['high']['teacher_conflict_mean']):.4f}; random: {float(q04_by_sel['random']['teacher_conflict_mean']):.4f}; low: {float(q04_by_sel['low']['teacher_conflict_mean']):.4f}.",
        "",
        "The utility ranking separates token diagnostics, but the absolute diagnostic gaps are small. Downstream generation metrics therefore only move slightly.",
        "",
        "### 3. Evaluation metrics are partly saturated or noisy",
        "",
        "- Validity and duplicate rate are already near-perfect for most cohorts, so they cannot separate methods.",
        "- TemBERTure medians for final variants are close to 0.95-0.98, so mean differences are dominated by a minority low-score tail.",
        "- Solubility means for aligned methods lie in a narrow band around 0.66-0.70.",
        "- Structure metrics have large sequence-level variance: in q_delta/smoothing variants, pLDDT range is far smaller than within-method SD.",
        "",
        "### 4. Sample size and seeds are too small for fine differences",
        "",
        "- Most cohorts contain about 128 sequences, often 32 per superfamily.",
        "- Training uses the same seed and roughly 48 update steps for many offline variants.",
        "- Cache construction uses max_samples=128, so each method is trained on a small, shared cache manifold.",
        "- The q0/q0.2/q0.4 ProLLaMA PPL diagnostic currently uses only 64 sequences per cohort.",
        "",
        "### 5. There are summary/protocol issues to clean up",
        "",
        "- `final_validation_eval_qdelta_smoothing/structure_hf/p0_hf_esmfold_structure_summary.csv` only contains q0.2, although per-sequence files exist for q0, q0.2, q0.4, and smoothing.",
        "- Some older basic PPL files use a legacy evaluator and should not be mixed with ProLLaMA diagnostic PPL.",
        "- MMseqs2 Novelty-T has many no-hit cases under the current conditional reference set and is not a reliable method separator.",
        "",
        "## What This Means For The Paper",
        "",
        "The strongest defensible story is not that every CacheOPD component creates a large downstream quality jump. The stronger story is:",
        "",
        "> Closed-support PoE is the main effective target. CacheOPD is a sparse support-aware variant that preserves much of PoE-Full quality at lower token budget, and clearly beats SFT / traditional average-teacher KD under matched offline training cost.",
        "",
        "## Immediate Fixes",
        "",
        "1. Rebuild all summary CSVs from per-sequence files; do not rely on stale partial summaries.",
        "2. Rerun ProLLaMA PPL diagnostics with all 128 sequences for q0/q0.2/q0.4/smoothing, or increase to 256-512 sequences.",
        "3. Add bootstrap confidence intervals to every main metric in the paper tables.",
        "4. Increase evaluation sample size and repeat generation with multiple seeds.",
        "5. Add per-superfamily breakdown; a method may separate on one family but vanish in the global mean.",
        "6. Add stronger stress tests where teachers disagree more, e.g. single-objective teacher baselines, skewed teacher weights, stricter property-conditioned prompts, or OOD/held-out superfamilies.",
        "7. Report cache-level effect size explicitly: teacher overlap, JS shift, support-gate selectivity, and target-in-top-K.",
        "",
        f"CSV source: `{OUT_CSV.relative_to(ROOT)}`",
        "",
    ]

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {OUT_CSV}")
    print(f"[OK] {OUT_REPORT}")


if __name__ == "__main__":
    main()
