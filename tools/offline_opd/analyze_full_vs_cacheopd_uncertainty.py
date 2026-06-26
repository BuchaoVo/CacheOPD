from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_md_report"
CSV_OUT = ROOT / "analysis_outputs/offline_proteinopd/supplemental_analysis/full_vs_cacheopd_bootstrap.csv"
MATCHED_CONDITIONAL = ROOT / "analysis_outputs/conditional_supplement/summaries/merged_per_sequence_metrics.csv"


def read_float_column(path: str, column: str, scale: float = 1.0, cohort: str | None = None) -> list[float]:
    fp = ROOT / path
    vals: list[float] = []
    with fp.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if cohort is not None and r.get("cohort") != cohort:
                continue
            raw = r.get(column, "")
            if raw == "":
                continue
            try:
                vals.append(float(raw) * scale)
            except ValueError:
                continue
    if not vals:
        raise RuntimeError(f"No values loaded from {fp} column={column} cohort={cohort}")
    return vals


def read_matched_method_column(method: str, column: str, scale: float = 1.0) -> list[float]:
    vals: list[float] = []
    with MATCHED_CONDITIONAL.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("method") != method:
                continue
            raw = r.get(column, "")
            if raw == "":
                continue
            try:
                vals.append(float(raw) * scale)
            except ValueError:
                continue
    if not vals:
        raise RuntimeError(f"No values loaded from {MATCHED_CONDITIONAL} method={method} column={column}")
    return vals


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def quantile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def boot_metric(full: list[float], cache: list[float], transform, better: str, n_boot: int = 10000):
    rng = random.Random(20260613)
    full_mean = transform(mean(full))
    cache_mean = transform(mean(cache))
    diff = cache_mean - full_mean
    boot_diffs = []
    cache_better = 0
    for _ in range(n_boot):
        fb = [full[rng.randrange(len(full))] for _ in range(len(full))]
        cb = [cache[rng.randrange(len(cache))] for _ in range(len(cache))]
        fd = transform(mean(fb))
        cd = transform(mean(cb))
        d = cd - fd
        boot_diffs.append(d)
        if better == "max":
            cache_better += int(cd >= fd)
        elif better == "min":
            cache_better += int(cd <= fd)
        else:
            raise ValueError(better)
    return {
        "full_mean": full_mean,
        "cache_mean": cache_mean,
        "delta_cache_minus_full": diff,
        "ci_low": quantile(boot_diffs, 0.025),
        "ci_high": quantile(boot_diffs, 0.975),
        "prob_cache_better": cache_better / n_boot,
        "n_full": len(full),
        "n_cache": len(cache),
    }


def main() -> None:
    def metric_row(
        metric: str,
        column: str,
        better: str,
        transform: Callable[[float], float],
        note: str,
        scale: float = 1.0,
    ) -> dict:
        return {
            "metric": metric,
            "better": better,
            "full": read_matched_method_column("CacheOPD Full", column, scale=scale),
            "cache": read_matched_method_column("CacheOPD Sparse", column, scale=scale),
            "transform": transform,
            "note": note,
        }

    if not MATCHED_CONDITIONAL.exists():
        raise FileNotFoundError(f"Missing matched conditional metrics: {MATCHED_CONDITIONAL}")

    metrics = [
        metric_row(
            "ProLLaMA PPL",
            "ppl_loss",
            "min",
            math.exp,
            "bootstrap over sequence-level NLL, then exponentiated mean NLL",
        ),
        metric_row("pLDDT", "plddt", "max", lambda x: x, "ESMFold pLDDT on the matched conditional cohort"),
        metric_row("pAE", "pae", "min", lambda x: x, "lower is better"),
        metric_row("pTM", "ptm", "max", lambda x: x, "higher is better"),
        metric_row("Sol", "sol", "max", lambda x: x, "Protein-Sol score"),
        metric_row("Thermo", "thermo", "max", lambda x: x, "TemBERTure score"),
    ]

    rows = []
    for m in metrics:
        stats = boot_metric(m["full"], m["cache"], m["transform"], m["better"])
        direction = "higher" if m["better"] == "max" else "lower"
        rel_token = 5628 / 16055
        rows.append({
            "metric": m["metric"],
            "direction": direction,
            "full_mean": stats["full_mean"],
            "cacheopd_mean": stats["cache_mean"],
            "delta_cache_minus_full": stats["delta_cache_minus_full"],
            "bootstrap_95ci_low": stats["ci_low"],
            "bootstrap_95ci_high": stats["ci_high"],
            "prob_cacheopd_better": stats["prob_cache_better"],
            "n_full": stats["n_full"],
            "n_cacheopd": stats["n_cache"],
            "cacheopd_token_ratio_vs_full": rel_token,
            "note": m["note"],
        })

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = OUT_DIR / "CacheOPD_full_vs_sparse_uncertainty_analysis.md"
    lines = [
        "# Offline cached PoE-Full vs CacheOPD Uncertainty Analysis",
        "",
        "## Purpose",
        "",
        "This diagnostic tests whether the apparent advantage of Offline cached PoE-Full over CacheOPD is a stable quality gap, a small-sample fluctuation, or an expected quality-budget trade-off.",
        "",
        "## Setup",
        "",
        "- Offline cached PoE-Full uses all cached PoE token positions.",
        "- CacheOPD uses q_delta=0.4, shift+entropy High50 selection, and gamma_nll=0.1.",
        "- CacheOPD uses 5,628 training tokens, whereas Offline cached PoE-Full uses 16,055 tokens; the token ratio is 35.1%.",
        "- Bootstrap confidence intervals use independent sequence-level resampling with 10,000 replicates.",
        "- ProLLaMA PPL is bootstrapped over sequence-level NLL and then exponentiated.",
        f"- Sequence-level metrics are loaded from `{MATCHED_CONDITIONAL.relative_to(ROOT)}`.",
        "",
        "## Bootstrap Results",
        "",
        "| Metric | Direction | Full mean | CacheOPD mean | Delta Cache-Full | 95% CI | P(CacheOPD better) | n Full / Cache |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['metric']} | {r['direction']} | "
            f"{r['full_mean']:.4f} | {r['cacheopd_mean']:.4f} | "
            f"{r['delta_cache_minus_full']:.4f} | "
            f"[{r['bootstrap_95ci_low']:.4f}, {r['bootstrap_95ci_high']:.4f}] | "
            f"{r['prob_cacheopd_better']:.3f} | "
            f"{r['n_full']} / {r['n_cacheopd']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "1. Offline cached PoE-Full is generally stronger on PPL, pLDDT, pAE, pTM, and Sol, which is expected because it trains on all cached token positions.",
        "2. CacheOPD is not designed to dominate PoE-Full on every quality metric. Its defensible claim is that sparse support-aware selection preserves most PoE-Full quality while using far fewer training tokens.",
        "3. If the confidence interval of CacheOPD-Full includes zero, the observed gap is not stable at this sample size. If it excludes zero, the gap should be described as a real quality-budget trade-off rather than as CacheOPD superiority.",
        "4. The current evaluation cohort is small enough that single-number differences below roughly one pLDDT point or a few hundredths in property score should be treated cautiously unless bootstrap intervals are clearly separated.",
        "",
        "## Manuscript Consequence",
        "",
        "Do not write that CacheOPD outperforms Offline cached PoE-Full. Write instead:",
        "",
        "> CacheOPD retains competitive quality at approximately one third of the cached token budget, while substantially outperforming SFT and traditional average-teacher offline KD under comparable offline training cost.",
        "",
        "The strongest main baseline for quality remains Offline cached PoE-Full; CacheOPD is the sparse efficient variant.",
        "",
        f"CSV source: `{CSV_OUT.relative_to(ROOT)}`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {CSV_OUT}")
    print(f"[OK] {report}")


if __name__ == "__main__":
    main()
