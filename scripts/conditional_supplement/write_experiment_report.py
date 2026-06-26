from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import OUT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out_root)
    status_dir = out_root / "status"
    statuses = []
    for path in sorted(status_dir.glob("*status*.json")):
        statuses.append(f"- `{path.name}`")
    internal_status = status_dir / "internal_generation_status.csv"
    internal_note = ""
    if internal_status.exists():
        df = pd.read_csv(internal_status)
        internal_note = df.to_markdown(index=False)
    text = f"""# Conditional Supplement Experiment Setup

## Goal

This supplementary experiment implements a fair conditional comparison for CacheOPD. Internal methods are kept in one matched conditional setting; external condition-aware models are handled in a separate reference block and are not ranked against internal methods.

## Fairness Constraints

- Internal methods must use the same four superfamily conditions, requested sample count per condition, decoding protocol, filtering rules, and evaluation pipeline.
- External baselines use standardized natural-language prompts derived from the same conditions.
- Unconditional and conditional results must not be mixed in a ranked table.
- Missing external baselines are recorded as status JSON files rather than filled with fabricated numbers.
- Per-sequence raw outputs, cleaned cohorts, metric files, per-superfamily summaries, bootstrap CIs, and figures are stored under `{out_root}`.

## Conditions

1. lysozyme-like domain superfamily
2. immunoglobulin-like beta-sandwich superfamily
3. TIM beta/alpha-barrel domain superfamily
4. Rossmann-like alpha/beta/alpha sandwich fold superfamily

## Implemented Scripts

- `build_condition_prompts.py`
- `run_internal_matched.py`
- `run_proteindt.py`
- `run_prodva.py`
- `run_esm3.py`
- `build_single_teacher_caches.py`
- `write_single_teacher_experiment_commands.py`
- `clean_generated_sequences.py`
- `merge_eval_metrics.py`
- `summarize_internal_matched.py`
- `summarize_external_condition_aware.py`
- `summarize_single_teacher_specificity.py`
- `bootstrap_ci.py`
- `plot_quality_cost_tradeoff.py`
- `plot_single_teacher_property_profile.py`
- `plot_condition_consistency_vs_quality.py`

## Internal Generation Status

{internal_note if internal_note else 'Not run yet.'}

## External Baseline Status Files

{chr(10).join(statuses) if statuses else 'No external status files yet.'}

## Evaluation Notes

The repository already contains reusable evaluators for ProLLaMA PPL, ESMFold structure metrics, Protein-Sol, TemBERTure, and condition-adherence NLL. Following the current CacheOPD evaluation decision, Novelty-T is not included in this supplement's result schema. This supplement pipeline organizes outputs under a unified per-sequence schema keyed by `method`, `condition_id`, `sample_id`, and `sequence_id`.
"""
    out_path = out_root / "summaries" / "conditional_supplement_experiment_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"[OK] report -> {out_path}")


if __name__ == "__main__":
    main()
