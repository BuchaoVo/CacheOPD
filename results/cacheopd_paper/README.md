# CacheOPD Paper Supplemental Outputs

This directory contains the compact reproducible analysis workflow for the CacheOPD experiments.

## Scope

The workflow reuses existing generated sequences, evaluation summaries, training logs, cache diagnostics, and teacher-distribution diagnostics. It does not rerun sequence generation, teacher scoring, ESMFold, Protein-Sol, TemBERTure, or PPL evaluation.

Directly ranked conditional methods are named consistently as:

- `ProLLaMA`
- `SFT`
- `ProbAvg-KD`
- `LogitAvg-KD`
- `Online OPD`
- `CacheOPD-Full`
- `CacheOPD-Sparse`

`CacheOPD-Full` denotes training on all valid cached PoE target positions. `CacheOPD-Sparse` denotes the support-aware sparse approximation using selected teacher-supported candidate tokens.

## Main Script

```bash
Rscript results/cacheopd_paper/scripts/build_cacheopd_paper_outputs.R \
  --repo-root /home/zbc/data/opd/ProteinOPD \
  --out-dir results/cacheopd_paper
```

Optional arguments:

```bash
--width 7.2
--height 4.6
--dpi 450
```

The script requires R with `ggplot2`.

The current workspace has a ready-to-use conda R environment:

```bash
/home/zbc/data/software/miniconda/envs/cacheopd-r/bin/Rscript \
  results/cacheopd_paper/scripts/build_cacheopd_paper_outputs.R \
  --repo-root /home/zbc/data/opd/ProteinOPD \
  --out-dir results/cacheopd_paper
```

## Data Sources

The core branch tracks compact CSV summaries in this directory and figure source
CSV files under `paper_figures_cacheopd/source_data/`. The builder can also read
larger local-only files from `analysis_outputs/` when they exist, but those raw
workspaces are intentionally not committed to `main`.

## Required Outputs

The script can write the following files locally. In the core repository branch,
we track compact CSV summaries and scripts; generated LaTeX tables and rendered
figures may be regenerated locally and need not be committed to `main`.

- `cost_decomposition.csv`
- `sparse_full_bootstrap.csv`
- `fusion_variant_results.csv`
- `quality_cost_plot.csv`
- `quality_cost_scatter.pdf`
- `quality_cost_scatter.png`
- `sparse_ratio_results.csv`, when sparse-ratio rows exist
- `sparse_ratio_curve.pdf`, when sparse-ratio rows exist
- `sparse_ratio_curve.png`, when sparse-ratio rows exist
- `cache_position_diagnostics.csv`
- `cache_diagnostics.pdf`
- `cache_diagnostics.png`
- `selected_unselected_stats.csv`
- `teacher_pairwise_diagnostics.csv`
- `teacher_pairwise_summary.csv`
- `teacher_overlap_heatmap.pdf`
- `teacher_overlap_heatmap.png`
- `pareto_hv_summary.csv`, when aggregate Pareto data exist
- `pareto_tradeoff.pdf`, when aggregate Pareto data exist
- `pareto_tradeoff.png`, when aggregate Pareto data exist
- `qualitative_cases.csv`, only if per-case structure images are found
- `qualitative_examples.pdf/png`, only if structure images can be linked to method-level case metadata
- `skipped_outputs.csv`, documenting missing optional inputs

## Figure Contracts

- `quality_cost_scatter`: shows quality-cost trade-off across directly ranked conditional methods.
- `sparse_ratio_curve`: shows how sparse token ratio affects PPL, pLDDT, and thermostability for available sparse-ratio/q-delta rows.
- `cache_diagnostics`: compares selected and unselected cached token positions by utility and support overlap.
- `teacher_overlap_heatmap`: shows low pairwise teacher JS divergence.
- `pareto_tradeoff`: visualizes solubility-thermostability trade-offs using available aggregate method-level metrics.

## Notes

The deprecated novelty column is intentionally excluded from generated main tables and figures. If a future protocol-aligned diversity diagnostic is restored, add it as a separate appendix diagnostic rather than mixing it into the main quality-cost tables.

Optional outputs are skipped rather than fabricated when the required source data are absent. See `skipped_outputs.csv` after running the script.
