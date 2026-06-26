# CacheOPD Reproducibility Package

This repository branch contains the code and scripts needed to reproduce the CacheOPD experiments. Large checkpoints, teacher adapters, raw caches, compact CSV/JSON/JSONL result files, source data tables, manuscript drafts, paper table bundles, rendered figure bundles, generated-sequence workspaces, ESMFold intermediate outputs, and nohup logs are intentionally excluded from git.

## Main Claims Covered

- Cached PoE targets over teacher-supported candidate tokens can replace online teacher scoring for the matched conditional ProteinOPD setting.
- CacheOPD-Sparse is a token-efficient approximation of CacheOPD-Full.
- The strongest reproducible contribution is teacher-call efficiency plus preference-token budget efficiency.
- Teacher-disagreement modeling and stronger teacher separation are treated as future work, not as the main experimental claim.

## Included Reproduction Assets

| Path | Purpose |
|---|---|
| `tools/offline_opd/` | Cache construction, cached-target fusion, utility scoring, offline OPD training, diagnostics, and analysis helpers. |
| `tools/p0_eval/` | PPL, ESMFold structure, Protein-Sol, TemBERTure, and related evaluation helpers. |
| `conditional/` | Conditional ProteinOPD/CacheOPD generation, teacher construction, ProLLaMA OPD training code, and configs. |
| `unconditional/` | Unconditional ProteinOPD teacher construction, generation, training code, and configs. |
| `scripts/conditional_supplement/` | Matched conditional comparison utilities, external baseline wrappers, bootstrap, and report writers. |
| `scripts/proteinopd_reproduction/` | Reproduction helpers for cohort preparation, metric merging, ESM2 diagnostic generation, PDB/3D visualization export, and solubility-structure analysis. |
| `scripts_local/` | Cluster-oriented shell entry points used for the final validation and sparse/fusion/anchor runs. |
| `results/cacheopd_paper/` | Scripts and documentation for supplemental analysis. Compact CSV summaries are local generated artifacts and are not tracked here. |
| `paper_figures_cacheopd/` | R plotting scripts for regenerating editable SVG/PDF/PNG figures from local source CSV files. Figure source data are not tracked here. |

## Files Not Included

The following files are required only for full end-to-end reruns and should be regenerated or downloaded locally:

- ProLLaMA, ProtGPT2, teacher LoRA adapters, and student checkpoints.
- Raw offline caches and token-level teacher logits.
- Raw generated sequences and raw ESMFold/Protein-Sol/TemBERTure outputs.
- Compact CSV/JSON/JSONL result tables, property datasets, instruction-tuning datasets, HuggingFace Arrow dataset caches, and expanded `*.full.jsonl` instruction dumps.
- `analysis_outputs/`, `outputs/`, `logs_nohup/`, manuscript draft directories, paper table bundles, and local model directories.

This branch intentionally tracks code rather than experiment data. Manuscript drafts, paper table bundles, compact result CSVs, source-data tables, and rendered figures are omitted from the main branch; regenerate them locally when needed or distribute them separately as release artifacts.

## Rebuild Supplemental Analyses and Figures

From the repository root:

```bash
Rscript results/cacheopd_paper/scripts/build_cacheopd_paper_outputs.R \
  --repo-root . \
  --out-dir results/cacheopd_paper
```

If using the local conda environment from the original workspace:

```bash
conda run -n cacheopd-r Rscript results/cacheopd_paper/scripts/build_cacheopd_paper_outputs.R \
  --repo-root . \
  --out-dir results/cacheopd_paper
```

After local evaluation summaries and cache diagnostics are available, this regenerates compact analysis CSVs and, if desired, local-only figure outputs:

- `cost_decomposition.csv`
- `sparse_full_bootstrap.csv`
- `fusion_variant_results.csv`
- `quality_cost_plot.csv` and local `quality_cost_scatter.pdf/png`
- `sparse_ratio_results.csv` and local `sparse_ratio_curve.pdf/png`
- `cache_position_diagnostics.csv`, local `cache_diagnostics.pdf/png`, and `selected_unselected_stats.csv`
- `teacher_pairwise_diagnostics.csv`, `teacher_pairwise_summary.csv`, and local `teacher_overlap_heatmap.pdf/png`
- optional Pareto and qualitative outputs when the corresponding source data exist

## Rebuild Editable Manuscript Figures

```bash
Rscript paper_figures_cacheopd/scripts/build_all_editable_figures.R --root .
```

The workflow writes editable SVG/PDF/PNG files to `paper_figures_cacheopd/figures_editable/` and optional LaTeX-ready copies to a local manuscript workspace. Source CSV files are local generated artifacts and are not tracked in this branch.

## Rebuild Method and Motivation Diagrams

```bash
Rscript paper_figures_cacheopd/scripts/build_scipilot_method_motivation_figures.R --root .
Rscript paper_figures_cacheopd/scripts/build_memt_style_method_figure.R --root .
```

The manuscript-ready method diagrams are also copied under `paper_md_report/Report MD/figures/`.

## Rerun Offline CacheOPD Experiments

The main training implementation is:

```bash
python tools/offline_opd/train_offline_opd.py --help
```

Typical preprocessing and diagnostic entry points are:

```bash
python tools/offline_opd/prepare_offline_generated_cohorts.py --help
python tools/offline_opd/build_fusion_cache_variant.py --help
python tools/offline_opd/score_cache_token_utility.py --help
python tools/offline_opd/diagnose_teacher_distribution_signal.py --help
```

Cluster shell wrappers used for the final experiments are available in `scripts_local/`, including:

- `run_final_validation_generate_4superfamily.sh`
- `run_final_utility_generate_4superfamily.sh`
- `run_sparse_kappa75_generate_4superfamily.sh`
- `run_experiment3_fusion_generate_4superfamily.sh`
- `run_experiment6_anchor_generate_4superfamily.sh`

These scripts assume local model/checkpoint paths and GPU assignments from the original lab environment. Update paths before running on a new machine.

## Conditional and Unconditional ProteinOPD Entrypoints

The original conditional and unconditional ProteinOPD workflows are included as
code/config/data scaffolds:

```bash
python conditional/generate/generate.py --help
python conditional/proteinopd/prollama_opd_train.py --help
python conditional/teacher_construct/scripts/instruction_tune.py --help
python unconditional/generate/generate.py --help
python unconditional/proteinopd/protein_opd_train.py --help
python unconditional/teacher_construct/prefix_tuning_prot.py --help
```

Property datasets and high-preference instruction datasets are not tracked in
the main branch. They can be regenerated locally with the data-construction
scripts or distributed separately as release artifacts.

## Evaluation Pipeline

The aligned final metric set is:

- PPL under the ProLLaMA evaluator
- pLDDT, pAE, and pTM from ESMFold
- solubility from Protein-Sol
- thermostability from TemBERTure
- teacher calls, GPU hours, and preference-token budget

Reference-dependent novelty scores are excluded from the final main tables.

Evaluation helper scripts are in `tools/p0_eval/`:

```bash
python tools/p0_eval/eval_p0_basic_ppl.py --help
python tools/p0_eval/score_esmfold_structure_hf.py --help
python tools/p0_eval/score_protein_sol_cmd.py --help
python tools/p0_eval/score_temberture_cls.py --help
```

ProteinOPD reproduction and 3D visualization helpers are in `scripts/proteinopd_reproduction/`:

```bash
python scripts/proteinopd_reproduction/prepare_reproduction_cohorts.py --help
python scripts/proteinopd_reproduction/merge_reproduction_metrics.py --help
python scripts/proteinopd_reproduction/export_3d_structure_batch.py --help
python scripts/proteinopd_reproduction/export_pdb_visualization_cases.py --help
Rscript scripts/proteinopd_reproduction/plot_sol3d_diagnostics.R --help
```

## Reproduction Boundary

The committed package is intended to reproduce code-level analyses and figures from compact source data. A full raw rerun requires external model weights and large intermediate files that are not suitable for normal git storage. If releasing trained adapters, raw caches, manuscript drafts, or rendered paper artifacts is required, use a separate release asset, Hugging Face Hub, Zenodo, or another artifact store rather than committing them to the main code branch.
