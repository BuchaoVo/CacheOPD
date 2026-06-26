# CacheOPD

CacheOPD is a reproducibility-focused codebase for offline multi-teacher protein
preference alignment. It builds on the ProteinOPD setting and studies whether
cached Product-of-Experts (PoE) teacher targets can replace online teacher
scoring while preserving the main preference-alignment signal.

The repository is intentionally organized as a code-only release. Model
checkpoints, training/evaluation data, raw caches, generated sequences, compact
CSV result tables, rendered figures, manuscript drafts, and paper table bundles
are not committed to the main branch.

## What This Repository Contains

- `conditional/`: conditional ProteinOPD/CacheOPD generation, ProLLaMA OPD
  training, teacher construction, and configs.
- `unconditional/`: unconditional ProteinOPD teacher construction, generation,
  training code, and configs.
- `tools/offline_opd/`: offline cache construction, cached-target fusion,
  token-utility scoring, student training, and diagnostics.
- `tools/p0_eval/`: PPL, structure, solubility, thermostability, and related
  evaluation helpers.
- `scripts/conditional_supplement/`: matched conditional comparison utilities,
  external-baseline wrappers, bootstrap analysis, and report writers.
- `scripts/proteinopd_reproduction/`: ProteinOPD reproduction helpers,
  metric merging, ESM2 diagnostic generation, PDB export, and 3D/solubility
  analysis.
- `paper_figures_cacheopd/`: R scripts for regenerating editable figures from
  local source CSV files.
- `results/cacheopd_paper/`: reproducible analysis scripts and documentation.

## Main Reproducible Claims

- Cached PoE targets over teacher-supported candidate tokens can replace online
  teacher scoring in the matched conditional ProteinOPD setting.
- CacheOPD-Sparse is a token-efficient approximation of CacheOPD-Full.
- The main practical contribution is teacher-call efficiency plus
  preference-token budget efficiency.
- Stronger teacher separation and explicit teacher-disagreement modeling are
  treated as future work rather than as the central experimental claim.

## Quick Start

Install the original ProteinOPD dependencies first:

```bash
pip install -r requirements.txt
```

Some evaluation and plotting scripts require optional external tools or models,
including ProLLaMA/ProtGPT2 checkpoints, ESMFold, Protein-Sol, TemBERTure, R,
and `ggplot2`. Paths in the shell scripts are examples from the original lab
environment and should be updated before rerunning on a new machine.

## Key Entrypoints

Offline CacheOPD training and diagnostics:

```bash
python tools/offline_opd/train_offline_opd.py --help
python tools/offline_opd/prepare_offline_generated_cohorts.py --help
python tools/offline_opd/build_fusion_cache_variant.py --help
python tools/offline_opd/score_cache_token_utility.py --help
python tools/offline_opd/diagnose_teacher_distribution_signal.py --help
```

Conditional and unconditional ProteinOPD workflows:

```bash
python conditional/generate/generate.py --help
python conditional/proteinopd/prollama_opd_train.py --help
python conditional/teacher_construct/scripts/instruction_tune.py --help
python unconditional/generate/generate.py --help
python unconditional/proteinopd/protein_opd_train.py --help
python unconditional/teacher_construct/prefix_tuning_prot.py --help
```

Evaluation helpers:

```bash
python tools/p0_eval/eval_p0_basic_ppl.py --help
python tools/p0_eval/score_esmfold_structure_hf.py --help
python tools/p0_eval/score_protein_sol_cmd.py --help
python tools/p0_eval/score_temberture_cls.py --help
```

## Regenerate Compact Analyses and Figures

Supplemental analysis CSVs and local figure outputs can be regenerated after
placing local evaluation summaries in the expected paths:

```bash
Rscript results/cacheopd_paper/scripts/build_cacheopd_paper_outputs.R \
  --repo-root . \
  --out-dir results/cacheopd_paper
```

Editable figure outputs can be regenerated after local source CSV files are
available:

```bash
Rscript paper_figures_cacheopd/scripts/build_all_editable_figures.R --root .
Rscript paper_figures_cacheopd/scripts/build_scipilot_method_motivation_figures.R --root .
Rscript paper_figures_cacheopd/scripts/build_memt_style_method_figure.R --root .
```

Rendered figures are local outputs and are ignored by git.

## What Is Not Tracked

The following files are intentionally excluded:

- model checkpoints, LoRA adapters, optimizer states, and raw teacher/student
  outputs;
- offline caches and token-level teacher logits;
- compact CSV/JSON/JSONL result tables, property datasets, instruction-tuning
  datasets, HuggingFace Arrow cache directories, and expanded `*.full.jsonl`
  dumps;
- raw generated sequences and raw structure/evaluator outputs;
- manuscript drafts, paper table bundles, and rendered figure files;
- `analysis_outputs/`, `outputs/`, `logs_nohup/`, local model directories, and
  other machine-specific artifacts.

For full details, see `REPRODUCIBILITY.md`.

## Acknowledgements

We sincerely thank the authors of ProteinOPD for releasing the original codebase
and experimental setting that this reproducibility package builds upon. We also
acknowledge the open protein language model and evaluation ecosystems used by
the project, including ProLLaMA, ProtGPT2, ESM/ESMFold, Protein-Sol, and
TemBERTure, as well as the broader PyTorch, HuggingFace, DeepSpeed, and R
communities for the software infrastructure that made these experiments
possible.

This repository is prepared as a code-only release for reproducible research.
Any errors in the CacheOPD implementation, experiment scripts, or documentation
are our responsibility.
