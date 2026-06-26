# CacheOPD Reproducible Figure Source Data

This lightweight repository branch tracks the scripts needed to rebuild the
paper figures. Figure source CSV files and rendered PDF/PNG/SVG/TIFF files are
intentionally omitted from git and should be generated or supplied locally.

## Tracked Source Assets

- `scripts/build_all_editable_figures.R`
- `scripts/build_memt_style_method_figure.R`
- `scripts/build_scipilot_method_motivation_figures.R`
- local-only `source_data/*.csv` files when generated

## Rebuild Editable Figures

```bash
Rscript paper_figures_cacheopd/scripts/build_all_editable_figures.R --root .
```

The script writes editable outputs to:

- `paper_figures_cacheopd/figures_editable/*.svg`
- `paper_figures_cacheopd/figures_editable/*.pdf`
- `paper_figures_cacheopd/figures_editable/*.png`
- optional LaTeX-ready copies under `paper_md_report/images/` if that local
  manuscript workspace exists.

## Rebuild Method and Motivation Diagrams

```bash
Rscript paper_figures_cacheopd/scripts/build_scipilot_method_motivation_figures.R --root .
Rscript paper_figures_cacheopd/scripts/build_memt_style_method_figure.R --root .
```

The generated method diagrams can also be copied to a local manuscript figure
directory if one exists.

## Notes

- Cross-block best values are not bolded because unconditional and conditional
  rows are not directly comparable.
- Composite scores in the quality-cost figure are used only for visualization;
  tables preserve individual metrics.
- `figure1_cacheopd_memt_style_framework` follows a staged ProteinOPD/LightODP-
  style method layout: offline cache construction, token utility selection, and
  zero-teacher distillation with explicit efficiency annotations.
