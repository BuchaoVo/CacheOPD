#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default = NULL) {
  hit <- which(args == name)
  if (length(hit) == 0 || hit == length(args)) return(default)
  args[[hit + 1]]
}

input_dir <- get_arg("--input_dir", "analysis_outputs/proteinopd_reproduction/sol3d_diagnostics")
out_dir <- get_arg("--out_dir", "paper_figures_cacheopd/figures/sol3d")
width_mm <- as.numeric(get_arg("--width_mm", "183"))
height_mm <- as.numeric(get_arg("--height_mm", "132"))
dpi <- as.numeric(get_arg("--dpi", "600"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

features_path <- file.path(input_dir, "sol3d_merged_metrics.csv")
summary_path <- file.path(input_dir, "sol3d_method_summary.csv")
corr_path <- file.path(input_dir, "sol3d_feature_correlations.csv")

stopifnot(file.exists(features_path), file.exists(summary_path), file.exists(corr_path))

df <- read.csv(features_path, check.names = FALSE)
summary_df <- read.csv(summary_path, check.names = FALSE)
corr_df <- read.csv(corr_path, check.names = FALSE)

method_order <- c(
  "ProLLaMA", "SFT", "ProbAvg-KD", "LogitAvg-KD", "Online OPD",
  "CacheOPD-Full", "CacheOPD-Sparse",
  "Fold-teacher KD", "Sol-teacher KD", "Thermo-teacher KD"
)

method_family <- function(x) {
  ifelse(x %in% c("CacheOPD-Full", "CacheOPD-Sparse"), "CacheOPD",
    ifelse(x %in% c("Online OPD"), "Online OPD",
      ifelse(grepl("KD", x), "KD", ifelse(x == "SFT", "SFT", "Base"))
    )
  )
}

family_cols <- c(
  "Base" = "#6B7280",
  "SFT" = "#9CA3AF",
  "KD" = "#7AA6C2",
  "Online OPD" = "#D08C45",
  "CacheOPD" = "#3A8F84"
)

cache_cols <- c(
  "ProLLaMA" = "#737373",
  "SFT" = "#A3A3A3",
  "ProbAvg-KD" = "#8AB6D6",
  "LogitAvg-KD" = "#5B8DB8",
  "Online OPD" = "#D08C45",
  "CacheOPD-Full" = "#2D7F73",
  "CacheOPD-Sparse" = "#49A397",
  "Fold-teacher KD" = "#9DBFD2",
  "Sol-teacher KD" = "#6EA5C4",
  "Thermo-teacher KD" = "#4E7EA4"
)

df$method <- factor(df$method, levels = method_order)
df$family <- method_family(as.character(df$method))
df$family <- factor(df$family, levels = c("Base", "SFT", "KD", "Online OPD", "CacheOPD"))
summary_df$method <- factor(summary_df$method, levels = method_order)
summary_df$family <- method_family(as.character(summary_df$method))

numeric_cols <- c(
  "sol", "plddt_100", "ppl", "ptm", "acidic_frac", "hydrophobic_frac",
  "exposed_hydrophobic_frac_proxy", "exposed_charged_frac_proxy",
  "sol3d_surface_balance_proxy", "sol3d_hydrophobic_risk_proxy"
)
for (nm in numeric_cols) {
  if (nm %in% names(df)) df[[nm]] <- as.numeric(df[[nm]])
}

theme_nature <- function(base_size = 6.6, base_family = "Helvetica") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = "black"),
      axis.ticks = element_line(linewidth = 0.30, colour = "black"),
      axis.ticks.length = unit(1.5, "mm"),
      axis.title = element_text(size = base_size, colour = "black"),
      axis.text = element_text(size = base_size - 0.6, colour = "black"),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.8),
      legend.key.height = unit(3.2, "mm"),
      legend.key.width = unit(3.8, "mm"),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size - 0.2, face = "bold"),
      plot.title = element_text(size = base_size + 0.4, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size - 0.8, colour = "#4B5563", hjust = 0),
      panel.grid.major.y = element_line(linewidth = 0.16, colour = "#E5E7EB"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(4, 5, 4, 5)
    )
}

theme_set(theme_nature())

panel_label <- function(label) {
  annotate("text", x = -Inf, y = Inf, label = label, hjust = -0.45, vjust = 1.25,
           size = 3.0, fontface = "bold", family = "Helvetica")
}

cor_label <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return("")
  r <- suppressWarnings(cor(rank(x[ok]), rank(y[ok]), method = "pearson"))
  paste0("Spearman rho = ", sprintf("%.2f", r), ", n = ", sum(ok))
}

scatter_df <- df[is.finite(df$acidic_frac) & is.finite(df$sol), ]

p_a <- ggplot(scatter_df, aes(x = acidic_frac, y = sol)) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.45, colour = "#2F6F73",
              fill = "#CFE4E1", alpha = 0.55) +
  geom_point(aes(fill = family), shape = 21, size = 2.0, stroke = 0.25,
             colour = "white", alpha = 0.92) +
  scale_fill_manual(values = family_cols, drop = FALSE) +
  scale_x_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0.04, 0.08))) +
  scale_y_continuous(expand = expansion(mult = c(0.04, 0.08))) +
  coord_cartesian(ylim = c(0.42, 0.91), clip = "off") +
  labs(
    title = "Acidic composition tracks solubility",
    subtitle = cor_label(scatter_df$acidic_frac, scatter_df$sol),
    x = "Acidic residues",
    y = "Protein-Sol score"
  ) +
  panel_label("a") +
  theme(legend.position = c(0.05, 0.08), legend.justification = c(0, 0))

sol_corr <- corr_df[corr_df$target == "sol", ]
sol_corr <- sol_corr[is.finite(sol_corr$spearman), ]
keep_features <- c(
  "acidic_frac",
  "mean_exposure_score_proxy",
  "exposed_acidic_frac_proxy",
  "exposed_residue_frac_proxy",
  "hydrophobic_frac",
  "exposed_basic_frac_proxy",
  "mean_ca_neighbor_10a",
  "contact_density_8a",
  "exposed_net_charge_per_residue_proxy"
)
feature_labels <- c(
  "acidic_frac" = "Acidic fraction",
  "mean_exposure_score_proxy" = "Exposure proxy",
  "exposed_acidic_frac_proxy" = "Exposed acidic proxy",
  "exposed_residue_frac_proxy" = "Exposed residues proxy",
  "hydrophobic_frac" = "Hydrophobic fraction",
  "exposed_basic_frac_proxy" = "Exposed basic proxy",
  "mean_ca_neighbor_10a" = "CA neighbours, 10 A",
  "contact_density_8a" = "Contact density, 8 A",
  "exposed_net_charge_per_residue_proxy" = "Exposed net charge proxy"
)
sol_corr <- sol_corr[sol_corr$feature %in% keep_features, ]
sol_corr$feature_label <- feature_labels[sol_corr$feature]
sol_corr$feature_label <- factor(sol_corr$feature_label, levels = rev(feature_labels[keep_features]))
sol_corr$direction <- ifelse(sol_corr$spearman >= 0, "positive", "negative")

p_b <- ggplot(sol_corr, aes(x = spearman, y = feature_label)) +
  geom_vline(xintercept = 0, linewidth = 0.32, colour = "#8A8A8A") +
  geom_segment(aes(x = 0, xend = spearman, yend = feature_label, colour = direction),
               linewidth = 0.55, lineend = "round") +
  geom_point(aes(colour = direction), size = 2.0) +
  scale_colour_manual(values = c("positive" = "#2F6F73", "negative" = "#C85A4A")) +
  scale_x_continuous(limits = c(-0.72, 0.68), breaks = seq(-0.6, 0.6, 0.3)) +
  labs(
    title = "3D-derived proxies explain Sol variation",
    subtitle = "Spearman correlation with Protein-Sol",
    x = "Spearman rho",
    y = NULL
  ) +
  panel_label("b") +
  theme(legend.position = "none", panel.grid.major.y = element_blank())

risk_col <- "sol3d_hydrophobic_risk_proxy"
risk_summary <- aggregate(df[[risk_col]], by = list(method = df$method, family = df$family), FUN = function(z) {
  z <- z[is.finite(z)]
  c(mean = mean(z), se = sd(z) / sqrt(length(z)), n = length(z))
})
risk_summary <- data.frame(
  method = risk_summary$method,
  family = risk_summary$family,
  mean = risk_summary$x[, "mean"],
  se = risk_summary$x[, "se"],
  n = risk_summary$x[, "n"]
)
risk_summary$method <- factor(risk_summary$method, levels = rev(method_order))

p_c <- ggplot(risk_summary, aes(x = mean, y = method)) +
  geom_segment(aes(x = mean - se, xend = mean + se, yend = method), linewidth = 0.35, colour = "#777777") +
  geom_point(aes(fill = family), shape = 21, size = 2.4, stroke = 0.25, colour = "white") +
  scale_fill_manual(values = family_cols, drop = FALSE) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.12))) +
  labs(
    title = "Surface hydrophobic risk",
    subtitle = "Mean +/- s.e.m.; lower is better",
    x = "Hydrophobic-risk proxy",
    y = NULL
  ) +
  panel_label("c") +
  theme(legend.position = "none", panel.grid.major.y = element_line(linewidth = 0.16, colour = "#ECECEC"))

method_plot <- summary_df
method_plot$method_chr <- as.character(method_plot$method)
method_plot$label <- ifelse(method_plot$method_chr %in% c("CacheOPD-Full", "CacheOPD-Sparse", "Online OPD", "SFT"),
                            method_plot$method_chr, "")
method_plot$family <- factor(method_plot$family, levels = c("Base", "SFT", "KD", "Online OPD", "CacheOPD"))

p_d <- ggplot(method_plot, aes(x = sol3d_hydrophobic_risk_proxy_mean, y = plddt_100_mean)) +
  geom_point(aes(fill = family, size = sol_mean), shape = 21, stroke = 0.28, colour = "white", alpha = 0.95) +
  geom_text(aes(label = label), nudge_x = 0.006, nudge_y = 1.2, size = 2.0, family = "Helvetica",
            colour = "#232323", check_overlap = TRUE) +
  scale_fill_manual(values = family_cols, drop = FALSE) +
  scale_size_continuous(range = c(1.6, 4.0)) +
  scale_x_continuous(expand = expansion(mult = c(0.08, 0.14))) +
  scale_y_continuous(expand = expansion(mult = c(0.08, 0.12))) +
  labs(
    title = "Quality-surface trade-off",
    subtitle = "Point size encodes mean Protein-Sol",
    x = "Hydrophobic-risk proxy",
    y = "Mean pLDDT"
  ) +
  panel_label("d") +
  theme(legend.position = "right")

source_data <- data.frame(
  panel = c(rep("a", nrow(df)), rep("b", nrow(sol_corr)), rep("c", nrow(risk_summary)), rep("d", nrow(method_plot))),
  object = c(as.character(df$method), as.character(sol_corr$feature), as.character(risk_summary$method), as.character(method_plot$method_chr)),
  x = c(df$acidic_frac, sol_corr$spearman, risk_summary$mean, method_plot$sol3d_hydrophobic_risk_proxy_mean),
  y = c(df$sol, seq_len(nrow(sol_corr)), seq_len(nrow(risk_summary)), method_plot$plddt_100_mean)
)
write.csv(source_data, file.path(out_dir, "figure_sol3d_source_data.csv"), row.names = FALSE)

draw_figure <- function() {
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(
    nrow = 2, ncol = 2,
    widths = unit(c(1.25, 1.0), "null"),
    heights = unit(c(1.05, 1.0), "null")
  )))
  print(p_a, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
  print(p_b, vp = viewport(layout.pos.row = 1, layout.pos.col = 2))
  print(p_c, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  print(p_d, vp = viewport(layout.pos.row = 2, layout.pos.col = 2))
}

base_name <- file.path(out_dir, "figure_sol3d_diagnostics")
w <- width_mm / 25.4
h <- height_mm / 25.4

svg(paste0(base_name, ".svg"), width = w, height = h, family = "Helvetica", onefile = TRUE)
draw_figure()
dev.off()

grDevices::cairo_pdf(paste0(base_name, ".pdf"), width = w, height = h, family = "Helvetica")
draw_figure()
dev.off()

png(paste0(base_name, ".png"), width = w, height = h, units = "in", res = dpi, type = "cairo")
draw_figure()
dev.off()

qa <- c(
  "Figure: Sol3D diagnostics",
  "Core conclusion: ESMFold-derived 3D surface/exposure proxies provide a structural diagnostic layer for Protein-Sol variation.",
  "Archetype: quantitative grid with one hero scatter panel and three supporting evidence panels.",
  "Backend: R only; ggplot2 + grid; exported as SVG, PDF and PNG.",
  "Source data: figure_sol3d_source_data.csv.",
  "Statistics: panel a uses per-sequence n from the merged Sol3D table; panel b reports Spearman correlations; panel c reports mean +/- s.e.m. over n=8 per method.",
  "Integrity note: *_proxy features are CA-neighborhood exposure proxies, not exact physical SASA."
)
writeLines(qa, file.path(out_dir, "figure_sol3d_QA_notes.txt"))

cat("[OK] wrote -> ", paste(paste0(base_name, c(".svg", ".pdf", ".png")), collapse = ", "), "\n", sep = "")
cat("[OK] source data -> ", file.path(out_dir, "figure_sol3d_source_data.csv"), "\n", sep = "")
cat("[OK] QA notes -> ", file.path(out_dir, "figure_sol3d_QA_notes.txt"), "\n", sep = "")
