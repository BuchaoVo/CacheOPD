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

root <- normalizePath(get_arg("--root", "."), mustWork = TRUE)
out_root <- file.path(root, "paper_figures_cacheopd")
fig_dir <- file.path(out_root, "figures_editable")
src_dir <- file.path(out_root, "source_data")
qa_dir <- file.path(out_root, "figure_QA")
paper_image_dir <- file.path(root, "paper_md_report", "images")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(src_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(qa_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(paper_image_dir, recursive = TRUE, showWarnings = FALSE)

width_mm <- as.numeric(get_arg("--width_mm", "183"))
height_mm <- as.numeric(get_arg("--height_mm", "122"))
dpi <- as.numeric(get_arg("--dpi", "600"))

read_csv <- function(rel) {
  path <- file.path(root, rel)
  if (!file.exists(path)) stop("Missing input file: ", path)
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

num <- function(x) suppressWarnings(as.numeric(x))

method_alias <- function(x) {
  x <- as.character(x)
  x[x == "CacheOPD Full"] <- "CacheOPD-Full"
  x[x == "CacheOPD Sparse"] <- "CacheOPD-Sparse"
  x[x == "Offline cached PoE-Full"] <- "CacheOPD-Full"
  x[x == "FinalUtility-ShiftEntropy-q04-High50"] <- "CacheOPD-Sparse"
  x[x == "ProteinOPD"] <- "Online OPD"
  x
}

method_family <- function(x) {
  x <- method_alias(x)
  ifelse(x %in% c("CacheOPD-Full", "CacheOPD-Sparse", "CacheOPD"), "CacheOPD",
    ifelse(x %in% c("Online OPD"), "Online OPD",
      ifelse(grepl("KD|teacher", x), "KD", ifelse(grepl("SFT", x), "SFT", "Base"))
    )
  )
}

method_order <- c(
  "ProLLaMA", "SFT", "ProbAvg-KD", "LogitAvg-KD", "Online OPD",
  "CacheOPD-Full", "CacheOPD-Sparse",
  "Fold-teacher KD", "Sol-teacher KD", "Thermo-teacher KD"
)

family_cols <- c(
  "Base" = "#6B7280",
  "SFT" = "#A7A7A7",
  "KD" = "#7AA6C2",
  "Online OPD" = "#D08C45",
  "CacheOPD" = "#3A8F84",
  "Other" = "#B0B0B0"
)

signal_cols <- c(
  "CacheOPD-Full" = "#2D7F73",
  "CacheOPD-Sparse" = "#49A397",
  "Online OPD" = "#D08C45",
  "SFT" = "#A7A7A7",
  "ProLLaMA" = "#6B7280",
  "ProbAvg-KD" = "#8AB6D6",
  "LogitAvg-KD" = "#5B8DB8"
)

theme_nature <- function(base_size = 6.6, base_family = "Helvetica") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = "black"),
      axis.ticks = element_line(linewidth = 0.30, colour = "black"),
      axis.ticks.length = unit(1.45, "mm"),
      axis.title = element_text(size = base_size, colour = "black"),
      axis.text = element_text(size = base_size - 0.6, colour = "black"),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.8),
      legend.key.height = unit(3.0, "mm"),
      legend.key.width = unit(3.8, "mm"),
      strip.background = element_blank(),
      strip.text = element_text(size = base_size - 0.2, face = "bold"),
      plot.title = element_text(size = base_size + 0.4, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size - 0.8, colour = "#4B5563", hjust = 0),
      panel.grid.major.y = element_line(linewidth = 0.16, colour = "#E8EAED"),
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

draw_grid <- function(plots, nrow, ncol, widths = NULL, heights = NULL) {
  grid.newpage()
  if (is.null(widths)) widths <- rep(1, ncol)
  if (is.null(heights)) heights <- rep(1, nrow)
  pushViewport(viewport(layout = grid.layout(
    nrow = nrow, ncol = ncol,
    widths = unit(widths, "null"),
    heights = unit(heights, "null")
  )))
  for (item in plots) {
    print(item$plot, vp = viewport(layout.pos.row = item$row, layout.pos.col = item$col))
  }
}

save_grid <- function(name, plots, nrow, ncol, widths = NULL, heights = NULL,
                      width_mm_use = width_mm, height_mm_use = height_mm) {
  base <- file.path(fig_dir, name)
  w <- width_mm_use / 25.4
  h <- height_mm_use / 25.4
  svg(paste0(base, ".svg"), width = w, height = h, family = "Helvetica", onefile = TRUE)
  draw_grid(plots, nrow, ncol, widths, heights)
  dev.off()
  grDevices::cairo_pdf(paste0(base, ".pdf"), width = w, height = h, family = "Helvetica")
  draw_grid(plots, nrow, ncol, widths, heights)
  dev.off()
  png(paste0(base, ".png"), width = w, height = h, units = "in", res = dpi, type = "cairo")
  draw_grid(plots, nrow, ncol, widths, heights)
  dev.off()
}

write_qa <- function(name, lines) {
  writeLines(lines, file.path(qa_dir, paste0(name, "_QA_notes.txt")))
}

copy_for_latex <- function(src_name, dest_stem) {
  for (ext in c("pdf", "png")) {
    from <- file.path(fig_dir, paste0(src_name, ".", ext))
    to <- file.path(paper_image_dir, paste0(dest_stem, ".", ext))
    if (file.exists(from)) file.copy(from, to, overwrite = TRUE)
  }
}

metric_long <- function(df, method_col = "method", metrics) {
  rows <- list()
  k <- 1
  for (m in metrics) {
    if (!m %in% names(df)) next
    tmp <- data.frame(method = df[[method_col]], metric = m, value = num(df[[m]]), stringsAsFactors = FALSE)
    rows[[k]] <- tmp
    k <- k + 1
  }
  do.call(rbind, rows)
}

## Figure 1: staged CacheOPD framework
framework_nodes <- data.frame(
  lane = c(
    rep("SFT", 3),
    rep("Traditional offline KD", 4),
    rep("Online OPD", 5),
    rep("CacheOPD", 5)
  ),
  x = c(
    1, 2.55, 4.1,
    1, 2.25, 3.45, 4.65,
    1, 2.05, 3.1, 4.15, 5.2,
    1, 2.1, 3.25, 4.35, 5.35
  ),
  y = c(
    rep(4, 3), rep(3, 4), rep(2, 5), rep(1, 5)
  ),
  w = c(
    1.14, 1.18, 0.98,
    0.92, 1.05, 1.05, 1.05,
    0.82, 0.95, 0.95, 0.95, 0.85,
    0.90, 1.05, 1.45, 1.20, 1.25
  ),
  h = c(
    rep(0.42, 3),
    rep(0.42, 4),
    rep(0.42, 5),
    rep(0.48, 5)
  ),
  label = c(
    "Prompts and reference sequences",
    "One-hot reference tokens",
    "Student SFT",
    "Reference prefixes",
    "Cached teacher distributions",
    "Probability/logit averaging",
    "Student KD",
    "Student rollout",
    "Real-time teacher scoring",
    "Normalized PoE target",
    "Student update",
    "Repeated teacher calls",
    "Reference rollouts",
    "Offline multi-teacher cache",
    "Cached PoE target over teacher-supported candidate tokens",
    "Sparse token-position selection",
    "Student training with zero training-stage teacher calls"
  ),
  group = c(
    rep("Base", 3),
    rep("KD", 4),
    rep("Online OPD", 5),
    rep("CacheOPD", 5)
  ),
  stringsAsFactors = FALSE
)
framework_nodes$xmin <- framework_nodes$x - framework_nodes$w / 2
framework_nodes$xmax <- framework_nodes$x + framework_nodes$w / 2
framework_nodes$ymin <- framework_nodes$y - framework_nodes$h / 2
framework_nodes$ymax <- framework_nodes$y + framework_nodes$h / 2
framework_nodes$plot_label <- gsub(" over ", "\nover ", framework_nodes$label, fixed = TRUE)
framework_nodes$plot_label <- gsub(" with zero ", "\nwith zero ", framework_nodes$plot_label, fixed = TRUE)
framework_nodes$plot_label <- gsub(" teacher ", "\nteacher ", framework_nodes$plot_label, fixed = TRUE)
framework_nodes$plot_label <- gsub(" token-position ", "\ntoken-position ", framework_nodes$plot_label, fixed = TRUE)

make_edges <- function(nodes) {
  rows <- list()
  k <- 1
  for (ln in unique(nodes$lane)) {
    sub <- nodes[nodes$lane == ln, ]
    sub <- sub[order(sub$x), ]
    if (nrow(sub) < 2) next
    for (i in seq_len(nrow(sub) - 1)) {
      rows[[k]] <- data.frame(
        lane = ln,
        x = sub$xmax[i] + 0.04,
        xend = sub$xmin[i + 1] - 0.04,
        y = sub$y[i],
        yend = sub$y[i + 1],
        stringsAsFactors = FALSE
      )
      k <- k + 1
    }
  }
  do.call(rbind, rows)
}
framework_edges <- make_edges(framework_nodes)

framework_claims <- data.frame(
  x = c(2.1, 3.25, 4.35, 5.35),
  y = c(0.42, 0.42, 0.42, 0.42),
  label = c(
    "amortized teacher inference",
    "multi-property PoE consensus",
    "token-budget reduction",
    "zero teacher calls during optimization"
  ),
  stringsAsFactors = FALSE
)

write.csv(framework_nodes[, c("lane", "x", "y", "w", "h", "label", "group")],
          file.path(src_dir, "figure1_framework_nodes_source_data.csv"), row.names = FALSE)
write.csv(framework_edges, file.path(src_dir, "figure1_framework_edges_source_data.csv"), row.names = FALSE)
write.csv(framework_claims, file.path(src_dir, "figure1_framework_claims_source_data.csv"), row.names = FALSE)

framework_cols <- c(
  "Base" = "#F1F3F5",
  "KD" = "#DFECF5",
  "Online OPD" = "#F3E1CC",
  "CacheOPD" = "#DDEFEA"
)
framework_border <- c(
  "Base" = "#8A8F98",
  "KD" = "#5B8DB8",
  "Online OPD" = "#D08C45",
  "CacheOPD" = "#2D7F73"
)

p1 <- ggplot() +
  geom_rect(data = data.frame(xmin = 0.44, xmax = 5.96, ymin = 0.67, ymax = 1.33),
            aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
            fill = "#F3FAF7", colour = "#2D7F73", linewidth = 0.42) +
  geom_segment(
    data = framework_edges,
    aes(x = x, xend = xend, y = y, yend = yend, colour = lane),
    arrow = arrow(length = unit(1.6, "mm"), type = "closed"),
    linewidth = 0.42,
    lineend = "round"
  ) +
  geom_rect(
    data = framework_nodes,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = group, colour = group),
    linewidth = 0.42
  ) +
  geom_text(
    data = framework_nodes,
    aes(x = x, y = y, label = plot_label),
    size = 1.85,
    lineheight = 0.88,
    family = "Helvetica",
    colour = "#111827"
  ) +
  geom_text(
    data = data.frame(
      x = 0.18,
      y = c(4, 3, 2, 1),
      label = c("SFT", "Offline KD", "Online OPD", "CacheOPD")
    ),
    aes(x = x, y = y, label = label),
    hjust = 0,
    size = 2.45,
    fontface = "bold",
    family = "Helvetica"
  ) +
  geom_text(
    data = framework_claims,
    aes(x = x, y = y, label = label),
    size = 1.75,
    family = "Helvetica",
    colour = "#2D7F73"
  ) +
  scale_fill_manual(values = framework_cols) +
  scale_colour_manual(values = c(framework_border, "SFT" = "#8A8F98", "Traditional offline KD" = "#5B8DB8", "Online OPD" = "#D08C45", "CacheOPD" = "#2D7F73")) +
  coord_cartesian(xlim = c(0.12, 6.03), ylim = c(0.24, 4.52), clip = "off") +
  labs(title = "CacheOPD staged workflow", subtitle = "Offline multi-teacher cache construction followed by cached student optimization") +
  theme_void(base_size = 6.8, base_family = "Helvetica") +
  theme(
    plot.title = element_text(size = 8.2, face = "bold", hjust = 0),
    plot.subtitle = element_text(size = 6.2, colour = "#4B5563", hjust = 0),
    plot.margin = margin(5, 7, 5, 7),
    legend.position = "none"
  )

save_grid(
  "figure1_cacheopd_framework_editable",
  list(list(plot = p1, row = 1, col = 1)),
  nrow = 1, ncol = 1, width_mm_use = 183, height_mm_use = 94
)
write_qa("figure1_cacheopd_framework_editable", c(
  "Figure: staged CacheOPD method overview.",
  "Source data: figure1_framework_nodes_source_data.csv, figure1_framework_edges_source_data.csv, figure1_framework_claims_source_data.csv.",
  "The schematic contrasts SFT, traditional offline KD, Online OPD, and CacheOPD.",
  "Required terminology is used: cached PoE target over teacher-supported candidate tokens; offline multi-teacher cache; zero training-stage teacher calls; sparse token-position selection.",
  "Editable outputs: SVG/PDF/PNG generated by R."
))

## Figure 2: quality-cost and cost decomposition
quality <- read_csv("results/cacheopd_paper/quality_cost_plot.csv")
cost <- read_csv("results/cacheopd_paper/cost_decomposition.csv")
quality$method <- method_alias(quality$method)
cost$method <- method_alias(cost$method)
quality$family <- method_family(quality$method)
cost$family <- method_family(cost$method)
quality$teacher_calls_plot <- pmax(num(quality$teacher_calls), 1)
quality$effective_tokens_num <- num(quality$effective_tokens)
quality$quality_index <- num(quality$quality_index)
quality$PPL <- num(quality$PPL)
quality$pLDDT <- num(quality$pLDDT)
quality$Sol <- num(quality$Sol)
quality$Thermo <- num(quality$Thermo)
cost$total_teacher_calls <- num(cost$total_teacher_calls)
cost$effective_tokens <- num(cost$effective_tokens)
quality_labels <- quality[quality$method %in% c("ProLLaMA", "SFT", "Online OPD", "CacheOPD-Full", "CacheOPD-Sparse"), ]
quality_labels$label_x <- quality_labels$teacher_calls_plot
quality_labels$label_x[quality_labels$label_x <= 1] <- 1.22
quality_labels$label_hjust <- ifelse(quality_labels$method == "Online OPD", 1, 0)

fig2_source <- merge(
  quality,
  cost[, c("method", "cache_teacher_calls", "training_teacher_calls", "total_teacher_calls", "student_gpu_hours", "effective_tokens")],
  by = "method",
  all.x = TRUE,
  suffixes = c("", "_cost")
)
write.csv(fig2_source, file.path(src_dir, "figure2_quality_cost_source_data.csv"), row.names = FALSE)

p2a <- ggplot(quality, aes(x = teacher_calls_plot, y = quality_index)) +
  geom_point(aes(fill = family), shape = 21, size = 3.1, stroke = 0.35, colour = "white") +
  geom_text(data = quality_labels, aes(x = label_x, y = quality_index, label = method, hjust = label_hjust),
            nudge_y = 0.018, size = 2.0, family = "Helvetica", check_overlap = TRUE) +
  scale_x_log10(breaks = c(1, 100, 1000, 6144), labels = c("0", "100", "1k", "6.1k"), expand = expansion(mult = c(0.05, 0.14))) +
  scale_fill_manual(values = family_cols, drop = FALSE) +
  labs(title = "Quality-cost frontier", subtitle = "Teacher calls shown on log scale", x = "Teacher calls", y = "Quality index") +
  annotate("text", x = 1, y = Inf, label = "a", hjust = -0.6, vjust = 1.25,
           size = 3.0, fontface = "bold", family = "Helvetica") +
  theme(legend.position = c(0.05, 0.08), legend.justification = c(0, 0))

cost_plot <- cost[!is.na(cost$total_teacher_calls), ]
cost_plot$method <- factor(cost_plot$method, levels = rev(unique(cost_plot$method)))
p2b <- ggplot(cost_plot, aes(x = total_teacher_calls, y = method)) +
  geom_col(aes(fill = family), width = 0.62, colour = "white", linewidth = 0.15) +
  scale_fill_manual(values = family_cols, drop = FALSE) +
  scale_x_continuous(labels = comma) +
  labs(title = "Teacher-call decomposition", subtitle = "Online scoring is amortized by cached targets", x = "Total teacher calls", y = NULL) +
  panel_label("b") +
  theme(legend.position = "none", panel.grid.major.y = element_blank())

metrics2 <- metric_long(quality, metrics = c("PPL", "pLDDT", "Sol", "Thermo"))
metrics2$method <- factor(metrics2$method, levels = method_order)
metrics2$metric <- factor(metrics2$metric, levels = c("PPL", "pLDDT", "Sol", "Thermo"))
p2c <- ggplot(metrics2, aes(x = method, y = value, fill = method_family(method))) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.12) +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = family_cols, drop = FALSE) +
  labs(title = "Matched conditional metrics", subtitle = "Same backbone, prompts, decoding and evaluators", x = NULL, y = "Mean score") +
  geom_text(data = data.frame(metric = factor("PPL", levels = levels(metrics2$metric)), x = -Inf, y = Inf, label = "c"),
            aes(x = x, y = y, label = label), inherit.aes = FALSE,
            hjust = -0.45, vjust = 1.25, size = 3.0, fontface = "bold", family = "Helvetica") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

save_grid(
  "figure2_quality_cost_tradeoff_editable",
  list(list(plot = p2a, row = 1, col = 1), list(plot = p2b, row = 1, col = 2), list(plot = p2c, row = 2, col = 1)),
  nrow = 2, ncol = 2, widths = c(1.2, 1), heights = c(1, 1.0)
)
write_qa("figure2_quality_cost_tradeoff_editable", c(
  "Figure: main quality-cost tradeoff.",
  "Source data: source_data/figure2_quality_cost_source_data.csv.",
  "Panels: a quality index vs teacher calls; b total teacher calls; c matched conditional metrics.",
  "Editable outputs: SVG/PDF/PNG generated by R.",
  "Caveat: teacher_calls=0 is plotted at 1 on log scale and labelled as 0."
))

## Figure 3: Sparse/Full and token selection diagnostics
sparse_boot <- read_csv("results/cacheopd_paper/sparse_full_bootstrap.csv")
sel_stats <- read_csv("results/cacheopd_paper/selected_unselected_stats.csv")
cache_diag <- read_csv("results/cacheopd_paper/cache_position_diagnostics.csv")
sparse_ratio_table <- read_csv("paper_md_report/tables/table4_sparse_budget_quality.csv")

sparse_boot$delta_sparse_minus_full <- num(sparse_boot$delta_sparse_minus_full)
sparse_boot$ci_low <- num(sparse_boot$ci_low)
sparse_boot$ci_high <- num(sparse_boot$ci_high)
sparse_boot$prob_sparse_better <- num(sparse_boot$prob_sparse_better)
sparse_boot$metric <- factor(sparse_boot$metric, levels = rev(sparse_boot$metric))
sel_stats$selected_label <- ifelse(as.character(sel_stats$selected) == "TRUE", "Selected", "Unselected")
cache_diag$selected_label <- ifelse(as.character(cache_diag$selected) == "TRUE", "Selected", "Unselected")
cache_diag$utility <- num(cache_diag$utility)
if ("utility_clean" %in% names(cache_diag)) {
  cache_diag$utility_clean <- num(cache_diag$utility_clean)
} else {
  cache_diag$utility_clean <- cache_diag$utility
}
cache_diag$preference_shift_js <- num(cache_diag$preference_shift_js)
cache_diag$teacher_conflict <- num(cache_diag$teacher_conflict)
cache_diag$poe_entropy <- num(cache_diag$poe_entropy)
sparse_ratio <- data.frame(
  method = sparse_ratio_table$Variant,
  effective_tokens = num(gsub(",", "", sparse_ratio_table$Tokens)),
  sparse_ratio_vs_full = num(gsub("%", "", sparse_ratio_table$`Preference-token ratio`)) / 100,
  ppl = num(sparse_ratio_table$`PPL↓`),
  plddt = num(sparse_ratio_table$`pLDDT↑`),
  pae = num(sparse_ratio_table$`pAE↓`),
  ptm = num(sparse_ratio_table$`pTM↑`),
  sol = num(sparse_ratio_table$`Sol↑`),
  thermo = num(sparse_ratio_table$`Thermo↑`),
  status = sparse_ratio_table$Status,
  stringsAsFactors = FALSE
)
sparse_ratio <- sparse_ratio[order(sparse_ratio$sparse_ratio_vs_full), ]
sparse_ratio$display <- paste0(sprintf("%.1f", 100 * sparse_ratio$sparse_ratio_vs_full), "% sparse")
sparse_ratio$display[sparse_ratio$method == "CacheOPD-Full"] <- "100% full"

fig3_source <- rbind(
  data.frame(panel = "bootstrap", sparse_boot[, c("metric", "delta_sparse_minus_full", "ci_low", "ci_high", "prob_sparse_better")]),
  data.frame(panel = "bootstrap", sparse_boot[, c("metric", "delta_sparse_minus_full", "ci_low", "ci_high", "prob_sparse_better")])
)
write.csv(sparse_boot, file.path(src_dir, "figure3_sparse_full_bootstrap_source_data.csv"), row.names = FALSE)
write.csv(sel_stats, file.path(src_dir, "figure3_selected_unselected_source_data.csv"), row.names = FALSE)
write.csv(cache_diag, file.path(src_dir, "figure3_cache_position_source_data.csv"), row.names = FALSE)
write.csv(sparse_ratio, file.path(src_dir, "figure3_sparse_ratio_source_data.csv"), row.names = FALSE)

p3a <- ggplot(sparse_boot, aes(x = delta_sparse_minus_full, y = metric)) +
  geom_vline(xintercept = 0, linewidth = 0.32, colour = "#777777") +
  geom_segment(aes(x = ci_low, xend = ci_high, yend = metric), linewidth = 0.5, colour = "#6B7280") +
  geom_point(aes(fill = ifelse(prob_sparse_better >= 0.5, "Sparse better", "Full better")), shape = 21, size = 2.4, colour = "white", stroke = 0.25) +
  scale_fill_manual(values = c("Sparse better" = "#49A397", "Full better" = "#6B7280")) +
  labs(title = "Sparse approximation uncertainty", subtitle = "Delta = Sparse - Full; bootstrap 95% CI", x = "Delta", y = NULL) +
  panel_label("a") +
  theme(legend.position = c(0.03, 0.05), legend.justification = c(0, 0), panel.grid.major.y = element_blank())

sel_long <- data.frame(
  selected_label = rep(sel_stats$selected_label, 4),
  metric = rep(c("Utility", "Pref. shift", "Conflict", "PoE entropy"), each = nrow(sel_stats)),
  value = c(num(sel_stats$utility_mean), num(sel_stats$preference_shift_js_mean), num(sel_stats$teacher_conflict_mean), num(sel_stats$poe_entropy_mean))
)
sel_long$metric <- factor(sel_long$metric, levels = c("Utility", "Pref. shift", "Conflict", "PoE entropy"))
p3b <- ggplot(sel_long, aes(x = selected_label, y = value, fill = selected_label)) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.15) +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c("Selected" = "#3A8F84", "Unselected" = "#B7BEC7")) +
  labs(title = "Selected token positions carry stronger utility", subtitle = "Selected vs unselected cached positions", x = NULL, y = "Mean value") +
  geom_text(data = data.frame(metric = factor("Utility", levels = levels(sel_long$metric)), x = -Inf, y = Inf, label = "b"),
            aes(x = x, y = y, label = label), inherit.aes = FALSE,
            hjust = -0.45, vjust = 1.25, size = 3.0, fontface = "bold", family = "Helvetica") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 25, hjust = 1))

sampled <- cache_diag[is.finite(cache_diag$utility_clean), ]
if (nrow(sampled) > 6000) sampled <- sampled[seq(1, nrow(sampled), length.out = 6000), ]
p3c <- ggplot(sampled, aes(x = preference_shift_js, y = utility_clean, colour = selected_label)) +
  geom_point(size = 0.45, alpha = 0.22) +
  scale_colour_manual(values = c("Selected" = "#2D7F73", "Unselected" = "#9CA3AF")) +
  labs(title = "Token utility landscape", subtitle = "Subsampled token positions for readability", x = "Preference shift JS", y = "Clean utility") +
  panel_label("c") +
  theme(legend.position = c(0.05, 0.08), legend.justification = c(0, 0))

p3d <- ggplot(sparse_ratio, aes(x = sparse_ratio_vs_full, y = plddt)) +
  geom_path(colour = "#3A8F84", linewidth = 0.45) +
  geom_point(fill = "#3A8F84", shape = 21, size = 2.3, colour = "white", stroke = 0.25) +
  scale_x_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0.10, 0.10))) +
  labs(title = "Sparse-ratio sensitivity", subtitle = "Available sparse-ratio runs", x = "Token ratio vs Full", y = "pLDDT") +
  panel_label("d")

save_grid(
  "figure3_token_selection_sparse_editable",
  list(list(plot = p3a, row = 1, col = 1), list(plot = p3b, row = 1, col = 2), list(plot = p3c, row = 2, col = 1), list(plot = p3d, row = 2, col = 2)),
  nrow = 2, ncol = 2, widths = c(1.15, 1), heights = c(1, 1)
)
write_qa("figure3_token_selection_sparse_editable", c(
  "Figure: Sparse/Full approximation and token-selection diagnostics.",
  "Source data: figure3_sparse_full_bootstrap_source_data.csv, figure3_selected_unselected_source_data.csv, figure3_cache_position_source_data.csv, figure3_sparse_ratio_source_data.csv.",
  "Panels: a bootstrap deltas; b selected/unselected token means; c clean token utility landscape; d sparse-ratio sensitivity if available.",
  "Editable outputs: SVG/PDF/PNG generated by R.",
  "Caveat: panel c is visually subsampled when token rows exceed 6000."
))

sparse_ratio_long <- metric_long(sparse_ratio, metrics = c("ppl", "plddt", "pae", "ptm", "sol", "thermo"))
sparse_ratio_long$display <- rep(sparse_ratio$display, 6)
sparse_ratio_long$sparse_ratio_vs_full <- rep(sparse_ratio$sparse_ratio_vs_full, 6)
sparse_ratio_long$metric_label <- sparse_ratio_long$metric
sparse_ratio_long$metric_label[sparse_ratio_long$metric == "ppl"] <- "PPL"
sparse_ratio_long$metric_label[sparse_ratio_long$metric == "plddt"] <- "pLDDT"
sparse_ratio_long$metric_label[sparse_ratio_long$metric == "pae"] <- "pAE"
sparse_ratio_long$metric_label[sparse_ratio_long$metric == "ptm"] <- "pTM"
sparse_ratio_long$metric_label[sparse_ratio_long$metric == "sol"] <- "Sol"
sparse_ratio_long$metric_label[sparse_ratio_long$metric == "thermo"] <- "Thermo"
write.csv(sparse_ratio_long, file.path(src_dir, "appendix_sparse_ratio_curve_source_data.csv"), row.names = FALSE)

p_sparse_ratio <- ggplot(sparse_ratio_long, aes(x = sparse_ratio_vs_full, y = value)) +
  geom_line(colour = "#3A8F84", linewidth = 0.42) +
  geom_point(shape = 21, size = 2.1, fill = "#3A8F84", colour = "white", stroke = 0.25) +
  facet_wrap(~ metric_label, scales = "free_y", nrow = 2) +
  scale_x_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0.10, 0.10))) +
  labs(title = "Sparse-ratio sensitivity across evaluation metrics",
       subtitle = "Available token-budget runs; PPL and pAE are lower-is-better",
       x = "Token ratio vs CacheOPD-Full", y = "Mean score") +
  theme(legend.position = "none")

save_grid(
  "appendix_sparse_ratio_curve_editable",
  list(list(plot = p_sparse_ratio, row = 1, col = 1)),
  nrow = 1, ncol = 1, width_mm_use = 183, height_mm_use = 92
)
write_qa("appendix_sparse_ratio_curve_editable", c(
  "Figure: sparse-ratio sensitivity curve.",
  "Source data: appendix_sparse_ratio_curve_source_data.csv.",
  "This figure exposes all available sparse-ratio runs rather than only the default CacheOPD-Sparse point.",
  "Editable outputs: SVG/PDF/PNG generated by R."
))

## Figure 4: final validation and ablations
fv <- read_csv("paper_figures_cacheopd/tables/table3_final_validation_ablation.csv")
fv$Tokens <- num(fv$Tokens)
for (m in c("PPL", "Sol", "Thermo", "Thermophilic", "pLDDT", "pAE", "pTM")) fv[[m]] <- num(fv[[m]])
write.csv(fv, file.path(src_dir, "figure4_final_validation_source_data.csv"), row.names = FALSE)

fv_long <- metric_long(fv, method_col = "Display", metrics = c("PPL", "Sol", "Thermo", "pLDDT", "pAE", "pTM"))
fv_long$Block <- rep(fv$Block, 6)
fv_long$Display <- factor(fv_long$method, levels = rev(fv$Display))

fv_structure_ready <- fv[is.finite(fv$Tokens) & is.finite(fv$pLDDT), ]
fv_long_finite <- fv_long[is.finite(fv_long$value), ]
fv_structure_ready$point_label <- ifelse(
  fv_structure_ready$Display %in% c("Shift+Entropy q0.2 High50", "Shift+Entropy q0.2 Random50", "Anchor gamma=1"),
  fv_structure_ready$Display,
  ""
)

p4a <- ggplot(fv_structure_ready, aes(x = Tokens, y = pLDDT, fill = Block)) +
  geom_point(shape = 21, size = 2.6, colour = "white", stroke = 0.25) +
  geom_text(aes(label = point_label), size = 1.75, family = "Helvetica", nudge_y = 0.32, check_overlap = TRUE) +
  scale_fill_manual(values = c("utility_controls" = "#3A8F84", "smoothing" = "#7AA6C2", "q_delta" = "#D08C45", "anchor" = "#8E78B9")) +
  scale_x_continuous(expand = expansion(mult = c(0.10, 0.14))) +
  labs(title = "Final validation quality-token trade-off", subtitle = "Each point is one final-validation design", x = "Effective tokens", y = "pLDDT") +
  panel_label("a") +
  theme(legend.position = c(0.05, 0.08), legend.justification = c(0, 0))

p4b <- ggplot(fv_long_finite[fv_long_finite$metric %in% c("PPL", "Sol", "Thermo"), ], aes(x = Display, y = value, fill = Block)) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.12) +
  coord_flip() +
  facet_wrap(~ metric, scales = "free_x", nrow = 1) +
  scale_fill_manual(values = c("utility_controls" = "#3A8F84", "smoothing" = "#7AA6C2", "q_delta" = "#D08C45", "anchor" = "#8E78B9")) +
  labs(title = "Utility controls and sensitivity", subtitle = "PPL lower is better; Sol/Thermo higher is better", x = NULL, y = "Score") +
  geom_text(data = data.frame(metric = "PPL", x = -Inf, y = Inf, label = "b"),
            aes(x = x, y = y, label = label), inherit.aes = FALSE,
            hjust = -0.45, vjust = 1.25, size = 3.0, fontface = "bold", family = "Helvetica") +
  theme(legend.position = "none", panel.grid.major.y = element_blank())

p4c <- ggplot(fv_long_finite[fv_long_finite$metric %in% c("pLDDT", "pAE", "pTM"), ], aes(x = Display, y = value, fill = Block)) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.12) +
  coord_flip() +
  facet_wrap(~ metric, scales = "free_x", nrow = 1) +
  scale_fill_manual(values = c("utility_controls" = "#3A8F84", "smoothing" = "#7AA6C2", "q_delta" = "#D08C45", "anchor" = "#8E78B9")) +
  labs(title = "Structure-side validation", subtitle = "pLDDT/pTM higher; pAE lower", x = NULL, y = "Score") +
  geom_text(data = data.frame(metric = "pLDDT", x = -Inf, y = Inf, label = "c"),
            aes(x = x, y = y, label = label), inherit.aes = FALSE,
            hjust = -0.45, vjust = 1.25, size = 3.0, fontface = "bold", family = "Helvetica") +
  theme(legend.position = "none", panel.grid.major.y = element_blank())

save_grid(
  "figure4_final_validation_editable",
  list(list(plot = p4a, row = 1, col = 1), list(plot = p4b, row = 2, col = 1), list(plot = p4c, row = 2, col = 2)),
  nrow = 2, ncol = 2, widths = c(1, 1), heights = c(0.95, 1.2)
)
write_qa("figure4_final_validation_editable", c(
  "Figure: final-validation ablations.",
  "Source data: source_data/figure4_final_validation_source_data.csv.",
  "Panels: a token-quality tradeoff; b utility/property metrics; c structure metrics.",
  "Editable outputs: SVG/PDF/PNG generated by R.",
  "Caveat: the figure summarizes available final-validation rows rather than multi-seed retraining."
))

## Appendix: teacher/cache diagnostics
teacher <- read_csv("results/cacheopd_paper/teacher_pairwise_summary.csv")
teacher$topk_overlap_mean <- num(teacher$topk_overlap_mean)
teacher$js_mean <- num(teacher$js_mean)
teacher$top1_agreement <- num(teacher$top1_agreement)
teacher_long <- data.frame(
  teacher_pair = rep(teacher$teacher_pair, 3),
  metric = rep(c("JSD", "Top-k overlap", "Top-1 agreement"), each = nrow(teacher)),
  value = c(teacher$js_mean, teacher$topk_overlap_mean, teacher$top1_agreement)
)
teacher_long$teacher_pair <- factor(teacher_long$teacher_pair, levels = teacher$teacher_pair)
write.csv(teacher_long, file.path(src_dir, "appendix_teacher_cache_source_data.csv"), row.names = FALSE)

p5a <- ggplot(teacher_long, aes(x = teacher_pair, y = metric, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.35) +
  geom_text(aes(label = sprintf("%.3f", value)), size = 2.1, family = "Helvetica") +
  scale_fill_gradient(low = "#F2F4F7", high = "#2D7F73") +
  labs(title = "Teacher pairwise diagnostics", subtitle = "High overlap indicates weak teacher separation", x = NULL, y = NULL) +
  panel_label("a") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "right", panel.grid = element_blank())

cache_sample <- cache_diag
if (nrow(cache_sample) > 8000) cache_sample <- cache_sample[seq(1, nrow(cache_sample), length.out = 8000), ]
p5b <- ggplot(cache_sample, aes(x = poe_entropy, fill = selected_label)) +
  geom_density(alpha = 0.45, linewidth = 0.35) +
  scale_fill_manual(values = c("Selected" = "#3A8F84", "Unselected" = "#B7BEC7")) +
  labs(title = "PoE target entropy distribution", subtitle = "Selected positions are compared with unselected cache positions", x = "PoE entropy", y = "Density") +
  panel_label("b") +
  theme(legend.position = c(0.05, 0.86), legend.justification = c(0, 1))

p5c <- ggplot(cache_sample, aes(x = teacher_conflict, y = preference_shift_js, colour = selected_label)) +
  geom_point(size = 0.45, alpha = 0.22) +
  scale_colour_manual(values = c("Selected" = "#2D7F73", "Unselected" = "#9CA3AF")) +
  labs(title = "Conflict and preference shift", subtitle = "Token-level cache diagnostics", x = "Teacher conflict", y = "Preference shift JS") +
  panel_label("c") +
  theme(legend.position = "none")

save_grid(
  "appendix_teacher_cache_diagnostics_editable",
  list(list(plot = p5a, row = 1, col = 1), list(plot = p5b, row = 1, col = 2), list(plot = p5c, row = 2, col = 1)),
  nrow = 2, ncol = 2, widths = c(1.05, 1), heights = c(1, 1)
)
write_qa("appendix_teacher_cache_diagnostics_editable", c(
  "Figure: teacher/cache diagnostics.",
  "Source data: source_data/appendix_teacher_cache_source_data.csv plus results/cacheopd_paper/cache_position_diagnostics.csv.",
  "Panels: a teacher pairwise heatmap; b PoE entropy distribution; c conflict-preference shift landscape.",
  "Editable outputs: SVG/PDF/PNG generated by R.",
  "Caveat: teacher-disagreement modeling is diagnostic/future-work framing, not the main method claim."
))

## Appendix: sequence and metric diagnostics
seq_df <- read_csv("analysis_outputs/conditional_supplement/summaries/merged_per_sequence_metrics.csv")
seq_df$method <- method_alias(seq_df$method)
seq_df$family <- method_family(seq_df$method)
seq_df$method <- factor(seq_df$method, levels = method_order)
seq_df$length <- num(seq_df$length)
for (m in c("ppl", "plddt", "pae", "ptm", "sol", "thermo")) seq_df[[m]] <- num(seq_df[[m]])

aa_classes <- list(
  hydrophobic = strsplit("AVILMFWYC", "")[[1]],
  polar = strsplit("STNQCGP", "")[[1]],
  charged = strsplit("DEKRH", "")[[1]],
  acidic = strsplit("DE", "")[[1]],
  basic = strsplit("KRH", "")[[1]]
)

class_fraction <- function(seq, cls) {
  chars <- strsplit(as.character(seq), "")[[1]]
  if (length(chars) == 0) return(NA_real_)
  sum(chars %in% cls) / length(chars)
}

comp_rows <- list()
k <- 1
for (i in seq_len(nrow(seq_df))) {
  seq <- seq_df$sequence[i]
  for (cls_name in names(aa_classes)) {
    comp_rows[[k]] <- data.frame(
      method = as.character(seq_df$method[i]),
      family = seq_df$family[i],
      aa_class = cls_name,
      fraction = class_fraction(seq, aa_classes[[cls_name]]),
      stringsAsFactors = FALSE
    )
    k <- k + 1
  }
}
comp <- do.call(rbind, comp_rows)
comp_summary <- aggregate(fraction ~ method + family + aa_class, data = comp, FUN = mean, na.rm = TRUE)
comp_summary$method <- factor(comp_summary$method, levels = method_order)
comp_summary$aa_class <- factor(comp_summary$aa_class, levels = c("hydrophobic", "polar", "charged", "acidic", "basic"))

metrics <- c("length", "ppl", "plddt", "pae", "ptm", "sol", "thermo")
corr_rows <- list()
k <- 1
for (a in metrics) {
  for (b in metrics) {
    ok <- is.finite(seq_df[[a]]) & is.finite(seq_df[[b]])
    val <- if (sum(ok) >= 3) cor(rank(seq_df[[a]][ok]), rank(seq_df[[b]][ok]), method = "pearson") else NA_real_
    corr_rows[[k]] <- data.frame(metric_x = a, metric_y = b, spearman = val, stringsAsFactors = FALSE)
    k <- k + 1
  }
}
corr_metrics <- do.call(rbind, corr_rows)
corr_metrics$metric_x <- factor(corr_metrics$metric_x, levels = metrics)
corr_metrics$metric_y <- factor(corr_metrics$metric_y, levels = rev(metrics))

write.csv(seq_df, file.path(src_dir, "appendix_sequence_per_sequence_source_data.csv"), row.names = FALSE)
write.csv(comp_summary, file.path(src_dir, "appendix_sequence_composition_source_data.csv"), row.names = FALSE)
write.csv(corr_metrics, file.path(src_dir, "appendix_metric_correlation_source_data.csv"), row.names = FALSE)

p6a <- ggplot(seq_df, aes(x = method, y = length, fill = family)) +
  geom_violin(width = 0.82, alpha = 0.62, colour = NA, trim = TRUE) +
  geom_boxplot(width = 0.16, outlier.size = 0.25, linewidth = 0.22, colour = "#303030", fill = "white") +
  scale_fill_manual(values = family_cols, drop = FALSE) +
  labs(title = "Generated sequence length distribution", subtitle = "Matched conditional cohorts, n = 128 per method", x = NULL, y = "Length") +
  panel_label("a") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

p6b <- ggplot(comp_summary, aes(x = aa_class, y = method, fill = fraction)) +
  geom_tile(colour = "white", linewidth = 0.28) +
  geom_text(aes(label = sprintf("%.2f", fraction)), size = 1.8, family = "Helvetica") +
  scale_fill_gradient(low = "#F4F5F7", high = "#3A8F84", labels = percent_format(accuracy = 1)) +
  labs(title = "Amino-acid class composition", subtitle = "Mean fraction by method", x = NULL, y = NULL) +
  panel_label("b") +
  theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 25, hjust = 1))

p6c <- ggplot(corr_metrics, aes(x = metric_x, y = metric_y, fill = spearman)) +
  geom_tile(colour = "white", linewidth = 0.28) +
  geom_text(aes(label = sprintf("%.2f", spearman)), size = 1.9, family = "Helvetica") +
  scale_fill_gradient2(low = "#C85A4A", mid = "#F7F7F7", high = "#2F6F73", midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Metric correlation structure", subtitle = "Spearman correlation over matched conditional sequences", x = NULL, y = NULL) +
  panel_label("c") +
  theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 35, hjust = 1))

save_grid(
  "appendix_sequence_metric_diagnostics_editable",
  list(list(plot = p6a, row = 1, col = 1), list(plot = p6b, row = 1, col = 2), list(plot = p6c, row = 2, col = 1)),
  nrow = 2, ncol = 2, widths = c(1.2, 1), heights = c(1, 1)
)
write_qa("appendix_sequence_metric_diagnostics_editable", c(
  "Figure: sequence-level and metric diagnostics.",
  "Source data: appendix_sequence_per_sequence_source_data.csv, appendix_sequence_composition_source_data.csv, appendix_metric_correlation_source_data.csv.",
  "Panels: a length distribution; b amino-acid class composition; c metric correlation heatmap.",
  "Editable outputs: SVG/PDF/PNG generated by R.",
  "Caveat: composition classes are coarse physicochemical summaries, not a substitute for structural or evolutionary diversity metrics."
))

## Appendix: fusion ablation from the manuscript table source
fusion <- read_csv("paper_md_report/tables/tableA3_fusion_variant_ablation.csv")
col_ppl <- grep("^PPL", names(fusion), value = TRUE)[1]
col_plddt <- grep("^pLDDT", names(fusion), value = TRUE)[1]
col_pae <- grep("^pAE", names(fusion), value = TRUE)[1]
col_ptm <- grep("^pTM", names(fusion), value = TRUE)[1]
col_sol <- grep("^Sol", names(fusion), value = TRUE)[1]
col_thermo <- grep("^Thermo", names(fusion), value = TRUE)[1]
fusion_source <- data.frame(
  method = fusion$Variant,
  fusion_rule = fusion$`Fusion rule`,
  support = fusion$Support,
  tokens = num(gsub(",", "", fusion$Tokens)),
  entropy = num(fusion$Entropy),
  PPL = num(fusion[[col_ppl]]),
  pLDDT = num(fusion[[col_plddt]]),
  pAE = num(fusion[[col_pae]]),
  pTM = num(fusion[[col_ptm]]),
  Sol = num(fusion[[col_sol]]),
  Thermo = num(fusion[[col_thermo]]),
  stringsAsFactors = FALSE
)
fusion_source$family <- method_family(fusion_source$method)
write.csv(fusion_source, file.path(src_dir, "appendix_fusion_ablation_source_data.csv"), row.names = FALSE)

fusion_long <- metric_long(fusion_source, metrics = c("PPL", "pLDDT", "pAE", "pTM", "Sol", "Thermo"))
fusion_long$family <- rep(fusion_source$family, 6)
fusion_long$method <- factor(fusion_long$method, levels = rev(fusion_source$method))
fusion_long$metric <- factor(fusion_long$metric, levels = c("PPL", "pLDDT", "pAE", "pTM", "Sol", "Thermo"))

p_fusion_a <- ggplot(fusion_source, aes(x = entropy, y = pLDDT, fill = family)) +
  geom_point(shape = 21, size = 2.7, colour = "white", stroke = 0.25) +
  geom_text(aes(label = method), size = 1.65, family = "Helvetica", nudge_y = 0.35, check_overlap = TRUE) +
  scale_fill_manual(values = family_cols, drop = FALSE) +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.22))) +
  coord_cartesian(clip = "off") +
  labs(title = "Fusion rule ablation", subtitle = "Cached PoE variants preserve stronger structure quality", x = "Target entropy", y = "pLDDT") +
  panel_label("a") +
  theme(legend.position = c(0.05, 0.08), legend.justification = c(0, 0))

p_fusion_b <- ggplot(fusion_long, aes(x = method, y = value, fill = family)) +
  geom_col(width = 0.68, colour = "white", linewidth = 0.12) +
  coord_flip() +
  facet_wrap(~ metric, scales = "free_x", nrow = 2) +
  scale_fill_manual(values = family_cols, drop = FALSE) +
  labs(title = "Metric profile by fusion variant", subtitle = "PPL and pAE lower; other metrics higher", x = NULL, y = "Mean score") +
  geom_text(data = data.frame(metric = factor("PPL", levels = levels(fusion_long$metric)), x = -Inf, y = Inf, label = "b"),
            aes(x = x, y = y, label = label), inherit.aes = FALSE,
            hjust = -0.45, vjust = 1.25, size = 3.0, fontface = "bold", family = "Helvetica") +
  theme(legend.position = "none", panel.grid.major.y = element_blank())

save_grid(
  "appendix_fusion_ablation_editable",
  list(list(plot = p_fusion_a, row = 1, col = 1), list(plot = p_fusion_b, row = 1, col = 2)),
  nrow = 1, ncol = 2, widths = c(0.95, 1.35), width_mm_use = 183, height_mm_use = 94
)
write_qa("appendix_fusion_ablation_editable", c(
  "Figure: fusion variant ablation.",
  "Source data: appendix_fusion_ablation_source_data.csv, derived from paper_md_report/tables/tableA3_fusion_variant_ablation.csv.",
  "The figure visualizes cached PoE target variants against SFT and traditional offline KD baselines.",
  "Editable outputs: SVG/PDF/PNG generated by R."
))

manifest <- data.frame(
  figure = c(
    "fig:cacheopd_framework",
    "fig:quality_cost",
    "fig:sparse_curve",
    "fig:final_validation",
    "fig:cache_diagnostics",
    "fig:sequence_metric_diagnostics",
    "fig:fusion_ablation"
  ),
  editable_stem = c(
    "figure1_cacheopd_framework_editable",
    "figure2_quality_cost_tradeoff_editable",
    "figure3_token_selection_sparse_editable; appendix_sparse_ratio_curve_editable",
    "figure4_final_validation_editable",
    "appendix_teacher_cache_diagnostics_editable",
    "appendix_sequence_metric_diagnostics_editable",
    "appendix_fusion_ablation_editable"
  ),
  primary_source_data = c(
    "figure1_framework_nodes_source_data.csv; figure1_framework_edges_source_data.csv; figure1_framework_claims_source_data.csv",
    "figure2_quality_cost_source_data.csv",
    "figure3_sparse_full_bootstrap_source_data.csv; figure3_selected_unselected_source_data.csv; figure3_cache_position_source_data.csv; figure3_sparse_ratio_source_data.csv",
    "figure4_final_validation_source_data.csv",
    "appendix_teacher_cache_source_data.csv; figure3_cache_position_source_data.csv",
    "appendix_sequence_per_sequence_source_data.csv; appendix_sequence_composition_source_data.csv; appendix_metric_correlation_source_data.csv",
    "appendix_fusion_ablation_source_data.csv"
  ),
  script = "paper_figures_cacheopd/scripts/build_all_editable_figures.R",
  output_formats = "svg; pdf; png",
  stringsAsFactors = FALSE
)
write.csv(manifest, file.path(src_dir, "cacheopd_reproducible_figure_manifest.csv"), row.names = FALSE)

copy_for_latex("figure1_cacheopd_framework_editable", "cacheopd_framework")
copy_for_latex("figure2_quality_cost_tradeoff_editable", "cacheopd_quality_cost")
copy_for_latex("figure3_token_selection_sparse_editable", "cacheopd_token_selection")
copy_for_latex("appendix_sparse_ratio_curve_editable", "cacheopd_sparse_curve")
copy_for_latex("appendix_teacher_cache_diagnostics_editable", "cacheopd_cache_diagnostics")
copy_for_latex("appendix_fusion_ablation_editable", "cacheopd_fusion_ablation")
copy_for_latex("figure4_final_validation_editable", "cacheopd_final_validation")
copy_for_latex("appendix_sequence_metric_diagnostics_editable", "cacheopd_sequence_metric_diagnostics")

cat("[OK] editable figures -> ", fig_dir, "\n", sep = "")
cat("[OK] source data -> ", src_dir, "\n", sep = "")
cat("[OK] QA notes -> ", qa_dir, "\n", sep = "")
cat("[OK] LaTeX-ready figure copies -> ", paper_image_dir, "\n", sep = "")
