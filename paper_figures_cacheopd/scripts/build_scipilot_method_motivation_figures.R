#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else getwd()
root <- normalizePath(root, mustWork = TRUE)

fig_dir <- file.path(root, "paper_figures_cacheopd", "figures_scipilot")
src_dir <- file.path(root, "paper_figures_cacheopd", "source_data")
md_fig_dir <- file.path(root, "paper_md_report", "Report MD", "figures")
md_src_dir <- file.path(root, "paper_md_report", "Report MD", "source_data")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(src_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(md_fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(md_src_dir, recursive = TRUE, showWarnings = FALSE)

read_csv <- function(path) {
  read.csv(file.path(root, path), check.names = FALSE, stringsAsFactors = FALSE)
}

num <- function(x) {
  as.numeric(gsub("[% ,]", "", as.character(x)))
}

pick <- function(df, key_col, key, value_col) {
  v <- df[df[[key_col]] == key, value_col, drop = TRUE]
  if (length(v) == 0) return(NA)
  v[[1]]
}

main_tbl <- read_csv("paper_md_report/tables/table2_main_conditional_results.csv")
cost_tbl <- read_csv("paper_md_report/tables/table3_cost_decomposition.csv")
sparse_tbl <- read_csv("paper_md_report/tables/table4_sparse_budget_quality.csv")
selected_tbl <- read_csv("paper_figures_cacheopd/source_data/figure3_selected_unselected_source_data.csv")
teacher_tbl <- read_csv("results/cacheopd_paper/teacher_pairwise_summary.csv")

online_calls <- num(pick(cost_tbl, "Method", "Online OPD", "Total calls"))
cache_calls <- num(pick(cost_tbl, "Method", "CacheOPD-Sparse", "Total calls"))
call_reduction <- pick(cost_tbl, "Method", "CacheOPD-Sparse", "Call reduction vs Online")
full_tokens <- num(pick(cost_tbl, "Method", "CacheOPD-Full", "Tokens"))
sparse_tokens <- num(pick(cost_tbl, "Method", "CacheOPD-Sparse", "Tokens"))
full_quality <- as.numeric(pick(cost_tbl, "Method", "CacheOPD-Full", "Quality index"))
sparse_quality <- as.numeric(pick(cost_tbl, "Method", "CacheOPD-Sparse", "Quality index"))

cache_full <- main_tbl[main_tbl$Method == "CacheOPD-Full", ]
cache_sparse <- main_tbl[main_tbl$Method == "CacheOPD-Sparse", ]
online <- main_tbl[main_tbl$Method == "Online OPD", ]
probavg <- main_tbl[main_tbl$Method == "ProbAvg-KD", ]
logitavg <- main_tbl[main_tbl$Method == "LogitAvg-KD", ]

mean_teacher_overlap <- mean(as.numeric(teacher_tbl$topk_overlap_mean), na.rm = TRUE)
mean_teacher_js <- mean(as.numeric(teacher_tbl$js_mean), na.rm = TRUE)

method_nodes <- data.frame(
  panel = c(rep("Phase I", 3), rep("Phase II", 3), rep("Phase III", 3)),
  node = c(
    "Reference prompts and rollouts",
    "Three property teachers",
    "Cached teacher supports",
    "Temperature-calibrated PoE",
    "Support gate and utility",
    "Global sparse mask",
    "Static cache minibatches",
    "Student optimization",
    "Conditional generation"
  ),
  evidence = c(
    "128 matched conditional samples; reference visitation distribution",
    "Foldability, solubility, thermostability teachers",
    "Top-K teacher-supported candidate tokens plus exact log probabilities",
    "q(v) proportional to product_m p_m(v)^(lambda_m/tau_m)",
    "Reference support mass gate; U = KL(cached PoE target || reference)",
    paste0("CacheOPD-Sparse keeps ", format(sparse_tokens, big.mark = ","), " tokens vs ",
           format(full_tokens, big.mark = ","), " full-cache tokens"),
    "No teacher model is called during optimization",
    "Preference loss plus LM and EOS anchors",
    "Same prompts, decoding, post-processing and evaluators as baselines"
  ),
  stringsAsFactors = FALSE
)

method_edges <- data.frame(
  from = method_nodes$node[-nrow(method_nodes)],
  to = method_nodes$node[-1],
  relation = c(
    "teacher scoring", "support union", "PoE fusion", "utility scoring",
    "sparsification", "cache loader", "student update", "generation"
  ),
  stringsAsFactors = FALSE
)

method_callouts <- data.frame(
  claim = c(
    "Teacher-call efficiency",
    "Token-budget efficiency",
    "Matched quality",
    "Teacher-overlap diagnosis"
  ),
  value = c(
    paste0(format(cache_calls, big.mark = ","), " vs ", format(online_calls, big.mark = ","), " calls"),
    paste0(round(100 * sparse_tokens / full_tokens, 1), "% of full-cache tokens"),
    paste0("Sparse quality index ", sprintf("%.3f", sparse_quality),
           " vs full ", sprintf("%.3f", full_quality)),
    paste0("Mean teacher top-K overlap ", sprintf("%.3f", mean_teacher_overlap))
  ),
  role = c(
    "Main hard contribution",
    "Sparse approximation",
    "Non-inferiority support",
    "Future stronger-teacher direction"
  ),
  stringsAsFactors = FALSE
)

motivation_metrics <- data.frame(
  panel = c(
    "A. Online cost bottleneck", "A. Online cost bottleneck",
    "B. Fusion target matters", "B. Fusion target matters", "B. Fusion target matters",
    "C. Token utility is selective", "C. Token utility is selective",
    "D. Sparse approximates full", "D. Sparse approximates full"
  ),
  item = c(
    "Online OPD train calls", "CacheOPD cache calls",
    "ProbAvg-KD quality index", "LogitAvg-KD quality index", "CacheOPD-Full quality index",
    "Unselected utility", "Selected utility",
    "CacheOPD-Full quality index", "CacheOPD-Sparse quality index"
  ),
  value = c(
    online_calls, cache_calls,
    as.numeric(pick(cost_tbl, "Method", "ProbAvg-KD", "Quality index")),
    as.numeric(pick(cost_tbl, "Method", "LogitAvg-KD", "Quality index")),
    full_quality,
    as.numeric(selected_tbl[selected_tbl$selected == "FALSE", "utility_mean"]),
    as.numeric(selected_tbl[selected_tbl$selected == "TRUE", "utility_mean"]),
    full_quality, sparse_quality
  ),
  annotation = c(
    "teacher served in every optimization loop",
    paste0(call_reduction, " fewer teacher calls than online OPD"),
    "linear probability averaging",
    "linear logit averaging",
    "cached PoE target over teacher-supported candidate tokens",
    "lower preference signal",
    "higher preference signal",
    paste0(format(full_tokens, big.mark = ","), " training tokens"),
    paste0(format(sparse_tokens, big.mark = ","), " training tokens")
  ),
  stringsAsFactors = FALSE
)

write.csv(method_nodes, file.path(src_dir, "figure_method_framework_nodes_scipilot.csv"), row.names = FALSE)
write.csv(method_edges, file.path(src_dir, "figure_method_framework_edges_scipilot.csv"), row.names = FALSE)
write.csv(method_callouts, file.path(src_dir, "figure_method_framework_callouts_scipilot.csv"), row.names = FALSE)
write.csv(motivation_metrics, file.path(src_dir, "figure_motivation_scipilot_source_data.csv"), row.names = FALSE)
write.csv(method_nodes, file.path(md_src_dir, "figure_method_framework_nodes_scipilot.csv"), row.names = FALSE)
write.csv(method_edges, file.path(md_src_dir, "figure_method_framework_edges_scipilot.csv"), row.names = FALSE)
write.csv(method_callouts, file.path(md_src_dir, "figure_method_framework_callouts_scipilot.csv"), row.names = FALSE)
write.csv(motivation_metrics, file.path(md_src_dir, "figure_motivation_scipilot_source_data.csv"), row.names = FALSE)

pal <- list(
  ink = "#17202A",
  muted = "#5D6D7E",
  pale = "#F7F9FB",
  line = "#C7D0D9",
  blue = "#2F5D8C",
  teal = "#2A9D8F",
  amber = "#D9A441",
  red = "#C75D4D",
  purple = "#756BB1",
  green = "#4C9A6A",
  slate = "#E8EDF3",
  cream = "#FFF7E6"
)

open_dev <- function(path, width, height, type) {
  if (type == "svg") {
    grDevices::svg(path, width = width, height = height, pointsize = 8, onefile = TRUE)
  } else if (type == "pdf") {
    grDevices::pdf(path, width = width, height = height, pointsize = 8, useDingbats = FALSE)
  } else if (type == "png") {
    grDevices::png(path, width = width, height = height, units = "in", res = 600, type = "cairo")
  } else if (type == "tiff") {
    grDevices::tiff(path, width = width, height = height, units = "in", res = 600, compression = "lzw", type = "cairo")
  }
}

txt <- function(label, x, y, size = 8, col = pal$ink, fontface = "plain",
                just = "centre", lineheight = 0.9) {
  grid.text(label, x = unit(x, "npc"), y = unit(y, "npc"),
            gp = gpar(fontsize = size, col = col, fontface = fontface, lineheight = lineheight),
            just = just)
}

box <- function(x, y, w, h, fill, col = pal$line, lwd = 0.7, r = 0.012) {
  grid.roundrect(x = unit(x, "npc"), y = unit(y, "npc"),
                 width = unit(w, "npc"), height = unit(h, "npc"),
                 r = unit(r, "npc"),
                 gp = gpar(fill = fill, col = col, lwd = lwd))
}

arrow_seg <- function(x0, y0, x1, y1, col = pal$muted, lwd = 0.9) {
  grid.segments(unit(x0, "npc"), unit(y0, "npc"), unit(x1, "npc"), unit(y1, "npc"),
                arrow = arrow(length = unit(2.2, "mm"), type = "closed"),
                gp = gpar(col = col, lwd = lwd, lineend = "round"))
}

small_bar <- function(x, y, w, h, values, labels, cols, maxv = max(values, na.rm = TRUE),
                      label_size = 5.8, value_fmt = function(z) format(round(z, 1), big.mark = ",")) {
  n <- length(values)
  gap <- 0.012
  bw <- (w - gap * (n - 1)) / n
  for (i in seq_len(n)) {
    bx <- x + (i - 1) * (bw + gap)
    bh <- h * values[i] / maxv
    grid.rect(unit(bx + bw / 2, "npc"), unit(y + bh / 2, "npc"),
              width = unit(bw, "npc"), height = unit(bh, "npc"),
              gp = gpar(fill = cols[i], col = NA))
    txt(labels[i], bx + bw / 2, y - 0.028, size = label_size, col = pal$muted)
    txt(value_fmt(values[i]), bx + bw / 2, y + bh + 0.02, size = label_size, col = pal$ink, fontface = "bold")
  }
  grid.lines(unit(c(x, x + w), "npc"), unit(c(y, y), "npc"), gp = gpar(col = pal$line, lwd = 0.5))
}

draw_method_framework <- function() {
  grid.newpage()
  grid.rect(gp = gpar(fill = "white", col = NA))

  txt("CacheOPD method framework", 0.04, 0.965, size = 12, fontface = "bold", just = "left")
  txt("Three-stage offline alignment with cached PoE targets and teacher-supported sparse training",
      0.04, 0.935, size = 7.2, col = pal$muted, just = "left")

  phase_x <- c(0.18, 0.50, 0.82)
  phase_w <- 0.28
  phase_titles <- c("Phase I: cache teacher distributions",
                    "Phase II: fuse and select tokens",
                    "Phase III: train without teachers")
  phase_cols <- c("#EAF2FA", "#EAF8F5", "#FFF4DA")
  for (i in 1:3) {
    box(phase_x[i], 0.55, phase_w, 0.66, fill = phase_cols[i], col = "#D6DEE8", lwd = 0.8)
    txt(phase_titles[i], phase_x[i], 0.855, size = 7.4, fontface = "bold", col = pal$ink)
  }

  # Phase I
  box(0.18, 0.755, 0.205, 0.095, "white", col = "#B7C7D8")
  txt("Reference prompts", 0.18, 0.785, size = 6.8, fontface = "bold")
  txt("same conditional rollouts", 0.18, 0.758, size = 5.5, col = pal$muted)

  box(0.18, 0.625, 0.205, 0.12, "white", col = "#B7C7D8")
  txt("Property teachers", 0.18, 0.663, size = 6.8, fontface = "bold")
  txt("foldability | solubility | thermostability", 0.18, 0.633, size = 5.1, col = pal$muted)

  box(0.18, 0.475, 0.205, 0.13, "white", col = "#B7C7D8")
  txt("Cached teacher supports", 0.18, 0.525, size = 6.8, fontface = "bold")
  txt("top-K candidate tokens\nexact teacher log-probs", 0.18, 0.485, size = 5.3, col = pal$muted)

  arrow_seg(0.18, 0.708, 0.18, 0.685)
  arrow_seg(0.18, 0.565, 0.18, 0.542)

  # Phase II
  box(0.50, 0.755, 0.21, 0.105, "white", col = "#ACD7D0")
  txt("Cached PoE target", 0.50, 0.792, size = 6.8, fontface = "bold")
  txt("soft intersection over teacher-supported tokens", 0.50, 0.763, size = 5.2, col = pal$muted)

  box(0.50, 0.625, 0.21, 0.11, "white", col = "#ACD7D0")
  txt("Support-aware utility", 0.50, 0.665, size = 6.8, fontface = "bold")
  txt("U = KL(q_PoE || p_ref)\nplus reference support gate", 0.50, 0.628, size = 5.3, col = pal$muted)

  box(0.50, 0.475, 0.21, 0.13, "white", col = "#ACD7D0")
  txt("Global sparse mask", 0.50, 0.525, size = 6.8, fontface = "bold")
  txt("retain high-utility positions\npreference-token budget kappa", 0.50, 0.485, size = 5.3, col = pal$muted)

  arrow_seg(0.285, 0.475, 0.395, 0.755)
  arrow_seg(0.50, 0.705, 0.50, 0.68)
  arrow_seg(0.50, 0.57, 0.50, 0.54)

  # Phase III
  box(0.82, 0.755, 0.205, 0.105, "white", col = "#E4C46E")
  txt("Static cache minibatches", 0.82, 0.792, size = 6.8, fontface = "bold")
  txt("teacher models bypassed during optimization", 0.82, 0.763, size = 5.2, col = pal$muted)

  box(0.82, 0.625, 0.205, 0.11, "white", col = "#E4C46E")
  txt("Student update", 0.82, 0.665, size = 6.8, fontface = "bold")
  txt("PoE preference CE\n+ LM and EOS anchors", 0.82, 0.628, size = 5.3, col = pal$muted)

  box(0.82, 0.475, 0.205, 0.13, "white", col = "#E4C46E")
  txt("Aligned generator", 0.82, 0.525, size = 6.8, fontface = "bold")
  txt("same prompts, decoding and evaluators\nas matched baselines", 0.82, 0.485, size = 5.3, col = pal$muted)

  arrow_seg(0.605, 0.475, 0.715, 0.755)
  arrow_seg(0.82, 0.705, 0.82, 0.68)
  arrow_seg(0.82, 0.57, 0.82, 0.54)

  # Innovation ribbon
  box(0.50, 0.245, 0.88, 0.255, fill = pal$pale, col = "#D8E0E8", lwd = 0.8)
  txt("Core design choices and measured consequences", 0.075, 0.35, size = 7.5, fontface = "bold", just = "left")

  call_x <- c(0.18, 0.39, 0.61, 0.82)
  call_titles <- method_callouts$claim
  call_values <- method_callouts$value
  call_cols <- c(pal$blue, pal$teal, pal$amber, pal$purple)
  for (i in seq_along(call_x)) {
    box(call_x[i], 0.245, 0.18, 0.15, fill = "white", col = call_cols[i], lwd = 1.0)
    txt(call_titles[i], call_x[i], 0.292, size = 6.2, fontface = "bold", col = call_cols[i])
    txt(call_values[i], call_x[i], 0.252, size = 5.1, col = pal$ink)
    txt(method_callouts$role[i], call_x[i], 0.215, size = 4.9, col = pal$muted)
  }

  txt("Teacher calls during optimization: zero", 0.04, 0.09, size = 7.0, fontface = "bold", col = pal$teal, just = "left")
  txt("The expensive teacher ensemble is used only once to populate the cache; all student updates consume static cached targets.",
      0.04, 0.06, size = 5.7, col = pal$muted, just = "left")
}

draw_motivation <- function() {
  grid.newpage()
  grid.rect(gp = gpar(fill = "white", col = NA))

  txt("Motivation: what CacheOPD fixes", 0.04, 0.965, size = 12, fontface = "bold", just = "left")
  txt("The evidence separates call efficiency, target construction, token selectivity and sparse approximation.",
      0.04, 0.935, size = 7.2, col = pal$muted, just = "left")

  # Panel A
  box(0.265, 0.69, 0.43, 0.37, fill = "#F7F9FB", col = "#D6DEE8")
  txt("A  Online OPD is teacher-call bound", 0.07, 0.84, size = 7.5, fontface = "bold", just = "left")
  txt("Real-time teacher serving appears inside the optimization loop.", 0.07, 0.812, size = 5.6, col = pal$muted, just = "left")
  small_bar(0.12, 0.60, 0.26, 0.18,
            values = c(online_calls, cache_calls),
            labels = c("Online", "Cache"),
            cols = c(pal$red, pal$teal),
            maxv = online_calls,
            value_fmt = function(z) format(round(z), big.mark = ","))
  txt(paste0(call_reduction, " reduction"), 0.265, 0.555, size = 6.5, fontface = "bold", col = pal$teal)

  # Panel B
  box(0.735, 0.69, 0.43, 0.37, fill = "#F7F9FB", col = "#D6DEE8")
  txt("B  Offline KD is not the same target", 0.54, 0.84, size = 7.5, fontface = "bold", just = "left")
  txt("PoE behaves as a consensus-seeking target over cached teacher supports.", 0.54, 0.812, size = 5.6, col = pal$muted, just = "left")
  qvals <- c(
    as.numeric(pick(cost_tbl, "Method", "ProbAvg-KD", "Quality index")),
    as.numeric(pick(cost_tbl, "Method", "LogitAvg-KD", "Quality index")),
    full_quality
  )
  small_bar(0.57, 0.60, 0.30, 0.18,
            values = qvals,
            labels = c("ProbAvg", "LogitAvg", "PoE"),
            cols = c("#9DA8B2", "#7F8C8D", pal$blue),
            maxv = 1,
            value_fmt = function(z) sprintf("%.2f", z))
  txt("quality index", 0.735, 0.555, size = 5.6, col = pal$muted)

  # Panel C
  box(0.265, 0.27, 0.43, 0.37, fill = "#F7F9FB", col = "#D6DEE8")
  txt("C  Useful gradients concentrate at selected tokens", 0.07, 0.42, size = 7.5, fontface = "bold", just = "left")
  txt("Sparse selection keeps positions with larger preference utility.", 0.07, 0.392, size = 5.6, col = pal$muted, just = "left")
  uvals <- c(
    as.numeric(selected_tbl[selected_tbl$selected == "FALSE", "utility_mean"]),
    as.numeric(selected_tbl[selected_tbl$selected == "TRUE", "utility_mean"])
  )
  small_bar(0.14, 0.18, 0.22, 0.17,
            values = uvals,
            labels = c("Unselected", "Selected"),
            cols = c("#BFC7CF", pal$amber),
            maxv = max(uvals) * 1.15,
            value_fmt = function(z) sprintf("%.2f", z))
  txt("mean token utility", 0.265, 0.135, size = 5.6, col = pal$muted)

  # Panel D
  box(0.735, 0.27, 0.43, 0.37, fill = "#F7F9FB", col = "#D6DEE8")
  txt("D  Sparse CacheOPD is a compact full-cache proxy", 0.54, 0.42, size = 7.5, fontface = "bold", just = "left")
  txt("It keeps the optimization target but spends fewer preference tokens.", 0.54, 0.392, size = 5.6, col = pal$muted, just = "left")

  # Token-quality mini scatter
  x0 <- 0.60; y0 <- 0.16; ww <- 0.27; hh <- 0.15
  grid.lines(unit(c(x0, x0 + ww), "npc"), unit(c(y0, y0), "npc"), gp = gpar(col = pal$line, lwd = 0.5))
  grid.lines(unit(c(x0, x0), "npc"), unit(c(y0, y0 + hh), "npc"), gp = gpar(col = pal$line, lwd = 0.5))
  xs <- x0 + ww * c(sparse_tokens, full_tokens) / full_tokens
  quality_vals <- c(sparse_quality, full_quality)
  ys <- y0 + hh * (quality_vals - 0.84) / (1.00 - 0.84)
  grid.points(unit(xs, "npc"), unit(ys, "npc"), pch = 21, size = unit(4.8, "mm"),
              gp = gpar(fill = c(pal$teal, pal$blue), col = "white", lwd = 0.8))
  grid.segments(unit(xs[1], "npc"), unit(ys[1], "npc"), unit(xs[2], "npc"), unit(ys[2], "npc"),
                gp = gpar(col = "#94A3B8", lwd = 0.8, lty = 2))
  txt("Sparse", xs[1], ys[1] + 0.035, size = 5.4, col = pal$teal, fontface = "bold")
  txt("Full", xs[2] - 0.015, ys[2] + 0.035, size = 5.4, col = pal$blue, fontface = "bold")
  txt("tokens", x0 + ww / 2, y0 - 0.035, size = 5.5, col = pal$muted)
  txt("quality", x0 - 0.04, y0 + hh / 2, size = 5.5, col = pal$muted, just = "centre")
  txt(paste0(round(100 * sparse_tokens / full_tokens, 1), "% tokens; ",
             sprintf("%.3f", sparse_quality), " quality index"),
      0.735, 0.115, size = 5.6, col = pal$muted)

  txt(paste0("Current teacher ensemble is low-disagreement: mean JSD ",
             sprintf("%.4f", mean_teacher_js), ", mean top-K overlap ",
             sprintf("%.3f", mean_teacher_overlap), "."),
      0.04, 0.055, size = 5.7, col = pal$muted, just = "left")
}

save_all <- function(draw_fun, basename, width_mm = 183, height_mm = 112) {
  width <- width_mm / 25.4
  height <- height_mm / 25.4
  outputs <- list(
    svg = file.path(fig_dir, paste0(basename, ".svg")),
    pdf = file.path(fig_dir, paste0(basename, ".pdf")),
    png = file.path(fig_dir, paste0(basename, ".png")),
    tiff = file.path(fig_dir, paste0(basename, ".tiff"))
  )
  for (type in names(outputs)) {
    open_dev(outputs[[type]], width, height, type)
    draw_fun()
    dev.off()
  }
  file.copy(outputs$svg, file.path(md_fig_dir, paste0(basename, ".svg")), overwrite = TRUE)
  file.copy(outputs$pdf, file.path(md_fig_dir, paste0(basename, ".pdf")), overwrite = TRUE)
  file.copy(outputs$png, file.path(md_fig_dir, paste0(basename, ".png")), overwrite = TRUE)
  invisible(outputs)
}

save_all(draw_method_framework, "figure1_cacheopd_method_framework_scipilot", 183, 118)
save_all(draw_motivation, "figure2_cacheopd_motivation_scipilot", 183, 118)

qa <- data.frame(
  file = c("figure1_cacheopd_method_framework_scipilot", "figure2_cacheopd_motivation_scipilot"),
  core_conclusion = c(
    "CacheOPD decouples teacher acquisition from student optimization through cached PoE targets and support-aware sparse token training.",
    "The method is motivated by online teacher-call cost, the mismatch between ordinary offline KD and cached PoE targets, and concentration of useful preference signal in a sparse subset of tokens."
  ),
  source_data = c(
    "figure_method_framework_nodes_scipilot.csv; figure_method_framework_edges_scipilot.csv; figure_method_framework_callouts_scipilot.csv",
    "figure_motivation_scipilot_source_data.csv"
  ),
  export_formats = "svg, pdf, png, tiff",
  notes = c(
    "Schematic-led composite; all text remains editable in SVG/PDF.",
    "Schematic-led composite with quantitative mini-panels; no dual axes or pie charts."
  ),
  stringsAsFactors = FALSE
)
write.csv(qa, file.path(src_dir, "figure_method_motivation_scipilot_manifest.csv"), row.names = FALSE)
write.csv(qa, file.path(md_src_dir, "figure_method_motivation_scipilot_manifest.csv"), row.names = FALSE)

message("[OK] Wrote SciPilot-style method and motivation figures to: ", fig_dir)
