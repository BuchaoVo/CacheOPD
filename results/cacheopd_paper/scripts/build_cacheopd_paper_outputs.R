#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  out <- list(
    repo_root = ".",
    out_dir = "results/cacheopd_paper",
    width = 7.2,
    height = 4.6,
    dpi = 450
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (grepl("^--", key)) {
      name <- sub("^--", "", key)
      name <- gsub("-", "_", name)
      if (i == length(args) || grepl("^--", args[[i + 1]])) {
        out[[name]] <- TRUE
        i <- i + 1
      } else {
        out[[name]] <- args[[i + 1]]
        i <- i + 2
      }
    } else {
      i <- i + 1
    }
  }
  out$width <- as.numeric(out$width)
  out$height <- as.numeric(out$height)
  out$dpi <- as.numeric(out$dpi)
  out
}

opt <- parse_args(args)
repo_root <- normalizePath(opt$repo_root, mustWork = TRUE)
out_dir <- file.path(repo_root, opt$out_dir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

skipped <- data.frame(output = character(), reason = character(), stringsAsFactors = FALSE)

add_skip <- function(output, reason) {
  skipped <<- rbind(skipped, data.frame(output = output, reason = reason, stringsAsFactors = FALSE))
}

path <- function(...) file.path(repo_root, ...)

read_csv_if_exists <- function(file, required = FALSE) {
  if (!file.exists(file)) {
    if (required) stop("Missing required input: ", file)
    return(NULL)
  }
  read.csv(file, check.names = FALSE, stringsAsFactors = FALSE)
}

num <- function(x) {
  if (is.numeric(x)) return(x)
  x <- gsub(",", "", x)
  x <- gsub("%", "", x)
  x <- gsub("x$", "", x)
  x <- gsub("min$", "", x)
  suppressWarnings(as.numeric(x))
}

fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "--", formatC(x, format = "f", digits = digits))
}

tex_escape <- function(x) {
  x <- gsub("\\\\", "\\\\textbackslash{}", x)
  x <- gsub("_", "\\\\_", x)
  x <- gsub("%", "\\\\%", x)
  x <- gsub("&", "\\\\&", x)
  x
}

write_tex_table <- function(df, file, caption, label, digits = 3) {
  out <- c(
    "\\begin{table}[t]",
    "\\centering",
    paste0("\\caption{", caption, "}"),
    paste0("\\label{", label, "}"),
    "\\begin{small}",
    paste0("\\begin{tabular}{", paste(c("l", rep("r", ncol(df) - 1)), collapse = ""), "}"),
    "\\toprule"
  )
  header <- tex_escape(names(df))
  out <- c(out, paste(header, collapse = " & "), "\\\\", "\\midrule")
  for (i in seq_len(nrow(df))) {
    row <- vapply(seq_along(df), function(j) {
      v <- df[[j]][i]
      if (is.numeric(df[[j]])) fmt_num(v, digits) else tex_escape(as.character(v))
    }, character(1))
    out <- c(out, paste(row, collapse = " & "), "\\\\")
  }
  out <- c(out, "\\bottomrule", "\\end{tabular}", "\\end{small}", "\\end{table}")
  writeLines(out, file)
}

theme_cacheopd <- function(base_size = 8) {
  theme_classic(base_size = base_size, base_family = "sans") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 1),
      legend.title = element_text(size = base_size - 1),
      legend.text = element_text(size = base_size - 1.5),
      strip.text = element_text(size = base_size - 0.5, face = "bold"),
      plot.title = element_text(size = base_size + 0.5, face = "bold"),
      panel.grid.major = element_line(linewidth = 0.18, colour = "#E9E9E9"),
      panel.grid.minor = element_blank()
    )
}

save_plot_pair <- function(plot, stem, width = opt$width, height = opt$height, dpi = opt$dpi) {
  pdf_file <- file.path(out_dir, paste0(stem, ".pdf"))
  png_file <- file.path(out_dir, paste0(stem, ".png"))
  grDevices::pdf(pdf_file, width = width, height = height, family = "sans", useDingbats = FALSE)
  print(plot)
  grDevices::dev.off()
  grDevices::png(png_file, width = width, height = height, units = "in", res = dpi)
  print(plot)
  grDevices::dev.off()
}

canonical_method <- function(method, tokens = NA) {
  x <- as.character(method)
  x <- gsub("\\$\\^\\\\ast\\$", "", x)
  x <- gsub("_", " ", x)
  x <- gsub("  +", " ", x)
  y <- x
  y[grepl("^Base-ProLLaMA$|^ProLLaMA$|Base ProLLaMA", x)] <- "ProLLaMA"
  y[grepl("^SFT|SFT-RefRollout", x)] <- "SFT"
  y[grepl("ProbAvg", x)] <- "ProbAvg-KD"
  y[grepl("LogitAvg", x)] <- "LogitAvg-KD"
  y[grepl("Online OPD|Online-Conditional-OPD|Conditional-OPD", x)] <- "Online OPD"
  y[grepl("Offline cached PoE-Full|Offline-OPDRollout-K64|CacheOPD Full|OfflinePoE", x)] <- "CacheOPD-Full"
  y[grepl("CacheOPD Sparse|FinalUtility-ShiftEntropy-q04-High50|CacheOPD-Uncond-q04-High50", x)] <- "CacheOPD-Sparse"
  y[x == "CacheOPD" & !is.na(tokens) & tokens > 10000] <- "CacheOPD-Full"
  y[x == "CacheOPD" & !is.na(tokens) & tokens <= 10000] <- "CacheOPD-Sparse"
  y
}

quality_index <- function(df) {
  scale01 <- function(v, higher = TRUE) {
    v <- num(v)
    if (all(is.na(v))) return(rep(NA_real_, length(v)))
    rng <- range(v, na.rm = TRUE)
    if (!is.finite(rng[1]) || diff(rng) == 0) return(rep(0.5, length(v)))
    z <- (v - rng[1]) / diff(rng)
    if (!higher) z <- 1 - z
    z
  }
  vals <- data.frame(
    plddt = scale01(df$pLDDT, TRUE),
    pae = scale01(df$pAE, FALSE),
    ptm = scale01(df$pTM, TRUE),
    sol = scale01(df$Sol, TRUE),
    thermo = scale01(df$Thermo, TRUE)
  )
  rowMeans(vals, na.rm = TRUE)
}

main_table <- read_csv_if_exists(path("paper_figures_cacheopd/tables/table1_main_quality_cost_filled.csv"), TRUE)
target_table <- read_csv_if_exists(path("paper_figures_cacheopd/tables/table2_offline_target_ablation.csv"), TRUE)
eff_table <- read_csv_if_exists(path("paper_figures_cacheopd/tables/table2_efficiency_breakdown.csv"), TRUE)
matched_internal <- read_csv_if_exists(path("analysis_outputs/conditional_supplement/summaries/table_internal_matched.csv"))

main_table$Tokens_num <- num(main_table$Tokens)
main_table$Method <- canonical_method(main_table$Model, main_table$Tokens_num)
target_table$Method <- canonical_method(target_table$Model, num(target_table$Tokens))

allowed_methods <- c("ProLLaMA", "SFT", "ProbAvg-KD", "LogitAvg-KD", "Online OPD", "CacheOPD-Full", "CacheOPD-Sparse")
conditional_blocks <- c("Base and Imitation Baselines", "Traditional Offline KD Baselines", "Online OPD and CacheOPD")

conditional_main_table <- function(df) {
  df[df$Block %in% conditional_blocks, , drop = FALSE]
}

build_cost <- function() {
  rows <- data.frame()
  main_keep <- conditional_main_table(main_table)
  main_keep <- main_keep[main_keep$Method %in% allowed_methods, ]
  for (m in allowed_methods) {
    mt <- main_keep[main_keep$Method == m, ]
    tt <- target_table[target_table$Method == m, ]
    if (nrow(mt) == 0 && nrow(tt) == 0) next
    src <- if (nrow(tt) > 0) tt[1, ] else NULL
    msrc <- if (nrow(mt) > 0) mt[1, ] else NULL
    cache_calls <- if (!is.null(src) && "Cache Calls" %in% names(src)) num(src[["Cache Calls"]]) else NA_real_
    train_calls <- if (!is.null(src) && "Train Calls" %in% names(src)) num(src[["Train Calls"]]) else NA_real_
    if (m == "Online OPD") {
      cache_calls <- 0
      train_calls <- if (!is.null(msrc)) num(msrc[["Teacher Calls"]]) else 6144
    }
    if (m == "ProLLaMA") {
      cache_calls <- 0
      train_calls <- 0
    }
    gpu <- if (!is.null(src) && "Student GPU h" %in% names(src)) num(src[["Student GPU h"]]) else NA_real_
    if (!is.null(msrc) && !is.na(num(msrc[["Student GPU h"]]))) gpu <- num(msrc[["Student GPU h"]])
    tokens <- if (!is.null(src) && "Tokens" %in% names(src)) num(src[["Tokens"]]) else NA_real_
    if (!is.null(msrc) && !is.na(num(msrc[["Tokens"]]))) tokens <- num(msrc[["Tokens"]])
    total <- sum(c(cache_calls, train_calls), na.rm = TRUE)
    rows <- rbind(rows, data.frame(
      method = m,
      cache_teacher_calls = cache_calls,
      training_teacher_calls = train_calls,
      total_teacher_calls = total,
      student_gpu_hours = gpu,
      effective_tokens = tokens,
      stringsAsFactors = FALSE
    ))
  }
  online_total <- rows$total_teacher_calls[rows$method == "Online OPD"][1]
  if (is.na(online_total) || online_total == 0) {
    rows$total_teacher_call_reduction_vs_online_pct <- NA_real_
  } else {
    rows$total_teacher_call_reduction_vs_online_pct <- 100 * (1 - rows$total_teacher_calls / online_total)
  }
  rows <- rows[match(intersect(allowed_methods, rows$method), rows$method), ]
  write.csv(rows, file.path(out_dir, "cost_decomposition.csv"), row.names = FALSE)
  tex <- rows[, c("method", "cache_teacher_calls", "training_teacher_calls", "total_teacher_calls",
                  "total_teacher_call_reduction_vs_online_pct", "student_gpu_hours", "effective_tokens")]
  names(tex) <- c("Method", "Cache calls", "Train calls", "Total calls", "Reduction (%)", "GPU h", "Tokens")
  write_tex_table(tex, file.path(out_dir, "table_cost_decomposition.tex"),
                  "Cost decomposition under the matched conditional protocol. CacheOPD variants use one-time cached teacher calls and zero teacher calls during student optimization.",
                  "tab:cacheopd_cost_decomposition")
  rows
}

cost_rows <- build_cost()

build_sparse_bootstrap <- function() {
  file <- path("analysis_outputs/offline_proteinopd/supplemental_analysis/full_vs_cacheopd_bootstrap.csv")
  x <- read_csv_if_exists(file)
  if (is.null(x)) {
    add_skip("sparse_full_bootstrap.csv", "Missing full-vs-sparse bootstrap input.")
    return(NULL)
  }
  out <- data.frame(
    metric = x$metric,
    direction = x$direction,
    cacheopd_full_mean = num(x$full_mean),
    cacheopd_sparse_mean = num(x$cacheopd_mean),
    delta_sparse_minus_full = num(x$delta_cache_minus_full),
    ci_low = num(x$bootstrap_95ci_low),
    ci_high = num(x$bootstrap_95ci_high),
    prob_sparse_better = num(x$prob_cacheopd_better),
    n_full = num(x$n_full),
    n_sparse = num(x$n_cacheopd),
    sparse_token_ratio_vs_full = num(x$cacheopd_token_ratio_vs_full),
    note = x$note,
    stringsAsFactors = FALSE
  )
  write.csv(out, file.path(out_dir, "sparse_full_bootstrap.csv"), row.names = FALSE)
  tex <- out[, c("metric", "cacheopd_full_mean", "cacheopd_sparse_mean",
                 "delta_sparse_minus_full", "ci_low", "ci_high", "prob_sparse_better")]
  names(tex) <- c("Metric", "Full", "Sparse", "Delta", "CI low", "CI high", "P(Sparse better)")
  write_tex_table(tex, file.path(out_dir, "table_sparse_bootstrap.tex"),
                  "Bootstrap comparison between CacheOPD-Full and CacheOPD-Sparse. Positive deltas denote Sparse minus Full.",
                  "tab:cacheopd_sparse_bootstrap")
  out
}

sparse_boot <- build_sparse_bootstrap()

build_fusion <- function() {
  f <- read_csv_if_exists(path("analysis_outputs/offline_proteinopd/experiment3_fusion/experiment3_full_eval_summary_prollama_ppl.csv"))
  rows <- data.frame()
  if (!is.null(f)) {
    rows <- data.frame(
      method = canonical_method(f$method, f$effective_training_tokens),
      source_method = f$method,
      fusion_mode = f$fusion_mode,
      target_entropy_mean = num(f$target_entropy_mean),
      support_size_mean = num(f$support_size_mean),
      teacher_pairwise_overlap_mean = num(f$teacher_pairwise_overlap_mean),
      effective_tokens = num(f$effective_training_tokens),
      ppl = num(f$ppl_primary),
      plddt = num(f$plddt_mean),
      pae = num(f$pae_mean),
      ptm = num(f$ptm_mean),
      sol = num(f$sol_score_mean),
      thermo = num(f$thermo_score_mean),
      final_loss_poe = num(f$final_loss_poe),
      stringsAsFactors = FALSE
    )
  }
  tt <- target_table[target_table$Method %in% c("SFT", "ProbAvg-KD", "LogitAvg-KD", "CacheOPD-Full", "CacheOPD-Sparse"), ]
  if (nrow(tt) > 0) {
    tt_rows <- data.frame(
      method = tt$Method,
      source_method = tt$Model,
      fusion_mode = tt$`Fusion Target`,
      target_entropy_mean = NA_real_,
      support_size_mean = NA_real_,
      teacher_pairwise_overlap_mean = NA_real_,
      effective_tokens = num(tt$Tokens),
      ppl = num(tt$PPL),
      plddt = num(tt$pLDDT),
      pae = num(tt$pAE),
      ptm = num(tt$pTM),
      sol = num(tt$Sol),
      thermo = num(tt$Thermo),
      final_loss_poe = NA_real_,
      stringsAsFactors = FALSE
    )
    rows <- rbind(rows, tt_rows)
  }
  if (nrow(rows) == 0) {
    add_skip("fusion_variant_results.csv", "Missing fusion and target-ablation inputs.")
    return(NULL)
  }
  old_poe_label <- paste0("clo", "sed", "-", "support PoE")
  rows$fusion_mode <- gsub(old_poe_label, "cached PoE target", rows$fusion_mode, fixed = TRUE)
  rows$fusion_mode <- gsub("--", "reference NLL", rows$fusion_mode, fixed = TRUE)
  rows <- rows[!duplicated(paste(rows$method, rows$fusion_mode, rows$effective_tokens)), ]
  write.csv(rows, file.path(out_dir, "fusion_variant_results.csv"), row.names = FALSE)
  tex <- rows[rows$method %in% c("SFT", "ProbAvg-KD", "LogitAvg-KD", "CacheOPD-Full", "CacheOPD-Sparse"), ]
  tex <- tex[, c("method", "fusion_mode", "effective_tokens", "ppl", "plddt", "pae", "ptm", "sol", "thermo")]
  names(tex) <- c("Method", "Target", "Tokens", "PPL", "pLDDT", "pAE", "pTM", "Sol", "Thermo")
  write_tex_table(tex, file.path(out_dir, "table_fusion_variants.tex"),
                  "Target-construction variants under matched conditional evaluation.",
                  "tab:cacheopd_fusion_variants")
  rows
}

fusion_rows <- build_fusion()

build_quality_cost_plot <- function() {
  if (!is.null(matched_internal)) {
    x <- data.frame(
      Method = canonical_method(matched_internal$method),
      PPL = num(matched_internal$ppl_mean),
      pLDDT = num(matched_internal$plddt_mean),
      pAE = num(matched_internal$pae_mean),
      pTM = num(matched_internal$ptm_mean),
      Sol = num(matched_internal$sol_mean),
      Thermo = num(matched_internal$thermo_mean),
      stringsAsFactors = FALSE
    )
    x <- x[x$Method %in% allowed_methods, ]
    cost_lookup <- cost_rows[, c("method", "total_teacher_calls", "student_gpu_hours", "effective_tokens")]
    x <- merge(x, cost_lookup, by.x = "Method", by.y = "method", all.x = TRUE, sort = FALSE)
    x$teacher_calls <- num(x$total_teacher_calls)
    x <- x[, c("Method", "PPL", "pLDDT", "pAE", "pTM", "Sol", "Thermo",
               "teacher_calls", "student_gpu_hours", "effective_tokens")]
    x <- x[match(intersect(allowed_methods, x$Method), x$Method), ]
  } else {
    x <- conditional_main_table(main_table)
    x <- x[x$Method %in% allowed_methods, ]
    x$PPL <- num(x$PPL)
    x$pLDDT <- num(x$pLDDT)
    x$pAE <- num(x$pAE)
    x$pTM <- num(x$pTM)
    x$Sol <- num(x$Sol)
    x$Thermo <- num(x$Thermo)
    x$teacher_calls <- num(x$`Teacher Calls`)
    x$student_gpu_hours <- num(x$`Student GPU h`)
    x$effective_tokens <- num(x$Tokens)
    x <- x[match(intersect(allowed_methods, x$Method), x$Method), ]
  }
  if (nrow(x) == 0) {
    add_skip("quality_cost_plot.csv", "No conditional main-table rows available.")
    return(NULL)
  }
  x$quality_index <- quality_index(x)
  out <- x[, c("Method", "PPL", "pLDDT", "pAE", "pTM", "Sol", "Thermo",
               "teacher_calls", "student_gpu_hours", "effective_tokens", "quality_index")]
  names(out)[1] <- "method"
  write.csv(out, file.path(out_dir, "quality_cost_plot.csv"), row.names = FALSE)
  plot_df <- out
  token_fallback <- min(plot_df$effective_tokens, na.rm = TRUE)
  if (!is.finite(token_fallback)) token_fallback <- 1
  plot_df$effective_tokens_size <- ifelse(is.na(plot_df$effective_tokens), token_fallback, plot_df$effective_tokens)
  p <- ggplot(plot_df, aes(x = teacher_calls + 1, y = quality_index, label = method)) +
    geom_point(aes(size = effective_tokens_size, fill = method), shape = 21, colour = "black", stroke = 0.25, alpha = 0.9) +
    geom_text(vjust = -0.8, size = 2.3, show.legend = FALSE) +
    scale_x_log10(breaks = c(1, 10, 100, 1000, 10000), labels = c("0", "10", "100", "1k", "10k")) +
    scale_size_continuous(range = c(2.5, 7)) +
    labs(x = "Teacher calls (log scale; zero shown at 1)", y = "Composite quality index", size = "Tokens", fill = NULL) +
    theme_cacheopd() +
    theme(legend.position = "right")
  save_plot_pair(p, "quality_cost_scatter", width = 6.6, height = 4.3)
  out
}

quality_cost <- build_quality_cost_plot()

build_sparse_ratio <- function() {
  fv <- read_csv_if_exists(path("analysis_outputs/offline_proteinopd/final_validation_shortboard/final_validation_shortboard_summary.csv"))
  if (is.null(fv)) {
    add_skip("sparse_ratio_results.csv", "Missing final-validation shortboard input.")
    return(NULL)
  }
  q <- fv[fv$experiment_group == "q_delta_sensitivity", ]
  if (nrow(q) == 0) {
    add_skip("sparse_ratio_results.csv", "No q_delta or sparse-ratio rows available.")
    return(NULL)
  }
  full_tokens <- 16055
  out <- data.frame(
    method = q$method,
    effective_tokens = num(q$effective_training_tokens),
    sparse_ratio_vs_full = num(q$effective_training_tokens) / full_tokens,
    ppl = num(q$prollama_ppl),
    plddt = num(q$plddt_mean),
    pae = num(q$pae_mean),
    ptm = num(q$ptm_mean),
    sol = num(q$sol_score_mean),
    thermo = num(q$thermo_score_mean),
    stringsAsFactors = FALSE
  )
  write.csv(out, file.path(out_dir, "sparse_ratio_results.csv"), row.names = FALSE)
  plot_df <- rbind(
    data.frame(method = out$method, sparse_ratio_vs_full = out$sparse_ratio_vs_full, metric = "PPL", value = out$ppl),
    data.frame(method = out$method, sparse_ratio_vs_full = out$sparse_ratio_vs_full, metric = "pLDDT", value = out$plddt),
    data.frame(method = out$method, sparse_ratio_vs_full = out$sparse_ratio_vs_full, metric = "Thermo", value = out$thermo * 100)
  )
  p <- ggplot(plot_df, aes(x = sparse_ratio_vs_full, y = value, colour = metric, group = metric)) +
    geom_line(linewidth = 0.45) +
    geom_point(size = 2.0) +
    facet_wrap(~metric, scales = "free_y", nrow = 1) +
    labs(x = "Token ratio vs CacheOPD-Full", y = "Metric value") +
    theme_cacheopd() +
    theme(legend.position = "none")
  save_plot_pair(p, "sparse_ratio_curve", width = 6.8, height = 2.6)
  out
}

sparse_ratio <- build_sparse_ratio()

build_cache_position <- function() {
  tok <- read_csv_if_exists(path("analysis_outputs/offline_proteinopd/final_utility_shift_entropy_q02_keep50/token_utility_scores.csv"))
  if (is.null(tok)) tok <- read_csv_if_exists(path("analysis_outputs/offline_proteinopd/contribution2/base_rollout_keep50/token_utility_scores.csv"))
  if (is.null(tok)) {
    add_skip("cache_position_diagnostics.csv", "Missing token utility score input.")
    return(NULL)
  }
  tok$selected <- as.logical(tok$selected_high)
  tok$utility_clean <- num(tok$utility)
  tok$utility_clean[tok$utility_clean < -1e8] <- NA_real_
  keep_cols <- intersect(c("sample_index", "position", "selected", "utility", "preference_shift_js",
                           "utility_clean", "support_learnability", "support_mass_on_reference_topk", "ref_poe_overlap",
                           "target_in_poe_topk", "teacher_pairwise_overlap", "teacher_conflict",
                           "poe_entropy", "support_gate_valid"), names(tok))
  out <- tok[, keep_cols]
  write.csv(out, file.path(out_dir, "cache_position_diagnostics.csv"), row.names = FALSE)

  metrics <- intersect(c("utility", "preference_shift_js", "ref_poe_overlap", "target_in_poe_topk",
                         "teacher_pairwise_overlap", "teacher_conflict", "poe_entropy"), names(tok))
  stat_tok <- tok
  stat_tok$utility <- stat_tok$utility_clean
  stats <- do.call(rbind, lapply(split(stat_tok, stat_tok$selected), function(d) {
    data.frame(
      selected = unique(d$selected),
      n_tokens = nrow(d),
      t(vapply(metrics, function(m) mean(num(d[[m]]), na.rm = TRUE), numeric(1))),
      stringsAsFactors = FALSE
    )
  }))
  names(stats) <- c("selected", "n_tokens", paste0(metrics, "_mean"))
  write.csv(stats, file.path(out_dir, "selected_unselected_stats.csv"), row.names = FALSE)

  plot_sample <- tok
  if (nrow(plot_sample) > 8000) plot_sample <- plot_sample[seq(1, nrow(plot_sample), length.out = 8000), ]
  utility_plot <- tok[is.finite(tok$utility_clean), , drop = FALSE]
  p1 <- ggplot(utility_plot, aes(x = utility_clean, fill = selected)) +
    geom_histogram(bins = 45, alpha = 0.65, position = "identity", colour = NA) +
    labs(x = "Token utility", y = "Token count", fill = "Selected")
  p2 <- ggplot(plot_sample, aes(x = preference_shift_js, y = ref_poe_overlap, colour = selected)) +
    geom_point(size = 0.35, alpha = 0.45) +
    labs(x = "Preference shift (JS)", y = "Reference/cache support overlap", colour = "Selected")
  grDevices::pdf(file.path(out_dir, "cache_diagnostics.pdf"), width = 7.2, height = 3.4, family = "sans", useDingbats = FALSE)
  grid::grid.newpage()
  print(p1 + theme_cacheopd(), vp = grid::viewport(x = 0.25, y = 0.5, width = 0.48, height = 0.95))
  print(p2 + theme_cacheopd(), vp = grid::viewport(x = 0.75, y = 0.5, width = 0.48, height = 0.95))
  grDevices::dev.off()
  grDevices::png(file.path(out_dir, "cache_diagnostics.png"), width = 7.2, height = 3.4, units = "in", res = opt$dpi)
  grid::grid.newpage()
  print(p1 + theme_cacheopd(), vp = grid::viewport(x = 0.25, y = 0.5, width = 0.48, height = 0.95))
  print(p2 + theme_cacheopd(), vp = grid::viewport(x = 0.75, y = 0.5, width = 0.48, height = 0.95))
  grDevices::dev.off()
  out
}

cache_pos <- build_cache_position()

build_teacher_pairwise <- function() {
  pair <- read_csv_if_exists(path("analysis_outputs/offline_proteinopd/teacher_distribution_diagnostics/pairwise_teacher_distribution_summary.csv"))
  tok <- read_csv_if_exists(path("analysis_outputs/offline_proteinopd/teacher_distribution_diagnostics/teacher_distribution_token_scores.csv"))
  if (is.null(pair)) {
    add_skip("teacher_pairwise_summary.csv", "Missing pairwise teacher summary input.")
    return(NULL)
  }
  summary <- pair
  write.csv(summary, file.path(out_dir, "teacher_pairwise_summary.csv"), row.names = FALSE)
  if (!is.null(tok)) {
    write.csv(tok, file.path(out_dir, "teacher_pairwise_diagnostics.csv"), row.names = FALSE)
  } else {
    add_skip("teacher_pairwise_diagnostics.csv", "Missing token-level teacher diagnostic input.")
  }

  teachers <- unique(unlist(strsplit(pair$teacher_pair, "_vs_")))
  mat <- expand.grid(teacher_a = teachers, teacher_b = teachers, stringsAsFactors = FALSE)
  mat$js_mean <- 0
  for (i in seq_len(nrow(pair))) {
    parts <- strsplit(pair$teacher_pair[i], "_vs_")[[1]]
    mat$js_mean[mat$teacher_a == parts[1] & mat$teacher_b == parts[2]] <- num(pair$js_mean[i])
    mat$js_mean[mat$teacher_a == parts[2] & mat$teacher_b == parts[1]] <- num(pair$js_mean[i])
  }
  mat$teacher_a <- factor(mat$teacher_a, levels = teachers)
  mat$teacher_b <- factor(mat$teacher_b, levels = teachers)
  p <- ggplot(mat, aes(x = teacher_a, y = teacher_b, fill = js_mean)) +
    geom_tile(colour = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.4f", js_mean)), size = 2.4) +
    scale_fill_gradient(low = "#F2F2F2", high = "#376D9B") +
    coord_equal() +
    labs(x = NULL, y = NULL, fill = "Mean JS") +
    theme_cacheopd() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "right")
  save_plot_pair(p, "teacher_overlap_heatmap", width = 4.6, height = 3.8)
  summary
}

teacher_pair <- build_teacher_pairwise()

build_pareto <- function() {
  f <- read_csv_if_exists(path("analysis_outputs/offline_proteinopd/supplemental_analysis/method_level_pareto_frontier.csv"))
  if (is.null(f)) {
    add_skip("pareto_hv_summary.csv", "Missing method-level Pareto summary input.")
    return(NULL)
  }
  f$method_canonical <- canonical_method(f$method, f$effective_tokens)
  out <- data.frame(
    source = f$source,
    method = f$method_canonical,
    source_method = f$method,
    ppl = num(f$ppl_primary),
    plddt = num(f$plddt_mean),
    pae = num(f$pae_mean),
    ptm = num(f$ptm_mean),
    sol = num(f$sol_score_mean),
    thermo = num(f$thermo_score_mean),
    effective_tokens = num(f$effective_tokens),
    multi_objective_score = num(f$multi_objective_score),
    pareto_non_dominated = f$pareto_non_dominated,
    stringsAsFactors = FALSE
  )
  out <- out[!is.na(out$method) & out$method != "", ]
  write.csv(out, file.path(out_dir, "pareto_hv_summary.csv"), row.names = FALSE)
  p <- ggplot(out, aes(x = sol, y = thermo, label = method)) +
    geom_point(aes(size = plddt, colour = as.factor(pareto_non_dominated)), alpha = 0.85) +
    geom_text(vjust = -0.8, size = 2.2, show.legend = FALSE) +
    labs(x = "Solubility", y = "Thermostability", size = "pLDDT", colour = "Pareto") +
    theme_cacheopd() +
    theme(legend.position = "right")
  save_plot_pair(p, "pareto_tradeoff", width = 5.8, height = 4.2)
  out
}

pareto <- build_pareto()

build_qualitative <- function() {
  image_candidates <- list.files(path("analysis_outputs"), pattern = "\\.(png|jpg)$", recursive = TRUE, full.names = TRUE)
  image_candidates <- image_candidates[grepl("structure|esmfold|qualitative|case", image_candidates, ignore.case = TRUE)]
  if (length(image_candidates) == 0) {
    add_skip("qualitative_cases.csv", "No per-case structure images were found under analysis_outputs.")
    add_skip("qualitative_examples.pdf/png", "No per-case structure images were found under analysis_outputs.")
    return(NULL)
  }
  out <- data.frame(image_path = image_candidates, stringsAsFactors = FALSE)
  write.csv(out, file.path(out_dir, "qualitative_cases.csv"), row.names = FALSE)
  add_skip("qualitative_examples.pdf/png", "Structure image montage is not rendered by this script because the detected images are not linked to method-level case metadata.")
  out
}

qual_cases <- build_qualitative()

if (nrow(skipped) > 0) {
  write.csv(skipped, file.path(out_dir, "skipped_outputs.csv"), row.names = FALSE)
} else {
  write.csv(data.frame(output = character(), reason = character()), file.path(out_dir, "skipped_outputs.csv"), row.names = FALSE)
}

cat("CacheOPD paper outputs written to:", out_dir, "\n")
if (nrow(skipped) > 0) {
  cat("Skipped optional outputs:\n")
  print(skipped)
}
