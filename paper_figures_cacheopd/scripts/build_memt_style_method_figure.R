#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(grid)
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
report_fig_dir <- file.path(root, "paper_md_report", "Report MD", "figures")
report_src_dir <- file.path(root, "paper_md_report", "Report MD", "source_data")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(src_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(qa_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_src_dir, recursive = TRUE, showWarnings = FALSE)

stem <- "figure1_cacheopd_memt_style_framework"
width_mm <- as.numeric(get_arg("--width_mm", "183"))
height_mm <- as.numeric(get_arg("--height_mm", "112"))
dpi <- as.numeric(get_arg("--dpi", "600"))

col <- list(
  ink = "#101820",
  muted = "#596273",
  grid = "#E8EDF2",
  blue = "#86BBD8",
  blue_dark = "#2F6F8F",
  green = "#7FC6B2",
  green_dark = "#2D7F73",
  orange = "#E3A35C",
  orange_dark = "#B96F22",
  red = "#D77A61",
  red_dark = "#B95040",
  violet = "#A99AD6",
  gray = "#F3F5F7",
  gray2 = "#E9EDF2",
  white = "#FFFFFF"
)

nodes <- data.frame(
  id = c(
    "prompt", "rollout", "prefix", "teacher_fold", "teacher_sol", "teacher_thermo",
    "support", "cache", "poe", "utility", "select", "student", "anchors",
    "online_rollout", "online_teacher", "online_update", "online_repeat"
  ),
  stage = c(
    "I", "I", "I", "II", "II", "II", "II", "II", "II", "III", "III", "III", "III",
    "online", "online", "online", "online"
  ),
  x = c(
    0.095, 0.245, 0.395, 0.555, 0.555, 0.555,
    0.705, 0.855, 0.445, 0.605, 0.735, 0.875, 0.735,
    0.130, 0.375, 0.620, 0.850
  ),
  y = c(
    0.565, 0.565, 0.565, 0.675, 0.565, 0.455,
    0.565, 0.565, 0.245, 0.245, 0.245, 0.245, 0.115,
    0.875, 0.875, 0.875, 0.875
  ),
  w = c(
    0.105, 0.125, 0.125, 0.115, 0.115, 0.115,
    0.125, 0.135, 0.150, 0.115, 0.135, 0.145, 0.180,
    0.150, 0.175, 0.145, 0.150
  ),
  h = c(
    0.090, 0.090, 0.090, 0.075, 0.075, 0.075,
    0.090, 0.105, 0.095, 0.095, 0.095, 0.105, 0.075,
    0.080, 0.080, 0.080, 0.080
  ),
  label = c(
    "Condition prompts\nc",
    "Reference rollouts\nx ~ p_ref",
    "Prefix states\nh_{i,t}",
    "Foldability\nteacher",
    "Solubility\nteacher",
    "Thermostability\nteacher",
    "Teacher-supported\ncandidate tokens",
    "Offline multi-teacher\ncache",
    "Cached PoE target\nq(v | h)",
    "Utility score\nKL(q || p_ref)",
    "Sparse token-position\nselection",
    "Student optimization\nzero teacher calls",
    "LM / EOS anchors\nsequence validity",
    "Student rollout\np_theta",
    "Real-time\nteacher scoring",
    "Student update",
    "Repeated teacher calls\nduring training"
  ),
  family = c(
    "input", "input", "input", "teacher", "teacher", "teacher",
    "support", "cache", "poe", "utility", "select", "student", "anchor",
    "online", "online", "online", "online"
  ),
  stringsAsFactors = FALSE
)

edges <- data.frame(
  from = c(
    "prompt", "rollout", "prefix",
    "teacher_fold", "teacher_sol", "teacher_thermo", "support",
    "support", "poe", "utility", "select", "select",
    "online_rollout", "online_teacher", "online_update"
  ),
  to = c(
    "rollout", "prefix", "support",
    "support", "support", "support", "cache",
    "poe", "utility", "select", "student", "anchors",
    "online_teacher", "online_update", "online_repeat"
  ),
  type = c(
    rep("main", 7), "main", "main", "main", "main", "anchor",
    "online", "online", "online"
  ),
  stringsAsFactors = FALSE
)

callouts <- data.frame(
  id = c("phase1", "phase2", "phase3", "eff1", "eff2", "eff3", "formula"),
  x = c(0.235, 0.755, 0.800, 0.300, 0.545, 0.785, 0.455),
  y = c(0.705, 0.705, 0.360, 0.045, 0.045, 0.045, 0.370),
  label = c(
    "I. Reference visitation",
    "II. Offline teacher cache construction",
    "III. Sparse cached student distillation",
    "one-time cache calls",
    "training-stage teacher calls = 0",
    "35.1% preference-token budget",
    "cached PoE: normalized product over teacher-supported candidate tokens"
  ),
  family = c("phase", "phase", "phase", "eff", "eff", "eff", "formula"),
  stringsAsFactors = FALSE
)

write.csv(nodes, file.path(src_dir, paste0(stem, "_nodes.csv")), row.names = FALSE)
write.csv(edges, file.path(src_dir, paste0(stem, "_edges.csv")), row.names = FALSE)
write.csv(callouts, file.path(src_dir, paste0(stem, "_callouts.csv")), row.names = FALSE)
write.csv(nodes, file.path(report_src_dir, paste0(stem, "_nodes.csv")), row.names = FALSE)
write.csv(edges, file.path(report_src_dir, paste0(stem, "_edges.csv")), row.names = FALSE)
write.csv(callouts, file.path(report_src_dir, paste0(stem, "_callouts.csv")), row.names = FALSE)

family_fill <- c(
  input = "#F2F5F8",
  teacher = "#E8F2F7",
  support = "#EAF4F0",
  cache = "#DDEFEA",
  poe = "#E9F5F1",
  utility = "#F4F0FA",
  select = "#F2F6EA",
  student = "#E7F2EF",
  anchor = "#F6F3EA",
  online = "#FBEDE7"
)
family_border <- c(
  input = "#7A8493",
  teacher = col$blue_dark,
  support = col$green_dark,
  cache = col$green_dark,
  poe = col$green_dark,
  utility = "#7B66AA",
  select = "#799345",
  student = col$green_dark,
  anchor = "#9A7A3B",
  online = col$orange_dark
)

node_by_id <- function(id) nodes[nodes$id == id, ][1, ]

draw_text <- function(label, x, y, size = 6.2, col_text = col$ink, fontface = "plain",
                      just = "centre", lineheight = 0.90) {
  grid.text(
    label,
    x = unit(x, "npc"),
    y = unit(y, "npc"),
    just = just,
    gp = gpar(fontsize = size, col = col_text, fontface = fontface, fontfamily = "Helvetica", lineheight = lineheight)
  )
}

draw_round <- function(x, y, w, h, fill, border, lwd = 1.0, r = 0.012) {
  grid.roundrect(
    x = unit(x, "npc"), y = unit(y, "npc"),
    width = unit(w, "npc"), height = unit(h, "npc"),
    r = unit(r, "npc"),
    gp = gpar(fill = fill, col = border, lwd = lwd)
  )
}

draw_arrow <- function(x1, y1, x2, y2, colour = col$muted, lwd = 1.2, curved = FALSE) {
  if (curved) {
    grid.curve(
      x1 = unit(x1, "npc"), y1 = unit(y1, "npc"),
      x2 = unit(x2, "npc"), y2 = unit(y2, "npc"),
      curvature = 0.30,
      square = FALSE,
      arrow = arrow(type = "closed", length = unit(2.2, "mm")),
      gp = gpar(col = colour, lwd = lwd, lineend = "round")
    )
  } else {
    grid.segments(
      x0 = unit(x1, "npc"), y0 = unit(y1, "npc"),
      x1 = unit(x2, "npc"), y1 = unit(y2, "npc"),
      arrow = arrow(type = "closed", length = unit(2.0, "mm")),
      gp = gpar(col = colour, lwd = lwd, lineend = "round")
    )
  }
}

draw_stacked_cache <- function(x, y, w, h) {
  for (i in 3:1) {
    draw_round(
      x + (i - 1) * 0.006, y + (i - 1) * 0.006,
      w, h,
      fill = adjustcolor(col$green, alpha.f = 0.18 + i * 0.08),
      border = col$green_dark,
      lwd = 0.8,
      r = 0.010
    )
  }
}

draw_node <- function(n) {
  if (n$family == "cache") {
    draw_stacked_cache(n$x, n$y, n$w, n$h)
  } else {
    draw_round(n$x, n$y, n$w, n$h, family_fill[[n$family]], family_border[[n$family]], 0.95)
  }
  draw_text(n$label, n$x, n$y, size = ifelse(n$family == "online", 5.8, 5.9), lineheight = 0.92)
}

draw_edge <- function(e) {
  a <- node_by_id(e$from)
  b <- node_by_id(e$to)
  colour <- ifelse(e$type == "online", col$orange_dark, ifelse(e$type == "anchor", "#9A7A3B", col$green_dark))
  curved <- e$from == "online_repeat" && e$to == "online_rollout"
  if (curved) {
    draw_arrow(a$x, a$y + a$h * 0.52, b$x, b$y + b$h * 0.52, colour = colour, lwd = 1.15, curved = TRUE)
    return()
  }
  dx <- b$x - a$x
  dy <- b$y - a$y
  if (abs(dx) >= abs(dy)) {
    x1 <- a$x + sign(dx) * a$w / 2 + sign(dx) * 0.006
    y1 <- a$y
    x2 <- b$x - sign(dx) * b$w / 2 - sign(dx) * 0.006
    y2 <- b$y
  } else {
    x1 <- a$x
    y1 <- a$y + sign(dy) * a$h / 2 + sign(dy) * 0.006
    x2 <- b$x
    y2 <- b$y - sign(dy) * b$h / 2 - sign(dy) * 0.006
  }
  draw_arrow(x1, y1, x2, y2, colour = colour, lwd = 1.15)
}

draw_figure <- function() {
  grid.newpage()
  grid.rect(gp = gpar(fill = col$white, col = NA))

  draw_text("CacheOPD: cached multi-teacher preference distillation", 0.030, 0.965,
            size = 11.5, fontface = "bold", just = c("left", "centre"))
  draw_text("Offline cache construction turns expensive multi-teacher OPD into reusable cached PoE targets for sparse student training.",
            0.030, 0.928, size = 6.7, col_text = col$muted, just = c("left", "centre"))

  draw_round(0.500, 0.875, 0.940, 0.132, fill = "#FFF8F2", border = "#E8B27C", lwd = 0.9, r = 0.018)
  draw_round(0.500, 0.405, 0.940, 0.620, fill = "#FAFCFB", border = "#D8E5E0", lwd = 0.9, r = 0.018)
  draw_round(0.500, 0.047, 0.940, 0.070, fill = "#F5FAF8", border = "#CAE2DA", lwd = 0.8, r = 0.018)

  draw_text("Online OPD baseline: teacher-in-the-loop optimization", 0.048, 0.933,
            size = 6.4, fontface = "bold", col_text = col$orange_dark, just = c("left", "centre"))
  draw_text("CacheOPD pipeline", 0.048, 0.790,
            size = 7.2, fontface = "bold", col_text = col$green_dark, just = c("left", "centre"))

  for (i in seq_len(nrow(callouts))) {
    cc <- callouts[i, ]
    if (cc$family == "phase") {
      draw_round(cc$x, cc$y, 0.210, 0.036, fill = "#FFFFFF", border = "#D4E4DE", lwd = 0.8, r = 0.010)
      draw_text(cc$label, cc$x, cc$y, size = 5.6, fontface = "bold", col_text = col$green_dark)
    }
  }

  for (i in seq_len(nrow(edges))) draw_edge(edges[i, ])
  for (i in seq_len(nrow(nodes))) draw_node(nodes[i, ])

  draw_round(0.455, 0.370, 0.420, 0.046, fill = "#FFFFFF", border = "#D4E4DE", lwd = 0.75, r = 0.010)
  draw_text(callouts$label[callouts$id == "formula"], 0.455, 0.370, size = 5.5, col_text = col$green_dark)

  eff <- callouts[callouts$family == "eff", ]
  for (i in seq_len(nrow(eff))) {
    draw_round(eff$x[i], eff$y[i], 0.245, 0.040, fill = "#FFFFFF", border = "#C7DED6", lwd = 0.75, r = 0.012)
    draw_text(eff$label[i], eff$x[i], eff$y[i], size = 5.7, col_text = col$green_dark, fontface = "bold")
  }

  draw_text("Reference visitation distribution p_ref", 0.230, 0.485, size = 5.4, col_text = col$muted)
  draw_text("Top-K union support", 0.805, 0.485, size = 5.4, col_text = col$muted)
  draw_text("Full-vocabulary student softmax", 0.874, 0.170, size = 5.4, col_text = col$muted)
}

save_all <- function(base_path, width_mm_use = width_mm, height_mm_use = height_mm) {
  w <- width_mm_use / 25.4
  h <- height_mm_use / 25.4
  svg(paste0(base_path, ".svg"), width = w, height = h, family = "Helvetica", onefile = TRUE)
  draw_figure()
  dev.off()
  grDevices::cairo_pdf(paste0(base_path, ".pdf"), width = w, height = h, family = "Helvetica")
  draw_figure()
  dev.off()
  png(paste0(base_path, ".png"), width = w, height = h, units = "in", res = dpi, type = "cairo")
  draw_figure()
  dev.off()
}

base_editable <- file.path(fig_dir, stem)
save_all(base_editable)

for (ext in c("svg", "pdf", "png")) {
  file.copy(
    file.path(fig_dir, paste0(stem, ".", ext)),
    file.path(report_fig_dir, paste0(stem, ".", ext)),
    overwrite = TRUE
  )
}

writeLines(c(
  "Figure: Mem-T-style CacheOPD method framework.",
  "Core conclusion: CacheOPD moves multi-teacher preference acquisition to an offline cache and trains the student from cached PoE targets over teacher-supported candidate tokens.",
  "Source data: source_data/figure1_cacheopd_memt_style_framework_nodes.csv, edges.csv, callouts.csv.",
  "Outputs: SVG/PDF/PNG in paper_figures_cacheopd/figures_editable and paper_md_report/Report MD/figures.",
  "No expensive generation or scoring is performed by this script."
), file.path(qa_dir, paste0(stem, "_QA_notes.txt")))

cat("[OK] Mem-T-style method figure -> ", base_editable, ".svg\n", sep = "")
cat("[OK] Report copies -> ", file.path(report_fig_dir, paste0(stem, ".svg")), "\n", sep = "")
cat("[OK] Source data -> ", file.path(src_dir, paste0(stem, "_nodes.csv")), "\n", sep = "")
