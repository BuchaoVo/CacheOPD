#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TABLES_TO_MERGE = [
    "analysis_outputs/offline_proteinopd/p0_eval/final/offline_p0_full_summary.csv",
    "analysis_outputs/offline_proteinopd/experiment3_fusion/experiment3_full_eval_summary.csv",
    "analysis_outputs/offline_proteinopd/experiment3_fusion/experiment3_compact_results.csv",
    "analysis_outputs/offline_proteinopd/contribution2/base_rollout_keep50/contribution2_full_eval_summary.csv",
    "analysis_outputs/offline_proteinopd/contribution2/base_rollout_keep50/experiment4_selection_ablation_summary.csv",
    "analysis_outputs/offline_proteinopd/contribution3/base_rollout_conflict/contribution3_full_eval_summary.csv",
    "analysis_outputs/offline_proteinopd/anchor_ablation/anchor_ablation_summary.csv",
]


def method_alias(name):
    aliases = {
        "Offline-BaseRollout-K64-DefaultGamma0.1": "Offline-BaseRollout-K64",
        "Offline-OPD": "Offline-OPDRollout-K64",
        "Conditional-OPD": "Baseline",
        "Online OPD baseline": "Baseline",
    }
    return aliases.get(str(name), str(name))


def read_prollama_summary(path):
    ppl = pd.read_csv(path)
    extra_path = path.parent.parent / "prollama_ppl_final_utility" / "ppl_diagnostic_summary.csv"
    if extra_path.exists():
        ppl = pd.concat([ppl, pd.read_csv(extra_path)], ignore_index=True)
    ppl = ppl[(ppl["model"] == "ProLLaMA") & (ppl["mode"] == "raw")].copy()
    ppl["method_key"] = ppl["cohort"].map(method_alias)
    cols = [
        "method_key",
        "mean_nll",
        "median_nll",
        "ppl_individual_mean",
        "ppl_individual_median",
        "ppl_from_mean_nll",
        "ppl_from_median_nll",
    ]
    ppl = ppl[cols].rename(
        columns={
            "mean_nll": "prollama_mean_nll",
            "median_nll": "prollama_median_nll",
            "ppl_individual_mean": "prollama_ppl_individual_mean",
            "ppl_individual_median": "prollama_ppl_individual_median",
            "ppl_from_mean_nll": "prollama_ppl_from_mean_nll",
            "ppl_from_median_nll": "prollama_ppl_from_median_nll",
        }
    )
    return ppl.drop_duplicates("method_key")


def write_table(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    md_path = path.with_suffix(".md")
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")


def merge_ppl_into_tables(root, ppl):
    out_paths = []
    for rel in TABLES_TO_MERGE:
        in_path = root / rel
        if not in_path.exists():
            continue
        df = pd.read_csv(in_path)
        if "method" not in df.columns:
            continue
        df["method_key"] = df["method"].map(method_alias)
        merged = df.merge(ppl, on="method_key", how="left")
        merged["ppl_primary"] = merged["prollama_ppl_from_mean_nll"].combine_first(
            merged["ppl_mean"] if "ppl_mean" in merged.columns else pd.Series(np.nan, index=merged.index)
        )
        merged["ppl_primary_source"] = np.where(
            merged["prollama_ppl_from_mean_nll"].notna(),
            "ProLLaMA raw mean-NLL",
            "existing table",
        )
        out_path = in_path.with_name(in_path.stem + "_prollama_ppl.csv")
        write_table(merged.drop(columns=["method_key"]), out_path)
        out_paths.append(str(out_path))
    return out_paths


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def collect_rows(root, ppl):
    sources = [
        ("main_p0", root / "analysis_outputs/offline_proteinopd/p0_eval/final/offline_p0_full_summary.csv"),
        ("fusion", root / "analysis_outputs/offline_proteinopd/experiment3_fusion/experiment3_full_eval_summary.csv"),
        ("selection_old", root / "analysis_outputs/offline_proteinopd/contribution2/base_rollout_keep50/contribution2_full_eval_summary.csv"),
        ("conflict", root / "analysis_outputs/offline_proteinopd/contribution3/base_rollout_conflict/contribution3_full_eval_summary.csv"),
        ("anchor", root / "analysis_outputs/offline_proteinopd/anchor_ablation/anchor_ablation_summary.csv"),
        ("final_utility", root / "analysis_outputs/offline_proteinopd/final_utility_eval/final/final_utility_full_summary.csv"),
    ]
    rows = []
    for source, path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "method" not in df.columns:
            continue
        cols = {
            "ppl_loss_mean": pick_col(df, ["ppl_loss_mean"]),
            "ppl_mean_existing": pick_col(df, ["ppl_mean"]),
            "plddt_mean": pick_col(df, ["plddt_mean_100", "plddt_mean"]),
            "pae_mean": pick_col(df, ["pae_mean"]),
            "ptm_mean": pick_col(df, ["ptm_mean"]),
            "sol_score_mean": pick_col(df, ["sol_score_mean"]),
            "thermo_score_mean": pick_col(df, ["thermo_score_mean"]),
            "thermophilic_rate": pick_col(df, ["thermophilic_rate"]),
            "diversity_score_3mer": pick_col(df, ["diversity_score_3mer"]),
            "novelty_t_3mer_proxy_mean": pick_col(df, ["novelty_t_3mer_proxy_mean"]),
            "valid_rate": pick_col(df, ["valid_rate"]),
            "duplicate_rate": pick_col(df, ["duplicate_rate"]),
            "mean_length": pick_col(df, ["mean_length", "basic_mean_length"]),
            "effective_tokens": pick_col(df, ["effective_tokens", "effective_training_tokens", "selected_tokens"]),
            "train_runtime_sec": pick_col(df, ["train_runtime_sec", "total_elapsed_sec"]),
            "final_loss_poe": pick_col(df, ["final_loss_poe"]),
        }
        for _, r in df.iterrows():
            row = {"source": source, "method": r["method"], "method_key": method_alias(r["method"])}
            for out_col, in_col in cols.items():
                row[out_col] = r[in_col] if in_col else np.nan
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(ppl, on="method_key", how="left")
    out["ppl_primary"] = out["prollama_ppl_from_mean_nll"].combine_first(out["ppl_mean_existing"])
    return out


def normalize(series, higher_is_better):
    x = pd.to_numeric(series, errors="coerce")
    lo, hi = x.min(skipna=True), x.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(np.nan, index=series.index)
    z = (x - lo) / (hi - lo)
    if not higher_is_better:
        z = 1.0 - z
    return z


def add_pareto_columns(df):
    if df.empty:
        return df
    score_cols = {
        "score_plddt": ("plddt_mean", True),
        "score_pae": ("pae_mean", False),
        "score_sol": ("sol_score_mean", True),
        "score_thermo": ("thermo_score_mean", True),
        "score_ppl": ("ppl_primary", False),
    }
    out = df.copy()
    for new_col, (old_col, hib) in score_cols.items():
        out[new_col] = normalize(out[old_col], hib)
    score_matrix = out[list(score_cols)]
    out["n_score_objectives"] = score_matrix.notna().sum(axis=1)
    out["multi_objective_score"] = score_matrix.mean(axis=1, skipna=True)
    out.loc[out["n_score_objectives"] < 4, "multi_objective_score"] = np.nan

    objectives = ["plddt_mean", "sol_score_mean", "thermo_score_mean"]
    transformed = pd.DataFrame(
        {
            "plddt": pd.to_numeric(out["plddt_mean"], errors="coerce"),
            "neg_pae": -pd.to_numeric(out["pae_mean"], errors="coerce"),
            "sol": pd.to_numeric(out["sol_score_mean"], errors="coerce"),
            "thermo": pd.to_numeric(out["thermo_score_mean"], errors="coerce"),
            "neg_ppl": -pd.to_numeric(out["ppl_primary"], errors="coerce"),
        }
    )
    valid = transformed.notna().all(axis=1)
    dominated = np.zeros(len(out), dtype=bool)
    vals = transformed.to_numpy(dtype=float)
    for i in range(len(out)):
        if not valid.iloc[i]:
            dominated[i] = False
            continue
        for j in range(len(out)):
            if i == j or not valid.iloc[j]:
                continue
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                dominated[i] = True
                break
    out["pareto_non_dominated"] = valid & ~dominated
    out["pareto_objectives"] = "pLDDT, -pAE, Sol, Thermo, -ProLLaMAPPL"
    out["rank_multi_objective"] = out["multi_objective_score"].rank(ascending=False, method="min")
    return out.sort_values(["rank_multi_objective", "source", "method"], na_position="last")


def make_gap_status(out_dir):
    rows = [
        {
            "priority": "P0-1",
            "experiment": "Final CacheOPD combined run",
            "status": "not yet run",
            "current_evidence": "Components are separated: raw cached PoE, old utility selection, conflict analysis, anchor/gamma.",
            "next_action": "Build final target path with smoothing + support gate + final utility mask, then train/generate/evaluate.",
        },
        {
            "priority": "P0-2",
            "experiment": "Fusion / smoothing ablation",
            "status": "partially run",
            "current_evidence": "ProbAvg, LogitAvg, Raw PoE are available; smoothing/final support-gated target is missing.",
            "next_action": "Add PoE+smoothing and PoE+smoothing+support gate variants.",
        },
        {
            "priority": "P0-3",
            "experiment": "Anchor ablation + gamma sweep",
            "status": "partially run",
            "current_evidence": "gamma_nll = 0, 0.1, 1.0 downstream; gamma training diagnostics for 0, 0.03, 0.1, 0.3, 1.0.",
            "next_action": "Add LM-only vs LM+EOS if EOS anchor is implemented; evaluate gamma=0.3 downstream if needed.",
        },
        {
            "priority": "P0-4",
            "experiment": "Support gate q_delta sensitivity",
            "status": "cache-level + selection sweep run",
            "current_evidence": "q_delta support sweep, coverage distribution, and final shift+entropy cache selection sweep exist; downstream selected-model eval is missing.",
            "next_action": "Train/generate/evaluate q_delta in {0, 0.2, 0.4} under final utility.",
        },
        {
            "priority": "P0-5",
            "experiment": "Updated final utility selection",
            "status": "cache-level + first downstream run",
            "current_evidence": "U=shift+entropy with coverage hard gate has been scored; final high adapter has been trained and generated with basic sequence metrics + ProLLaMA PPL.",
            "next_action": "Complete ProteinSol/TemBERTure/ESMFold for final high, then train/generate/evaluate random/support-only/shift-only/low controls.",
        },
        {
            "priority": "P1-1",
            "experiment": "Downstream K sensitivity",
            "status": "cache-level run",
            "current_evidence": "K diagnostics exist for 8,16,32,64; downstream only K=64.",
            "next_action": "Train/generate/evaluate K=32/64/128 or at least K=32/64.",
        },
        {
            "priority": "P1-3",
            "experiment": "SFT / Avg-KD baselines",
            "status": "Avg-KD partially run",
            "current_evidence": "ProbAvg and LogitAvg KD variants exist; pure SFT and single-teacher KD missing.",
            "next_action": "Add objective=SFT and single-teacher cache variants.",
        },
        {
            "priority": "P1-4",
            "experiment": "Pareto frontier",
            "status": "analysis added",
            "current_evidence": "Method-level Pareto and multi-objective rank generated from existing downstream tables.",
            "next_action": "Optionally add per-sequence Pareto HV plots for final paper figures.",
        },
    ]
    df = pd.DataFrame(rows)
    write_table(df, out_dir / "supplemental_gap_status.csv")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zbc/data/opd/ProteinOPD")
    ap.add_argument(
        "--prollama_summary",
        default="analysis_outputs/offline_proteinopd/prollama_ppl_unified/ppl_diagnostic_summary.csv",
    )
    ap.add_argument("--out_dir", default="analysis_outputs/offline_proteinopd/supplemental_analysis")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ppl = read_prollama_summary(root / args.prollama_summary)
    write_table(ppl, out_dir / "unified_prollama_ppl_by_method.csv")

    merged_paths = merge_ppl_into_tables(root, ppl)
    consolidated = collect_rows(root, ppl)
    consolidated = add_pareto_columns(consolidated)

    core_cols = [
        "source",
        "method",
        "ppl_primary",
        "prollama_ppl_from_mean_nll",
        "ppl_loss_mean",
        "plddt_mean",
        "pae_mean",
        "ptm_mean",
        "sol_score_mean",
        "thermo_score_mean",
        "thermophilic_rate",
        "diversity_score_3mer",
        "novelty_t_3mer_proxy_mean",
        "mean_length",
        "effective_tokens",
        "final_loss_poe",
        "multi_objective_score",
        "n_score_objectives",
        "rank_multi_objective",
        "pareto_non_dominated",
    ]
    core_cols = [c for c in core_cols if c in consolidated.columns]
    write_table(consolidated[core_cols], out_dir / "cacheopd_consolidated_metrics_prollama_ppl.csv")

    pareto = consolidated[consolidated["pareto_non_dominated"] == True][core_cols]
    write_table(pareto, out_dir / "method_level_pareto_frontier.csv")
    gap = make_gap_status(out_dir)

    manifest = pd.DataFrame(
        {
            "artifact": [
                "unified_prollama_ppl_by_method",
                "cacheopd_consolidated_metrics_prollama_ppl",
                "method_level_pareto_frontier",
                "supplemental_gap_status",
            ]
            + [Path(p).stem for p in merged_paths],
            "path": [
                str(out_dir / "unified_prollama_ppl_by_method.csv"),
                str(out_dir / "cacheopd_consolidated_metrics_prollama_ppl.csv"),
                str(out_dir / "method_level_pareto_frontier.csv"),
                str(out_dir / "supplemental_gap_status.csv"),
            ]
            + merged_paths,
        }
    )
    write_table(manifest, out_dir / "supplemental_analysis_manifest.csv")
    print(f"[OK] wrote supplemental analysis to {out_dir}")
    print(f"[OK] merged ProLLaMA PPL into {len(merged_paths)} experiment tables")
    print(gap[["priority", "experiment", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
