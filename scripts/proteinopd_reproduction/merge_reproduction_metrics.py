from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_all_csv(directory: Path, suffix: str = ".csv") -> pd.DataFrame:
    if not directory.exists():
        return pd.DataFrame()
    frames = []
    for fp in sorted(directory.glob(f"*{suffix}")):
        if "summary" in fp.stem.lower() or "manifest" in fp.stem.lower():
            continue
        try:
            frames.append(pd.read_csv(fp))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_cohorts(out_root: Path) -> pd.DataFrame:
    frames = []
    for setting in ["unconditional", "conditional"]:
        directory = out_root / "eval_cohorts" / setting
        for fp in sorted(directory.glob("*.csv")):
            df = pd.read_csv(fp)
            if "setting" not in df.columns:
                df["setting"] = setting
            if "method" not in df.columns:
                df["method"] = fp.stem
            if "seq_id" not in df.columns:
                df["seq_id"] = [f"{fp.stem}_{i:05d}" for i in range(len(df))]
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["setting", "method", "seq_id", "sequence", "length"])
    base = pd.concat(frames, ignore_index=True)
    keep = [c for c in ["setting", "method", "seq_id", "sequence", "length", "source"] if c in base.columns]
    return base[keep].drop_duplicates(["setting", "method", "seq_id"])


def merge_metric(base: pd.DataFrame, metric: pd.DataFrame, keep: dict[str, str], setting: str) -> pd.DataFrame:
    if base.empty or metric.empty:
        return base
    metric = metric.copy()
    if "seq_id" not in metric.columns and "sequence_id" in metric.columns:
        metric = metric.rename(columns={"sequence_id": "seq_id"})
    if "method" not in metric.columns or "seq_id" not in metric.columns:
        return base
    metric["setting"] = setting
    rename = {src: dst for src, dst in keep.items() if src in metric.columns}
    metric = metric.rename(columns=rename)
    cols = ["setting", "method", "seq_id"] + list(rename.values())
    cols = [c for c in cols if c in metric.columns]
    if len(cols) <= 3:
        return base
    metric = metric[cols].drop_duplicates(["setting", "method", "seq_id"])

    merged = base.copy()
    key_cols = ["setting", "method", "seq_id"]
    metric = metric.set_index(key_cols)
    merged_index = pd.MultiIndex.from_frame(merged[key_cols])

    for col in [c for c in metric.columns if c not in key_cols]:
        if col not in merged.columns:
            merged[col] = pd.NA
        values = metric[col].reindex(merged_index).to_numpy()
        mask = pd.notna(values)
        if mask.any():
            merged.loc[mask, col] = values[mask]
    return merged


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    aa = set("ACDEFGHIKLMNPQRSTVWY")
    df["valid_aa"] = df["sequence"].astype(str).str.upper().apply(lambda x: bool(x) and set(x).issubset(aa))
    df["duplicate"] = df.duplicated(["setting", "method", "sequence"], keep=False)
    grouped = []
    metrics = {
        "ppl": "mean",
        "ppl_loss": "mean",
        "novelty_u": "mean",
        "novelty_t": "mean",
        "plddt": "mean",
        "pae": "mean",
        "ptm": "mean",
        "sol": "mean",
        "thermo": "mean",
    }
    for (setting, method), g in df.groupby(["setting", "method"], dropna=False):
        plddt_vals = pd.to_numeric(g.get("plddt"), errors="coerce") if "plddt" in g.columns else None
        if plddt_vals is not None and plddt_vals.notna().any() and plddt_vals.max() <= 1.0:
            plddt_vals = plddt_vals * 100.0
        row = {
            "setting": setting,
            "method": method,
            "n_total": len(g),
            "n_valid": int(g["valid_aa"].sum()),
            "valid_rate": float(g["valid_aa"].mean()),
            "unique_sequences": int(g["sequence"].nunique()),
            "duplicate_rate": float(1.0 - g["sequence"].nunique() / max(len(g), 1)),
            "mean_length": float(pd.to_numeric(g.get("length"), errors="coerce").mean()),
        }
        for metric, agg in metrics.items():
            if metric in g.columns:
                vals = plddt_vals if metric == "plddt" and plddt_vals is not None else pd.to_numeric(g[metric], errors="coerce")
                row[f"{metric}_{agg}"] = float(vals.mean()) if vals.notna().any() else None
                row[f"{metric}_n"] = int(vals.notna().sum())
            else:
                row[f"{metric}_{agg}"] = None
                row[f"{metric}_n"] = 0
        row["plddt_gt70_pct"] = float((plddt_vals > 70).mean() * 100) if plddt_vals is not None else None
        row["pae_lt10_pct"] = float((pd.to_numeric(g.get("pae"), errors="coerce") < 10).mean() * 100) if "pae" in g.columns else None
        grouped.append(row)
    return pd.DataFrame(grouped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-root", default="analysis_outputs/proteinopd_reproduction")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    raw = out_root / "raw_eval"
    summaries = out_root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)

    merged = load_cohorts(out_root)
    metric_specs = [
        ("unconditional", raw / "ppl_unconditional_protgpt2", {"ppl": "ppl", "ppl_loss": "ppl_loss", "nll": "ppl_loss"}),
        ("conditional", raw / "ppl_conditional_prollama", {"ppl": "ppl", "ppl_loss": "ppl_loss", "nll": "ppl_loss"}),
        ("unconditional", raw / "structure_unconditional", {"plddt": "plddt", "pae": "pae", "ptm": "ptm"}),
        ("conditional", raw / "structure_conditional", {"plddt": "plddt", "pae": "pae", "ptm": "ptm"}),
        ("unconditional", raw / "temberture_unconditional", {"thermo_score": "thermo"}),
        ("conditional", raw / "temberture_conditional", {"thermo_score": "thermo"}),
        ("unconditional", raw / "protein_sol_unconditional", {"sol_score": "sol", "scaled_sol": "sol"}),
        ("conditional", raw / "protein_sol_conditional", {"sol_score": "sol", "scaled_sol": "sol"}),
        ("unconditional", raw / "novelty_u_unconditional", {"novelty_mmseqs": "novelty_u"}),
        ("conditional", raw / "novelty_u_conditional", {"novelty_mmseqs": "novelty_u"}),
        ("unconditional", raw / "novelty_t_unconditional", {"novelty_mmseqs": "novelty_t"}),
    ]
    for setting, directory, keep in metric_specs:
        metric = read_all_csv(directory)
        merged = merge_metric(merged, metric, keep, setting)

    merged_path = summaries / "proteinopd_reproduction_per_sequence_metrics.csv"
    merged.to_csv(merged_path, index=False)
    summary = summarize(merged)
    summary_path = summaries / "proteinopd_reproduction_method_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] wrote {merged_path}")
    print(f"[OK] wrote {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
