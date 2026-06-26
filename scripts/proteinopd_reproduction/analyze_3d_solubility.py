from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

HYDROPHOBIC = set("AVILMFWYC")
POLAR = set("STNQCGP")
CHARGED = set("DEKRH")
ACIDIC = set("DE")
BASIC = set("KRH")

METHOD_ALIASES = {
    "CacheOPD Full": "CacheOPD-Full",
    "CacheOPD-Full": "CacheOPD-Full",
    "Offline cached PoE-Full": "CacheOPD-Full",
    "CacheOPD Sparse": "CacheOPD-Sparse",
    "CacheOPD-Sparse": "CacheOPD-Sparse",
    "FinalUtility-ShiftEntropy-q04-High50": "CacheOPD-Sparse",
    "ProteinOPD": "Online OPD",
    "Online OPD": "Online OPD",
    "Online_OPD": "Online OPD",
    "Fold-teacher_KD": "Fold-teacher KD",
    "Sol-teacher_KD": "Sol-teacher KD",
    "Thermo-teacher_KD": "Thermo-teacher KD",
}


def normalize_method(name: object) -> str:
    raw = str(name)
    if raw in METHOD_ALIASES:
        return METHOD_ALIASES[raw]
    repaired = raw.replace("_KD", " KD")
    repaired = repaired.replace("Online_OPD", "Online OPD")
    return METHOD_ALIASES.get(repaired, repaired)


def norm_key(setting: object, method: object, seq_id: object) -> str:
    method_norm = re.sub(r"[^a-z0-9]+", "", normalize_method(method).lower())
    setting_norm = re.sub(r"[^a-z0-9]+", "", str(setting).lower())
    return f"{setting_norm}::{method_norm}::{seq_id}"


def parse_pdb_filename(path: Path) -> dict[str, str]:
    stem = path.stem
    parts = stem.split("__")
    if len(parts) >= 4 and parts[2].startswith("rank"):
        return {"setting": parts[0], "method": normalize_method(parts[1]), "seq_id": "__".join(parts[3:]), "case_id": stem}
    if len(parts) >= 3:
        return {"setting": parts[0], "method": normalize_method(parts[1]), "seq_id": "__".join(parts[2:]), "case_id": stem}
    return {"setting": "", "method": "", "seq_id": stem, "case_id": stem}


def normalize_metric_table(path: Path, default_setting: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "seq_id" not in df.columns:
        if "sequence_id" in df.columns:
            df["seq_id"] = df["sequence_id"]
        elif "sample_id" in df.columns and "method" in df.columns:
            df["seq_id"] = df["method"].astype(str) + "_" + df["sample_id"].astype(str).str.zfill(5)
    if "setting" not in df.columns:
        df["setting"] = default_setting
    if "method" in df.columns:
        df["method"] = df["method"].map(normalize_method)
    if {"setting", "method", "seq_id"}.issubset(df.columns):
        df["_merge_key"] = [norm_key(r.setting, r.method, r.seq_id) for r in df.itertuples(index=False)]
    return df


def load_metadata(paths: list[str], default_setting: str) -> pd.DataFrame:
    frames = []
    for raw in paths:
        path = Path(raw)
        if path.exists():
            frames.append(normalize_metric_table(path, default_setting))
    if not frames:
        return pd.DataFrame()
    meta = pd.concat(frames, ignore_index=True)
    if "_merge_key" in meta.columns:
        meta = meta.drop_duplicates("_merge_key", keep="first")
    return meta


def parse_pdb(path: Path) -> list[dict]:
    residues: dict[tuple, dict] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b = float(line[60:66]) if len(line) >= 66 else np.nan
            except ValueError:
                continue
            aa = THREE_TO_ONE.get(resname)
            if aa is None:
                continue
            key = (chain, resseq, line[26].strip(), resname)
            rec = residues.setdefault(key, {"chain": chain, "resseq": resseq, "resname": resname, "aa": aa, "atoms": [], "bfactors": []})
            rec["atoms"].append((atom, np.array([x, y, z], dtype=float)))
            rec["bfactors"].append(b)

    ordered = []
    for key in sorted(residues):
        rec = residues[key]
        ca = None
        for atom, coord in rec["atoms"]:
            if atom == "CA":
                ca = coord
                break
        if ca is None:
            ca = np.mean([coord for _, coord in rec["atoms"]], axis=0)
        rec["ca"] = ca
        rec["centroid"] = np.mean([coord for _, coord in rec["atoms"]], axis=0)
        rec["mean_bfactor"] = float(np.nanmean(rec["bfactors"])) if rec["bfactors"] else np.nan
        ordered.append(rec)
    return ordered


def pairwise_dist(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def safe_frac(mask: np.ndarray, denom_mask: np.ndarray | None = None) -> float:
    if denom_mask is None:
        denom = len(mask)
        return float(np.sum(mask) / denom) if denom else np.nan
    denom = int(np.sum(denom_mask))
    return float(np.sum(mask & denom_mask) / denom) if denom else np.nan


def pdb_features(path: Path) -> dict:
    residues = parse_pdb(path)
    base = parse_pdb_filename(path)
    base.update({"pdb_file": str(path), "n_residues_pdb": len(residues)})
    if len(residues) < 2:
        return {**base, "status": "too_short"}

    aa = np.array([r["aa"] for r in residues])
    ca = np.vstack([r["ca"] for r in residues])
    dist = pairwise_dist(ca)
    np.fill_diagonal(dist, np.nan)
    neighbor_10 = np.nansum(dist <= 10.0, axis=1)
    contact_8 = np.nansum(dist <= 8.0) / 2.0

    centered = ca - ca.mean(axis=0, keepdims=True)
    rg = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    max_diameter = float(np.nanmax(dist))
    contact_density = float(contact_8 / len(residues))

    # This is a CA-neighborhood exposure proxy, not physical SASA.
    exposure_score = 1.0 - np.minimum(neighbor_10 / 24.0, 1.0)
    exposed = exposure_score >= 0.35
    if exposed.sum() < max(3, int(0.15 * len(residues))):
        threshold = np.nanquantile(exposure_score, 0.75)
        exposed = exposure_score >= threshold

    hydro = np.array([x in HYDROPHOBIC for x in aa])
    polar = np.array([x in POLAR for x in aa])
    charged = np.array([x in CHARGED for x in aa])
    acidic = np.array([x in ACIDIC for x in aa])
    basic = np.array([x in BASIC for x in aa])

    exposed_hydro_idx = np.where(exposed & hydro)[0]
    hydro_patch_contacts = 0.0
    if len(exposed_hydro_idx) >= 2:
        sub = dist[np.ix_(exposed_hydro_idx, exposed_hydro_idx)]
        hydro_patch_contacts = float(np.nansum(sub <= 8.0) / 2.0)
    hydro_patch_per_res = hydro_patch_contacts / max(1, len(exposed_hydro_idx))

    exposed_hydrophobic_frac = safe_frac(hydro, exposed)
    exposed_charged_frac = safe_frac(charged, exposed)
    exposed_polar_frac = safe_frac(polar, exposed)
    exposed_acidic_frac = safe_frac(acidic, exposed)
    exposed_basic_frac = safe_frac(basic, exposed)
    exposed_net_charge_per_residue = float((np.sum(basic & exposed) - np.sum(acidic & exposed)) / max(1, np.sum(exposed)))
    surface_balance = float(exposed_polar_frac + exposed_charged_frac - exposed_hydrophobic_frac)
    hydrophobic_risk = float(exposed_hydrophobic_frac + 0.05 * hydro_patch_per_res - 0.5 * exposed_charged_frac)

    features = {
        **base,
        "status": "ok",
        "sequence_from_pdb": "".join(aa.tolist()),
        "mean_bfactor": float(np.nanmean([r["mean_bfactor"] for r in residues])),
        "radius_gyration_ca": rg,
        "max_ca_diameter": max_diameter,
        "mean_ca_neighbor_10a": float(np.nanmean(neighbor_10)),
        "contact_density_8a": contact_density,
        "exposed_residue_frac_proxy": float(np.mean(exposed)),
        "mean_exposure_score_proxy": float(np.mean(exposure_score)),
        "hydrophobic_frac": safe_frac(hydro),
        "polar_frac": safe_frac(polar),
        "charged_frac": safe_frac(charged),
        "acidic_frac": safe_frac(acidic),
        "basic_frac": safe_frac(basic),
        "exposed_hydrophobic_frac_proxy": exposed_hydrophobic_frac,
        "exposed_polar_frac_proxy": exposed_polar_frac,
        "exposed_charged_frac_proxy": exposed_charged_frac,
        "exposed_acidic_frac_proxy": exposed_acidic_frac,
        "exposed_basic_frac_proxy": exposed_basic_frac,
        "exposed_net_charge_per_residue_proxy": exposed_net_charge_per_residue,
        "exposed_hydrophobic_patch_contacts_proxy": hydro_patch_contacts,
        "exposed_hydrophobic_patch_per_residue_proxy": hydro_patch_per_res,
        "sol3d_surface_balance_proxy": surface_balance,
        "sol3d_hydrophobic_risk_proxy": hydrophobic_risk,
    }
    return features


def corr_table(df: pd.DataFrame, target: str, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    if target not in df.columns:
        return pd.DataFrame(rows)
    y = pd.to_numeric(df[target], errors="coerce")
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 3:
            continue
        pearson = float(x[valid].corr(y[valid], method="pearson"))
        spearman = float(x[valid].rank().corr(y[valid].rank(), method="pearson"))
        rows.append({"target": target, "feature": col, "n": int(valid.sum()), "pearson": pearson, "spearman": spearman})
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)


def summarize_by_method(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    metrics = [c for c in ["sol", "plddt_100", "pae", "ptm", "ppl", "thermo"] if c in df.columns]
    cols = metrics + feature_cols
    numeric = df.copy()
    for col in cols:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    grouped = numeric.groupby(["setting", "method"], dropna=False)
    summary = grouped[cols].agg(["count", "mean", "median", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns.to_flat_index()]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract lightweight 3D solubility features from ESMFold PDBs and merge with CacheOPD metrics.")
    parser.add_argument("--pdb_dir", action="append", required=True, help="Directory containing PDB files. Can be repeated.")
    parser.add_argument("--metrics_csv", action="append", default=[], help="Per-sequence metric CSV for merging. Can be repeated.")
    parser.add_argument("--metadata_csv", action="append", default=[], help="Selected sequence CSV with case_id/method/seq_id metadata. Can be repeated.")
    parser.add_argument("--out_dir", default="analysis_outputs/proteinopd_reproduction/sol3d_diagnostics")
    parser.add_argument("--default_setting", default="conditional")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = []
    for raw_dir in args.pdb_dir:
        pdb_files.extend(sorted(Path(raw_dir).glob("*.pdb")))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found under: {args.pdb_dir}")

    features = pd.DataFrame([pdb_features(path) for path in pdb_files])
    features["method"] = features["method"].map(normalize_method)
    features["_merge_key"] = [norm_key(r.setting, r.method, r.seq_id) for r in features.itertuples(index=False)]

    meta = load_metadata(args.metadata_csv + args.metrics_csv, args.default_setting)
    merged = features.copy()
    if len(meta) and "case_id" in meta.columns:
        case_meta = meta.drop_duplicates("case_id", keep="first")
        value_cols = [c for c in case_meta.columns if c not in {"pdb_file"} and c not in merged.columns]
        value_cols = ["case_id"] + value_cols
        merged = merged.merge(case_meta[value_cols], on="case_id", how="left")
    if len(meta) and "_merge_key" in meta.columns:
        value_cols = [c for c in meta.columns if c not in {"case_id", "pdb_file"} and c not in merged.columns]
        value_cols = ["_merge_key"] + value_cols
        merged = merged.merge(meta[value_cols], on="_merge_key", how="left")
    if "sequence" not in merged.columns and "sequence_from_pdb" in merged.columns:
        merged["sequence"] = merged["sequence_from_pdb"]
    if "plddt" in merged.columns:
        plddt = pd.to_numeric(merged["plddt"], errors="coerce")
        merged["plddt_100"] = np.where(plddt <= 1.0, plddt * 100.0, plddt)

    feature_cols = [
        c
        for c in features.columns
        if c.endswith("_proxy")
        or c
        in {
            "mean_bfactor",
            "radius_gyration_ca",
            "max_ca_diameter",
            "mean_ca_neighbor_10a",
            "contact_density_8a",
            "hydrophobic_frac",
            "polar_frac",
            "charged_frac",
            "acidic_frac",
            "basic_frac",
        }
    ]

    features_path = out_dir / "sol3d_features.csv"
    merged_path = out_dir / "sol3d_merged_metrics.csv"
    summary_path = out_dir / "sol3d_method_summary.csv"
    corr_path = out_dir / "sol3d_feature_correlations.csv"
    readme_path = out_dir / "README.md"

    features.drop(columns=["_merge_key"], errors="ignore").to_csv(features_path, index=False)
    merged.drop(columns=["_merge_key"], errors="ignore").to_csv(merged_path, index=False)
    summarize_by_method(merged, feature_cols).to_csv(summary_path, index=False)

    corr_frames = []
    for target in ["sol", "plddt_100", "ptm", "pae", "thermo", "ppl"]:
        table = corr_table(merged, target, feature_cols)
        if len(table):
            corr_frames.append(table)
    corr = pd.concat(corr_frames, ignore_index=True) if corr_frames else pd.DataFrame()
    corr.to_csv(corr_path, index=False)

    readme_path.write_text(
        "\n".join(
            [
                "# CacheOPD 3D Solubility Diagnostics",
                "",
                "This directory contains ESMFold-PDB-derived structure/solubility co-analysis outputs.",
                "",
                "Important: columns ending with `_proxy` are lightweight geometry proxies based on CA-neighborhood exposure, not physical SASA. They are suitable for diagnostic trend analysis and should not be described as exact solvent-accessible surface area.",
                "",
                "Outputs:",
                f"- `{features_path.name}`: PDB-only 3D features.",
                f"- `{merged_path.name}`: 3D features merged with available per-sequence metrics such as Sol, pLDDT, pAE, pTM, PPL, and Thermo.",
                f"- `{summary_path.name}`: method-level means/medians/stds for metrics and 3D features.",
                f"- `{corr_path.name}`: Pearson/Spearman correlations between 3D features and scalar metrics.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[OK] PDB files analyzed: {len(pdb_files)}")
    print(f"[OK] features -> {features_path}")
    print(f"[OK] merged -> {merged_path}")
    print(f"[OK] method summary -> {summary_path}")
    print(f"[OK] correlations -> {corr_path}")


if __name__ == "__main__":
    main()
