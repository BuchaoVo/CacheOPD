from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding


AA = set("ACDEFGHIKLMNPQRSTVWY")


def safe_name(text: object) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def clean_sequence(seq: object) -> str:
    return "".join(str(seq).split()).upper()


def normalize_method_name(name: object) -> str:
    name = str(name)
    aliases = {
        "CacheOPD Full": "CacheOPD-Full",
        "CacheOPD Sparse": "CacheOPD-Sparse",
        "Online OPD": "Online OPD",
        "ProteinOPD": "Online OPD",
        "ProLLaMA": "ProLLaMA",
    }
    return aliases.get(name, name)


def normalize_metrics_table(path: Path, default_setting: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise ValueError(f"{path} does not contain a sequence column")
    if "method" not in df.columns:
        raise ValueError(f"{path} does not contain a method column")

    if "seq_id" not in df.columns:
        if "sequence_id" in df.columns:
            df["seq_id"] = df["sequence_id"]
        elif "sample_id" in df.columns:
            df["seq_id"] = df["method"].astype(str) + "_" + df["sample_id"].astype(str).str.zfill(5)
        else:
            df["seq_id"] = [f"{Path(path).stem}_{i:05d}" for i in range(len(df))]

    if "setting" not in df.columns:
        df["setting"] = default_setting

    df["sequence"] = df["sequence"].map(clean_sequence)
    df["method"] = df["method"].map(normalize_method_name)
    df["length"] = df["sequence"].str.len()
    df = df[df["sequence"].map(lambda s: len(s) > 0 and set(s).issubset(AA))].copy()
    return df


def select_rows(df: pd.DataFrame, methods: list[str], max_per_method: int, max_len: int, strategy: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df[df["length"] <= max_len].copy()
    if methods:
        wanted = {normalize_method_name(m) for m in methods}
        df = df[df["method"].isin(wanted)].copy()

    rows = []
    for method, sub in df.groupby("method", sort=False):
        sub = sub.copy()
        if max_per_method <= 0 or len(sub) <= max_per_method:
            rows.append(sub)
            continue

        if strategy == "top_plddt" and "plddt" in sub.columns:
            rows.append(sub.sort_values("plddt", ascending=False).head(max_per_method))
        elif strategy == "stratified_sol" and "sol" in sub.columns and sub["sol"].notna().sum() >= 3:
            sub["_sol_rank"] = pd.to_numeric(sub["sol"], errors="coerce").rank(method="first", pct=True)
            bins = pd.cut(sub["_sol_rank"], bins=[0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"], include_lowest=True)
            sub["_bin"] = bins.astype(str)
            take_parts = []
            per_bin = max(1, max_per_method // 3)
            for _, bin_df in sub.groupby("_bin", sort=False):
                take = min(per_bin, len(bin_df))
                take_parts.append(bin_df.sample(n=take, random_state=int(rng.integers(1_000_000))))
            picked = pd.concat(take_parts, ignore_index=False)
            if len(picked) < max_per_method:
                rest = sub.drop(index=picked.index, errors="ignore")
                if len(rest):
                    picked = pd.concat(
                        [
                            picked,
                            rest.sample(
                                n=min(max_per_method - len(picked), len(rest)),
                                random_state=int(rng.integers(1_000_000)),
                            ),
                        ],
                        ignore_index=False,
                    )
            rows.append(picked.head(max_per_method).drop(columns=["_sol_rank", "_bin"], errors="ignore"))
        else:
            rows.append(sub.sample(n=max_per_method, random_state=int(rng.integers(1_000_000))))

    if not rows:
        raise RuntimeError("No valid sequences selected for folding.")
    out = pd.concat(rows, ignore_index=True)
    out["case_id"] = [
        f"{safe_name(r.setting)}__{safe_name(r.method)}__{safe_name(r.seq_id)}"
        for r in out.itertuples(index=False)
    ]
    return out


def patch_hf_compute_tm() -> None:
    import transformers.models.esm.modeling_esmfold as modeling_esmfold
    import transformers.models.esm.openfold_utils.loss as loss_mod

    if getattr(modeling_esmfold.compute_tm, "_cacheopd_safe_patch", False):
        return
    original_compute_tm = modeling_esmfold.compute_tm

    def safe_compute_tm(*args, **kwargs):
        try:
            value = original_compute_tm(*args, **kwargs)
            if torch.is_tensor(value) and value.numel() == 0:
                return args[0].new_tensor(float("nan"))
            return value
        except IndexError:
            return args[0].new_tensor(float("nan"))

    safe_compute_tm._cacheopd_safe_patch = True
    modeling_esmfold.compute_tm = safe_compute_tm
    loss_mod.compute_tm = safe_compute_tm


def load_esmfold(model_path: Path, device: torch.device, chunk_size: int):
    patch_hf_compute_tm()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = EsmForProteinFolding.from_pretrained(
        str(model_path),
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"):
        model.trunk.set_chunk_size(chunk_size)
    model.eval().to(device)
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold selected CacheOPD/ProteinOPD sequences into PDB files with local HF ESMFold.")
    parser.add_argument("--metrics_csv", action="append", required=True, help="Per-sequence metric CSV. Can be repeated.")
    parser.add_argument("--out_dir", default="analysis_outputs/proteinopd_reproduction/sol3d_diagnostics/pdbs")
    parser.add_argument("--selected_csv", default="analysis_outputs/proteinopd_reproduction/sol3d_diagnostics/sol3d_selected_sequences.csv")
    parser.add_argument("--model_path", default="/home/zbc/data/models/esmfold_v1_hf")
    parser.add_argument("--method", action="append", default=[], help="Method to include. Can be repeated.")
    parser.add_argument("--default_setting", default="conditional")
    parser.add_argument("--max_per_method", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--strategy", choices=["random", "top_plddt", "stratified_sol"], default="stratified_sol")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tables = [normalize_metrics_table(Path(p), args.default_setting) for p in args.metrics_csv]
    df = pd.concat(tables, ignore_index=True)
    selected = select_rows(df, args.method, args.max_per_method, args.max_len, args.strategy, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = Path(args.selected_csv)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected["pdb_file"] = [str(out_dir / f"{case_id}.pdb") for case_id in selected["case_id"]]
    selected.to_csv(selected_path, index=False)
    print(f"[OK] selected {len(selected)} sequences -> {selected_path}")

    if args.dry_run:
        print("[DRY-RUN] Folding skipped.")
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] device={device}")
    _, model = load_esmfold(Path(args.model_path), device=device, chunk_size=args.chunk_size)

    status_rows = []
    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="ESMFold PDB"):
        pdb_path = Path(row.pdb_file)
        status = {"case_id": row.case_id, "method": row.method, "seq_id": row.seq_id, "pdb_file": str(pdb_path), "status": "ok", "error": ""}
        if pdb_path.exists() and not args.force:
            status_rows.append(status)
            continue
        try:
            pdb = model.infer_pdb(str(row.sequence))
            pdb_path.write_text(pdb, encoding="utf-8")
        except RuntimeError as exc:
            status["status"] = "runtime_error"
            status["error"] = repr(exc)
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            status["status"] = "error"
            status["error"] = repr(exc)
        status_rows.append(status)
        pd.DataFrame(status_rows).to_csv(out_dir.parent / "sol3d_pdb_export_status.csv", index=False)

    print(f"[OK] PDB directory -> {out_dir}")
    print(f"[OK] status -> {out_dir.parent / 'sol3d_pdb_export_status.csv'}")


if __name__ == "__main__":
    main()
