from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_ROOT, ROOT, ensure_dirs, write_status


TEACHERS = {
    "fold": {
        "method": "Fold-teacher KD",
        "prefix": "foldability",
        "expected_primary_metric": "pLDDT/pAE/pTM",
    },
    "sol": {
        "method": "Sol-teacher KD",
        "prefix": "solubility",
        "expected_primary_metric": "Protein-Sol",
    },
    "thermo": {
        "method": "Thermo-teacher KD",
        "prefix": "thermostability",
        "expected_primary_metric": "TemBERTure",
    },
}

COMMON_FILES = [
    "rollouts_used.csv",
    "reference_lengths.npy",
    "reference_meta.json",
    "reference_target_ids.npy",
    "reference_topk_ids.npy",
    "reference_topk_logprobs.npy",
    "reference_valid_mask.npy",
]


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        rel = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown materialization mode: {mode}")


def load_shape(path: Path) -> tuple[int, ...]:
    return tuple(np.load(path, mmap_mode="r").shape)


def count_effective_tokens(cache_dir: Path) -> int:
    mask = np.load(cache_dir / "reference_valid_mask.npy", mmap_mode="r")
    return int(mask.sum())


def build_one(source: Path, dest_root: Path, teacher_key: str, mode: str) -> dict:
    info = TEACHERS[teacher_key]
    prefix = info["prefix"]
    dest = dest_root / teacher_key
    dest.mkdir(parents=True, exist_ok=True)

    for name in COMMON_FILES:
        link_or_copy(source / name, dest / name, mode)

    teacher_files = [
        f"{prefix}_lengths.npy",
        f"{prefix}_meta.json",
        f"{prefix}_target_ids.npy",
        f"{prefix}_topk_ids.npy",
        f"{prefix}_topk_logprobs.npy",
        f"{prefix}_valid_mask.npy",
    ]
    for name in teacher_files:
        link_or_copy(source / name, dest / name, mode)

    link_or_copy(source / f"{prefix}_topk_ids.npy", dest / "poe_topk_ids.npy", mode)
    link_or_copy(source / f"{prefix}_topk_logprobs.npy", dest / "poe_topk_logprobs.npy", mode)

    manifest = {
        "cache_type": "single_teacher_closed_support",
        "teacher_key": teacher_key,
        "method": info["method"],
        "source_cache": str(source),
        "cache_dir": str(dest),
        "teacher_target_ids": f"{prefix}_topk_ids.npy",
        "teacher_target_logprobs": f"{prefix}_topk_logprobs.npy",
        "poe_topk_ids_alias": f"{prefix}_topk_ids.npy",
        "poe_topk_logprobs_alias": f"{prefix}_topk_logprobs.npy",
        "expected_primary_metric": info["expected_primary_metric"],
        "effective_reference_tokens": count_effective_tokens(source),
        "target_shape": load_shape(source / f"{prefix}_topk_ids.npy"),
        "reference_mask_shape": load_shape(source / "reference_valid_mask.npy"),
        "materialization": mode,
    }
    (dest / "single_teacher_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_status(
        info["method"].replace(" ", "_"),
        "cache_prepared",
        "Single-teacher cache prepared; run training and generation commands to obtain a matched cohort.",
        OUT_ROOT / "status",
        cache_dir=str(dest),
        teacher_key=teacher_key,
        effective_reference_tokens=manifest["effective_reference_tokens"],
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Build fair single-teacher offline KD caches from cached teacher top-k targets.")
    ap.add_argument(
        "--source_cache",
        default=str(ROOT / "analysis_outputs/offline_proteinopd/cache_diag/base_rollout_k64"),
        help="Cache containing reference arrays and fold/sol/thermo teacher top-k arrays.",
    )
    ap.add_argument(
        "--dest_root",
        default=str(OUT_ROOT / "single_teacher_caches"),
        help="Output directory for single-teacher cache variants.",
    )
    ap.add_argument("--materialization", choices=["hardlink", "symlink", "copy"], default="hardlink")
    args = ap.parse_args()

    ensure_dirs(OUT_ROOT)
    source = Path(args.source_cache)
    dest_root = Path(args.dest_root)
    if not source.exists():
        raise FileNotFoundError(f"Source cache not found: {source}")

    rows = [build_one(source, dest_root, key, args.materialization) for key in TEACHERS]
    out_csv = dest_root / "single_teacher_cache_manifest.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] single-teacher caches -> {dest_root}")
    print(f"[OK] manifest -> {out_csv}")


if __name__ == "__main__":
    main()
