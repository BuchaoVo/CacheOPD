from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import OUT_ROOT, ROOT, ensure_dirs, write_status


INTERNAL_SOURCES = {
    "ProLLaMA": ROOT / "analysis_outputs/p0_current_eval/cohorts/Base-ProLLaMA.csv",
    "SFT": ROOT / "analysis_outputs/offline_proteinopd/eval_cohorts_main_table/SFT-RefRollout.csv",
    "ProbAvg-KD": ROOT / "analysis_outputs/offline_proteinopd/eval_cohorts_experiment3/Exp3-Fusion-ProbAvg.csv",
    "LogitAvg-KD": ROOT / "analysis_outputs/offline_proteinopd/eval_cohorts_experiment3/Exp3-Fusion-LogitAvg.csv",
    "Online OPD": ROOT / "analysis_outputs/p0_current_eval/cohorts/Conditional-OPD.csv",
    "CacheOPD Full": ROOT / "analysis_outputs/offline_proteinopd/eval_cohorts_p0/Offline-OPDRollout-K64.csv",
    "CacheOPD Sparse": ROOT / "analysis_outputs/offline_proteinopd/eval_cohorts_final_validation_qdelta_smoothing/FinalUtility-ShiftEntropy-q04-High50.csv",
}

SINGLE_TEACHER_SOURCES = {
    "Fold-teacher KD": OUT_ROOT / "generated_cohorts_single_teacher/Fold-teacher KD.csv",
    "Sol-teacher KD": OUT_ROOT / "generated_cohorts_single_teacher/Sol-teacher KD.csv",
    "Thermo-teacher KD": OUT_ROOT / "generated_cohorts_single_teacher/Thermo-teacher KD.csv",
}

SINGLE_TEACHER_META = {
    "Fold-teacher KD": {
        "cache": OUT_ROOT / "single_teacher_caches/fold",
        "adapter": ROOT / "outputs/offline_proteinopd/conditional_supplement_single_teacher/single_teacher_fold_kd",
    },
    "Sol-teacher KD": {
        "cache": OUT_ROOT / "single_teacher_caches/sol",
        "adapter": ROOT / "outputs/offline_proteinopd/conditional_supplement_single_teacher/single_teacher_sol_kd",
    },
    "Thermo-teacher KD": {
        "cache": OUT_ROOT / "single_teacher_caches/thermo",
        "adapter": ROOT / "outputs/offline_proteinopd/conditional_supplement_single_teacher/single_teacher_thermo_kd",
    },
}


def normalize_internal(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df.copy()
    if "seq_id" in out.columns and "sequence_id" not in out.columns:
        out = out.rename(columns={"seq_id": "sequence_id"})
    if "superfamily_file" in out.columns and "condition_id" not in out.columns:
        out = out.rename(columns={"superfamily_file": "condition_id"})
    if "condition" in out.columns and "condition_id" not in out.columns:
        out = out.rename(columns={"condition": "condition_id"})
    out["method"] = method
    if "sequence_id" in out.columns and "seq_id" not in out.columns:
        out["seq_id"] = out["sequence_id"]
    if "sample_id" not in out.columns:
        out["sample_id"] = out.groupby("condition_id").cumcount() if "condition_id" in out.columns else range(len(out))
    keep = [c for c in ["method", "condition_id", "sample_id", "sequence_id", "seq_id", "sequence", "length", "source_json"] if c in out.columns]
    return out[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--samples_per_condition", type=int, default=32)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)
    rows = []
    status_rows = []
    for method, path in INTERNAL_SOURCES.items():
        if not path.exists():
            status_rows.append({"method": method, "status": "missing", "path": str(path)})
            write_status(method.replace(" ", "_"), "missing", f"Internal cohort not found: {path}", out_root / "status")
            continue
        df = pd.read_csv(path)
        norm = normalize_internal(df, method)
        if "condition_id" in norm.columns:
            norm = norm.groupby("condition_id", group_keys=False).head(args.samples_per_condition).copy()
        norm.to_csv(out_root / "generations" / "internal" / f"{method.replace(' ', '_')}.csv", index=False)
        rows.append(norm)
        status_rows.append({"method": method, "status": "ok", "path": str(path), "n": len(norm)})
        write_status(method.replace(" ", "_"), "ok", "Matched internal cohort copied into conditional supplement generations.", out_root / "status", source=str(path), n=int(len(norm)))

    for method, path in SINGLE_TEACHER_SOURCES.items():
        if path.exists():
            df = pd.read_csv(path)
            norm = normalize_internal(df, method)
            if "condition_id" in norm.columns:
                norm = norm.groupby("condition_id", group_keys=False).head(args.samples_per_condition).copy()
            norm.to_csv(out_root / "generations" / "internal" / f"{method.replace(' ', '_')}.csv", index=False)
            rows.append(norm)
            status_rows.append({"method": method, "status": "ok", "path": str(path), "n": len(norm)})
            write_status(method.replace(" ", "_"), "ok", "Matched single-teacher cohort copied into conditional supplement generations.", out_root / "status", source=str(path), n=int(len(norm)))
            continue

        meta = SINGLE_TEACHER_META[method]
        cache_ready = meta["cache"].exists() and (meta["cache"] / "single_teacher_cache_manifest.json").exists()
        adapter_ready = meta["adapter"].exists() and (meta["adapter"] / "adapter_config.json").exists()
        if adapter_ready:
            status = "adapter_exists_no_cohort"
            reason = "Single-teacher adapter exists, but matched generated cohort is not prepared. Run generate_single_teacher_all.sh and postprocess_single_teacher.sh."
        elif cache_ready:
            status = "cache_prepared_no_adapter"
            reason = "Single-teacher cache exists, but the adapter has not been trained yet. Run train_single_teacher_all.sh or train_single_teacher_parallel.sh."
        else:
            status = "missing"
            reason = "No matched generated cohort or prepared single-teacher cache was found. Run build_single_teacher_caches.py first."
        status_rows.append({"method": method, "status": status, "path": str(path), "reason": reason})
        write_status(
            method.replace(" ", "_"),
            status,
            reason,
            out_root / "status",
            expected_cohort=str(path),
            cache_dir=str(meta["cache"]),
            adapter_dir=str(meta["adapter"]),
        )

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_root / "generations" / "internal" / "internal_matched_generations.csv", index=False)
    pd.DataFrame(status_rows).to_csv(out_root / "status" / "internal_generation_status.csv", index=False)
    print(f"[OK] internal status -> {out_root / 'status' / 'internal_generation_status.csv'}")


if __name__ == "__main__":
    main()
