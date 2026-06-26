import json
import re
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/data/users/zbc/opd/ProteinOPD")
GEN_ROOT = ROOT / "outputs/conditional/offline_opd_4superfamily"
OUT_DIR = ROOT / "analysis_outputs/offline_proteinopd/eval_cohorts"

AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 50
MAX_LEN = 1024

METHOD_MAP = {
    "offline_baserollout_k64": "Offline-BaseRollout-K64",
    "offline_opdrollout_k64": "Offline-OPDRollout-K64",
    "contrib2_high50": "Contrib2-High50",
    "contrib2_random50": "Contrib2-Random50",
    "contrib2_low50": "Contrib2-Low50",
    "contrib2_base_rollout_k64_keep50_high_gamma01": "Contrib2-High50",
    "contrib2_base_rollout_k64_keep50_random_gamma01": "Contrib2-Random50",
    "contrib2_base_rollout_k64_keep50_low_gamma01": "Contrib2-Low50",
    "contrib3_conflictup": "Contrib3-ConflictUp",
    "contrib3_conflictdown": "Contrib3-ConflictDown",
    "contrib3_base_rollout_k64_conflict_up_gamma01": "Contrib3-ConflictUp",
    "contrib3_base_rollout_k64_conflict_down_gamma01": "Contrib3-ConflictDown",
    "exp3_fusion_probavg": "Exp3-Fusion-ProbAvg",
    "exp3_fusion_logitavg": "Exp3-Fusion-LogitAvg",
    "exp3_fusion_prob_avg_gamma01": "Exp3-Fusion-ProbAvg",
    "exp3_fusion_logit_avg_gamma01": "Exp3-Fusion-LogitAvg",
    "anchor_gamma0": "Exp6-Anchor-Gamma0",
    "anchor_gamma1": "Exp6-Anchor-Gamma1",
    "anchor_ablation_gamma0": "Exp6-Anchor-Gamma0",
    "anchor_ablation_gamma1": "Exp6-Anchor-Gamma1",
    "final_utility_shift_entropy_q02_high50": "FinalUtility-ShiftEntropy-q02-High50",
    "final_utility_shift_entropy_q02_random50": "FinalUtility-ShiftEntropy-q02-Random50",
    "final_utility_shift_entropy_q02_low50": "FinalUtility-ShiftEntropy-q02-Low50",
    "final_utility_shift_only_q02_high50": "FinalUtility-ShiftOnly-q02-High50",
    "final_utility_shift_entropy_q0_high50": "FinalUtility-ShiftEntropy-q0-High50",
    "final_utility_shift_entropy_q04_high50": "FinalUtility-ShiftEntropy-q04-High50",
    "smoothing_conflict_alpha05": "Smoothing-ConflictAlpha05",
    "cacheopd_sparse_kappa25_high": "CacheOPD-Sparse-k25",
    "cacheopd_sparse_kappa75_high": "CacheOPD-Sparse-k75",
    "sft_ref_rollout": "SFT-RefRollout",
    "single_teacher_fold_kd": "Fold-teacher KD",
    "single_teacher_sol_kd": "Sol-teacher KD",
    "single_teacher_thermo_kd": "Thermo-teacher KD",
}

def collect_strings(obj):
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(collect_strings(v))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(collect_strings(x))
    return out

def is_valid_seq(seq):
    return (
        MIN_LEN <= len(seq) <= MAX_LEN
        and set(seq).issubset(AA)
    )

def extract_sequences_from_string(s):
    s = str(s)
    candidates = []

    # Case 1: explicit Seq=<...>
    for m in re.finditer(r"Seq=<([^>]*)>", s, flags=re.IGNORECASE):
        seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", m.group(1).upper())
        if is_valid_seq(seq):
            candidates.append(seq)

    # Case 2: whole string is already a protein sequence
    stripped = s.strip().upper()
    if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", stripped):
        if is_valid_seq(stripped):
            candidates.append(stripped)

    # Case 3: generated text contains long contiguous AA runs
    for m in re.finditer(r"[ACDEFGHIKLMNPQRSTVWY]{%d,}" % MIN_LEN, s.upper()):
        seq = m.group(0)
        if is_valid_seq(seq):
            candidates.append(seq)

    return candidates

def infer_superfamily_from_filename(fp):
    name = fp.stem
    # remove method prefix if present
    for prefix in METHOD_MAP.keys():
        if name.startswith(prefix + "_"):
            return name[len(prefix) + 1:]
    return name

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare generated protein cohorts for offline OPD evaluation.")
    parser.add_argument("--gen_root", default=str(GEN_ROOT), help="Directory containing one subdirectory per method.")
    parser.add_argument("--out_dir", default=str(OUT_DIR), help="Output directory for cohort CSV files.")
    parser.add_argument(
        "--include_method",
        action="append",
        default=None,
        help="Optional method directory name to include. Can be provided multiple times.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gen_root = Path(args.gen_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_methods = set(args.include_method or [])
    manifest = []

    for method_dir in sorted(gen_root.glob("*")):
        if not method_dir.is_dir():
            continue

        method_safe = method_dir.name
        if include_methods and method_safe not in include_methods:
            continue
        method = METHOD_MAP.get(method_safe, method_safe)

        rows = []
        seen = set()

        json_files = sorted(method_dir.glob("*.json"))
        print(f"\n[METHOD] {method}")
        print(f"DIR={method_dir}")
        print(f"json_files={len(json_files)}")

        for fp in json_files:
            try:
                obj = json.loads(fp.read_text())
            except Exception as e:
                print(f"[WARN] failed to read {fp}: {repr(e)}")
                continue

            strings = collect_strings(obj)
            superfamily = infer_superfamily_from_filename(fp)

            file_count = 0
            for s in strings:
                for seq in extract_sequences_from_string(s):
                    if seq in seen:
                        continue
                    seen.add(seq)
                    rows.append({
                        "method": method,
                        "seq_id": f"{method}_{len(rows):05d}",
                        "sequence": seq,
                        "length": len(seq),
                        "source_json": str(fp),
                        "superfamily_file": superfamily,
                    })
                    file_count += 1

            print(f"  {fp.name}: strings={len(strings)}, extracted_unique={file_count}")

        df = pd.DataFrame(rows)
        out_csv = out_dir / f"{method}.csv"
        df.to_csv(out_csv, index=False)

        print(f"[OK] {method}: {len(df)} unique valid sequences -> {out_csv}")
        if len(df):
            print(df[["seq_id", "length", "sequence"]].head(3).to_string(index=False))

        manifest.append({
            "method": method,
            "n_unique_valid": len(df),
            "out_csv": str(out_csv),
            "json_files": len(json_files),
        })

    manifest_df = pd.DataFrame(manifest)
    manifest_path = out_dir / "offline_generated_cohort_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print("\n[MANIFEST]")
    print(manifest_df.to_string(index=False))
    print(f"[OK] saved: {manifest_path}")

if __name__ == "__main__":
    main()
