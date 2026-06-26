import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(seq):
    return "".join(str(seq).split()).upper()


def is_valid_aa(seq):
    return len(seq) > 0 and set(seq).issubset(AA)


def tensor_to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (float, int)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().float().mean().cpu().item())
    try:
        return float(np.asarray(x).mean())
    except Exception:
        return np.nan


def extract_structure_scores(output):
    """
    Robust extractor for ESMFold output dict.
    Different fair-esm/openfold versions expose slightly different keys.
    """
    keys = list(output.keys()) if isinstance(output, dict) else []

    # pLDDT
    plddt = np.nan
    for key in ["mean_plddt", "plddt"]:
        if key in output:
            x = output[key]
            if key == "plddt":
                plddt = tensor_to_float(x)
            else:
                plddt = tensor_to_float(x)
            break

    # pAE
    pae = np.nan
    for key in ["predicted_aligned_error", "pae", "aligned_error"]:
        if key in output:
            pae = tensor_to_float(output[key])
            break

    # ptm optional
    ptm = np.nan
    for key in ["ptm", "mean_ptm"]:
        if key in output:
            ptm = tensor_to_float(output[key])
            break

    return {
        "plddt": plddt,
        "pae": pae,
        "ptm": ptm,
        "output_keys": ",".join(keys),
    }


def load_model(chunk_size, cpu_offload=False):
    import esm

    model = esm.pretrained.esmfold_v1()
    model = model.eval()

    if hasattr(model, "set_chunk_size") and chunk_size is not None and chunk_size > 0:
        model.set_chunk_size(chunk_size)

    if torch.cuda.is_available() and not cpu_offload:
        model = model.cuda()

    return model


def score_sequence(model, seq, num_recycles):
    t0 = time.time()

    with torch.no_grad():
        try:
            output = model.infer([seq], num_recycles=num_recycles)
        except TypeError:
            output = model.infer([seq])

    scores = extract_structure_scores(output)
    scores["runtime_sec"] = time.time() - t0
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_per_method", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--num_recycles", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cohort_dir = Path(args.cohort_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_files = sorted([
        p for p in cohort_dir.glob("*.csv")
        if p.name != "p0_cohort_manifest.csv"
    ])

    if not cohort_files:
        raise FileNotFoundError(f"No cohort CSV files found in {cohort_dir}")

    print("[INFO] loading ESMFold model...")
    model = load_model(chunk_size=args.chunk_size, cpu_offload=args.cpu_offload)
    print("[INFO] model loaded")
    print("[INFO] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] cuda device:", torch.cuda.get_device_name(0))

    all_summary = []

    for cohort_file in cohort_files:
        df = pd.read_csv(cohort_file)
        if "method" not in df.columns:
            df["method"] = cohort_file.stem
        if "seq_id" not in df.columns:
            df["seq_id"] = [f"{cohort_file.stem}_{i:05d}" for i in range(len(df))]
        if "sequence" not in df.columns:
            raise ValueError(f"No sequence column in {cohort_file}")

        method = str(df["method"].iloc[0])
        out_csv = out_dir / f"{method}_esmfold_structure_scores.csv"

        df["sequence"] = df["sequence"].apply(clean_seq)
        df = df[df["sequence"].apply(is_valid_aa)].copy()
        df = df[df["sequence"].str.len() <= args.max_len].copy()

        if args.max_per_method and args.max_per_method > 0:
            df = df.head(args.max_per_method).copy()

        done_ids = set()
        existing_rows = []

        if out_csv.exists() and not args.force:
            old = pd.read_csv(out_csv)
            if "seq_id" in old.columns:
                done_ids = set(old["seq_id"].astype(str))
                existing_rows = old.to_dict("records")
                print(f"[RESUME] {method}: loaded {len(done_ids)} scored rows from {out_csv}")

        rows = existing_rows[:]

        todo = df[~df["seq_id"].astype(str).isin(done_ids)].copy()
        print(f"\n[RUN] {method}: total={len(df)}, todo={len(todo)}, output={out_csv}")

        for _, r in tqdm(todo.iterrows(), total=len(todo), desc=f"ESMFold {method}"):
            seq_id = str(r["seq_id"])
            seq = str(r["sequence"])

            row = {
                "method": method,
                "seq_id": seq_id,
                "sequence": seq,
                "length": len(seq),
                "status": "ok",
                "error": "",
                "plddt": np.nan,
                "pae": np.nan,
                "ptm": np.nan,
                "runtime_sec": np.nan,
                "output_keys": "",
            }

            try:
                scores = score_sequence(model, seq, args.num_recycles)
                row.update(scores)
            except RuntimeError as e:
                row["status"] = "runtime_error"
                row["error"] = repr(e)
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
            except Exception as e:
                row["status"] = "error"
                row["error"] = repr(e)

            rows.append(row)

            # incremental save
            pd.DataFrame(rows).to_csv(out_csv, index=False)

        scored = pd.DataFrame(rows)
        ok = scored[scored["status"] == "ok"].copy()

        summary = {
            "method": method,
            "n_total": len(df),
            "n_scored_ok": len(ok),
            "n_failed": int((scored["status"] != "ok").sum()),
            "mean_length": ok["length"].mean() if len(ok) else np.nan,
            "plddt_mean": ok["plddt"].mean() if len(ok) else np.nan,
            "plddt_median": ok["plddt"].median() if len(ok) else np.nan,
            "pct_plddt_gt70": 100.0 * (ok["plddt"] > 70).mean() if len(ok) else np.nan,
            "pae_mean": ok["pae"].mean() if len(ok) else np.nan,
            "pae_median": ok["pae"].median() if len(ok) else np.nan,
            "pct_pae_lt10": 100.0 * (ok["pae"] < 10).mean() if len(ok) and ok["pae"].notna().any() else np.nan,
            "ptm_mean": ok["ptm"].mean() if len(ok) else np.nan,
            "runtime_sec_mean": ok["runtime_sec"].mean() if len(ok) else np.nan,
        }

        all_summary.append(summary)

        print("[SUMMARY]", json.dumps(summary, indent=2, ensure_ascii=False))

    summary_df = pd.DataFrame(all_summary)
    summary_path = out_dir / "p0_esmfold_structure_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
