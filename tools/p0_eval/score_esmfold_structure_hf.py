import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

AA = set("ACDEFGHIKLMNPQRSTVWY")
_PATCHED_COMPUTE_TM = False

def clean_seq(seq):
    return "".join(str(seq).split()).upper()

def is_valid_aa(seq):
    return len(seq) > 0 and set(seq).issubset(AA)

def get_attr_or_key(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return None

def tensor_mean(x):
    if x is None:
        return np.nan
    if torch.is_tensor(x):
        return float(x.detach().float().mean().cpu().item())
    try:
        return float(np.asarray(x).mean())
    except Exception:
        return np.nan

def plddt_summary_series(values):
    vals = pd.to_numeric(values, errors="coerce")
    if vals.notna().any() and vals.max() <= 1.0:
        vals = vals * 100.0
    return vals

def load_model(model_path, device, chunk_size=64, fp16=False):
    patch_hf_compute_tm()
    model_dir = Path(model_path).expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"HF ESMFold local model path does not exist: {model_dir}. "
            "Please download facebook/esmfold_v1 to this directory first."
        )

    model_dir = str(model_dir.resolve())

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    model = EsmForProteinFolding.from_pretrained(
        model_dir,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    if hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"):
        model.trunk.set_chunk_size(chunk_size)

    model.eval()
    model.to(device)

    if fp16 and device.type == "cuda":
        model.half()

    return tokenizer, model


def patch_hf_compute_tm():
    """Avoid HF ESMFold aborting pLDDT/PAE scoring when pTM is undefined.

    transformers 4.57 can raise IndexError inside compute_tm when the weighted
    pTM vector has no finite argmax. We still want per-sequence pLDDT/PAE, so
    only this pTM failure is converted to NaN.
    """
    global _PATCHED_COMPUTE_TM
    if _PATCHED_COMPUTE_TM:
        return
    import transformers.models.esm.modeling_esmfold as modeling_esmfold
    import transformers.models.esm.openfold_utils.loss as loss_mod

    original_compute_tm = modeling_esmfold.compute_tm

    def safe_compute_tm(*args, **kwargs):
        try:
            value = original_compute_tm(*args, **kwargs)
            if torch.is_tensor(value) and value.numel() == 0:
                logits = args[0]
                return logits.new_tensor(float("nan"))
            return value
        except IndexError:
            logits = args[0]
            return logits.new_tensor(float("nan"))

    modeling_esmfold.compute_tm = safe_compute_tm
    loss_mod.compute_tm = safe_compute_tm
    _PATCHED_COMPUTE_TM = True

def score_one(tokenizer, model, seq, device):
    t0 = time.time()

    inputs = tokenizer(
        [seq],
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)

    return {
        "plddt": tensor_mean(get_attr_or_key(outputs, "plddt")),
        "pae": tensor_mean(get_attr_or_key(outputs, "predicted_aligned_error")),
        "ptm": tensor_mean(get_attr_or_key(outputs, "ptm")),
        "runtime_sec": time.time() - t0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_path", default="/home/zbc/data/models/esmfold_v1_hf")
    parser.add_argument("--max_per_method", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cohort_dir = Path(args.cohort_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_files = sorted([
        p for p in cohort_dir.glob("*.csv")
        if "manifest" not in p.stem.lower() and "summary" not in p.stem.lower()
    ])
    if not cohort_files:
        raise FileNotFoundError(f"No cohort CSV files found in {cohort_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device, flush=True)
    if device.type == "cuda":
        print("[INFO] gpu:", torch.cuda.get_device_name(0), flush=True)

    print("[INFO] loading HF ESMFold:", args.model_path, flush=True)
    tokenizer, model = load_model(
        args.model_path,
        device=device,
        chunk_size=args.chunk_size,
        fp16=args.fp16,
    )
    print("[INFO] model loaded", flush=True)

    summaries = []

    for cohort_file in cohort_files:
        df = pd.read_csv(cohort_file)

        if "method" not in df.columns:
            df["method"] = cohort_file.stem
        if "seq_id" not in df.columns:
            df["seq_id"] = [f"{cohort_file.stem}_{i:05d}" for i in range(len(df))]
        if "sequence" not in df.columns:
            raise ValueError(f"No sequence column in {cohort_file}")

        method = str(df["method"].iloc[0])
        out_csv = out_dir / f"{method}_hf_esmfold_structure_scores.csv"

        df["sequence"] = df["sequence"].apply(clean_seq)
        df = df[df["sequence"].apply(is_valid_aa)].copy()
        df = df[df["sequence"].str.len() <= args.max_len].copy()

        if args.max_per_method > 0:
            df = df.head(args.max_per_method).copy()

        rows = []
        done_ids = set()

        if out_csv.exists() and not args.force:
            old = pd.read_csv(out_csv)
            rows = old.to_dict("records")
            done_ids = set(old["seq_id"].astype(str))
            print(f"[RESUME] {method}: {len(done_ids)} existing rows", flush=True)

        todo = df[~df["seq_id"].astype(str).isin(done_ids)].copy()
        print(f"\n[RUN] {method}: total={len(df)}, todo={len(todo)}", flush=True)

        for _, r in tqdm(todo.iterrows(), total=len(todo), desc=f"HF-ESMFold {method}"):
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
            }

            try:
                scores = score_one(tokenizer, model, seq, device)
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
            pd.DataFrame(rows).to_csv(out_csv, index=False)

        scored = pd.DataFrame(rows)
        ok = scored[scored["status"] == "ok"].copy()
        plddt_vals = plddt_summary_series(ok["plddt"]) if len(ok) else pd.Series(dtype=float)

        summary = {
            "method": method,
            "n_total": len(df),
            "n_scored_ok": len(ok),
            "n_failed": int((scored["status"] != "ok").sum()) if len(scored) else 0,
            "mean_length": ok["length"].mean() if len(ok) else np.nan,
            "plddt_mean": plddt_vals.mean() if len(ok) else np.nan,
            "plddt_median": plddt_vals.median() if len(ok) else np.nan,
            "pct_plddt_gt70": 100.0 * (plddt_vals > 70).mean() if len(ok) else np.nan,
            "pae_mean": ok["pae"].mean() if len(ok) else np.nan,
            "pae_median": ok["pae"].median() if len(ok) else np.nan,
            "pct_pae_lt10": 100.0 * (ok["pae"] < 10).mean() if len(ok) and ok["pae"].notna().any() else np.nan,
            "ptm_mean": ok["ptm"].mean() if len(ok) else np.nan,
            "runtime_sec_mean": ok["runtime_sec"].mean() if len(ok) else np.nan,
        }

        summaries.append(summary)
        print(pd.DataFrame([summary]).to_string(index=False), flush=True)

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "p0_hf_esmfold_structure_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n[OK] saved summary:", summary_path, flush=True)

if __name__ == "__main__":
    main()
