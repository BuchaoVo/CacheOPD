import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(seq):
    return "".join(str(seq).split()).upper()

def is_valid_aa(seq):
    return len(seq) > 0 and set(seq).issubset(AA)

def parse_temberture_output(x):
    """
    Official README example:
    model.predict(seq) -> ['Thermophilic', 0.999...]
    """
    label = ""
    score = np.nan

    if isinstance(x, (list, tuple)):
        if len(x) >= 1:
            label = str(x[0])
        if len(x) >= 2:
            try:
                score = float(x[1])
            except Exception:
                score = np.nan
    elif isinstance(x, dict):
        for k in ["label", "class", "thermal_class"]:
            if k in x:
                label = str(x[k])
                break
        for k in ["score", "thermophilicity_score", "probability", "prob"]:
            if k in x:
                try:
                    score = float(x[k])
                except Exception:
                    pass
                break
    else:
        label = str(x)

    return label, score

def is_thermophilic_label(label):
    label = str(label).lower()
    if "non" in label:
        return False
    return "thermophilic" in label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--temberture_root", default="/home/zbc/data/models/TemBERTure")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_per_method", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.temberture_root)
    sys.path.insert(0, str(Path(args.temberture_root) / "temBERTure"))

    from temBERTure import TemBERTure

    cohort_dir = Path(args.cohort_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if adapter_path.name == "AdapterBERT_adapter":
            adapter_path = adapter_path.parent
    else:
        root = Path(args.temberture_root)
        adapter_dirs = [p for p in root.rglob("AdapterBERT_adapter") if "CLS" in str(p) or "temBERTure_CLS" in str(p)]
        if not adapter_dirs:
            adapter_dirs = list(root.rglob("AdapterBERT_adapter"))
        if not adapter_dirs:
            raise FileNotFoundError(
                "Cannot find local AdapterBERT_adapter under TemBERTure root. "
                "The repository may not have been cloned with its adapter weights. "
                "Run `find /home/zbc/data/models/TemBERTure -type d -name AdapterBERT_adapter -print`."
            )
        adapter_path = adapter_dirs[0].parent

    # TemBERTure internally does: adapter_path + 'AdapterBERT_adapter'
    # so adapter_path must end with a slash.
    adapter_path_str = str(adapter_path.resolve()) + "/"

    expected_adapter = Path(adapter_path_str) / "AdapterBERT_adapter"
    if not expected_adapter.exists():
        raise FileNotFoundError(
            f"Expected local adapter directory not found: {expected_adapter}. "
            f"Current adapter_path_str={adapter_path_str}"
        )

    print("[INFO] TemBERTure root:", args.temberture_root, flush=True)
    print("[INFO] CLS adapter prefix:", adapter_path_str, flush=True)
    print("[INFO] expected adapter dir:", expected_adapter, flush=True)
    print("[INFO] device:", args.device, flush=True)

    model = TemBERTure(
        adapter_path=adapter_path_str,
        device=args.device,
        batch_size=args.batch_size,
        task="classification",
    )

    cohort_files = sorted([
        p for p in cohort_dir.glob("*.csv")
        if "manifest" not in p.stem.lower() and "summary" not in p.stem.lower()
    ])
    if not cohort_files:
        raise FileNotFoundError(f"No cohort CSV files found in {cohort_dir}")

    summaries = []

    for fp in cohort_files:
        df = pd.read_csv(fp)
        if "method" not in df.columns:
            df["method"] = fp.stem
        if "seq_id" not in df.columns:
            df["seq_id"] = [f"{fp.stem}_{i:05d}" for i in range(len(df))]
        if "sequence" not in df.columns:
            raise ValueError(f"No sequence column in {fp}")

        method = str(df["method"].iloc[0])
        out_csv = out_dir / f"{method}_temberture_scores.csv"

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

        for _, r in tqdm(todo.iterrows(), total=len(todo), desc=f"TemBERTure {method}"):
            seq_id = str(r["seq_id"])
            seq = str(r["sequence"])
            row = {
                "method": method,
                "seq_id": seq_id,
                "sequence": seq,
                "length": len(seq),
                "status": "ok",
                "error": "",
                "thermo_label": "",
                "thermo_score": np.nan,
                "runtime_sec": np.nan,
            }

            t0 = time.time()
            try:
                pred = model.predict(seq)
                label, score = parse_temberture_output(pred)
                row["thermo_label"] = label
                row["thermo_score"] = score
                row["runtime_sec"] = time.time() - t0
            except Exception as e:
                row["status"] = "error"
                row["error"] = repr(e)
                row["runtime_sec"] = time.time() - t0

            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

        scored = pd.DataFrame(rows)
        ok = scored[scored["status"] == "ok"].copy()

        summary = {
            "method": method,
            "n_total": len(df),
            "n_scored_ok": len(ok),
            "n_failed": int((scored["status"] != "ok").sum()) if len(scored) else 0,
            "thermo_score_mean": ok["thermo_score"].mean() if len(ok) else np.nan,
            "thermo_score_median": ok["thermo_score"].median() if len(ok) else np.nan,
            "thermophilic_rate": float(ok["thermo_label"].apply(is_thermophilic_label).mean()) if len(ok) else np.nan,
            "runtime_sec_mean": ok["runtime_sec"].mean() if len(ok) else np.nan,
        }

        summaries.append(summary)
        print(pd.DataFrame([summary]).to_string(index=False), flush=True)

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "p0_temberture_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n[OK] saved summary:", summary_path, flush=True)

if __name__ == "__main__":
    main()
