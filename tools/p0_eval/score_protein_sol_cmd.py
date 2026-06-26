import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(seq):
    return "".join(str(seq).split()).upper()

def is_valid_aa(seq):
    return len(seq) > 0 and set(seq).issubset(AA)

def write_fasta(path, seq_id, seq):
    with open(path, "w") as f:
        f.write(f">{seq_id}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "sequence"

def parse_sol_score(text):
    """
    Strict Protein-Sol parser.

    Protein-Sol wrapper emits lines such as:
    SEQUENCE PREDICTIONS,>seq_id,76.002, 0.655, 0.446, 5.080

    We use the second numeric field as sol_score because it is the
    scaled solubility score in the 0-1 range. The first numeric value
    is kept out of sol_score to avoid mixing scales.
    """
    import re
    import csv
    import io
    import numpy as np

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        low = line.lower()

        if low.startswith("legend"):
            continue
        if "features used for prediction are scaled" in low:
            continue
        if any(x in low for x in ["cp:", "mv:", "rm:", "cannot open", "no such file", "try '"]):
            continue

        if line.startswith("SEQUENCE PREDICTIONS"):
            try:
                parts = next(csv.reader(io.StringIO(line)))
            except Exception:
                parts = line.split(",")

            nums = []
            for x in parts[2:]:
                try:
                    nums.append(float(str(x).strip()))
                except Exception:
                    pass

            if len(nums) >= 2:
                # nums[0] is the raw Protein-Sol value; nums[1] is scaled solubility.
                return nums[1], line
            elif len(nums) == 1:
                return nums[0], line

    # Fallback: CSV-like header/data parsing
    for i, line in enumerate(lines[:-1]):
        low = line.lower()
        if low.startswith("legend"):
            continue
        if not any(k in low for k in ["solub", "protein-sol", "protein sol", "scaled", "prediction", "score"]):
            continue

        try:
            header = next(csv.reader(io.StringIO(line)))
        except Exception:
            header = re.split(r"[\t,]+|\s{2,}", line)

        header_norm = [h.strip().lower().replace(" ", "_").replace("-", "_") for h in header]

        candidate_idx = []
        for j, h in enumerate(header_norm):
            if (
                "solub" in h
                or "protein_sol" in h
                or h in ["score", "prediction", "scaled_score", "scaled_solubility"]
                or ("scaled" in h and "score" in h)
            ):
                candidate_idx.append(j)

        if not candidate_idx:
            continue

        for data_line in lines[i + 1:i + 6]:
            if data_line.lower().startswith("legend"):
                continue

            try:
                parts = next(csv.reader(io.StringIO(data_line)))
            except Exception:
                parts = re.split(r"[\t,]+|\s{2,}", data_line)

            for j in candidate_idx:
                if j < len(parts):
                    try:
                        return float(parts[j]), f"HEADER={line} | DATA={data_line}"
                    except Exception:
                        pass

    return np.nan, ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cmd_template", required=True)
    parser.add_argument("--max_per_method", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
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
        out_csv = out_dir / f"{method}_protein_sol_scores.csv"

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

        for _, r in tqdm(todo.iterrows(), total=len(todo), desc=f"Protein-Sol {method}"):
            seq_id = str(r["seq_id"])
            seq = str(r["sequence"])

            row = {
                "method": method,
                "seq_id": seq_id,
                "sequence": seq,
                "length": len(seq),
                "status": "ok",
                "error": "",
                "sol_score": np.nan,
                "parsed_line": "",
                "stdout_tail": "",
                "stderr_tail": "",
            }

            try:
                with tempfile.TemporaryDirectory() as td:
                    fasta = Path(td) / f"{safe_filename(seq_id)}.fasta"
                    write_fasta(fasta, seq_id, seq)

                    cmd = args.cmd_template.format(fasta=str(fasta), seq_id=seq_id)
                    p = subprocess.run(
                        cmd,
                        shell=True,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=args.timeout,
                    )

                    text = p.stdout + "\n" + p.stderr
                    row["stdout_tail"] = p.stdout[-1000:]
                    row["stderr_tail"] = p.stderr[-1000:]

                    if p.returncode != 0:
                        row["status"] = "returncode_nonzero"
                        row["error"] = f"returncode={p.returncode}"

                    score, src_line = parse_sol_score(text)
                    row["sol_score"] = score
                    row["parsed_line"] = src_line

                    if np.isnan(score) and row["status"] == "ok":
                        row["status"] = "parse_failed"
                        row["error"] = "No numeric solubility score parsed."

            except Exception as e:
                row["status"] = "error"
                row["error"] = repr(e)

            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

        scored = pd.DataFrame(rows)
        ok = scored[(scored["status"] == "ok") & scored["sol_score"].notna()].copy()

        summary = {
            "method": method,
            "n_total": len(df),
            "n_scored_ok": len(ok),
            "n_failed": int((scored["status"] != "ok").sum()) if len(scored) else 0,
            "sol_score_mean": ok["sol_score"].mean() if len(ok) else np.nan,
            "sol_score_median": ok["sol_score"].median() if len(ok) else np.nan,
            "sol_score_min": ok["sol_score"].min() if len(ok) else np.nan,
            "sol_score_max": ok["sol_score"].max() if len(ok) else np.nan,
        }

        summaries.append(summary)
        print(pd.DataFrame([summary]).to_string(index=False), flush=True)

    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "p0_protein_sol_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n[OK] saved summary:", summary_path, flush=True)

if __name__ == "__main__":
    main()
