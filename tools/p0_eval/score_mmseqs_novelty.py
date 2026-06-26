import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


AA = set("ACDEFGHIKLMNPQRSTVWY")
SEQ_RE = re.compile(r"Seq=<([^>]*)>")


def clean_seq(seq):
    return "".join(str(seq).split()).upper()


def is_valid_aa(seq):
    return len(seq) > 0 and set(seq).issubset(AA)


def extract_json_sequence(x):
    if isinstance(x, dict):
        for key in ["sequence", "seq", "protein_sequence"]:
            if key in x:
                return clean_seq(x[key])
        out = str(x.get("output", ""))
        m = SEQ_RE.search(out)
        if m:
            return clean_seq(m.group(1))
    return ""


def load_reference_sequences(paths):
    refs = []
    for fp in paths:
        p = Path(fp)
        if not p.exists():
            raise FileNotFoundError(f"Missing reference file: {p}")
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("data", data.get("records", []))
            for item in data:
                seq = extract_json_sequence(item)
                if is_valid_aa(seq):
                    refs.append(seq)
        else:
            df = pd.read_csv(p)
            seq_col = None
            for cand in ["sequence", "seq", "protein_sequence"]:
                if cand in df.columns:
                    seq_col = cand
                    break
            if seq_col is None:
                raise ValueError(f"No sequence column in reference CSV: {p}")
            for seq in df[seq_col].dropna().astype(str):
                seq = clean_seq(seq)
                if is_valid_aa(seq):
                    refs.append(seq)
    return list(dict.fromkeys(refs))


def write_fasta(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for seq_id, seq in records:
            f.write(f">{seq_id}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def run(cmd, cwd=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed with return code "
            f"{proc.returncode}: {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
        )
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ref", action="append", required=True)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--min_seq_id", type=float, default=0.0)
    parser.add_argument("--max_seqs", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if shutil.which(args.mmseqs) is None:
        raise FileNotFoundError(f"Cannot find mmseqs executable: {args.mmseqs}")

    cohort_dir = Path(args.cohort_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_files = sorted([
        p for p in cohort_dir.glob("*.csv")
        if "manifest" not in p.stem.lower() and "summary" not in p.stem.lower()
    ])
    if not cohort_files:
        raise FileNotFoundError(f"No cohort CSV files found in {cohort_dir}")

    refs = load_reference_sequences(args.ref)
    if not refs:
        raise RuntimeError("No valid reference sequences loaded.")

    summary_rows = []
    with tempfile.TemporaryDirectory(prefix="mmseqs_novelty_") as td:
        td = Path(td)
        ref_fasta = td / "reference.fasta"
        write_fasta(ref_fasta, [(f"ref_{i:06d}", seq) for i, seq in enumerate(refs)])

        ref_db = td / "ref_db"
        run([args.mmseqs, "createdb", str(ref_fasta), str(ref_db)])

        for fp in cohort_files:
            df = pd.read_csv(fp)
            if "method" not in df.columns:
                df["method"] = fp.stem
            if "seq_id" not in df.columns:
                df["seq_id"] = [f"{fp.stem}_{i:05d}" for i in range(len(df))]
            if "sequence" not in df.columns:
                raise ValueError(f"No sequence column in {fp}")

            method = str(df["method"].iloc[0])
            out_csv = out_dir / f"{method}_mmseqs_novelty.csv"
            if out_csv.exists() and not args.force:
                scored = pd.read_csv(out_csv)
            else:
                df["sequence"] = df["sequence"].apply(clean_seq)
                valid = df[df["sequence"].apply(is_valid_aa)].copy()
                if args.max_seqs and args.max_seqs > 0:
                    valid = valid.head(args.max_seqs).copy()

                query_fasta = td / f"{method}.fasta"
                query_db = td / f"{method}_query_db"
                result_db = td / f"{method}_result_db"
                result_tsv = td / f"{method}_result.tsv"
                tmp_dir = td / f"{method}_tmp"

                records = list(zip(valid["seq_id"].astype(str), valid["sequence"].astype(str)))
                write_fasta(query_fasta, records)
                run([args.mmseqs, "createdb", str(query_fasta), str(query_db)])
                run([
                    args.mmseqs,
                    "search",
                    str(query_db),
                    str(ref_db),
                    str(result_db),
                    str(tmp_dir),
                    "--min-seq-id",
                    str(args.min_seq_id),
                    "--threads",
                    str(args.threads),
                ])
                run([
                    args.mmseqs,
                    "convertalis",
                    str(query_db),
                    str(ref_db),
                    str(result_db),
                    str(result_tsv),
                    "--format-output",
                    "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits",
                    "--threads",
                    str(args.threads),
                ])

                if result_tsv.exists() and result_tsv.stat().st_size > 0:
                    hits = pd.read_csv(result_tsv, sep="\t", header=None)
                    hits.columns = [
                        "query",
                        "target",
                        "pident",
                        "alnlen",
                        "mismatch",
                        "gapopen",
                        "qstart",
                        "qend",
                        "tstart",
                        "tend",
                        "evalue",
                        "bits",
                    ]
                    best = (
                        hits.sort_values(["query", "pident", "bits"], ascending=[True, False, False])
                        .drop_duplicates("query")
                        .set_index("query")
                    )
                else:
                    best = pd.DataFrame()

                rows = []
                for _, r in valid.iterrows():
                    seq_id = str(r["seq_id"])
                    hit = best.loc[seq_id] if len(best) and seq_id in best.index else None
                    max_identity = float(hit["pident"]) / 100.0 if hit is not None else 0.0
                    rows.append({
                        "method": method,
                        "seq_id": seq_id,
                        "sequence": r["sequence"],
                        "length": len(r["sequence"]),
                        "mmseqs_identity_max": max_identity,
                        "novelty_mmseqs": 1.0 - max_identity,
                        "best_ref": str(hit["target"]) if hit is not None else "",
                        "best_bits": float(hit["bits"]) if hit is not None else np.nan,
                        "best_evalue": float(hit["evalue"]) if hit is not None else np.nan,
                    })
                scored = pd.DataFrame(rows)
                scored.to_csv(out_csv, index=False)

            summary_rows.append({
                "method": method,
                "n_total": len(scored),
                "novelty_mmseqs_mean": scored["novelty_mmseqs"].mean(),
                "novelty_mmseqs_median": scored["novelty_mmseqs"].median(),
                "max_identity_mean": scored["mmseqs_identity_max"].mean(),
                "max_identity_median": scored["mmseqs_identity_max"].median(),
                "pct_identity_ge_0.5": float((scored["mmseqs_identity_max"] >= 0.5).mean()),
                "pct_identity_ge_0.8": float((scored["mmseqs_identity_max"] >= 0.8).mean()),
            })

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "mmseqs_novelty_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"[OK] summary -> {summary_path}")


if __name__ == "__main__":
    main()
