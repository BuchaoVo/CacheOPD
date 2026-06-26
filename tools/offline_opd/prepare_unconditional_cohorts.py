import argparse
import json
import re
from pathlib import Path

import pandas as pd


AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 50
MAX_LEN = 1024
SEQUENCE_KEYS = ("sequence", "response#1", "response", "generated", "generation", "text")


def collect_record_strings(obj):
    if isinstance(obj, dict):
        for key in SEQUENCE_KEYS:
            value = obj.get(key)
            if isinstance(value, str):
                return [value]
    return collect_strings(obj)


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


def clean_candidate(text):
    text = str(text)
    match = re.search(r"Seq=<([^>]*)>", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1)
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", text.upper())


def extract_sequences(text, allow_regex=True):
    seqs = []
    cleaned = clean_candidate(text)
    if MIN_LEN <= len(cleaned) <= MAX_LEN and set(cleaned).issubset(AA):
        seqs.append(cleaned)
    if not allow_regex:
        return seqs
    for match in re.finditer(r"[ACDEFGHIKLMNPQRSTVWY]{%d,}" % MIN_LEN, str(text).upper()):
        seq = match.group(0)
        if len(seq) <= MAX_LEN and set(seq).issubset(AA):
            seqs.append(seq)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", action="append", required=True, help="METHOD=path.json")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for spec in args.input_json:
        if "=" not in spec:
            raise ValueError(f"input_json must be METHOD=path.json, got {spec}")
        method, path = spec.split("=", 1)
        fp = Path(path)
        obj = json.loads(fp.read_text())
        rows = []
        seen = set()
        candidates = []
        if isinstance(obj, list):
            for item in obj:
                candidates.extend(collect_record_strings(item))
        else:
            candidates.extend(collect_record_strings(obj))
        for s in candidates:
            for seq in extract_sequences(s, allow_regex=False):
                if seq in seen:
                    continue
                seen.add(seq)
                rows.append({
                    "method": method,
                    "seq_id": f"{method}_{len(rows):05d}",
                    "sequence": seq,
                    "length": len(seq),
                    "source_json": str(fp),
                })
        out_csv = out_dir / f"{method}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        manifest.append({"method": method, "n_unique_valid": len(rows), "out_csv": str(out_csv), "source": str(fp)})
        print(f"[OK] {method}: {len(rows)} -> {out_csv}")

    pd.DataFrame(manifest).to_csv(out_dir / "unconditional_cohort_manifest.csv", index=False)


if __name__ == "__main__":
    main()
