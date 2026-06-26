from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from common import OUT_ROOT, ensure_dirs


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare one-file-per-method cohorts for legacy evaluators.")
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--source_dir", default="")
    ap.add_argument("--cohort_dir", default="")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)
    source_dir = Path(args.source_dir) if args.source_dir else out_root / "generations" / "internal"
    cohort_dir = Path(args.cohort_dir) if args.cohort_dir else out_root / "eval_cohorts_internal"
    cohort_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(source_dir.glob("*.csv")):
        if path.name == "internal_matched_generations.csv":
            continue
        df = pd.read_csv(path)
        if df.empty or "sequence" not in df.columns:
            continue
        out_path = cohort_dir / path.name
        shutil.copy2(path, out_path)
        rows.append({"method_file": path.name, "rows": len(df), "out_csv": str(out_path)})

    manifest = pd.DataFrame(rows)
    manifest.to_csv(cohort_dir / "eval_cohort_manifest.csv", index=False)
    print(f"[OK] eval cohorts -> {cohort_dir}")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
