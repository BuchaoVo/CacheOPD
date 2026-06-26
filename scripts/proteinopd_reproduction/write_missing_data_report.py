from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-root", default="analysis_outputs/proteinopd_reproduction")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    status_dir = out_root / "status"
    availability = status_dir / "input_availability.csv"
    cohorts = status_dir / "prepared_cohort_manifest.csv"
    out_md = status_dir / "proteinopd_reproduction_gap_report.md"

    lines = [
        "# ProteinOPD Reproduction Gap Report",
        "",
        "This report is generated from local file availability and prepared cohort manifests.",
        "",
    ]

    if availability.exists():
        df = pd.read_csv(availability)
        lines.extend(["## Input Availability", ""])
        counts = df.groupby(["priority", "status"]).size().reset_index(name="n")
        lines.append(counts.to_markdown(index=False))
        lines.extend(["", "### Missing Or Metadata-Only Inputs", ""])
        miss = df[df["status"].isin(["missing", "metadata_only"])].copy()
        if miss.empty:
            lines.append("No missing inputs in the expected manifest.")
        else:
            lines.append(miss[["setting", "method", "priority", "status", "notes"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append(f"Missing availability manifest: `{availability}`")

    if cohorts.exists():
        cdf = pd.read_csv(cohorts)
        lines.extend(["## Prepared Cohorts", ""])
        lines.append(cdf[["setting", "method", "status", "n_sequences", "source"]].to_markdown(index=False))
        lines.extend(["", "### Cohort Sufficiency", ""])
        cdf["target_128"] = cdf["n_sequences"] >= 128
        cdf["target_256"] = cdf["n_sequences"] >= 256
        lines.append(cdf[["setting", "method", "n_sequences", "target_128", "target_256"]].to_markdown(index=False))
        lines.append("")
    else:
        lines.append(f"Missing prepared cohort manifest: `{cohorts}`")

    lines.extend(
        [
            "## Immediate Next Jobs",
            "",
            "1. Run `analysis_outputs/proteinopd_reproduction/commands/run_round1_unified_eval.sh` after confirming GPU assignment.",
            "2. Configure `PROTEIN_SOL_CMD_TEMPLATE` before running Protein-Sol, otherwise that section will be skipped.",
            "3. Pass UniProt/UniRef references to `write_reproduction_eval_commands.py --uniprot-ref` before enabling MMseqs2 Novelty-U.",
            "4. Treat `metadata_only` external methods as unavailable for local ranking until real generation CSVs are produced.",
            "",
        ]
    )

    status_dir.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
