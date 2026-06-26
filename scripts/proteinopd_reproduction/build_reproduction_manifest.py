from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpectedInput:
    setting: str
    method: str
    priority: str
    source_type: str
    candidates: tuple[str, ...]
    notes: str


EXPECTED = [
    ExpectedInput(
        "unconditional",
        "ProtGPT2",
        "P0",
        "generation_or_cohort",
        (
            "analysis_outputs/unconditional_cacheopd/cohorts_base",
            "outputs/unconditional_student_opd_4gpu/generations",
        ),
        "Base autoregressive PLM reference; use local generation if available.",
    ),
    ExpectedInput(
        "unconditional",
        "ProteinOPD",
        "P0",
        "generation_or_cohort",
        (
            "outputs/unconditional_student_opd_4gpu/generations",
            "analysis_outputs/unconditional_cacheopd/eval_cohorts_split/ProteinOPD-Uncond",
        ),
        "Original unconditional ProteinOPD or local reproduced adapter generation.",
    ),
    ExpectedInput(
        "unconditional",
        "CacheOPD-Uncond-Full",
        "P1",
        "generation_or_cohort",
        (
            "analysis_outputs/unconditional_cacheopd/eval_cohorts_split/OfflinePoE-Uncond-Full",
            "analysis_outputs/unconditional_cacheopd/eval_cohorts_shards/OfflinePoE-Uncond-Full",
        ),
        "Useful for CacheOPD bridge, not original ProteinOPD paper.",
    ),
    ExpectedInput(
        "unconditional",
        "CacheOPD-Uncond-Sparse",
        "P1",
        "generation_or_cohort",
        (
            "analysis_outputs/unconditional_cacheopd/eval_cohorts_split/CacheOPD-Uncond-q04-High50",
            "analysis_outputs/unconditional_cacheopd/eval_cohorts_shards/CacheOPD-Uncond-q04-High50",
        ),
        "Useful for CacheOPD bridge, not original ProteinOPD paper.",
    ),
    ExpectedInput(
        "conditional",
        "ProLLaMA",
        "P0",
        "generation_or_cohort",
        (
            "analysis_outputs/p0_current_eval/cohorts/Base-ProLLaMA.csv",
            "analysis_outputs/conditional_supplement/eval_cohorts_internal/ProLLaMA.csv",
        ),
        "Conditional base model cohort.",
    ),
    ExpectedInput(
        "conditional",
        "ProteinOPD",
        "P0",
        "generation_or_cohort",
        (
            "analysis_outputs/p0_current_eval/cohorts/Conditional-OPD.csv",
            "analysis_outputs/conditional_supplement/eval_cohorts_internal/Online_OPD.csv",
        ),
        "Local reproduced conditional OPD cohort; may differ from original Table 3 protocol.",
    ),
    ExpectedInput(
        "conditional",
        "ASPO",
        "P1",
        "generation_or_cohort",
        (
            "analysis_outputs/conditional_supplement/generations/external/ASPO.csv",
            "analysis_outputs/conditional_supplement/eval_cohorts_external/ASPO.csv",
        ),
        "Important original baseline; usually missing unless ASPO runner is configured.",
    ),
    ExpectedInput(
        "conditional",
        "ProDVa",
        "P1",
        "generation_or_cohort",
        (
            "analysis_outputs/conditional_supplement/generations/external/ProDVa.csv",
            "analysis_outputs/conditional_supplement/status/ProDVa_status.json",
        ),
        "Runner currently checks availability; generation adapter likely missing.",
    ),
    ExpectedInput(
        "conditional",
        "ProteinDT",
        "P2",
        "generation_or_cohort",
        (
            "analysis_outputs/conditional_supplement/generations/external/ProteinDT.csv",
            "analysis_outputs/conditional_supplement/status/ProteinDT_status.json",
        ),
        "Runner currently checks availability; generation adapter likely missing.",
    ),
    ExpectedInput(
        "unconditional",
        "ESM2",
        "P2",
        "generation_or_cohort",
        (
            "analysis_outputs/proteinopd_reproduction/generations/unconditional/ESM2.csv",
        ),
        "Requires iterative masked-token infilling protocol.",
    ),
    ExpectedInput(
        "unconditional",
        "ESM3",
        "P2",
        "generation_or_cohort",
        (
            "analysis_outputs/proteinopd_reproduction/generations/unconditional/ESM3.csv",
            "analysis_outputs/conditional_supplement/status/ESM3_status.json",
        ),
        "Requires local ESM3 checkpoint/client and generation adapter.",
    ),
    ExpectedInput(
        "unconditional",
        "ProGen2",
        "P2",
        "generation_or_cohort",
        (
            "analysis_outputs/proteinopd_reproduction/generations/unconditional/ProGen2.csv",
        ),
        "Requires local ProGen2 checkpoint.",
    ),
    ExpectedInput(
        "unconditional",
        "Pinal",
        "P3",
        "generation_or_cohort",
        (
            "analysis_outputs/proteinopd_reproduction/generations/unconditional/Pinal.csv",
        ),
        "Requires external Pinal implementation/checkpoint.",
    ),
    ExpectedInput(
        "unconditional",
        "MoMPNN",
        "P3",
        "generation_or_cohort",
        (
            "analysis_outputs/proteinopd_reproduction/generations/unconditional/MoMPNN.csv",
        ),
        "Cross-paradigm inverse-folding baseline; requires 256 backbones.",
    ),
    ExpectedInput(
        "unconditional",
        "ProtRL",
        "P3",
        "training_log_and_generation",
        (
            "analysis_outputs/proteinopd_reproduction/generations/unconditional/ProtRL.csv",
            "analysis_outputs/proteinopd_reproduction/logs/ProtRL",
        ),
        "Needed for Figure 2/6 strict reproduction; expensive and currently not configured.",
    ),
]


def count_sequences(path: Path) -> int | str:
    if not path.exists():
        return "missing"
    if path.is_dir():
        total = 0
        found = False
        for fp in path.rglob("*.csv"):
            try:
                with fp.open("r", encoding="utf-8", errors="ignore") as f:
                    n = max(sum(1 for _ in f) - 1, 0)
                total += n
                found = True
            except OSError:
                continue
        if found:
            return total
        json_files = list(path.rglob("*.json"))
        if json_files:
            return f"dir_json_files={len(json_files)}"
        return "dir_exists_no_csv"
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return max(sum(1 for _ in f) - 1, 0)
    if path.suffix.lower() == ".json":
        return "json_exists"
    return "exists"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-dir", default="analysis_outputs/proteinopd_reproduction/status")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "input_availability.csv"

    rows = []
    for item in EXPECTED:
        hits = []
        hit_counts = []
        data_hits = []
        for cand in item.candidates:
            p = repo / cand
            if p.exists():
                hits.append(cand)
                hit_counts.append(f"{cand}:{count_sequences(p)}")
                if not cand.endswith("_status.json"):
                    data_hits.append(cand)
        if data_hits:
            status = "available"
        elif hits:
            status = "metadata_only"
        else:
            status = "missing"
        rows.append(
            {
                "setting": item.setting,
                "method": item.method,
                "priority": item.priority,
                "source_type": item.source_type,
                "status": status,
                "available_candidates": " | ".join(hits),
                "candidate_counts": " | ".join(hit_counts),
                "all_candidates": " | ".join(item.candidates),
                "notes": item.notes,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
