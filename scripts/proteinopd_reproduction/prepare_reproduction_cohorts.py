from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


AA = set("ACDEFGHIKLMNPQRSTVWY")
SEQ_RE = re.compile(r"Seq=<([^>]*)>")
STEP_RE = re.compile(r"step[_-](\d+)")


def clean_sequence(seq: object) -> str:
    return "".join(str(seq).split()).upper()


def valid_aa(seq: str) -> bool:
    return bool(seq) and set(seq).issubset(AA)


def stable_rows(method: str, sequences: list[str], setting: str, source: str, max_n: int = 256) -> pd.DataFrame:
    seen = set()
    rows = []
    for seq in sequences:
        seq = clean_sequence(seq)
        if not valid_aa(seq):
            continue
        if seq in seen:
            continue
        seen.add(seq)
        idx = len(rows)
        rows.append(
            {
                "setting": setting,
                "method": method,
                "seq_id": f"{method}_{idx:05d}",
                "sequence": seq,
                "length": len(seq),
                "source": source,
            }
        )
        if len(rows) >= max_n:
            break
    return pd.DataFrame(rows)


def read_csv_sequences(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise ValueError(f"No sequence column in {path}")
    return df["sequence"].dropna().astype(str).tolist()


def extract_json_sequences(payload: object) -> list[str]:
    if isinstance(payload, dict):
        if isinstance(payload.get("generations"), list):
            payload = payload["generations"]
        elif isinstance(payload.get("data"), list):
            payload = payload["data"]
        elif isinstance(payload.get("records"), list):
            payload = payload["records"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        return []
    seqs: list[str] = []
    for item in payload:
        if isinstance(item, str):
            seqs.append(item)
        elif isinstance(item, dict):
            for key in ("sequence", "seq", "protein_sequence"):
                if key in item and item[key]:
                    seqs.append(str(item[key]))
                    break
            else:
                text = str(item.get("output", ""))
                match = SEQ_RE.search(text)
                if match:
                    seqs.append(match.group(1))
    return seqs


def latest_step_json(directory: Path) -> Path | None:
    files = sorted(directory.glob("*.json"))
    if not files:
        return None

    def step_of(path: Path) -> int:
        match = STEP_RE.search(path.stem)
        return int(match.group(1)) if match else -1

    return max(files, key=step_of)


def read_json_sequences(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return extract_json_sequences(payload)


def write_if_available(df: pd.DataFrame, out_path: Path) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return {"status": "empty", "out_csv": str(out_path), "n": 0}
    df.to_csv(out_path, index=False)
    return {"status": "ok", "out_csv": str(out_path), "n": len(df)}


def prepare(repo: Path, out_root: Path, max_n: int) -> pd.DataFrame:
    cohort_uncond = out_root / "eval_cohorts" / "unconditional"
    cohort_cond = out_root / "eval_cohorts" / "conditional"
    manifest_rows: list[dict[str, object]] = []

    jobs = [
        {
            "setting": "unconditional",
            "method": "ProtGPT2",
            "source": repo / "analysis_outputs/unconditional_cacheopd/cohorts_base/Base-ProtGPT2.csv",
            "kind": "csv",
            "out": cohort_uncond / "ProtGPT2.csv",
        },
        {
            "setting": "unconditional",
            "method": "ProteinOPD",
            "source": repo / "analysis_outputs/unconditional_cacheopd/eval_cohorts_split/ProteinOPD-Uncond/ProteinOPD-Uncond.csv",
            "fallback_dir": repo / "outputs/unconditional_student_opd_4gpu/generations",
            "kind": "csv_or_latest_json",
            "out": cohort_uncond / "ProteinOPD.csv",
        },
        {
            "setting": "unconditional",
            "method": "CacheOPD-Uncond-Full",
            "source": repo / "analysis_outputs/unconditional_cacheopd/eval_cohorts_split/OfflinePoE-Uncond-Full/OfflinePoE-Uncond-Full.csv",
            "kind": "csv",
            "out": cohort_uncond / "CacheOPD-Uncond-Full.csv",
        },
        {
            "setting": "unconditional",
            "method": "CacheOPD-Uncond-Sparse",
            "source": repo / "analysis_outputs/unconditional_cacheopd/eval_cohorts_split/CacheOPD-Uncond-q04-High50/CacheOPD-Uncond-q04-High50.csv",
            "kind": "csv",
            "out": cohort_uncond / "CacheOPD-Uncond-Sparse.csv",
        },
        {
            "setting": "conditional",
            "method": "ProLLaMA",
            "source": repo / "analysis_outputs/p0_current_eval/cohorts/Base-ProLLaMA.csv",
            "kind": "csv",
            "out": cohort_cond / "ProLLaMA.csv",
        },
        {
            "setting": "conditional",
            "method": "ProteinOPD",
            "source": repo / "analysis_outputs/p0_current_eval/cohorts/Conditional-OPD.csv",
            "kind": "csv",
            "out": cohort_cond / "ProteinOPD.csv",
        },
    ]

    for job in jobs:
        source = Path(job["source"])
        actual_source = source
        seqs: list[str] = []
        status = "missing"
        reason = ""
        try:
            if job["kind"] == "csv" and source.exists():
                seqs = read_csv_sequences(source)
            elif job["kind"] == "csv_or_latest_json":
                if source.exists():
                    seqs = read_csv_sequences(source)
                else:
                    latest = latest_step_json(Path(job["fallback_dir"]))
                    if latest is not None:
                        actual_source = latest
                        seqs = read_json_sequences(latest)
            if seqs:
                df = stable_rows(
                    str(job["method"]),
                    seqs,
                    str(job["setting"]),
                    str(actual_source.relative_to(repo)) if actual_source.is_relative_to(repo) else str(actual_source),
                    max_n=max_n,
                )
                result = write_if_available(df, Path(job["out"]))
                status = str(result["status"])
            else:
                df = pd.DataFrame()
                result = {"out_csv": str(job["out"]), "n": 0}
                reason = f"No sequences found at {source}"
        except Exception as exc:  # noqa: BLE001
            df = pd.DataFrame()
            result = {"out_csv": str(job["out"]), "n": 0}
            status = "error"
            reason = repr(exc)
        manifest_rows.append(
            {
                "setting": job["setting"],
                "method": job["method"],
                "status": status,
                "n_sequences": result["n"],
                "out_csv": result["out_csv"],
                "source": str(actual_source.relative_to(repo)) if actual_source.exists() and actual_source.is_relative_to(repo) else str(actual_source),
                "reason": reason,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    status_dir = out_root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(status_dir / "prepared_cohort_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-root", default="analysis_outputs/proteinopd_reproduction")
    parser.add_argument("--max-per-method", type=int, default=256)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = prepare(repo, out_root, args.max_per_method)
    print(manifest.to_string(index=False))
    print(f"[OK] wrote {out_root / 'status' / 'prepared_cohort_manifest.csv'}")


if __name__ == "__main__":
    main()
