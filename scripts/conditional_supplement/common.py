from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "analysis_outputs" / "conditional_supplement"
AA = set("ACDEFGHIKLMNPQRSTVWY")
SEQ_RE = re.compile(r"Seq=<([^>]*)>", re.IGNORECASE)

CONDITIONS = [
    {
        "condition_id": "lysozyme_like_domain_superfamily",
        "superfamily": "lysozyme-like domain superfamily",
    },
    {
        "condition_id": "immunoglobulin_like_beta_sandwich_superfamily",
        "superfamily": "immunoglobulin-like beta-sandwich superfamily",
    },
    {
        "condition_id": "tim_beta_alpha_barrel_domain_superfamily",
        "superfamily": "TIM beta/alpha-barrel domain superfamily",
    },
    {
        "condition_id": "rossmann_like_alpha_beta_alpha_sandwich_fold_superfamily",
        "superfamily": "Rossmann-like alpha/beta/alpha sandwich fold superfamily",
    },
]

PRIMARY_TEMPLATE = (
    "Design one amino-acid sequence for a protein belonging to the {SUPERFAMILY}. "
    "Return only the amino-acid sequence using the 20 standard amino-acid letters."
)
RICH_TEMPLATE = (
    "Design a well-folded globular protein sequence with the {SUPERFAMILY} architecture. "
    "The sequence should use only the 20 standard amino-acid letters and should not include any explanation or metadata."
)


def ensure_dirs(base: Path = OUT_ROOT) -> None:
    for rel in [
        "prompts",
        "generations/internal",
        "generations/external",
        "cleaned",
        "eval/ppl",
        "eval/structure",
        "eval/solubility",
        "eval/thermostability",
        "eval/novelty",
        "eval/diversity",
        "eval/condition_consistency",
        "summaries",
        "figures",
        "logs",
        "status",
    ]:
        (base / rel).mkdir(parents=True, exist_ok=True)


def clean_seq(x: object) -> str:
    s = str(x or "")
    match = SEQ_RE.search(s)
    if match:
        s = match.group(1)
    if set(s.strip().upper()).issubset(AA):
        return "".join(s.split()).upper()
    runs = re.findall(r"[ACDEFGHIKLMNPQRSTVWY]{10,}", s.upper())
    if runs:
        return max(runs, key=len)
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s.upper())


def is_valid_seq(seq: str, min_len: int = 30, max_len: int = 1024) -> bool:
    return min_len <= len(seq) <= max_len and set(seq).issubset(AA)


def read_json_any(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for v in obj.values():
            out.extend(collect_strings(v))
        return out
    if isinstance(obj, list):
        out: list[str] = []
        for v in obj:
            out.extend(collect_strings(v))
        return out
    return []


def infer_condition_from_name(path_or_name: str) -> str:
    name = Path(path_or_name).stem.lower()
    for c in CONDITIONS:
        cid = c["condition_id"]
        if cid.lower() in name:
            return cid
    return ""


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_status(model: str, status: str, reason: str, out_dir: Path | None = None, **extra) -> Path:
    out_dir = out_dir or OUT_ROOT / "status"
    payload = {
        "model": model,
        "status": status,
        "reason": reason,
        **extra,
    }
    path = out_dir / f"{model}_status.json"
    write_json(path, payload)
    return path


def require_module(module: str) -> tuple[bool, str]:
    try:
        __import__(module)
        return True, ""
    except Exception as exc:
        return False, repr(exc)


def executable_available(name: str) -> bool:
    return shutil.which(name) is not None


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def summarize_numeric(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        cols = list(group_cols) + ["valid_n"]
        for m in metrics:
            cols.extend([f"{m}_mean", f"{m}_std"])
        return pd.DataFrame(columns=cols)
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["valid_n"] = int(len(sub))
        for m in metrics:
            if m in sub.columns:
                vals = pd.to_numeric(sub[m], errors="coerce")
                row[f"{m}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
                row[f"{m}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_metric_file(path: Path, method_col: str = "method") -> pd.DataFrame:
    df = safe_read_csv(path)
    if df.empty:
        return df
    if method_col not in df.columns and "cohort" in df.columns:
        df = df.rename(columns={"cohort": method_col})
    if "sequence_id" not in df.columns and "seq_id" in df.columns:
        df = df.rename(columns={"seq_id": "sequence_id"})
    return df


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
