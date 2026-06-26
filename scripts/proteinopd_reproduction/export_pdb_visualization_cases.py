from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding


DEFAULT_METHODS = [
    "conditional::ProLLaMA",
    "conditional::ProteinOPD",
    "unconditional::ProtGPT2",
    "unconditional::CacheOPD-Uncond-Full",
    "unconditional::CacheOPD-Uncond-Sparse",
    "unconditional::ESM2-35M-MLM",
]


def safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def pick_cases(metrics_csv: Path, methods: list[str], top_k: int, max_len: int) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    required = {"setting", "method", "seq_id", "sequence", "length", "plddt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {metrics_csv}: {sorted(missing)}")

    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    df["plddt"] = pd.to_numeric(df["plddt"], errors="coerce")
    df = df[df["length"].notna() & df["plddt"].notna()].copy()
    df = df[df["length"] <= max_len].copy()
    df["setting_method"] = df["setting"].astype(str) + "::" + df["method"].astype(str)

    rows = []
    for key in methods:
        sub = df[df["setting_method"] == key].sort_values(["plddt", "ptm"], ascending=False, na_position="last")
        if sub.empty:
            print(f"[WARN] no case for {key}")
            continue
        rows.append(sub.head(top_k))
    if not rows:
        raise RuntimeError("No cases selected.")
    out = pd.concat(rows, ignore_index=True)
    out["case_id"] = [
        f"{safe_name(r.setting)}__{safe_name(r.method)}__rank{i % top_k + 1:02d}__{safe_name(r.seq_id)}"
        for i, r in out.iterrows()
    ]
    return out


def load_model(model_path: str, device: torch.device, chunk_size: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = EsmForProteinFolding.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"):
        model.trunk.set_chunk_size(chunk_size)
    model.eval().to(device)
    return tokenizer, model


def fold_to_pdb(model, sequence: str) -> str:
    if hasattr(model, "infer_pdb"):
        return model.infer_pdb(sequence)
    raise RuntimeError("Loaded ESMFold model does not expose infer_pdb().")


def write_viewer_html(cases: list[dict], out_html: Path) -> None:
    data = json.dumps(cases)
    cards = []
    for i, c in enumerate(cases):
        title = html.escape(f"{c['setting']} / {c['method']}")
        subtitle = html.escape(
            f"{c['seq_id']} | len={c['length']} | pLDDT={c['plddt']:.2f} | pAE={c['pae']:.2f} | pTM={c['ptm']:.3f}"
        )
        cards.append(
            f"""
<section class="case">
  <div class="meta">
    <h2>{title}</h2>
    <p>{subtitle}</p>
  </div>
  <div id="viewer-{i}" class="viewer"></div>
</section>
"""
        )

    out_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CacheOPD Protein Structure Cases</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f7f7f4;
      color: #1f2933;
    }}
    header {{
      padding: 24px 28px 12px;
      border-bottom: 1px solid #d8d8d0;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      font-weight: 700;
    }}
    .note {{
      margin: 0;
      max-width: 980px;
      color: #4d5965;
      font-size: 13px;
      line-height: 1.45;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 16px;
      padding: 18px;
    }}
    .case {{
      background: #ffffff;
      border: 1px solid #d8d8d0;
      border-radius: 8px;
      overflow: hidden;
    }}
    .meta {{
      padding: 12px 14px 8px;
      border-bottom: 1px solid #ecece6;
    }}
    h2 {{
      margin: 0 0 5px;
      font-size: 15px;
    }}
    p {{
      margin: 0;
      font-size: 12px;
      color: #52606d;
    }}
    .viewer {{
      width: 100%;
      height: 360px;
      position: relative;
    }}
  </style>
</head>
<body>
  <header>
    <h1>CacheOPD Representative ESMFold Structures</h1>
    <p class="note">Standalone 3Dmol.js viewer generated from locally folded PDB files. Structures are for qualitative visualization only; quantitative comparisons should use the CSV metrics.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
  <script>
    const cases = {data};
    cases.forEach((item, idx) => {{
      const el = document.getElementById(`viewer-${{idx}}`);
      const viewer = $3Dmol.createViewer(el, {{ backgroundColor: "white" }});
      viewer.addModel(item.pdb, "pdb");
      viewer.setStyle({{}}, {{ cartoon: {{ color: "spectrum" }} }});
      viewer.zoomTo();
      viewer.render();
    }});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", default="analysis_outputs/proteinopd_reproduction/summaries/proteinopd_reproduction_per_sequence_metrics.csv")
    parser.add_argument("--out_dir", default="analysis_outputs/proteinopd_reproduction/pdb_visualization")
    parser.add_argument("--model_path", default="/home/zbc/data/models/esmfold_v1_hf")
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pdb_dir = out_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    methods = args.method or DEFAULT_METHODS
    cases = pick_cases(Path(args.metrics_csv), methods, args.top_k, args.max_len)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    tokenizer, model = load_model(args.model_path, device, args.chunk_size)
    del tokenizer

    html_cases = []
    for _, row in tqdm(cases.iterrows(), total=len(cases), desc="Export PDB"):
        pdb_path = pdb_dir / f"{row.case_id}.pdb"
        if pdb_path.exists():
            pdb = pdb_path.read_text(encoding="utf-8")
        else:
            pdb = fold_to_pdb(model, str(row.sequence))
            pdb_path.write_text(pdb, encoding="utf-8")
        html_cases.append(
            {
                "case_id": row.case_id,
                "setting": row.setting,
                "method": row.method,
                "seq_id": row.seq_id,
                "length": int(row.length),
                "plddt": float(row.plddt),
                "pae": float(row.pae),
                "ptm": float(row.ptm),
                "pdb_file": str(pdb_path.relative_to(out_dir)),
                "pdb": pdb,
            }
        )

    cases_out = cases.copy()
    cases_out["pdb_file"] = [c["pdb_file"] for c in html_cases]
    cases_out.to_csv(out_dir / "selected_structure_cases.csv", index=False)
    write_viewer_html(html_cases, out_dir / "structure_viewer.html")
    print(f"[OK] selected cases -> {out_dir / 'selected_structure_cases.csv'}")
    print(f"[OK] PDB files -> {pdb_dir}")
    print(f"[OK] viewer -> {out_dir / 'structure_viewer.html'}")


if __name__ == "__main__":
    main()
