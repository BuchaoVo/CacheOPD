from __future__ import annotations

import argparse
from pathlib import Path

from common import CONDITIONS, OUT_ROOT, PRIMARY_TEMPLATE, RICH_TEMPLATE, ensure_dirs, write_jsonl


def build_rows(template: str, samples_per_condition: int, prompt_type: str) -> list[dict]:
    rows = []
    for cond in CONDITIONS:
        for sample_idx in range(samples_per_condition):
            rows.append({
                "condition_id": cond["condition_id"],
                "superfamily": cond["superfamily"],
                "sample_id": sample_idx,
                "prompt_type": prompt_type,
                "prompt": template.format(SUPERFAMILY=cond["superfamily"]),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--samples_per_condition", type=int, default=32)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)
    write_jsonl(
        out_root / "prompts" / "condition_prompts_primary.jsonl",
        build_rows(PRIMARY_TEMPLATE, args.samples_per_condition, "primary"),
    )
    write_jsonl(
        out_root / "prompts" / "condition_prompts_rich.jsonl",
        build_rows(RICH_TEMPLATE, args.samples_per_condition, "rich"),
    )
    print(f"[OK] wrote prompts under {out_root / 'prompts'}")


if __name__ == "__main__":
    main()

