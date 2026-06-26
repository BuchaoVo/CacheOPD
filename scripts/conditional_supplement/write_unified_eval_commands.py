from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from common import OUT_ROOT, ROOT, ensure_dirs, write_status


CONDA_LIB = "/home/zbc/data/software/miniconda/envs/proteinopd/lib"


def q(x: str | Path) -> str:
    return shlex.quote(str(x))


def cuda_env(gpu: str) -> str:
    return f"CUDA_VISIBLE_DEVICES={gpu} LD_LIBRARY_PATH={CONDA_LIB}:${{LD_LIBRARY_PATH:-}}"


def write_script(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write unified evaluator commands for the conditional supplement cohorts.")
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--cohort_dir", default=str(OUT_ROOT / "eval_cohorts_internal"))
    ap.add_argument("--ppl_model", default="/home/zbc/data/models/ProLLaMA")
    ap.add_argument("--esmfold_model", default="/home/zbc/data/models/esmfold_v1_hf")
    ap.add_argument("--temberture_root", default="/home/zbc/data/models/TemBERTure")
    ap.add_argument(
        "--protein_sol_cmd_template",
        default="",
        help="Shell template consumed by score_protein_sol_cmd.py. May also be supplied at run time via PROTEIN_SOL_CMD_TEMPLATE.",
    )
    ap.add_argument("--max_per_method", type=int, default=256)
    ap.add_argument("--eval_gpus", default="5,4", help="Comma-separated GPUs rotated across GPU-heavy evaluators.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)
    commands_dir = out_root / "commands"
    cohort_dir = Path(args.cohort_dir)
    gpus = [x.strip() for x in args.eval_gpus.split(",") if x.strip()]
    if not gpus:
        gpus = ["0"]

    ppl_gpu = gpus[0]
    struct_gpu = gpus[1 % len(gpus)]
    thermo_gpu = gpus[0]
    if args.protein_sol_cmd_template:
        protein_sol_check = "true"
        protein_sol_template_arg = q(args.protein_sol_cmd_template)
    else:
        protein_sol_check = '[ -n "${PROTEIN_SOL_CMD_TEMPLATE:-}" ]'
        protein_sol_template_arg = '"${PROTEIN_SOL_CMD_TEMPLATE}"'

    eval_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd " + q(ROOT),
        f"mkdir -p {q(out_root / 'raw_eval/ppl')} {q(out_root / 'raw_eval/structure_hf')} {q(out_root / 'raw_eval/protein_sol')} {q(out_root / 'raw_eval/temberture')}",
        f"conda run -n proteinopd python scripts/conditional_supplement/prepare_internal_eval_cohorts.py --out_root {q(out_root)} --cohort_dir {q(cohort_dir)}",
        (
            f"{cuda_env(ppl_gpu)} conda run -n proteinopd python tools/p0_eval/eval_p0_basic_ppl.py "
            f"--cohort_dir {q(cohort_dir)} --out_dir {q(out_root / 'raw_eval/ppl')} "
            f"--protgpt2 {q(args.ppl_model)} --max_length 1024"
        ),
        (
            f"{cuda_env(struct_gpu)} conda run -n proteinopd python tools/p0_eval/score_esmfold_structure_hf.py "
            f"--cohort_dir {q(cohort_dir)} --out_dir {q(out_root / 'raw_eval/structure_hf')} "
            f"--model_path {q(args.esmfold_model)} --max_per_method {args.max_per_method} --max_len 1024 --chunk_size 64 --fp16"
        ),
        (
            f"if {protein_sol_check}; then\n"
            f"  LD_LIBRARY_PATH={CONDA_LIB}:${{LD_LIBRARY_PATH:-}} conda run -n proteinopd python tools/p0_eval/score_protein_sol_cmd.py "
            f"--cohort_dir {q(cohort_dir)} --out_dir {q(out_root / 'raw_eval/protein_sol')} "
            f"--cmd_template {protein_sol_template_arg} --max_per_method {args.max_per_method} --max_len 1024\n"
            "else\n"
            "  echo '[SKIP] Protein-Sol: set PROTEIN_SOL_CMD_TEMPLATE or pass --protein_sol_cmd_template.'\n"
            "fi"
        ),
        (
            f"{cuda_env(thermo_gpu)} conda run -n proteinopd python tools/p0_eval/score_temberture_cls.py "
            f"--cohort_dir {q(cohort_dir)} --out_dir {q(out_root / 'raw_eval/temberture')} "
            f"--temberture_root {q(args.temberture_root)} --batch_size 1 --device cuda "
            f"--max_per_method {args.max_per_method} --max_len 1024"
        ),
        f"conda run -n proteinopd python scripts/conditional_supplement/collect_unified_eval_outputs.py --out_root {q(out_root)}",
        f"conda run -n proteinopd python scripts/conditional_supplement/merge_eval_metrics.py --out_root {q(out_root)}",
        f"conda run -n proteinopd python scripts/conditional_supplement/summarize_internal_matched.py --out_root {q(out_root)}",
        f"conda run -n proteinopd python scripts/conditional_supplement/summarize_external_condition_aware.py --out_root {q(out_root)}",
        f"conda run -n proteinopd python scripts/conditional_supplement/summarize_single_teacher_specificity.py --out_root {q(out_root)}",
        f"conda run -n proteinopd python scripts/conditional_supplement/bootstrap_ci.py --out_root {q(out_root)} --n_boot 5000",
        f"conda run -n proteinopd python scripts/conditional_supplement/write_experiment_report.py --out_root {q(out_root)}",
    ]
    script_path = commands_dir / "run_unified_eval_internal.sh"
    write_script(script_path, eval_lines)

    config = {
        "cohort_dir": str(cohort_dir),
        "ppl_model": args.ppl_model,
        "esmfold_model": args.esmfold_model,
        "temberture_root": args.temberture_root,
        "protein_sol_cmd_template": args.protein_sol_cmd_template or "ENV:PROTEIN_SOL_CMD_TEMPLATE",
        "max_per_method": args.max_per_method,
        "eval_gpus": gpus,
        "script": str(script_path),
    }
    config_path = commands_dir / "unified_eval_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    write_status(
        "Unified_Eval",
        "commands_ready",
        "Unified evaluator command script written. Protein-Sol is skipped unless a command template is configured.",
        out_root / "status",
        script=str(script_path),
        config=str(config_path),
    )
    print(f"[OK] unified eval script -> {script_path}")
    print(f"[OK] config -> {config_path}")


if __name__ == "__main__":
    main()
