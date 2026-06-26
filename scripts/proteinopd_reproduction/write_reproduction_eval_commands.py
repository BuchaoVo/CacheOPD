from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--out-root", default="analysis_outputs/proteinopd_reproduction")
    parser.add_argument("--protgpt2", default="/home/zbc/data/models/ProtGPT2")
    parser.add_argument("--prollama", default="/home/zbc/data/models/ProLLaMA")
    parser.add_argument("--esmfold-model", default="/home/zbc/data/models/esmfold_v1_hf")
    parser.add_argument("--temberture-root", default="/home/zbc/data/models/TemBERTure")
    parser.add_argument("--uniprot-ref", action="append", default=[])
    parser.add_argument("--train-ref", action="append", default=[])
    parser.add_argument("--protein-sol-cmd-template", default="")
    parser.add_argument("--max-per-method", type=int, default=256)
    parser.add_argument("--eval-gpus", default="5,6,7")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_root = (repo / args.out_root).resolve()
    commands = out_root / "commands"
    raw = out_root / "raw_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    gpus = [x.strip() for x in args.eval_gpus.split(",") if x.strip()]
    if not gpus:
        gpus = ["0"]
    ppl_gpu = gpus[0]
    thermo_gpu = gpus[1 % len(gpus)]
    struct_gpu = gpus[2 % len(gpus)]

    uncond_cohort = out_root / "eval_cohorts/unconditional"
    cond_cohort = out_root / "eval_cohorts/conditional"

    protein_sol_check = "true" if args.protein_sol_cmd_template else '[ -n "${PROTEIN_SOL_CMD_TEMPLATE:-}" ]'
    protein_sol_arg = q(args.protein_sol_cmd_template) if args.protein_sol_cmd_template else '"${PROTEIN_SOL_CMD_TEMPLATE}"'

    novelty_lines = []
    if args.uniprot_ref:
        refs = " ".join(f"--ref {q(x)}" for x in args.uniprot_ref)
        novelty_lines.append(
            f"conda run -n proteinopd python tools/p0_eval/score_mmseqs_novelty.py "
            f"--cohort_dir {q(uncond_cohort)} --out_dir {q(raw / 'novelty_u_unconditional')} "
            f"{refs} --threads 16"
        )
        novelty_lines.append(
            f"conda run -n proteinopd python tools/p0_eval/score_mmseqs_novelty.py "
            f"--cohort_dir {q(cond_cohort)} --out_dir {q(raw / 'novelty_u_conditional')} "
            f"{refs} --threads 16"
        )
    else:
        novelty_lines.append("echo '[SKIP] Novelty-U: pass --uniprot-ref REF_FASTA_OR_CSV to enable MMseqs2 novelty.'")

    if args.train_ref:
        refs = " ".join(f"--ref {q(x)}" for x in args.train_ref)
        novelty_lines.append(
            f"conda run -n proteinopd python tools/p0_eval/score_mmseqs_novelty.py "
            f"--cohort_dir {q(uncond_cohort)} --out_dir {q(raw / 'novelty_t_unconditional')} "
            f"{refs} --threads 16"
        )
    else:
        novelty_lines.append("echo '[SKIP] Novelty-T: pass --train-ref TRAIN_CSV/JSON to enable original ProteinOPD novelty-to-training diagnostic.'")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {q(repo)}",
        f"mkdir -p {q(raw)}",
        (
            "conda run -n proteinopd python scripts/proteinopd_reproduction/prepare_reproduction_cohorts.py "
            f"--repo-root {q(repo)} --out-root {q(out_root)} --max-per-method {args.max_per_method}"
        ),
        "",
        "# PPL: original unconditional table uses ProtGPT2-style designability scoring.",
        (
            f"{cuda_env(ppl_gpu)} conda run -n proteinopd python tools/p0_eval/eval_p0_basic_ppl.py "
            f"--cohort_dir {q(uncond_cohort)} --out_dir {q(raw / 'ppl_unconditional_protgpt2')} "
            f"--protgpt2 {q(args.protgpt2)} --max_length 1024"
        ),
        "",
        "# Conditional PPL: use ProLLaMA evaluator for locally recomputed conditional diagnostics.",
        (
            f"{cuda_env(ppl_gpu)} conda run -n proteinopd python tools/p0_eval/eval_p0_basic_ppl.py "
            f"--cohort_dir {q(cond_cohort)} --out_dir {q(raw / 'ppl_conditional_prollama')} "
            f"--protgpt2 {q(args.prollama)} --max_length 1024"
        ),
        "",
        "# Structure metrics.",
        (
            f"{cuda_env(struct_gpu)} conda run -n proteinopd python tools/p0_eval/score_esmfold_structure_hf.py "
            f"--cohort_dir {q(uncond_cohort)} --out_dir {q(raw / 'structure_unconditional')} "
            f"--model_path {q(args.esmfold_model)} --max_per_method {args.max_per_method} --max_len 1024 --chunk_size 16"
        ),
        (
            f"{cuda_env(struct_gpu)} conda run -n proteinopd python tools/p0_eval/score_esmfold_structure_hf.py "
            f"--cohort_dir {q(cond_cohort)} --out_dir {q(raw / 'structure_conditional')} "
            f"--model_path {q(args.esmfold_model)} --max_per_method {args.max_per_method} --max_len 1024 --chunk_size 16"
        ),
        "",
        "# Thermostability.",
        (
            f"{cuda_env(thermo_gpu)} conda run -n proteinopd python tools/p0_eval/score_temberture_cls.py "
            f"--cohort_dir {q(uncond_cohort)} --out_dir {q(raw / 'temberture_unconditional')} "
            f"--temberture_root {q(args.temberture_root)} --batch_size 1 --device cuda --max_per_method {args.max_per_method} --max_len 1024"
        ),
        (
            f"{cuda_env(thermo_gpu)} conda run -n proteinopd python tools/p0_eval/score_temberture_cls.py "
            f"--cohort_dir {q(cond_cohort)} --out_dir {q(raw / 'temberture_conditional')} "
            f"--temberture_root {q(args.temberture_root)} --batch_size 1 --device cuda --max_per_method {args.max_per_method} --max_len 1024"
        ),
        "",
        "# Solubility; skipped unless a Protein-Sol command template is configured.",
        (
            f"if {protein_sol_check}; then\n"
            f"  LD_LIBRARY_PATH={CONDA_LIB}:${{LD_LIBRARY_PATH:-}} conda run -n proteinopd python tools/p0_eval/score_protein_sol_cmd.py "
            f"--cohort_dir {q(uncond_cohort)} --out_dir {q(raw / 'protein_sol_unconditional')} --cmd_template {protein_sol_arg} "
            f"--max_per_method {args.max_per_method} --max_len 1024\n"
            f"  LD_LIBRARY_PATH={CONDA_LIB}:${{LD_LIBRARY_PATH:-}} conda run -n proteinopd python tools/p0_eval/score_protein_sol_cmd.py "
            f"--cohort_dir {q(cond_cohort)} --out_dir {q(raw / 'protein_sol_conditional')} --cmd_template {protein_sol_arg} "
            f"--max_per_method {args.max_per_method} --max_len 1024\n"
            "else\n"
            "  echo '[SKIP] Protein-Sol: set PROTEIN_SOL_CMD_TEMPLATE or pass --protein-sol-cmd-template.'\n"
            "fi"
        ),
        "",
        "# MMseqs2 novelty diagnostics.",
        *novelty_lines,
    ]

    script = commands / "run_round1_unified_eval.sh"
    write_script(script, lines)
    config = {
        "script": str(script),
        "unconditional_cohort": str(uncond_cohort),
        "conditional_cohort": str(cond_cohort),
        "raw_eval": str(raw),
        "protgpt2": args.protgpt2,
        "prollama": args.prollama,
        "esmfold_model": args.esmfold_model,
        "temberture_root": args.temberture_root,
        "eval_gpus": gpus,
        "protein_sol_cmd_template": args.protein_sol_cmd_template or "ENV:PROTEIN_SOL_CMD_TEMPLATE",
        "uniprot_ref": args.uniprot_ref,
        "train_ref": args.train_ref,
    }
    commands.mkdir(parents=True, exist_ok=True)
    (commands / "round1_unified_eval_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] wrote {script}")
    print(f"[OK] wrote {commands / 'round1_unified_eval_config.json'}")


if __name__ == "__main__":
    main()
