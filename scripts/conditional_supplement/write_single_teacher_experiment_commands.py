from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import pandas as pd

from common import CONDITIONS, OUT_ROOT, ROOT, ensure_dirs, write_status


TEACHERS = {
    "fold": {
        "method": "Fold-teacher KD",
        "safe": "single_teacher_fold_kd",
        "cache_subdir": "fold",
    },
    "sol": {
        "method": "Sol-teacher KD",
        "safe": "single_teacher_sol_kd",
        "cache_subdir": "sol",
    },
    "thermo": {
        "method": "Thermo-teacher KD",
        "safe": "single_teacher_thermo_kd",
        "cache_subdir": "thermo",
    },
}

CONDA_LIB = "/home/zbc/data/software/miniconda/envs/proteinopd/lib"


def q(x: str | Path) -> str:
    return shlex.quote(str(x))


def cuda_env(gpu: int) -> str:
    return f"CUDA_VISIBLE_DEVICES={gpu} LD_LIBRARY_PATH={CONDA_LIB}:${{LD_LIBRARY_PATH:-}}"


def write_script(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    path.chmod(0o755)


def parse_gpu_list(value: str, needed: int) -> list[int]:
    gpus = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required.")
    while len(gpus) < needed:
        gpus.extend(gpus)
    return gpus[:needed]


def train_command(
    teacher_key: str,
    base_model: Path,
    cache_root: Path,
    output_root: Path,
    epochs: int,
    max_samples: int,
    save_steps: int,
    seed: int,
    gpu: int,
) -> str:
    info = TEACHERS[teacher_key]
    cache_dir = cache_root / info["cache_subdir"]
    output_dir = output_root / info["safe"]
    return (
        f"{cuda_env(gpu)} conda run -n proteinopd python tools/offline_opd/train_offline_opd.py "
        f"--base_model {q(base_model)} "
        f"--cache_dir {q(cache_dir)} "
        f"--output_dir {q(output_dir)} "
        f"--max_length 512 --max_samples {max_samples} --epochs {epochs} "
        f"--batch_size 1 --grad_accum 8 --lr 2e-5 --weight_decay 0.0 --warmup_ratio 0.03 "
        f"--dtype bf16 --device cuda --lambda_poe 1.0 --gamma_nll 0.1 "
        f"--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 "
        f"--save_steps {save_steps} --log_steps 2 --seed {seed}"
    )


def generate_command(
    teacher_key: str,
    base_model: Path,
    adapter_root: Path,
    gen_root: Path,
    condition: dict,
    samples_per_condition: int,
    seed: int,
    gpu: int,
) -> str:
    info = TEACHERS[teacher_key]
    out_json = gen_root / info["safe"] / f"{info['safe']}_{condition['condition_id']}.json"
    adapter_path = adapter_root / info["safe"]
    return (
        f"{cuda_env(gpu)} conda run -n proteinopd python conditional/generate/generate.py "
        f"--config configs_local/conditional_generate_base.yaml "
        f"--load_mode lora "
        f"--model {q(base_model)} "
        f"--adapter_path {q(adapter_path)} "
        f"--superfamily {q(condition['superfamily'])} "
        f"--num_sequences {samples_per_condition} --generation_batch_size 8 "
        f"--output_json {q(out_json)} "
        f"--temperature 0.7 --top_k 200 --top_p 0.9 --repetition_penalty 1.2 "
        f"--max_new_tokens 512 --seed {seed}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Write fair single-teacher KD train/generate/postprocess command scripts.")
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--base_model", default="/home/zbc/data/models/ProLLaMA")
    ap.add_argument("--cache_root", default=str(OUT_ROOT / "single_teacher_caches"))
    ap.add_argument("--adapter_root", default=str(ROOT / "outputs/offline_proteinopd/conditional_supplement_single_teacher"))
    ap.add_argument("--generation_root", default=str(ROOT / "outputs/conditional/offline_opd_4superfamily"))
    ap.add_argument("--cohort_out_dir", default=str(OUT_ROOT / "generated_cohorts_single_teacher"))
    ap.add_argument("--samples_per_condition", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_samples", type=int, default=128)
    ap.add_argument("--save_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--train_gpus",
        default="0,1,2",
        help="Comma-separated GPU ids for Fold/Sol/Thermo training commands.",
    )
    ap.add_argument(
        "--generate_gpus",
        default="0,1,2,3",
        help="Comma-separated GPU ids rotated across generation commands.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)
    commands_dir = out_root / "commands"
    base_model = Path(args.base_model)
    cache_root = Path(args.cache_root)
    adapter_root = Path(args.adapter_root)
    generation_root = Path(args.generation_root)
    cohort_out_dir = Path(args.cohort_out_dir)
    train_gpus = parse_gpu_list(args.train_gpus, len(TEACHERS))
    generate_gpus = parse_gpu_list(args.generate_gpus, 4)

    train_rows = []
    train_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + q(ROOT)]
    train_parallel = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + q(ROOT)]
    parallel_groups: dict[int, list[str]] = {}
    for teacher_idx, (key, info) in enumerate(TEACHERS.items()):
        train_gpu = train_gpus[teacher_idx]
        cmd = train_command(
            key,
            base_model,
            cache_root,
            adapter_root,
            args.epochs,
            args.max_samples,
            args.save_steps,
            args.seed,
            train_gpu,
        )
        log = out_root / "logs" / f"{info['safe']}_train.log"
        train_lines.append(cmd + f" 2>&1 | tee {q(log)}")
        parallel_groups.setdefault(train_gpu, []).append(f"{cmd} > {q(log)} 2>&1")
        train_rows.append({
            "stage": "train",
            "method": info["method"],
            "teacher_key": key,
            "gpu": train_gpu,
            "cache_dir": str(cache_root / info["cache_subdir"]),
            "adapter_dir": str(adapter_root / info["safe"]),
            "command": cmd,
        })
    for gpu, grouped_commands in parallel_groups.items():
        train_parallel.append(f"# GPU {gpu}: commands in this block run sequentially on the same device.")
        train_parallel.append("(")
        for grouped_command in grouped_commands:
            train_parallel.append(f"  {grouped_command}")
        train_parallel.append(") &")
    train_parallel.append("wait")

    generate_rows = []
    generate_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + q(ROOT)]
    for teacher_idx, (key, info) in enumerate(TEACHERS.items()):
        for idx, cond in enumerate(CONDITIONS):
            gpu = generate_gpus[(teacher_idx + idx) % len(generate_gpus)]
            cmd = generate_command(
                key,
                base_model,
                adapter_root,
                generation_root,
                cond,
                args.samples_per_condition,
                args.seed + idx,
                gpu,
            )
            log = out_root / "logs" / f"{info['safe']}_{cond['condition_id']}_generate.log"
            generate_lines.append(cmd + f" 2>&1 | tee {q(log)}")
            generate_rows.append({
                "stage": "generate",
                "method": info["method"],
                "teacher_key": key,
                "condition_id": cond["condition_id"],
                "gpu": gpu,
                "output_json": str(generation_root / info["safe"] / f"{info['safe']}_{cond['condition_id']}.json"),
                "command": cmd,
            })

    post_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd " + q(ROOT),
        (
            "conda run -n proteinopd python tools/offline_opd/prepare_offline_generated_cohorts.py "
            f"--gen_root {q(generation_root)} --out_dir {q(cohort_out_dir)} "
            "--include_method single_teacher_fold_kd "
            "--include_method single_teacher_sol_kd "
            "--include_method single_teacher_thermo_kd"
        ),
        f"conda run -n proteinopd python scripts/conditional_supplement/run_internal_matched.py --out_root {q(out_root)} --samples_per_condition {args.samples_per_condition}",
        f"conda run -n proteinopd python scripts/conditional_supplement/clean_generated_sequences.py --input_dir {q(out_root / 'generations')} --out_dir {q(out_root / 'cleaned')}",
        f"conda run -n proteinopd python scripts/conditional_supplement/merge_eval_metrics.py --out_root {q(out_root)}",
        f"conda run -n proteinopd python scripts/conditional_supplement/summarize_internal_matched.py --out_root {q(out_root)}",
        f"conda run -n proteinopd python scripts/conditional_supplement/summarize_single_teacher_specificity.py --out_root {q(out_root)}",
    ]

    write_script(commands_dir / "train_single_teacher_all.sh", train_lines)
    write_script(commands_dir / "train_single_teacher_parallel.sh", train_parallel)
    write_script(commands_dir / "generate_single_teacher_all.sh", generate_lines)
    write_script(commands_dir / "postprocess_single_teacher.sh", post_lines)

    manifest = pd.DataFrame(train_rows + generate_rows)
    manifest_path = commands_dir / "single_teacher_command_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    meta = {
        "base_model": str(base_model),
        "cache_root": str(cache_root),
        "adapter_root": str(adapter_root),
        "generation_root": str(generation_root),
        "cohort_out_dir": str(cohort_out_dir),
        "samples_per_condition": args.samples_per_condition,
        "epochs": args.epochs,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "train_gpus": train_gpus,
        "generate_gpus": generate_gpus,
        "scripts": {
            "train_sequential": str(commands_dir / "train_single_teacher_all.sh"),
            "train_parallel": str(commands_dir / "train_single_teacher_parallel.sh"),
            "generate": str(commands_dir / "generate_single_teacher_all.sh"),
            "postprocess": str(commands_dir / "postprocess_single_teacher.sh"),
        },
    }
    (commands_dir / "single_teacher_experiment_config.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for info in TEACHERS.values():
        adapter_dir = adapter_root / info["safe"]
        status = "ready_to_train"
        reason = "Command scripts were written. Run training, generation, then postprocessing to add this method to the fair supplement."
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            status = "adapter_exists"
            reason = "Adapter already exists. Run generation and postprocessing if matched cohorts are missing."
        write_status(
            info["method"].replace(" ", "_"),
            status,
            reason,
            out_root / "status",
            adapter_dir=str(adapter_dir),
        )

    print(f"[OK] command scripts -> {commands_dir}")
    print(f"[OK] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
