#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/zbc/opd/ProteinOPD
cd "${PROJECT_DIR}"

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1
export PYTHONUNBUFFERED=1

CACHE_NAME=${CACHE_NAME:-base_rollout}
CACHE_DIR="analysis_outputs/offline_proteinopd/cache_diag/${CACHE_NAME}"
CONFLICT_DIR="analysis_outputs/offline_proteinopd/contribution3/${CACHE_NAME}_conflict"
BASE_MODEL=/home/zbc/data/models/ProLLaMA
OUT_PREFIX="outputs/offline_proteinopd/contrib3_${CACHE_NAME}_k64_conflict"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "CONFLICT_DIR=${CONFLICT_DIR}"
echo "Start contribution3 conflict experiment at $(date)"

python -u tools/offline_opd/score_cache_conflict_signal.py \
  --cache_dir "${CACHE_DIR}" \
  --out_dir "${CONFLICT_DIR}" \
  --up_lambda 1.0 \
  --down_lambda 0.8 \
  --down_min 0.2

declare -A WEIGHTS
WEIGHTS["up"]="${CONFLICT_DIR}/token_weight_conflict_up.npy"
WEIGHTS["down"]="${CONFLICT_DIR}/token_weight_conflict_down.npy"

for MODE in up down; do
  OUT_DIR="${OUT_PREFIX}_${MODE}_gamma01"
  echo "======================================================"
  echo "[TRAIN] MODE=${MODE}"
  echo "[WEIGHT] ${WEIGHTS[$MODE]}"
  echo "[OUT] ${OUT_DIR}"
  echo "======================================================"

  python -u tools/offline_opd/train_offline_opd.py \
    --base_model "${BASE_MODEL}" \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${OUT_DIR}" \
    --token_weight "${WEIGHTS[$MODE]}" \
    --max_samples 128 \
    --max_length 512 \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-5 \
    --gamma_nll 0.1 \
    --dtype bf16 \
    --device cuda \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --save_steps 20 \
    --log_steps 5
done

python - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("outputs/offline_proteinopd")
runs = [
    ("conflict_up", root / "contrib3_base_rollout_k64_conflict_up_gamma01"),
    ("conflict_down", root / "contrib3_base_rollout_k64_conflict_down_gamma01"),
]
rows = []
for mode, out_dir in runs:
    state_path = out_dir / "offline_trainer_state.json"
    log_path = out_dir / "training_log.csv"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    last = log.iloc[-1].to_dict() if len(log) else {}
    rows.append({
        "mode": mode,
        "output_dir": str(out_dir),
        "effective_tokens": state.get("effective_training_tokens"),
        "update_step": state.get("update_step"),
        "elapsed_sec": round(float(state.get("total_elapsed_sec", 0.0)), 3) if state else None,
        "final_loss": last.get("loss"),
        "final_loss_poe": last.get("loss_poe"),
        "final_loss_nll": last.get("loss_nll"),
        "adapter_exists": (out_dir / "adapter_model.safetensors").exists(),
    })

out = Path("analysis_outputs/offline_proteinopd/contribution3/base_rollout_conflict")
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / "training_summary.csv", index=False)
print(df.to_string(index=False))
print("[OK] wrote", out / "training_summary.csv")
PY

echo "Finished contribution3 conflict experiment at $(date)"
