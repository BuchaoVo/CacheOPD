#!/usr/bin/env bash
set -euo pipefail

cd /home/zbc/data/opd/ProteinOPD

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASE_MODEL=/home/zbc/data/models/ProLLaMA
ADAPTER=/home/zbc/data/models/ProteinOPD/prollama-adapter
OUT_DIR=/home/zbc/data/opd/ProteinOPD/outputs/conditional/superfamily_4gpu

mkdir -p ${OUT_DIR}

NUM_PER_GPU=32
MAX_NEW_TOKENS=512

run_one_superfamily () {
  GPU_ID=$1
  SEED=$2
  SUPERFAMILY=$3
  SAFE_NAME=$4

  OUT_JSON=${OUT_DIR}/${SAFE_NAME}.json
  LOG_FILE=${OUT_DIR}/${SAFE_NAME}.log

  echo "[GPU ${GPU_ID}] Start generation"
  echo "[GPU ${GPU_ID}] Superfamily: ${SUPERFAMILY}"
  echo "[GPU ${GPU_ID}] Output: ${OUT_JSON}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6 \
  LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-} \
  python -u conditional/generate/generate.py \
    --load_mode lora \
    --model ${BASE_MODEL} \
    --adapter_path ${ADAPTER} \
    --superfamily "${SUPERFAMILY}" \
    --num_sequences ${NUM_PER_GPU} \
    --generation_batch_size 1 \
    --output_json ${OUT_JSON} \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 200 \
    --repetition_penalty 1.2 \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --seed ${SEED} \
    > ${LOG_FILE} 2>&1

  echo "[GPU ${GPU_ID}] Finished: ${SUPERFAMILY}"
}

run_one_superfamily 0 42 "Lysozyme-like domain superfamily" "lysozyme_like_domain_superfamily" &
PID0=$!

run_one_superfamily 1 43 "Immunoglobulin-like beta-sandwich superfamily" "immunoglobulin_like_beta_sandwich_superfamily" &
PID1=$!

run_one_superfamily 2 44 "TIM beta/alpha-barrel domain superfamily" "tim_beta_alpha_barrel_domain_superfamily" &
PID2=$!

run_one_superfamily 3 45 "Rossmann-like alpha/beta/alpha sandwich fold superfamily" "rossmann_like_alpha_beta_alpha_sandwich_fold_superfamily" &
PID3=$!

wait ${PID0}
wait ${PID1}
wait ${PID2}
wait ${PID3}

echo "All 4-superfamily generation jobs finished."

python - <<'PY'
import json
from pathlib import Path

out_dir = Path("/home/zbc/data/opd/ProteinOPD/outputs/conditional/superfamily_4gpu")
files = sorted(out_dir.glob("*.json"))

merged = []
for fp in files:
    if fp.name == "merged_4superfamily.json":
        continue

    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                x["source_file"] = fp.name
                x["superfamily_file"] = fp.stem
            merged.append(x)
    elif isinstance(data, dict):
        data["source_file"] = fp.name
        data["superfamily_file"] = fp.stem
        merged.append(data)

merged_path = out_dir / "merged_4superfamily.json"
with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Merged {len(merged)} records -> {merged_path}")
PY

echo "Merged output saved."
