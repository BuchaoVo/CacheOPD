#!/usr/bin/env bash
set -euo pipefail

cd /home/zbc/data/opd/ProteinOPD

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

BASE_MODEL=/home/zbc/data/models/ProLLaMA
ADAPTER=/home/zbc/data/models/ProteinOPD/prollama-adapter
OUT_DIR=/home/zbc/data/opd/ProteinOPD/outputs/conditional/parallel

mkdir -p ${OUT_DIR}

SUPERFAMILY="Lysozyme-like domain superfamily"

NUM_PER_GPU=32
MAX_NEW_TOKENS=512

run_one_gpu () {
  GPU_ID=$1
  SEED=$2
  OUT_JSON=${OUT_DIR}/lysozyme_gpu${GPU_ID}.json
  LOG_FILE=${OUT_DIR}/lysozyme_gpu${GPU_ID}.log

  echo "[GPU ${GPU_ID}] Start generation -> ${OUT_JSON}"

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

  echo "[GPU ${GPU_ID}] Finished generation -> ${OUT_JSON}"
}

run_one_gpu 0 42 &
PID0=$!

run_one_gpu 1 43 &
PID1=$!

run_one_gpu 2 44 &
PID2=$!

run_one_gpu 3 45 &
PID3=$!

wait ${PID0}
wait ${PID1}
wait ${PID2}
wait ${PID3}

echo "All GPU generation jobs finished."

python - <<'PY'
import json
from pathlib import Path

out_dir = Path("/home/zbc/data/opd/ProteinOPD/outputs/conditional/parallel")
files = sorted(out_dir.glob("lysozyme_gpu*.json"))

merged = []
for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                x["source_file"] = fp.name
            merged.append(x)
    elif isinstance(data, dict):
        data["source_file"] = fp.name
        merged.append(data)

merged_path = out_dir / "lysozyme_4gpu_merged.json"
with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print(f"Merged {len(merged)} records -> {merged_path}")
PY

echo "Merged output saved."
