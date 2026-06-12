#!/usr/bin/env bash
set -euo pipefail

cd /home/zbc/data/opd/ProteinOPD

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export LD_PRELOAD=/home/zbc/data/software/miniconda/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/zbc/data/software/miniconda/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p /home/zbc/data/opd/ProteinOPD/outputs/conditional

python -u conditional/generate/generate.py \
  --load_mode lora \
  --model /home/zbc/data/models/ProLLaMA \
  --adapter_path /home/zbc/data/models/ProteinOPD/prollama-adapter \
  --superfamily "Lysozyme-like domain superfamily" \
  --num_sequences 8 \
  --generation_batch_size 1 \
  --output_json /home/zbc/data/opd/ProteinOPD/outputs/conditional/conditional_generate_released_8seq.json \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 200 \
  --repetition_penalty 1.2 \
  --max_new_tokens 256 \
  --seed 42
