#!/usr/bin/env bash
set -euo pipefail

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

unset LD_PRELOAD
unset LD_LIBRARY_PATH

export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1

# 如果官方 Hugging Face 下载慢，可以取消下一行注释
# export HF_ENDPOINT=https://hf-mirror.com

which python
python --version

python /home/zbc/data/opd/ProteinOPD/scripts_local/download_prollama_adapter_snapshot.py

echo "ProLLaMA adapter download finished."
find /home/zbc/data/models/ProteinOPD -maxdepth 3 -type f | sort
