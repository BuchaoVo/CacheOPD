#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/mnt/data/users/zbc/opd/ProteinOPD
cd ${PROJECT_DIR}

source /home/zbc/data/software/miniconda/etc/profile.d/conda.sh
conda activate proteinopd

export PYTHONUNBUFFERED=1

CMD_TEMPLATE='cd /home/zbc/data/software/protein_sol/extracted/protein-sol-sequence-prediction-software && rm -f seq_prediction.txt seq_prediction_OLD.txt reformat.in reformat_out composition.in composition_all.out seq_props.in seq_props.out STYprops.out bins.txt; bash multiple_prediction_wrapper_export.sh {fasta} > /tmp/proteinsol_{seq_id}.log 2>&1; cat seq_prediction.txt 2>/dev/null; cat seq_prediction_*.txt 2>/dev/null; cat /tmp/proteinsol_{seq_id}.log 2>/dev/null'

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "Start Protein-Sol evaluation at $(date)"

python -u tools/p0_eval/score_protein_sol_cmd.py \
  --cohort_dir analysis_outputs/p0_current_eval/cohorts \
  --out_dir analysis_outputs/p0_current_eval/property/protein_sol \
  --cmd_template "${CMD_TEMPLATE}" \
  --max_per_method 256 \
  --max_len 1024 \
  --timeout 180

echo "Finished Protein-Sol evaluation at $(date)"
