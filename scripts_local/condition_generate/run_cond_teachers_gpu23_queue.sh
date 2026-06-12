#!/usr/bin/env bash
set -euo pipefail

cd /home/zbc/data/opd/ProteinOPD

echo "Start conditional teacher training with GPU 2 and GPU 3"
date

EPOCHS=1

echo "[Stage 1] Launch foldability teacher on GPU 2"
bash scripts_local/run_cond_teacher_lora_one.sh \
  2 \
  /home/zbc/data/opd/ProteinOPD/conditional/data/foldability/train_plddt_200.json \
  /home/zbc/data/opd/ProteinOPD/conditional/data/foldability/test_plddt_50.json \
  /home/zbc/data/opd/ProteinOPD/conditional_teachers_local/foldability \
  ${EPOCHS} \
  > logs_nohup/cond_teacher_foldability_gpu2_inner_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID_FOLD=$!

echo "[Stage 1] Launch sol teacher on GPU 3"
bash scripts_local/run_cond_teacher_lora_one.sh \
  3 \
  /home/zbc/data/opd/ProteinOPD/conditional/data/sol/train_sol_200.json \
  /home/zbc/data/opd/ProteinOPD/conditional/data/sol/test_sol_50.json \
  /home/zbc/data/opd/ProteinOPD/conditional_teachers_local/sol \
  ${EPOCHS} \
  > logs_nohup/cond_teacher_sol_gpu3_inner_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID_SOL=$!

echo "foldability PID=${PID_FOLD}"
echo "sol PID=${PID_SOL}"

wait ${PID_FOLD}
echo "[Stage 1] foldability finished"
date

wait ${PID_SOL}
echo "[Stage 1] sol finished"
date

echo "[Stage 2] Launch thermo teacher on GPU 2"
bash scripts_local/run_cond_teacher_lora_one.sh \
  2 \
  /home/zbc/data/opd/ProteinOPD/conditional/data/thermo/train_thermo_200.json \
  /home/zbc/data/opd/ProteinOPD/conditional/data/thermo/test_thermo_50.json \
  /home/zbc/data/opd/ProteinOPD/conditional_teachers_local/thermo \
  ${EPOCHS} \
  > logs_nohup/cond_teacher_thermo_gpu2_inner_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "[Stage 2] thermo finished"
date

echo "All conditional teachers finished."

find /home/zbc/data/opd/ProteinOPD/conditional_teachers_local \
  -name adapter_config.json -print

find /home/zbc/data/opd/ProteinOPD/conditional_teachers_local \
  -name "adapter_model*" -print
