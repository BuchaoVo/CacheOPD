# CacheOPD High-Preference Instruction Data

This directory contains ProteinOPD-compatible instruction-tuning data copied
from the CacheOPD high-preference property teacher pipeline.

Datasets:

- `foldability_high_pref`
- `solubility_high_pref`
- `thermostability_high_pref`

Each dataset contains:

- `train_5k.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `balanced_test.jsonl`
- `sanity_check.jsonl`
- matching `*.full.jsonl` files with metadata
- `manifest.json`

The plain `.jsonl` files contain only the fields expected by
`instruction_tune.py`:

```json
{"instruction": "...", "input": "...", "output": "Seq=<...>"}
```

Train the three high-preference LoRA teachers from the ProteinOPD tree with:

```bash
cd /home/zbc/data/opd/ProteinOPD
GPU_FOLD=5 GPU_SOL=6 GPU_THERMO=7 \
bash conditional/teacher_construct/scripts/run_cacheopd_high_pref_lora_three_teachers.sh 1
```

