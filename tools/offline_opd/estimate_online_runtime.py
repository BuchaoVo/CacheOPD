from pathlib import Path
import json
import pandas as pd

ROOT = Path("/mnt/data/users/zbc/opd/ProteinOPD")

rows = []

for state_path in ROOT.glob("outputs/**/trainer_state.json"):
    run_dir = state_path
    # 如果在 checkpoint-x 下面，就取上一级作为 run_dir
    if state_path.parent.name.startswith("checkpoint-"):
        run_dir = state_path.parent.parent
    else:
        run_dir = state_path.parent

    try:
        state = json.loads(state_path.read_text())
    except Exception:
        continue

    runtime = None
    for item in reversed(state.get("log_history", [])):
        if isinstance(item, dict) and "train_runtime" in item:
            runtime = item["train_runtime"]
            break

    checkpoints = sorted(run_dir.glob("checkpoint-*"))
    mtimes = []
    for ckpt in checkpoints:
        files = list(ckpt.glob("*"))
        if files:
            mtimes.extend([f.stat().st_mtime for f in files])
        else:
            mtimes.append(ckpt.stat().st_mtime)

    mtime_span = None
    if len(mtimes) >= 2:
        mtime_span = max(mtimes) - min(mtimes)

    rows.append({
        "state_path": str(state_path.relative_to(ROOT)),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "global_step": state.get("global_step"),
        "train_runtime_from_state_sec": runtime,
        "checkpoint_mtime_span_sec": mtime_span,
        "n_checkpoints": len(checkpoints),
    })

df = pd.DataFrame(rows)
out = ROOT / "analysis_outputs/offline_proteinopd/efficiency/online_runtime_candidates.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(df.to_string(index=False))
print(f"\n[OK] saved: {out}")
