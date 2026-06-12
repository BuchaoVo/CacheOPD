from pathlib import Path
from huggingface_hub import snapshot_download

local_dir = Path("/home/zbc/data/models/ProteinOPD")
local_dir.mkdir(parents=True, exist_ok=True)

path = snapshot_download(
    repo_id="purilion/ProteinOPD",
    repo_type="model",
    local_dir=str(local_dir),
    allow_patterns=[
        "prollama-adapter/*",
    ],
    max_workers=1,
)

print("Downloaded to:", path)
print("Expected adapter dir:", local_dir / "prollama-adapter")
