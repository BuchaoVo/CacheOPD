from pathlib import Path
from huggingface_hub import snapshot_download

local_dir = Path("/home/zbc/data/models/esmfold_v1_hf")
local_dir.mkdir(parents=True, exist_ok=True)

path = snapshot_download(
    repo_id="facebook/esmfold_v1",
    repo_type="model",
    local_dir=str(local_dir),
    max_workers=1,
    resume_download=True,
)

print("Downloaded to:", path)
