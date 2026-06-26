from __future__ import annotations

import argparse
import os
from pathlib import Path

from common import OUT_ROOT, ensure_dirs, require_module, write_status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--prompt_file", default="")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    ok, err = require_module("esm")
    model_dir = os.environ.get("ESM3_MODEL_DIR", "")
    if not ok:
        write_status("ESM3", "unavailable", "Python module `esm` is not importable.", out_root / "status", import_error=err)
        return
    if not model_dir or not Path(model_dir).exists():
        write_status("ESM3", "unavailable", "Set ESM3_MODEL_DIR to a local runnable ESM3 checkpoint or client configuration.", out_root / "status")
        return
    write_status("ESM3", "available_not_run", "ESM package and model dir detected, but no repository-specific ESM3 text-to-sequence adapter is configured.", out_root / "status", model_dir=model_dir)


if __name__ == "__main__":
    main()

