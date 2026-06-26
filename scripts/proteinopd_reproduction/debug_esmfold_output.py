import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForProteinFolding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_csv", required=True)
    parser.add_argument("--model_path", default="/home/zbc/data/models/esmfold_v1_hf")
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--chunk_size", type=int, default=32)
    args = parser.parse_args()

    seq = str(pd.read_csv(args.cohort_csv)["sequence"].iloc[0])
    seq = "".join(seq.split()).upper()[: args.max_len]
    print("seq_len", len(seq), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = EsmForProteinFolding.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().cuda()

    if hasattr(model, "trunk") and hasattr(model.trunk, "set_chunk_size"):
        model.trunk.set_chunk_size(args.chunk_size)

    inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(inputs["input_ids"].cuda())

    print("type", type(out), flush=True)
    print("keys", list(out.keys()) if hasattr(out, "keys") else None, flush=True)
    for key in ["plddt", "predicted_aligned_error", "ptm", "mean_plddt", "pae"]:
        value = getattr(out, key, None)
        print("field", key, "none", value is None, flush=True)
        if torch.is_tensor(value):
            tensor = value.detach().float().cpu()
            print(
                key,
                "shape",
                tuple(tensor.shape),
                "nan_frac",
                float(torch.isnan(tensor).float().mean()),
                "mean",
                float(torch.nanmean(tensor)),
                flush=True,
            )


if __name__ == "__main__":
    main()
