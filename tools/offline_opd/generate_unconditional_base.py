import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(text):
    seq = "".join(str(text).split()).upper()
    return "".join(ch for ch in seq if ch in AA)


def postprocess(token_ids, tokenizer):
    ids = list(token_ids)
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if ids and eos is not None and ids[0] == eos:
        ids = ids[1:]
    while ids and ((eos is not None and ids[-1] == eos) or (pad is not None and ids[-1] == pad)):
        ids.pop()
    return clean_seq(tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--num_sequences", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=500)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    set_seed(args.seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()

    records = []
    eos = tokenizer.eos_token_id
    while len(records) < args.num_sequences:
        bs = min(args.batch_size, args.num_sequences - len(records))
        input_ids = torch.full((bs, 1), eos, dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row in out:
            seq = postprocess(row.detach().cpu().tolist(), tokenizer)
            records.append({"instruction": "N/A", "reference": "N/A", "response#1": seq})
        print(f"Generated {len(records)}/{args.num_sequences}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[OK] saved {len(records)} sequences -> {out_path}")


if __name__ == "__main__":
    main()
