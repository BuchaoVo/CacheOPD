from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CONDITIONS, OUT_ROOT


CONDITION_TEXT = {c["condition_id"]: c["superfamily"] for c in CONDITIONS}

LEGACY_P0_CONDITION_ORDER = [
    "immunoglobulin_like_beta_sandwich_superfamily",
    "lysozyme_like_domain_superfamily",
    "rossmann_like_alpha_beta_alpha_sandwich_fold_superfamily",
    "tim_beta_alpha_barrel_domain_superfamily",
]
LEGACY_P0_METHODS = {"ProLLaMA", "Online OPD"}


def build_prompt(superfamily: str) -> str:
    return f"[Generate by superfamily] Superfamily=<{superfamily}> Seq=<"


def score_nll(model, tokenizer, prompt: str, sequence: str, device, max_length: int):
    continuation = sequence + ">"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]

    if len(prompt_ids) + len(cont_ids) > max_length:
        keep = max_length - len(prompt_ids)
        if keep <= 4:
            return None
        cont_ids = cont_ids[:keep]

    input_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long, device=device)
    labels = torch.tensor([[-100] * len(prompt_ids) + cont_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        valid = shift_labels.view(-1) != -100
        if valid.sum().item() == 0:
            return None
        return float(loss[valid].mean().item())


def fill_legacy_p0_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Recover condition labels for legacy P0 cohorts that were saved without metadata.

    The current ProLLaMA and Online OPD cohorts are exact sequence-order matches to
    the archived 4-superfamily QC files. Those files were generated as four
    consecutive blocks of 32 sequences: Ig-like, Lysozyme, Rossmann, and TIM.
    """
    out = df.copy()
    for method in LEGACY_P0_METHODS:
        mask = out["method"].astype(str).eq(method)
        sub = out[mask].copy()
        if sub.empty or sub["condition_id"].isin(CONDITION_TEXT).all():
            continue
        order_col = "sequence_id" if "sequence_id" in sub.columns else None
        if order_col:
            sub = sub.sort_values(order_col)
        inferred = []
        for i in range(len(sub)):
            block = min(i // 32, len(LEGACY_P0_CONDITION_ORDER) - 1)
            inferred.append(LEGACY_P0_CONDITION_ORDER[block])
        out.loc[sub.index, "condition_id"] = inferred
        print(f"[INFO] recovered legacy P0 condition_id for {method}: {len(sub)} rows", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--base_model", default="/home/zbc/data/models/ProLLaMA")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_per_method_condition", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_dir = out_root / "eval/condition_consistency"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_sequence_condition_consistency.csv"
    long_csv = out_dir / "condition_nll_scores_long.csv"

    df = pd.read_csv(out_root / "cleaned/cleaned_sequences.csv")
    df = fill_legacy_p0_conditions(df)
    df = df[df["condition_id"].isin(CONDITION_TEXT)].copy()
    if args.max_per_method_condition > 0:
        df = df.groupby(["method", "condition_id"], group_keys=False).head(args.max_per_method_condition).copy()

    rows = []
    long_rows = []
    done = set()
    if out_csv.exists() and not args.force:
        old = pd.read_csv(out_csv)
        rows = old.to_dict("records")
        done = set(zip(old["method"].astype(str), old["sequence_id"].astype(str)))
        print(f"[RESUME] existing rows={len(rows)}", flush=True)
        if long_csv.exists():
            old_long = pd.read_csv(long_csv)
            long_rows = old_long.to_dict("records")
            print(f"[RESUME] existing long rows={len(long_rows)}", flush=True)

    if done:
        keys = list(zip(df["method"].astype(str), df["sequence_id"].astype(str)))
        todo = df[[key not in done for key in keys]].copy()
    else:
        todo = df.copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device", device, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.eval()

    for i, r in enumerate(todo.itertuples(index=False), start=1):
        method = str(getattr(r, "method"))
        sequence_id = str(getattr(r, "sequence_id"))
        target = str(getattr(r, "condition_id"))
        sequence = str(getattr(r, "sequence"))
        print(f"[{i}/{len(todo)}] {method} {sequence_id} target={target} len={len(sequence)}", flush=True)

        scores = {}
        for cond_id, superfamily in CONDITION_TEXT.items():
            nll = score_nll(model, tokenizer, build_prompt(superfamily), sequence, device, args.max_length)
            if nll is None:
                continue
            scores[cond_id] = nll
            long_rows.append({
                "method": method,
                "sequence_id": sequence_id,
                "condition_id": target,
                "scored_condition_id": cond_id,
                "nll": nll,
            })

        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1])
        pred = ranked[0][0]
        target_nll = scores.get(target)
        best_other = min([v for k, v in scores.items() if k != target], default=None)
        margin = best_other - target_nll if best_other is not None and target_nll is not None else pd.NA
        rows.append({
            "method": method,
            "sequence_id": sequence_id,
            "condition_id": target,
            "predicted_condition": pred,
            "condition_correct": pred == target,
            "target_nll": target_nll,
            "best_other_nll": best_other,
            "condition_margin": margin,
        })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        pd.DataFrame(long_rows).to_csv(long_csv, index=False)

    print(f"[OK] wrote {out_csv}", flush=True)
    print(f"[OK] wrote {long_csv}", flush=True)


if __name__ == "__main__":
    main()
