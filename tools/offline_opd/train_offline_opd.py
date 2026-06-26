import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model


AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(x):
    return "".join(str(x).split()).upper()


def is_valid_aa(seq):
    return len(seq) > 0 and set(seq).issubset(AA)


def get_dtype(name):
    name = str(name).lower()
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def load_tokenizer(base_model):
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<pad>"})
    return tok


def infer_lora_target_modules(model, requested):
    if requested:
        return [x.strip() for x in requested.split(",") if x.strip()]

    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj", "c_fc",
        "query_key_value", "dense", "fc1", "fc2",
    ]

    module_names = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        module_names.add(leaf)

    targets = [x for x in candidates if x in module_names]

    # LLaMA-style fallback
    if not targets:
        targets = ["q_proj", "v_proj"]

    return targets


class OfflinePoEDataset(Dataset):
    def __init__(
        self,
        cache_dir,
        base_model,
        tokenizer,
        max_length=512,
        max_samples=0,
        token_mask_path="",
        token_weight_path="",
    ):
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.token_mask_path = str(token_mask_path or "")
        self.token_weight_path = str(token_weight_path or "")

        rollouts_path = self.cache_dir / "rollouts_used.csv"
        if not rollouts_path.exists():
            raise FileNotFoundError(f"Missing rollouts_used.csv: {rollouts_path}")

        df = pd.read_csv(rollouts_path)
        if "sequence" not in df.columns:
            raise ValueError(f"rollouts_used.csv needs sequence column. columns={list(df.columns)}")

        if "seq_id" not in df.columns:
            df["seq_id"] = [f"sample_{i:05d}" for i in range(len(df))]

        df["sequence"] = df["sequence"].apply(clean_seq)
        df = df[df["sequence"].apply(is_valid_aa)].copy()
        if max_samples and max_samples > 0:
            df = df.head(max_samples).copy()

        self.df = df.reset_index(drop=True)

        self.poe_ids = np.load(self.cache_dir / "poe_topk_ids.npy", mmap_mode="r")
        self.poe_lps = np.load(self.cache_dir / "poe_topk_logprobs.npy", mmap_mode="r")
        self.ref_targets = np.load(self.cache_dir / "reference_target_ids.npy", mmap_mode="r")
        self.ref_mask = np.load(self.cache_dir / "reference_valid_mask.npy", mmap_mode="r")
        self.token_mask = None
        self.token_weight = None
        if self.token_mask_path:
            mask_path = Path(self.token_mask_path)
            if not mask_path.exists():
                cache_relative = self.cache_dir / self.token_mask_path
                if cache_relative.exists():
                    mask_path = cache_relative
                else:
                    raise FileNotFoundError(f"Missing token mask: {self.token_mask_path}")
            self.token_mask_path = str(mask_path)
            self.token_mask = np.load(mask_path, mmap_mode="r")
            if self.token_mask.shape[:2] != self.ref_mask.shape[:2]:
                raise ValueError(
                    f"token mask shape {self.token_mask.shape} does not match "
                    f"reference_valid_mask shape {self.ref_mask.shape}"
                )
        if self.token_weight_path:
            weight_path = Path(self.token_weight_path)
            if not weight_path.exists():
                cache_relative = self.cache_dir / self.token_weight_path
                if cache_relative.exists():
                    weight_path = cache_relative
                else:
                    raise FileNotFoundError(f"Missing token weight: {self.token_weight_path}")
            self.token_weight_path = str(weight_path)
            self.token_weight = np.load(weight_path, mmap_mode="r")
            if self.token_weight.shape[:2] != self.ref_mask.shape[:2]:
                raise ValueError(
                    f"token weight shape {self.token_weight.shape} does not match "
                    f"reference_valid_mask shape {self.ref_mask.shape}"
                )

        if len(self.df) > self.poe_ids.shape[0]:
            raise ValueError(
                f"rollout rows {len(self.df)} > cache rows {self.poe_ids.shape[0]}"
            )

    def __len__(self):
        return len(self.df)

    def encode(self, seq):
        enc = self.tokenizer(
            seq,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        return ids

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["sequence"]
        ids = self.encode(seq)

        if ids.numel() < 2:
            input_ids = ids[:0]
            labels = ids[:0]
        else:
            input_ids = ids[:-1]
            labels = ids[1:]

        T = input_ids.numel()
        cache_T = self.poe_ids.shape[1]
        T_eff = min(T, cache_T)

        item = {
            "idx": idx,
            "seq_id": str(row["seq_id"]),
            "input_ids": input_ids[:T_eff].long(),
            "labels": labels[:T_eff].long(),
            "poe_ids": torch.from_numpy(np.array(self.poe_ids[idx, :T_eff], dtype=np.int64)),
            "poe_lps": torch.from_numpy(np.array(self.poe_lps[idx, :T_eff], dtype=np.float32)),
            "valid_mask": torch.from_numpy(self._load_effective_mask(idx, T_eff)),
            "token_weight": torch.from_numpy(self._load_token_weight(idx, T_eff)),
        }
        return item

    def _load_effective_mask(self, idx, T_eff):
        mask = np.array(self.ref_mask[idx, :T_eff], dtype=bool)
        if self.token_mask is not None:
            mask &= np.array(self.token_mask[idx, :T_eff], dtype=bool)
        return mask

    def _load_token_weight(self, idx, T_eff):
        if self.token_weight is None:
            return np.ones(T_eff, dtype=np.float32)
        weight = np.array(self.token_weight[idx, :T_eff], dtype=np.float32)
        weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        weight = np.clip(weight, 0.0, None)
        return weight

    def count_effective_tokens(self):
        total = 0
        for idx in range(len(self.df)):
            seq = self.df.iloc[idx]["sequence"]
            ids = self.encode(seq)
            if ids.numel() < 2:
                continue
            T_eff = min(ids.numel() - 1, self.poe_ids.shape[1])
            total += int(self._load_effective_mask(idx, T_eff).sum())
        return total


def collate_fn(batch, pad_token_id):
    max_t = max(x["input_ids"].numel() for x in batch)
    K = batch[0]["poe_ids"].shape[-1]
    B = len(batch)

    input_ids = torch.full((B, max_t), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_t), dtype=torch.long)
    labels = torch.full((B, max_t), -100, dtype=torch.long)
    poe_ids = torch.full((B, max_t, K), -1, dtype=torch.long)
    poe_lps = torch.full((B, max_t, K), -1e4, dtype=torch.float32)
    valid_mask = torch.zeros((B, max_t), dtype=torch.bool)
    token_weight = torch.ones((B, max_t), dtype=torch.float32)

    seq_ids = []

    for i, x in enumerate(batch):
        T = x["input_ids"].numel()
        seq_ids.append(x["seq_id"])
        if T == 0:
            continue
        input_ids[i, :T] = x["input_ids"]
        attention_mask[i, :T] = 1
        labels[i, :T] = x["labels"]
        poe_ids[i, :T] = x["poe_ids"]
        poe_lps[i, :T] = x["poe_lps"]
        valid_mask[i, :T] = x["valid_mask"]
        token_weight[i, :T] = x["token_weight"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "poe_ids": poe_ids,
        "poe_lps": poe_lps,
        "valid_mask": valid_mask,
        "token_weight": token_weight,
        "seq_ids": seq_ids,
    }


def offline_poe_loss(logits, poe_ids, poe_lps, valid_mask, token_weight=None):
    """
    CE(q_cached, p_student) on cached support.

    poe_lps are logprobs from PoE over union support, then top-k truncated.
    We renormalize inside top-k support before CE.
    """
    # logits: [B, T, V]
    B, T, V = logits.shape
    K = poe_ids.shape[-1]

    safe_ids = poe_ids.clamp(min=0)
    student_logp = F.log_softmax(logits.float(), dim=-1)
    gathered = torch.gather(student_logp, dim=-1, index=safe_ids)

    support_mask = (poe_ids >= 0) & valid_mask.unsqueeze(-1)

    # Renormalize q on cached top-k support
    q_logp = poe_lps.float().masked_fill(~support_mask, -1e9)
    q_logp = q_logp - torch.logsumexp(q_logp, dim=-1, keepdim=True)
    q = torch.exp(q_logp).masked_fill(~support_mask, 0.0)

    token_ce = -(q * gathered).sum(dim=-1)
    token_ce = token_ce.masked_fill(~valid_mask, 0.0)

    if token_weight is None:
        weight = valid_mask.float()
    else:
        weight = token_weight.float().masked_fill(~valid_mask, 0.0)

    denom = weight.sum().clamp_min(1.0)
    return (token_ce * weight).sum() / denom


def rollout_nll_loss(logits, labels, valid_mask, token_weight=None):
    V = logits.shape[-1]
    loss = F.cross_entropy(
        logits.float().view(-1, V),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape)

    mask = valid_mask & (labels != -100)
    if token_weight is None:
        weight = mask.float()
    else:
        weight = token_weight.float().masked_fill(~mask, 0.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--token_mask", default="", help="Optional .npy bool mask selecting token positions to train on.")
    ap.add_argument("--token_weight", default="", help="Optional .npy float weights for token-level loss weighting.")

    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_samples", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lambda_poe", type=float, default=1.0)
    ap.add_argument("--gamma_nll", type=float, default=0.1)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="")

    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--log_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "offline_train_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print("[INFO] Offline OPD training")
    print("[INFO] No teachers are loaded during this script.")
    print("[INFO] base_model:", args.base_model)
    print("[INFO] cache_dir:", args.cache_dir)
    print("[INFO] output_dir:", args.output_dir)
    print("[INFO] token_mask:", args.token_mask or "NONE")
    print("[INFO] token_weight:", args.token_weight or "NONE")

    dtype = get_dtype(args.dtype)
    tokenizer = load_tokenizer(args.base_model)

    dataset = OfflinePoEDataset(
        cache_dir=args.cache_dir,
        base_model=args.base_model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
        token_mask_path=args.token_mask,
        token_weight_path=args.token_weight,
    )
    effective_token_count = dataset.count_effective_tokens()
    if effective_token_count <= 0:
        raise ValueError("No effective training tokens after applying token mask.")
    print("[INFO] effective training tokens:", effective_token_count)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target_modules = infer_lora_target_modules(model, args.target_modules)
    print("[INFO] LoRA target_modules:", target_modules)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_update_steps = math.ceil(len(loader) * args.epochs / args.grad_accum)
    warmup_steps = int(total_update_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    use_amp = args.device == "cuda" and dtype in [torch.float16, torch.bfloat16]
    amp_dtype = dtype if dtype in [torch.float16, torch.bfloat16] else torch.float32

    log_rows = []
    global_step = 0
    update_step = 0
    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            global_step += 1

            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            poe_ids = batch["poe_ids"].to(args.device)
            poe_lps = batch["poe_lps"].to(args.device)
            valid_mask = batch["valid_mask"].to(args.device)
            token_weight = batch["token_weight"].to(args.device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits

                loss_poe = offline_poe_loss(logits, poe_ids, poe_lps, valid_mask, token_weight)
                loss_nll = rollout_nll_loss(logits, labels, valid_mask, token_weight)
                loss = args.lambda_poe * loss_poe + args.gamma_nll * loss_nll

            (loss / args.grad_accum).backward()

            do_update = (global_step % args.grad_accum == 0)

            if do_update:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

            if global_step % args.log_steps == 0:
                row = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "epoch": epoch + 1,
                    "loss": float(loss.detach().cpu()),
                    "loss_poe": float(loss_poe.detach().cpu()),
                    "loss_nll": float(loss_nll.detach().cpu()),
                    "batch_selected_tokens": int(valid_mask.detach().sum().cpu()),
                    "batch_weight_sum": float((token_weight * valid_mask.float()).detach().sum().cpu()),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_sec": time.time() - t0,
                }
                log_rows.append(row)
                pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)

                pbar.set_postfix({
                    "loss": f"{row['loss']:.4f}",
                    "poe": f"{row['loss_poe']:.4f}",
                    "nll": f"{row['loss_nll']:.4f}",
                    "lr": f"{row['lr']:.2e}",
                })

            if args.save_steps > 0 and update_step > 0 and do_update and update_step % args.save_steps == 0:
                ckpt_dir = out_dir / f"checkpoint-{update_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                pd.DataFrame(log_rows).to_csv(ckpt_dir / "training_log.csv", index=False)
                print(f"[SAVE] {ckpt_dir}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False)

    final_state = {
        "global_step": global_step,
        "update_step": update_step,
        "epochs": args.epochs,
        "total_elapsed_sec": time.time() - t0,
        "teacher_forward_calls_during_training": 0,
        "token_mask": args.token_mask or None,
        "token_weight": args.token_weight or None,
        "effective_training_tokens": int(effective_token_count),
        "final_output_dir": str(out_dir),
    }
    (out_dir / "offline_trainer_state.json").write_text(
        json.dumps(final_state, indent=2),
        encoding="utf-8",
    )

    print("[OK] saved adapter:", out_dir)
    print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    main()
