# -*- coding: utf-8 -*-
"""
train_ddp_multifolder.py

在你原 train_ddp.py 基础上，最小改动实现 jsonl.gz 输入 + 自动 split + wandb。

关键：保持原数据流不变
- Dataset 仍然返回 signal_str（字符串），由 BasecallModel.tokenizer 编码成 input_ids
- reference 仍按 ref_row[ref_row>0] 取 labels

新增：
- --jsonl_paths 多文件夹/多文件输入
- 自动 split（--train_ratio/--val_ratio/--test_ratio，按 folder 或 file group）
- wandb（仅 rank0）
- CTC 使用 batch["input_lengths"]（来自 tokenizer attention_mask）
- ✅ 新增 checkpoint 保存/恢复：
  - 每 epoch 保存 ckpt_last.pt
  - val_acc 最优保存 ckpt_best.pt
  - 支持 --resume_ckpt 恢复 model/optimizer/scheduler/epoch/best 指标
"""

import os
import argparse
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional, Any, Dict, List
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from .utils import seed_everything, BLANK_IDX, ID2BASE, resolve_input_lengths
from .model import BasecallModel
from .metrics import (
    koi_beam_search_decode,
    batch_bonito_accuracy,
    plot_curves,
    save_metrics_csv,
    ctc_crf_loss,
)
from .data_multifolder import (
    scan_jsonl_files,
    split_jsonl_files_by_group,
    MultiJsonlSignalRefDataset,
    scan_npy_pairs,
    split_npy_pairs_by_group,
    MultiNpySignalRefDataset,
    create_collate_fn,
)
from .callback import plot_alignment_heatmap

try:
    import wandb
except Exception:
    wandb = None


# -------------------- distributed helpers --------------------

def init_distributed() -> Tuple[int, int, int, torch.device, bool]:
    ddp_env = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)

    if ddp_env:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if ddp_env and world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        ddp_enabled = True
    else:
        ddp_enabled = False

    return rank, world_size, local_rank, device, ddp_enabled


def cleanup_distributed(ddp_enabled: bool):
    if ddp_enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def setup_logger(log_file: str, rank: int) -> logging.Logger:
    logger = logging.getLogger("basecaller_ddp_multifolder")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    if not is_main_process(rank):
        logger.addHandler(logging.NullHandler())
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# -------------------- optimizer/scheduler --------------------

def build_adamw_with_no_decay(named_params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    decay_params, no_decay_params = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if any(k in n for k in no_decay_keywords):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr,
    )


def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr: float,
                    logger: Optional[logging.Logger], rank: int):
    try:
        from transformers import get_cosine_schedule_with_warmup
        if logger is not None and rank == 0:
            logger.info("[Scheduler] Using transformers.get_cosine_schedule_with_warmup")
        sched = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return sched, "hf_cosine_warmup"
    except Exception as e:
        if logger is not None and rank == 0:
            logger.warning(f"[Scheduler] transformers not available ({e}); fallback to CosineAnnealingLR (NO warmup).")
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=min_lr
        )
        return sched, "torch_cosine_no_warmup"


# -------------------- checkpoint helpers --------------------

def get_raw_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: str,
                    model,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    epoch: Optional[int] = None,
                    best_pbma: Optional[float] = None,
                    extra: Optional[Dict[str, Any]] = None):
    raw = get_raw_model(model)
    ckpt = {
        "epoch": epoch,
        "best_pbma": best_pbma,
        "model_state_dict": raw.state_dict(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str,
                    model,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    map_location: str = "cpu",
                    logger: Optional[logging.Logger] = None):
    if logger is not None:
        logger.info(f"[Resume] loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=map_location)

    # state_dict (strip possible module.)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    raw = get_raw_model(model)
    missing, unexpected = raw.load_state_dict(new_state, strict=False)

    if logger is not None:
        logger.info(f"[Resume] load model: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            logger.info(f"[Resume] missing keys (first 20): {missing[:20]}")
        if unexpected:
            logger.info(f"[Resume] unexpected keys (first 20): {unexpected[:20]}")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if logger is not None:
            logger.info("[Resume] optimizer state loaded")

    if scheduler is not None and "scheduler_state_dict" in ckpt and hasattr(scheduler, "load_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if logger is not None:
                logger.info("[Resume] scheduler state loaded")
        except Exception as e:
            if logger is not None:
                logger.warning(f"[Resume] scheduler load failed: {e}")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_pbma = ckpt.get("best_pbma", None)
    return start_epoch, best_pbma


# -------------------- train/eval --------------------

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    scheduler,
    device,
    rank: int,
    log_interval: int,
    use_wandb: bool,
    ctc_crf_blank_score: float,
):
    model.train()
    total_loss, n_batches = 0.0, 0

    it = tqdm(enumerate(data_loader, start=1), total=len(data_loader),
              disable=not is_main_process(rank), desc="[train]")
    for step, batch in it:
        input_ids = batch["input_ids"].to(device)
        input_lengths = resolve_input_lengths(
            input_ids,
            attention_mask=batch.get("attention_mask"),
            input_lengths=batch.get("input_lengths"),
        )

        target_labels = batch["target_labels"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        logits_btc = model(input_ids, attention_mask=attention_mask)


        # logits_btc = model(input_ids)              # [B,T,C]
        logits_tbc = logits_btc.transpose(0, 1)    # [T,B,C]

        loss = ctc_crf_loss(
            logits_tbc,
            target_labels,
            input_lengths,
            target_lengths,
            blank_idx=BLANK_IDX,
            pad_blank=False,
            blank_score=float(ctc_crf_blank_score),
        )
        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1

        if is_main_process(rank) and (step % log_interval == 0):
            lr = optimizer.param_groups[0]["lr"]
            msg = f"[Train] step={step}/{len(data_loader)} loss={loss.item():.4f} lr={lr:.6g}"
            print(msg)
            if use_wandb and wandb is not None:
                wandb.log({"train/loss": float(loss.item()), "lr": float(lr), "step": step})

    avg = total_loss / max(n_batches, 1)
    avg = float(reduce_mean(torch.tensor(avg, device=device)).item())
    return avg


@torch.no_grad()
def eval_one_epoch(
    model,
    data_loader,
    device,
    rank: int,
    split_name: str,
    ctc_crf_blank_score: float,
    acc_balanced: bool,
    acc_min_coverage: float,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    total_acc, n_acc = 0.0, 0
    total_cov, n_cov = 0.0, 0
    blank_ratios: List[float] = []
    nonzero_lengths: List[float] = []

    it = tqdm(data_loader, total=len(data_loader),
              disable=not is_main_process(rank), desc=f"[{split_name}]")
    for batch in it:
        input_ids = batch["input_ids"].to(device)
        input_lengths = resolve_input_lengths(
            input_ids,
            attention_mask=batch.get("attention_mask"),
            input_lengths=batch.get("input_lengths"),
        )

        target_labels = batch["target_labels"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        logits_btc = model(input_ids, attention_mask=attention_mask)

        # logits_btc = model(input_ids)              # [B,T,C]
        logits_tbc = logits_btc.transpose(0, 1)    # [T,B,C]

        loss = ctc_crf_loss(
            logits_tbc,
            target_labels,
            input_lengths,
            target_lengths,
            blank_idx=BLANK_IDX,
            pad_blank=False,
            blank_score=float(ctc_crf_blank_score),
        )
        total_loss += float(loss.item())
        n_batches += 1
        pred_seqs = koi_beam_search_decode(
            logits_tbc,
            blank_score=float(ctc_crf_blank_score),
            input_lengths=input_lengths,
        )
        acc = batch_bonito_accuracy(
            pred_seqs,
            batch["target_seqs"],
            balanced=acc_balanced,
            min_coverage=acc_min_coverage,
        )
        total_acc += float(acc)
        n_acc += 1
        input_len_list = input_lengths.detach().cpu().tolist()
        for p_ids, r_ids, input_len in zip(pred_seqs, batch["target_seqs"], input_len_list):
            seq_len = len(p_ids)
            zero_count = sum(1 for token in p_ids if token == BLANK_IDX)
            blank_ratio = zero_count / max(seq_len, 1)
            blank_ratios.append(blank_ratio)
            nonzero_len = max(float(input_len) * (1.0 - blank_ratio), 0.0)
            nonzero_lengths.append(nonzero_len)
            ref_len = max(len(r_ids), 1)
            total_cov += nonzero_len / ref_len
            n_cov += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_acc, 1)
    avg_cov = total_cov / max(n_cov, 1)
    avg_blank = float(np.mean(blank_ratios)) if blank_ratios else 0.0
    avg_nonzero_len = float(np.mean(nonzero_lengths)) if nonzero_lengths else 0.0

    avg_loss = float(reduce_mean(torch.tensor(avg_loss, device=device)).item())
    avg_acc = float(reduce_mean(torch.tensor(avg_acc, device=device)).item())
    avg_cov = float(reduce_mean(torch.tensor(avg_cov, device=device)).item())
    avg_blank = float(reduce_mean(torch.tensor(avg_blank, device=device)).item())
    avg_nonzero_len = float(reduce_mean(torch.tensor(avg_nonzero_len, device=device)).item())
    return avg_loss, avg_acc, avg_cov, avg_blank, avg_nonzero_len


# -------------------- pretrained loader (keep) --------------------

def load_pretrained_weights(model, ckpt_path: str, strict: bool = False,
                            key: str | None = None, logger: Optional[logging.Logger] = None):
    if ckpt_path is None:
        return

    if logger is not None:
        logger.info(f"[Pretrained] loading from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = None
    if isinstance(ckpt, dict):
        if key is not None and key in ckpt:
            state = ckpt[key]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            if all(isinstance(k, str) for k in ckpt.keys()):
                state = ckpt
    else:
        state = ckpt

    if state is None:
        raise ValueError(f"Cannot find a state_dict in checkpoint: {ckpt_path}. Try --pretrained_key.")

    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    target = get_raw_model(model)
    missing, unexpected = target.load_state_dict(new_state, strict=strict)

    if logger is not None:
        logger.info(f"[Pretrained] strict={strict} missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            logger.info(f"[Pretrained] missing keys (first 20): {missing[:20]}")
        if unexpected:
            logger.info(f"[Pretrained] unexpected keys (first 20): {unexpected[:20]}")


# -------------------- args --------------------

def _parse_folder_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--jsonl_paths", type=str, default=None,
                   help="Comma-separated .jsonl.gz files or folders (uses text/bases fields).")
    p.add_argument("--train_jsonl_paths", type=str, default=None,
                   help="Comma-separated .jsonl.gz files or folders for training set (skip auto split).")
    p.add_argument("--val_jsonl_paths", type=str, default=None,
                   help="Comma-separated .jsonl.gz files or folders for validation set (skip auto split).")
    p.add_argument("--test_jsonl_paths", type=str, default=None,
                   help="Comma-separated .jsonl.gz files or folders for test set (skip auto split).")
    p.add_argument("--npy_paths", type=str, default=None,
                   help="Comma-separated folders or tokens_*.npy/reference_*.npy files (uses token/reference pairs).")
    p.add_argument("--train_npy_paths", type=str, default=None,
                   help="Comma-separated folders or tokens_*.npy/reference_*.npy files for training set.")
    p.add_argument("--val_npy_paths", type=str, default=None,
                   help="Comma-separated folders or tokens_*.npy/reference_*.npy files for validation set.")
    p.add_argument("--test_npy_paths", type=str, default=None,
                   help="Comma-separated folders or tokens_*.npy/reference_*.npy files for test set.")
    p.add_argument("--group_by", type=str, default="folder", choices=["folder", "file"])
    p.add_argument("--recursive", action="store_true",
                   help="Scan subfolders for .jsonl.gz or tokens/reference .npy inputs.")

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)

    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--hidden-layer",type=int,default=-1, help="Which backbone hidden layer to use for CTC (-1=last, -2=second last, etc.)")


    p.add_argument("--pretrained_ckpt", type=str, default=None,
                   help="Optional path to a .pt/.pth checkpoint to load into the model (state_dict or dict containing one).")
    p.add_argument("--pretrained_strict", action="store_true",
                   help="Use strict=True when loading pretrained weights (default strict=False).")
    p.add_argument("--pretrained_key", type=str, default=None,
                   help="If checkpoint is a dict, optionally choose the key that contains the model state_dict (e.g. model/state_dict).")

    # ✅ resume from a full checkpoint (model+optim+sched+epoch)
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Path to ckpt_last.pt to resume training (restores model/optimizer/scheduler/epoch/best_acc).")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--quick_train", action="store_true",
                   help="Quick train mode: freeze backbone and force ctc_crf_state_len=5 (5120 classes).")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--min_lr", type=float, default=1e-5)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="./outputs_ddp")

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="basecaller")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    p.add_argument("--find_unused_parameters", action="store_true",
                   help="Enable DDP unused parameter detection (fix reduction error).")

    # ✅ save frequency controls (minimal)
    p.add_argument("--save_every", type=int, default=1,
                   help="Save ckpt_last.pt every N epochs (default 1).")
    p.add_argument("--save_best", action="store_true",
                   help="Save ckpt_best.pt based on best val_acc (default off unless provided).")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze backbone parameters; train only base_head.")
    p.add_argument("--reset_backbone_weights", action="store_true",
                   help="Reinitialize backbone weights for ablation (ignores pretrained backbone init).")
    p.add_argument("--unfreeze_last_n_layers", type=int, default=0,
                   help="Unfreeze only the last N backbone layers (default: 0).")
    p.add_argument("--unfreeze_layer_start", type=int, default=None,
                   help="Unfreeze backbone layers in [start, end). Optional finer control.")
    p.add_argument("--unfreeze_layer_end", type=int, default=None,
                   help="Unfreeze backbone layers in [start, end). Optional finer control.")

    p.add_argument("--head_output_activation", choices=["tanh", "relu"], default=None,
                   help="Optional activation applied to head output logits.")
    p.add_argument("--head_output_scale", type=float, default=None,
                   help="Optional scalar applied to head output logits (after activation).")

    p.add_argument("--acc_balanced", action="store_true",
                   help="Use Bonito balanced accuracy: (match - ins) / (match + mismatch + del).")
    p.add_argument("--acc_min_coverage", type=float, default=0.0,
                   help="Minimum reference coverage required to count a read for accuracy.")
    p.add_argument("--ctc_crf_state_len", type=int, default=5,
                   help="State length for Bonito CTC-CRF (used to set output classes).")
    p.add_argument("--ctc_crf_blank_score", type=float, default=2.0,
                   help="Fixed blank score for CTC-CRF (blank is not trained).")


    return p.parse_args()


# -------------------- main --------------------

def apply_quick_train_overrides(args) -> None:
    if not args.quick_train:
        return
    args.freeze_backbone = True
    args.ctc_crf_state_len = 5


def main():
    args = parse_args()
    apply_quick_train_overrides(args)
    rank, world_size, local_rank, device, ddp_enabled = init_distributed()

    seed_everything(args.seed + rank)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, "train.log"), rank)
    if is_main_process(rank):
        logger.info(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}")
        logger.info(f"[Args] {vars(args)}")
        if args.quick_train:
            logger.info("[Quick Train] enabled: freeze_backbone=True, ctc_crf_state_len=5 (5120 classes)")

    # ---- model (先建模型拿 tokenizer，保持原数据逻辑) ----
    import os as _os
    _os.environ["CTC_CRF_STATE_LEN"] = str(args.ctc_crf_state_len)
    n_base = len(ID2BASE) - 1
    if n_base <= 0:
        raise ValueError("CTC-CRF alphabet must include at least one non-blank base.")
    # CTC-CRF head emits full (blank+base) scores; blank score is overwritten during forward.
    num_classes = (n_base ** args.ctc_crf_state_len) * (n_base + 1)

    base_model = BasecallModel(
        model_path=args.model_name_or_path,
        num_classes=num_classes if num_classes is not None else None,
        hidden_layer=args.hidden_layer,
        freeze_backbone=bool(args.freeze_backbone),
        reset_backbone_weights=bool(args.reset_backbone_weights),
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        unfreeze_layer_start=args.unfreeze_layer_start,
        unfreeze_layer_end=args.unfreeze_layer_end,
        head_output_activation=args.head_output_activation,
        head_output_scale=args.head_output_scale,
        head_crf_blank_score=float(args.ctc_crf_blank_score),
        head_crf_n_base=n_base,
        head_crf_state_len=int(args.ctc_crf_state_len),
    ).to(device)

    model = base_model

    if ddp_enabled:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.find_unused_parameters),
        )

    tokenizer = model.module.tokenizer if hasattr(model, "module") else model.tokenizer

    # ---- scan + split ----
    train_jsonl_paths = _parse_folder_list(args.train_jsonl_paths)
    val_jsonl_paths = _parse_folder_list(args.val_jsonl_paths)
    test_jsonl_paths = _parse_folder_list(args.test_jsonl_paths)
    train_npy_paths = _parse_folder_list(args.train_npy_paths)
    val_npy_paths = _parse_folder_list(args.val_npy_paths)
    test_npy_paths = _parse_folder_list(args.test_npy_paths)

    using_jsonl = bool(args.jsonl_paths or train_jsonl_paths or val_jsonl_paths or test_jsonl_paths)
    using_npy = bool(args.npy_paths or train_npy_paths or val_npy_paths or test_npy_paths)
    if using_jsonl and using_npy:
        raise ValueError("Do not mix jsonl inputs with tokens/reference npy inputs in the same run.")

    if using_npy:
        if train_npy_paths or val_npy_paths or test_npy_paths:
            train_pairs = scan_npy_pairs(train_npy_paths, group_by=args.group_by, recursive=args.recursive)
            val_pairs = scan_npy_pairs(val_npy_paths, group_by=args.group_by, recursive=args.recursive) if val_npy_paths else []
            test_pairs = scan_npy_pairs(test_npy_paths, group_by=args.group_by, recursive=args.recursive) if test_npy_paths else []
        else:
            if not args.npy_paths:
                raise ValueError("Provide --npy_paths or explicit --train_npy_paths/--val_npy_paths/--test_npy_paths.")
            npy_paths = [x.strip() for x in args.npy_paths.split(",") if x.strip()]
            npy_pairs = scan_npy_pairs(npy_paths, group_by=args.group_by, recursive=args.recursive)
            train_pairs, val_pairs, test_pairs = split_npy_pairs_by_group(
                npy_pairs,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.split_seed,
            )

        if is_main_process(rank):
            logger.info(f"[Data] train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} test_pairs={len(test_pairs)}")

        train_dataset = MultiNpySignalRefDataset(train_pairs)
        val_dataset = MultiNpySignalRefDataset(val_pairs) if len(val_pairs) else None
        test_dataset = MultiNpySignalRefDataset(test_pairs) if len(test_pairs) else None
    else:
        if train_jsonl_paths or val_jsonl_paths or test_jsonl_paths:
            train_files = scan_jsonl_files(train_jsonl_paths, group_by=args.group_by, recursive=args.recursive)
            val_files = scan_jsonl_files(val_jsonl_paths, group_by=args.group_by, recursive=args.recursive) if val_jsonl_paths else []
            test_files = scan_jsonl_files(test_jsonl_paths, group_by=args.group_by, recursive=args.recursive) if test_jsonl_paths else []
        else:
            if not args.jsonl_paths:
                raise ValueError("Provide --jsonl_paths or explicit --train_jsonl_paths/--val_jsonl_paths/--test_jsonl_paths.")
            jsonl_paths = [x.strip() for x in args.jsonl_paths.split(",") if x.strip()]
            jsonl_files = scan_jsonl_files(jsonl_paths, group_by=args.group_by, recursive=args.recursive)
            train_files, val_files, test_files = split_jsonl_files_by_group(
                jsonl_files,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.split_seed,
            )

        if is_main_process(rank):
            logger.info(f"[Data] train_files={len(train_files)} val_files={len(val_files)} test_files={len(test_files)}")

        train_dataset = MultiJsonlSignalRefDataset(train_files)
        val_dataset = MultiJsonlSignalRefDataset(val_files) if len(val_files) else None
        test_dataset = MultiJsonlSignalRefDataset(test_files) if len(test_files) else None

    collate_fn = create_collate_fn(tokenizer)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp_enabled else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if (ddp_enabled and val_dataset is not None) else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if (ddp_enabled and test_dataset is not None) else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=collate_fn,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=collate_fn,
        )

    # ---- optimizer/scheduler/loss ----
    optimizer = build_adamw_with_no_decay(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler, sched_name = build_scheduler(optimizer, total_steps, warmup_steps, args.min_lr, logger, rank)

    if is_main_process(rank):
        logger.info(f"[Scheduler] steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
                    f"warmup_steps={warmup_steps} scheduler={sched_name}")

    # ---- optional pretrained weights (only if NOT resuming) ----
    if args.pretrained_ckpt and not args.resume_ckpt:
        load_pretrained_weights(
            model,
            args.pretrained_ckpt,
            strict=bool(args.pretrained_strict),
            key=args.pretrained_key,
            logger=logger,
        )

    # ---- wandb ----
    use_wandb = bool(args.use_wandb and wandb is not None and is_main_process(rank))
    if args.use_wandb and wandb is None and is_main_process(rank):
        logger.warning("[wandb] wandb not installed; pip install wandb")

    if use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))

    # ---- resume (after model+optim+sched created) ----
    start_epoch = 1
    best_pbma = -1.0
    if args.resume_ckpt:
        se, bp = load_checkpoint(
            args.resume_ckpt,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
            logger=logger if is_main_process(rank) else None,
        )
        start_epoch = se
        if bp is not None:
            try:
                best_pbma = float(bp)
            except Exception:
                pass
        if is_main_process(rank):
            logger.info(f"[Resume] start_epoch={start_epoch}, best_acc={best_pbma}")

    # ---- loop ----
    train_losses, val_losses, val_accs = [], [], []

    # if resuming mid-run, you may want to pad arrays; we keep it simple:
    # curves will reflect only epochs run in this session.
    for epoch in range(start_epoch, args.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            rank,
            args.log_interval,
            use_wandb,
            args.ctc_crf_blank_score,
        )
        train_losses.append(tr_loss)

        if is_main_process(rank):
            logger.info(f"[Train] epoch={epoch} avg_loss={tr_loss:.4f}")

        val_loss, val_acc = None, None
        if val_loader is not None:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            val_loss, val_acc, val_cov, val_blank, val_nonzero_len = eval_one_epoch(
                model,
                val_loader,
                device,
                rank,
                "val",
                args.ctc_crf_blank_score,
                args.acc_balanced,
                args.acc_min_coverage,
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if is_main_process(rank):
                logger.info(
                    f"[Val] epoch={epoch} loss={val_loss:.4f} acc={val_acc:.4f} "
                    f"coverage={val_cov:.4f} blank={val_blank:.4f} nonzero_len={val_nonzero_len:.2f}"
                )
                if use_wandb and wandb is not None:
                    wandb.log(
                        {
                            "val/loss": float(val_loss),
                            "val/acc": float(val_acc),
                            "val/coverage": float(val_cov),
                            "val/blank": float(val_blank),
                            "val/nonzero_len": float(val_nonzero_len),
                            "epoch": epoch,
                        }
                    )

        # ---- checkpoint save ----
        if is_main_process(rank) and (epoch % max(args.save_every, 1) == 0):
            last_path = os.path.join(args.output_dir, "ckpt_last.pt")
            save_checkpoint(
                last_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_pbma=best_pbma,
                extra={"train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc},
            )
            logger.info(f"[CKPT] saved {last_path}")

        if is_main_process(rank) and args.save_best and (val_acc is not None):
            if float(val_acc) > float(best_pbma):
                best_pbma = float(val_acc)
                best_path = os.path.join(args.output_dir, "ckpt_best.pt")
                save_checkpoint(
                    best_path,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_pbma=best_pbma,
                    extra={"train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc},
                )
                logger.info(f"[CKPT] new best acc={best_pbma:.4f} @ epoch={epoch}, saved {best_path}")

        if use_wandb and wandb is not None and is_main_process(rank):
            wandb.log({"epoch": epoch, "train/epoch_loss": float(tr_loss)})

    # ---- test ----
    if test_loader is not None:
        test_loss, test_acc, test_cov, test_blank, test_nonzero_len = eval_one_epoch(
            model,
            test_loader,
            device,
            rank,
            "test",
            args.ctc_crf_blank_score,
            args.acc_balanced,
            args.acc_min_coverage,
        )
        if is_main_process(rank):
            logger.info(
                f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} "
                f"coverage={test_cov:.4f} blank={test_blank:.4f} nonzero_len={test_nonzero_len:.2f}"
            )
            if use_wandb and wandb is not None:
                wandb.log(
                    {
                        "test/loss": float(test_loss),
                        "test/acc": float(test_acc),
                        "test/coverage": float(test_cov),
                        "test/blank": float(test_blank),
                        "test/nonzero_len": float(test_nonzero_len),
                    }
                )

    # ---- final save curves/csv ----
    if is_main_process(rank):
        try:
            if val_losses and val_accs:
                plot_curves(train_losses, val_losses, val_accs, save_path=os.path.join(args.output_dir, "curves.png"))
                save_metrics_csv(train_losses, val_losses, val_accs, os.path.join(args.output_dir, "metrics.csv"))
        except Exception as e:
            logger.warning(f"[Final Plot/CSV] failed: {e}")

    # ---- log a final alignment heatmap to wandb ----
    if use_wandb and wandb is not None and is_main_process(rank):
        loader = test_loader if test_loader is not None else val_loader
        if loader is not None:
            batch = next(iter(loader))
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with torch.no_grad():
                logits_btc = model(input_ids, attention_mask=attention_mask)
                logits_tbc = logits_btc.transpose(0, 1)
            input_lengths = resolve_input_lengths(
                input_ids,
                attention_mask=attention_mask,
                input_lengths=batch.get("input_lengths"),
            )
            pred_seqs = koi_beam_search_decode(
                logits_tbc,
                blank_score=float(args.ctc_crf_blank_score),
                input_lengths=input_lengths,
            )
            ref_seqs = batch["target_seqs"]
            fig = plot_alignment_heatmap(pred_seqs, ref_seqs, max_reads=32, max_len=80)
            wandb.log({"final/base_alignment": wandb.Image(fig)})
            plt.close(fig)

    if use_wandb and wandb is not None:
        wandb.finish()

    cleanup_distributed(ddp_enabled)


if __name__ == "__main__":
    main()
