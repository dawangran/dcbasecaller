# -*- coding: utf-8 -*-
import random
from typing import Dict, Any

import numpy as np
import torch

# 标签定义
LABELS = ["N", "A", "C", "G", "T"]  # 0,1,2,3,4
NUM_CLASSES = len(LABELS)
BLANK_IDX = 0  # CTC blank

ID2BASE = {
    0: "N",
    1: "A",
    2: "C",
    3: "G",
    4: "T",
}

BASE2ID = {b: i for i, b in ID2BASE.items()}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_input_lengths(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    input_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if input_lengths is not None:
        lengths = input_lengths.to(device=input_ids.device, dtype=torch.long)
    elif attention_mask is not None:
        lengths = attention_mask.sum(dim=1).to(device=input_ids.device, dtype=torch.long)
    else:
        lengths = torch.full(
            (input_ids.size(0),),
            input_ids.size(1),
            dtype=torch.long,
            device=input_ids.device,
        )

    if lengths.numel() == 0:
        return lengths

    if torch.any(lengths < 0):
        raise ValueError("input_lengths must be non-negative.")

    max_len = int(input_ids.size(1))
    if torch.any(lengths > max_len):
        lengths = torch.clamp(lengths, max=max_len)
    return lengths


def infer_head_config_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    default_layers: int = 2,
    default_kernel: int = 5,
    default_use_pointwise: bool = True,
    default_num_classes: int | None = None,
) -> Dict[str, Any]:
    if default_num_classes is None:
        default_num_classes = len(ID2BASE)

    def _infer_head_layers() -> int:
        indices = set()
        for key in state_dict.keys():
            if key.startswith("base_head.blocks."):
                parts = key.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    indices.add(int(parts[2]))
        if not indices:
            return 0
        return max(indices) + 1

    def _infer_kernel_size() -> int:
        weight = state_dict.get("base_head.blocks.0.0.weight")
        if isinstance(weight, torch.Tensor) and weight.dim() == 3:
            return int(weight.shape[-1])
        return default_kernel

    def _infer_use_pointwise() -> bool:
        if not any(key.startswith("base_head.blocks.") for key in state_dict.keys()):
            return False
        if "base_head.blocks.0.1.weight" in state_dict:
            return True
        if any(key.startswith("base_head.blocks.") and ".1.weight" in key for key in state_dict):
            return True
        return bool(default_use_pointwise)

    def _infer_num_classes() -> int:
        weight = state_dict.get("base_head.proj.weight")
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            return int(weight.shape[0])
        return int(default_num_classes)

    def _infer_transformer_layers() -> int:
        indices = set()
        for key in state_dict.keys():
            if key.startswith("base_head.transformer.layers."):
                parts = key.split(".")
                if len(parts) > 3 and parts[3].isdigit():
                    indices.add(int(parts[3]))
        if not indices:
            return 0
        return max(indices) + 1

    inferred_transformer_layers = _infer_transformer_layers()
    return {
        "head_layers": int(_infer_head_layers()),
        "head_kernel_size": int(_infer_kernel_size()),
        "head_use_pointwise": bool(_infer_use_pointwise()),
        "head_use_transformer": inferred_transformer_layers > 0,
        "head_transformer_layers": int(inferred_transformer_layers),
        "head_transformer_heads": 4,
        "num_classes": int(_infer_num_classes()),
    }
