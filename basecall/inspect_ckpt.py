# -*- coding: utf-8 -*-
"""
Inspect checkpoint head config and print suggested eval/infer flags.

Example:
  python inspect_ckpt.py --ckpt ckpt_best.pt
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import torch


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        return state
    raise ValueError(f"Unsupported checkpoint format: {path}")


def infer_head_layers(state_dict: Dict[str, torch.Tensor], default_layers: int = 2) -> int:
    indices = set()
    for key in state_dict.keys():
        if key.startswith("base_head.blocks."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                indices.add(int(parts[2]))
    if not indices:
        return default_layers
    return max(indices) + 1


def infer_kernel_size(state_dict: Dict[str, torch.Tensor], default_kernel: int = 5) -> int:
    weight = state_dict.get("base_head.blocks.0.0.weight")
    if isinstance(weight, torch.Tensor) and weight.dim() == 3:
        return int(weight.shape[-1])
    return default_kernel


def infer_use_pointwise(state_dict: Dict[str, torch.Tensor], default_value: bool = True) -> bool:
    if "base_head.blocks.0.1.weight" in state_dict:
        return True
    if any(key.startswith("base_head.blocks.") and ".1.weight" in key for key in state_dict):
        return True
    return default_value


def infer_transformer_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    indices = set()
    for key in state_dict.keys():
        if key.startswith("base_head.transformer.layers."):
            parts = key.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                indices.add(int(parts[3]))
    if not indices:
        return 0
    return max(indices) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    sd = load_checkpoint_state(args.ckpt)
    head_layers = infer_head_layers(sd)
    head_kernel_size = infer_kernel_size(sd)
    head_use_pointwise = infer_use_pointwise(sd)
    head_transformer_layers = infer_transformer_layers(sd)
    head_use_transformer = head_transformer_layers > 0

    summary = {
        "head_layers": head_layers,
        "head_kernel_size": head_kernel_size,
        "head_use_pointwise": head_use_pointwise,
        "head_use_transformer": head_use_transformer,
        "head_transformer_layers": head_transformer_layers,
    }
    print(json.dumps(summary, indent=2))

    flags = [
        f"--head_layers {head_layers}",
        f"--head_kernel_size {head_kernel_size}",
        f"--head_use_pointwise {str(head_use_pointwise).lower()}",
    ]
    if head_use_transformer:
        flags.append("--head_use_transformer")
        flags.append(f"--head_transformer_layers {head_transformer_layers}")
    else:
        flags.append("--head_disable_transformer")

    print("\nSuggested flags:")
    print(" ".join(flags))


if __name__ == "__main__":
    main()
