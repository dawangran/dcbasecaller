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

from .utils import infer_head_config_from_state_dict


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        return state
    raise ValueError(f"Unsupported checkpoint format: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    sd = load_checkpoint_state(args.ckpt)
    head_config = infer_head_config_from_state_dict(sd)
    head_layers = head_config["head_layers"]
    head_kernel_size = head_config["head_kernel_size"]
    head_use_pointwise = head_config["head_use_pointwise"]
    head_use_transformer = head_config["head_use_transformer"]
    head_transformer_layers = head_config["head_transformer_layers"]

    summary = {
        "head_layers": head_layers,
        "head_kernel_size": head_kernel_size,
        "head_use_pointwise": head_use_pointwise,
        "head_use_transformer": head_use_transformer,
        "head_transformer_layers": head_transformer_layers,
    }
    print(json.dumps(summary, indent=2))

    flags = []
    if head_layers == 0 and not head_use_transformer:
        flags.append("--head_linear")
    else:
        flags.append(f"--head_layers {head_layers}")
        flags.append(f"--head_kernel_size {head_kernel_size}")
        if not head_use_pointwise:
            flags.append("--head_disable_pointwise")
        if head_use_transformer:
            flags.append("--head_use_transformer")
            flags.append(f"--head_transformer_layers {head_transformer_layers}")

    print("\nSuggested flags:")
    print(" ".join(flags))


if __name__ == "__main__":
    main()
