# -*- coding: utf-8 -*-
"""
Inspect checkpoint head config for the CTC-CRF linear head.

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
    summary = {
        "num_classes": head_config["num_classes"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
