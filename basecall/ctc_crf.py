# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import List, Tuple

import torch


def _load_k2():
    try:
        import k2  # type: ignore
    except Exception as exc:
        raise ImportError(
            "CTC-CRF requires k2. Install it (e.g. pip install k2) and ensure it "
            "matches your PyTorch/CUDA version."
        ) from exc
    return k2


def _load_koi():
    try:
        import koi  # type: ignore
    except Exception as exc:
        raise ImportError(
            "CTC-CRF (koi) requires the ont-koi package. Install it and ensure it "
            "matches your PyTorch/CUDA version."
        ) from exc
    return koi


def _load_backend() -> Tuple[str, object]:
    backend = os.environ.get("CTC_CRF_BACKEND", "k2").lower()
    if backend == "koi":
        return "koi", _load_koi()
    return "k2", _load_k2()


def ctc_crf_loss(
    logits_tbc: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_idx: int = 0,
) -> torch.Tensor:
    backend, lib = _load_backend()
    log_probs = torch.log_softmax(logits_tbc.transpose(0, 1), dim=-1)  # [B,T,C]
    device = log_probs.device
    batch_size = log_probs.size(0)
    targets = targets.to(device=device)

    if backend == "k2":
        supervisions = torch.stack(
            [
                torch.arange(batch_size, device=device, dtype=torch.int32),
                torch.zeros(batch_size, device=device, dtype=torch.int32),
                input_lengths.to(device=device, dtype=torch.int32),
            ],
            dim=1,
        )
        return lib.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            supervisions=supervisions,
            blank=blank_idx,
            reduction="mean",
        )

    if not hasattr(lib, "ctc_loss"):
        raise AttributeError("koi backend missing ctc_loss; provide a compatible koi build.")
    return lib.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths.to(device=device),
        target_lengths=target_lengths.to(device=device),
        blank=blank_idx,
    )


def decode(logits_tbc: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    backend, lib = _load_backend()
    log_probs = torch.log_softmax(logits_tbc.transpose(0, 1), dim=-1)  # [B,T,C]
    if backend == "k2":
        paths = lib.ctc_greedy_search(log_probs, blank=blank_idx)
        return paths.tolist()
    if not hasattr(lib, "ctc_greedy_search"):
        raise AttributeError("koi backend missing ctc_greedy_search; provide a compatible koi build.")
    paths = lib.ctc_greedy_search(log_probs, blank=blank_idx)
    return paths.tolist()
