# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def ctc_loss(
    logits_tbc: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_idx: int = 0,
) -> torch.Tensor:
    """
    Plain CTC loss on logits with shape [T, B, C].
    """
    log_probs = F.log_softmax(logits_tbc.to(torch.float32), dim=-1)
    targets = targets.to(dtype=torch.long)
    input_lengths = input_lengths.to(dtype=torch.long, device="cpu")
    target_lengths = target_lengths.to(dtype=torch.long, device="cpu")
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_idx,
        reduction="mean",
        zero_infinity=True,
    )


def decode(
    logits_tbc: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    blank_idx: int = 0,
) -> List[List[int]]:
    """
    Bonito-style CTC Viterbi path collapse (beamsize=1 equivalent):
    timestep argmax -> collapse repeats -> remove blank.
    """
    if input_lengths is None:
        lengths = [logits_tbc.shape[0]] * logits_tbc.shape[1]
    else:
        lengths = [min(int(x), logits_tbc.shape[0]) for x in input_lengths.cpu().tolist()]

    pred_tbc = torch.argmax(logits_tbc, dim=-1)  # [T, B]
    decoded: List[List[int]] = []
    for b, length in enumerate(lengths):
        if length <= 0:
            decoded.append([])
            continue
        seq = pred_tbc[:length, b].detach().cpu().tolist()
        out: List[int] = []
        prev = None
        for token in seq:
            token = int(token)
            if token == blank_idx:
                prev = token
                continue
            if token != prev:
                out.append(token)
            prev = token
        decoded.append(out)
    return decoded
