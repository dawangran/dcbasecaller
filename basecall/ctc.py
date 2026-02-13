# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .utils import BASE2ID, ID2BASE


def _ctc_alphabet() -> str:
    max_id = max(ID2BASE.keys())
    return "".join(ID2BASE.get(i, "") for i in range(max_id + 1))


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


def ctc_label_smoothing_loss(
    logits_tbc: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_idx: int = 0,
    weights: Optional[torch.Tensor] = None,
) -> dict:
    """
    Bonito-style CTC + label smoothing loss.
    """
    log_probs = F.log_softmax(logits_tbc.to(torch.float32), dim=-1)
    _, batch_size, num_classes = log_probs.shape

    if weights is None:
        weights = torch.cat(
            [torch.tensor([0.4]), (0.1 / max(num_classes - 1, 1)) * torch.ones(max(num_classes - 1, 1))]
        )[:num_classes]

    # Follow Bonito's behavior: CTC input lengths are full T for each sample.
    full_input_lengths = torch.full(
        size=(batch_size,),
        fill_value=log_probs.shape[0],
        dtype=torch.long,
        device="cpu",
    )
    targets = targets.to(dtype=torch.long)
    target_lengths = target_lengths.to(dtype=torch.long, device="cpu")

    base_loss = F.ctc_loss(
        log_probs,
        targets,
        full_input_lengths,
        target_lengths,
        blank=blank_idx,
        reduction="mean",
        zero_infinity=True,
    )
    label_smooth_loss = -((log_probs * weights.to(log_probs.device)).mean())
    return {
        "total_loss": base_loss + label_smooth_loss,
        "loss": base_loss,
        "label_smooth_loss": label_smooth_loss,
    }


def decode(
    logits_tbc: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    blank_idx: int = 0,
    beamsize: int = 1,
    threshold: float = 1e-3,
    qscores: bool = False,
) -> List[List[int]]:
    """
    Bonito-style decode via fast_ctc_decode.
    Returns decoded token ids for each batch item.
    """
    from fast_ctc_decode import beam_search, viterbi_search  # type: ignore

    if input_lengths is None:
        lengths = [logits_tbc.shape[0]] * logits_tbc.shape[1]
    else:
        lengths = [min(int(x), logits_tbc.shape[0]) for x in input_lengths.cpu().tolist()]

    alphabet = _ctc_alphabet()
    posteriors = F.log_softmax(logits_tbc.to(torch.float32), dim=-1).exp()

    out_ids: List[List[int]] = []
    for b, length in enumerate(lengths):
        if length <= 0:
            out_ids.append([])
            continue

        probs = posteriors[:length, b, :].detach().cpu().numpy().astype(np.float32)
        if beamsize == 1 or qscores:
            seq, _path = viterbi_search(probs, alphabet)
        else:
            seq, _path = beam_search(
                probs,
                alphabet,
                beam_size=beamsize,
                beam_cut_threshold=threshold,
            )

        out_ids.append([BASE2ID.get(ch, blank_idx) for ch in str(seq)])

    return out_ids
