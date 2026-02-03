# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional, Tuple
import csv
import os
import re
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import edlib

from .utils import BLANK_IDX, ID2BASE, BASE2ID

_PARASAIL_AVAILABLE = importlib.util.find_spec("parasail") is not None
if _PARASAIL_AVAILABLE:
    import parasail  # type: ignore

_SPLIT_CIGAR = re.compile(r"(\d+)([=XID])")


def koi_beam_search_decode(
    logits_tbc: torch.Tensor,
    beam_width: int = 32,
    beam_cut: float = 100.0,
    scale: float = 1.0,
    offset: float = 0.0,
    blank_score: float = 2.0,
    reverse: bool = False,
    input_lengths: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    try:
        from koi.decode import beam_search, to_str  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Koi beam_search decoder requested but ont-koi is not installed. "
            "Install ont-koi and ensure it matches your PyTorch/CUDA version."
        ) from exc

    if input_lengths is None:
        lengths = [logits_tbc.shape[0]] * logits_tbc.shape[1]
    else:
        lengths = [min(int(x), logits_tbc.shape[0]) for x in input_lengths.cpu().tolist()]
    decoded: List[List[int]] = []
    for b, length in enumerate(lengths):
        if length <= 0:
            decoded.append([])
            continue
        scores = logits_tbc[:length, b : b + 1, :]
        if scores.is_cuda:
            scores = scores.half()
        sequence, _qstring, _moves = beam_search(
            scores,
            beam_width=beam_width,
            beam_cut=beam_cut,
            scale=scale,
            offset=offset,
            blank_score=blank_score,
        )
        if reverse:
            sequence = sequence[::-1]
        seq_str = sequence if isinstance(sequence, str) else to_str(sequence)
        decoded.append([BASE2ID.get(base, BLANK_IDX) for base in seq_str])
    return decoded




def ctc_crf_loss(
    logits_tbc: torch.Tensor,
    target_labels: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_idx: int = BLANK_IDX,
    pad_blank: bool = False,
    blank_score: float = 0.0,
) -> torch.Tensor:
    try:
        from . import ctc_crf  # type: ignore
    except Exception as exc:
        raise ImportError(
            "ctc-crf loss requested but ctc_crf is not installed. "
            "Install a CTC-CRF implementation and expose a ctc_crf_loss() API."
        ) from exc

    if not hasattr(ctc_crf, "ctc_crf_loss"):
        raise ImportError(
            "ctc_crf.ctc_crf_loss not found. Provide a CTC-CRF library with "
            "ctc_crf_loss(logits_tbc, targets, input_lengths, target_lengths, blank_idx)."
        )

    if pad_blank:
        logits_tbc = _pad_ctc_crf_blank(logits_tbc, blank_score)

    return ctc_crf.ctc_crf_loss(
        logits_tbc, target_labels, input_lengths, target_lengths, blank_idx=blank_idx
    )


def _pad_ctc_crf_blank(logits_tbc: torch.Tensor, blank_score: float) -> torch.Tensor:
    state_len = int(os.environ.get("CTC_CRF_STATE_LEN", "5"))
    n_base = len(ID2BASE) - 1
    if n_base <= 0:
        raise ValueError("CTC-CRF alphabet must include at least one non-blank base.")
    no_blank_dim = (n_base ** state_len) * n_base
    full_dim = (n_base + 1) * (n_base ** state_len)
    if logits_tbc.size(-1) == full_dim:
        t_len, batch, _ = logits_tbc.shape
        reshaped = logits_tbc.view(t_len, batch, n_base ** state_len, n_base + 1)
        reshaped = reshaped.clone()
        reshaped[..., 0] = float(blank_score)
        return reshaped.view(t_len, batch, -1)
    if logits_tbc.size(-1) != no_blank_dim:
        raise ValueError(
            f"CTC-CRF logits dim mismatch: got {logits_tbc.size(-1)}, "
            f"expected {no_blank_dim} (no-blank) or {full_dim} (full)."
        )
    t_len, batch, _ = logits_tbc.shape
    reshaped = logits_tbc.view(t_len, batch, no_blank_dim // n_base, n_base)
    padded = F.pad(reshaped, (1, 0), value=float(blank_score))
    return padded.view(t_len, batch, -1)


# ---------------- PBMA ----------------

def _parse_cigar(cigar: str) -> List[str]:
    """
    将 CIGAR 字符串 (如 '10=1X3=2I1=1D') 展开成操作序列:
    返回类似 ['=', '=', ..., 'X', '=', ..., 'I', ...]
    """
    ops = []
    num = ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            n = int(num) if num else 1
            ops.extend([ch] * n)
            num = ""
    return ops


def _alignment_counts(pred_seq: str, ref_seq: str) -> Dict[str, int]:
    """
    统计对齐结果中的 '=' 'X' 'I' 'D' 数量。
    返回 dict: match, mismatch, ins, del
    """
    counts = {"match": 0, "mismatch": 0, "ins": 0, "del": 0}
    if not ref_seq and not pred_seq:
        return counts

    result = edlib.align(pred_seq, ref_seq, task="path")
    cigar = result.get("cigar", None)
    if not cigar:
        return counts

    ops = _parse_cigar(cigar)
    for op in ops:
        if op in {"=", "M"}:
            counts["match"] += 1
        elif op == "X":
            counts["mismatch"] += 1
        elif op == "D":
            counts["del"] += 1
        elif op == "I":
            counts["ins"] += 1
    return counts


def _parasail_to_sam(result, seq: str) -> tuple[int, str]:
    cigstr = result.cigar.decode.decode()
    first = _SPLIT_CIGAR.search(cigstr)
    if first is None:
        return result.cigar.beg_ref, cigstr

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = "" if cliplen == 0 else f"{cliplen}S"
    if first_op == "I":
        pre = f"{int(first_count) + cliplen}S"
    elif first_op == "D":
        pre = clip
        rstart = int(first_count)
    else:
        pre = f"{clip}{prefix}"

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = f"{end_clip}S" if end_clip > 0 else ""
    new_cigstr = "".join((pre, mid, suf))
    return rstart, new_cigstr


def _alignment_counts_parasail(pred_seq: str, ref_seq: str) -> tuple[Dict[str, int], float]:
    alignment = parasail.sw_trace_striped_32(pred_seq, ref_seq, 8, 4, parasail.dnafull)
    _, cigar = _parasail_to_sam(alignment, pred_seq)
    counts = {"match": 0, "mismatch": 0, "ins": 0, "del": 0}
    for count, op in _SPLIT_CIGAR.findall(cigar):
        if op == "=":
            counts["match"] += int(count)
        elif op == "X":
            counts["mismatch"] += int(count)
        elif op == "D":
            counts["del"] += int(count)
        elif op == "I":
            counts["ins"] += int(count)
    r_coverage = len(alignment.traceback.ref) / max(len(ref_seq), 1)
    return counts, r_coverage


def _pbma_counts(pred_seq: str, ref_seq: str) -> Tuple[int, int]:
    """
    返回 (match, ref_len)，其中 ref_len = match + mismatch + del.
    以参考序列为基准计算 PBMA，忽略插入项对分母的影响。
    """
    if not ref_seq:
        return 0, 0
    if not pred_seq:
        return 0, len(ref_seq)

    counts = _alignment_counts(pred_seq, ref_seq)
    ref_len = counts["match"] + counts["mismatch"] + counts["del"]
    return counts["match"], ref_len


def cal_per_base_match_accuracy(pred_seq: str, ref_seq: str) -> float:
    """PBMA = match / ref_len (ref_len = match + mismatch + del)"""
    match, ref_len = _pbma_counts(pred_seq, ref_seq)
    return match / ref_len if ref_len > 0 else 0.0


def _ids_to_bases(ids: List[int], drop_blank: bool = True) -> str:
    bases: List[str] = []
    for i in ids:
        if drop_blank and i == BLANK_IDX:
            continue
        base = ID2BASE.get(i, "")
        if base:
            bases.append(base)
    return "".join(bases)


def cal_bonito_accuracy(
    pred_seq: str,
    ref_seq: str,
    balanced: bool = False,
    min_coverage: float = 0.0,
) -> float:
    """
    Bonito-style accuracy:
      - default: match / (match + ins + mismatch + del)
      - balanced: (match - ins) / (match + mismatch + del)
    结果返回百分比（0-100）。
    """
    if not pred_seq or not ref_seq:
        return 0.0

    if _PARASAIL_AVAILABLE:
        counts, r_coverage = _alignment_counts_parasail(pred_seq, ref_seq)
    else:
        counts = _alignment_counts(pred_seq, ref_seq)
        ref_len = counts["match"] + counts["mismatch"] + counts["del"]
        if ref_len <= 0:
            return 0.0
        r_coverage = ref_len / max(len(ref_seq), 1)
    if r_coverage < min_coverage:
        return 0.0

    if balanced:
        denom = counts["match"] + counts["mismatch"] + counts["del"]
        score = (counts["match"] - counts["ins"]) / denom if denom > 0 else 0.0
    else:
        denom = counts["match"] + counts["mismatch"] + counts["del"] + counts["ins"]
        score = counts["match"] / denom if denom > 0 else 0.0
    return float(score * 100.0)


def batch_pbma(
    pred_seqs: List[List[int]],
    ref_seqs: List[List[int]],
) -> float:
    """
    计算一个 batch 的 PBMA（按参考长度加权）
    pred_seqs/ref_seqs: list[list[int]]，标签 1..4
    """
    assert len(pred_seqs) == len(ref_seqs)
    total_match = 0
    total_ref = 0
    for p_ids, r_ids in zip(pred_seqs, ref_seqs):
        p_str = _ids_to_bases(p_ids, drop_blank=True)
        r_str = _ids_to_bases(r_ids, drop_blank=True)
        match, ref_len = _pbma_counts(p_str, r_str)
        total_match += match
        total_ref += ref_len
    return float(total_match) / float(total_ref) if total_ref > 0 else 0.0


def batch_bonito_accuracy(
    pred_seqs: List[List[int]],
    ref_seqs: List[List[int]],
    balanced: bool = False,
    min_coverage: float = 0.0,
) -> float:
    """
    Bonito-style accuracy across a batch, averaged by per-read accuracy.
    返回百分比（0-100）。
    """
    if not pred_seqs or not ref_seqs:
        return 0.0
    scores = []
    for p_ids, r_ids in zip(pred_seqs, ref_seqs):
        p_str = _ids_to_bases(p_ids, drop_blank=True)
        r_str = _ids_to_bases(r_ids, drop_blank=True)
        scores.append(cal_bonito_accuracy(p_str, r_str, balanced=balanced, min_coverage=min_coverage))
    return float(np.mean(scores)) if scores else 0.0


# ---------------- 曲线绘图 & metrics 保存 ----------------

def plot_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
):
    """画 Loss + Accuracy 曲线"""
    max_len = min(len(train_losses), len(val_losses), len(val_accs))
    if max_len == 0:
        print("[plot_curves] Empty metrics.")
        return

    train_losses = train_losses[:max_len]
    val_losses = val_losses[:max_len]
    val_accs = val_accs[:max_len]

    epochs = range(1, max_len + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Training Metrics: Loss & Validation Accuracy",
        fontsize=14,
        fontweight="bold",
    )

    ax1.grid(True, linestyle="--", alpha=0.4, axis="both")

    color_loss_train = "#1f77b4"
    color_loss_val = "#ff7f0e"

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color=color_loss_train, fontsize=12)

    line1 = ax1.plot(
        epochs,
        train_losses,
        label="Train Loss",
        color=color_loss_train,
        linewidth=2,
    )
    line2 = ax1.plot(
        epochs,
        val_losses,
        label="Val Loss",
        color=color_loss_val,
        linestyle="--",
        linewidth=2,
    )
    ax1.tick_params(axis="y", labelcolor=color_loss_train)

    # PBMA on twin axis
    ax2 = ax1.twinx()
    color_acc = "#2ca02c"
    ax2.set_ylabel("Validation Accuracy", color=color_acc, fontsize=12)
    line3 = ax2.plot(
        epochs,
        val_accs,
        label="Val Acc",
        color=color_acc,
        linestyle=":",
        linewidth=2,
    )
    ax2.tick_params(axis="y", labelcolor=color_acc)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlim(left=1, right=max_len)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper right",
        frameon=True,
        fancybox=True,
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[Plot] Saved training curves to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def save_metrics_csv(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    csv_path: str,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        for e, tr, vl, pa in zip(
            range(1, len(train_losses) + 1),
            train_losses,
            val_losses,
            val_accs,
        ):
            writer.writerow([e, tr, vl, pa])
    print(f"[Metrics] Saved to {csv_path}")
