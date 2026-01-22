# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional, Tuple
import csv
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import edlib

from .utils import BLANK_IDX, ID2BASE


# ---------------- CTC 解码 ----------------

def ctc_greedy_decode(
    logits_tbc: torch.Tensor,
    blank_idx: int = BLANK_IDX,
    input_lengths: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """
    logits_tbc: [T, B, C] (模型 forward 推理的输出)
    返回: list长度为 B, 每个元素是 list[int] 的预测碱基 ID 序列
    """
    pred_ids = torch.argmax(logits_tbc, dim=2)  # [T, B]
    pred_ids = pred_ids.cpu().numpy()

    B = pred_ids.shape[1]
    if input_lengths is None:
        lengths = [pred_ids.shape[0]] * B
    else:
        lengths = [min(int(x), pred_ids.shape[0]) for x in input_lengths.cpu().tolist()]
    decoded: List[List[int]] = []

    for b in range(B):
        length = lengths[b]
        if length <= 0:
            decoded.append([])
            continue
        seq = pred_ids[:length, b].tolist()
        # CTC: 去重复 + 去 blank
        new_seq = []
        prev = None
        for x in seq:
            if x == blank_idx:
                prev = x
                continue
            if prev is not None and x == prev:
                prev = x
                continue
            new_seq.append(x)
            prev = x
        decoded.append(new_seq)
    return decoded


def _logsumexp(a: float, b: float) -> float:
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = a if a > b else b
    return m + float(np.log(np.exp(a - m) + np.exp(b - m)))


def _ctc_beam_search_single(
    log_probs_tc: np.ndarray,
    beam_width: int,
    blank_idx: int,
) -> List[int]:
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -np.inf)}
    for t in range(log_probs_tc.shape[0]):
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        for prefix, (p_b, p_nb) in beams.items():
            for c in range(log_probs_tc.shape[1]):
                p = float(log_probs_tc[t, c])
                if c == blank_idx:
                    nb = next_beams.get(prefix, (-np.inf, -np.inf))
                    next_beams[prefix] = (
                        _logsumexp(nb[0], _logsumexp(p_b + p, p_nb + p)),
                        nb[1],
                    )
                    continue

                new_prefix = prefix + (c,)
                nb_new = next_beams.get(new_prefix, (-np.inf, -np.inf))
                if prefix and c == prefix[-1]:
                    next_beams[new_prefix] = (nb_new[0], _logsumexp(nb_new[1], p_b + p))
                    nb_same = next_beams.get(prefix, (-np.inf, -np.inf))
                    next_beams[prefix] = (nb_same[0], _logsumexp(nb_same[1], p_nb + p))
                else:
                    next_beams[new_prefix] = (
                        nb_new[0],
                        _logsumexp(nb_new[1], _logsumexp(p_b + p, p_nb + p)),
                    )

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[: max(1, beam_width)]
        )

    best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    return list(best)


def ctc_beam_search_decode(
    logits_tbc: torch.Tensor,
    beam_width: int = 5,
    blank_idx: int = BLANK_IDX,
    input_lengths: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    if beam_width <= 1:
        return ctc_greedy_decode(logits_tbc, blank_idx=blank_idx)
    log_probs = torch.log_softmax(logits_tbc, dim=2).cpu().numpy()
    if input_lengths is None:
        lengths = [log_probs.shape[0]] * log_probs.shape[1]
    else:
        lengths = [min(int(x), log_probs.shape[0]) for x in input_lengths.cpu().tolist()]
    decoded = []
    for b in range(log_probs.shape[1]):
        length = lengths[b]
        if length <= 0:
            decoded.append([])
            continue
        decoded.append(_ctc_beam_search_single(log_probs[:length, b, :], beam_width, blank_idx))
    return decoded


def ctc_crf_decode(
    logits_tbc: torch.Tensor,
    blank_idx: int = BLANK_IDX,
    input_lengths: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    try:
        from . import ctc_crf  # type: ignore
    except Exception as exc:
        raise ImportError(
            "ctc-crf decoder requested but ctc_crf is not installed. "
            "Install a CTC-CRF implementation and expose a decode() API."
        ) from exc

    if not hasattr(ctc_crf, "decode"):
        raise ImportError(
            "ctc_crf.decode not found. Provide a CTC-CRF library with a decode(logits, blank_idx) API."
        )

    if input_lengths is None:
        return ctc_crf.decode(logits_tbc, blank_idx=blank_idx)
    lengths = [min(int(x), logits_tbc.size(0)) for x in input_lengths.cpu().tolist()]
    decoded: List[List[int]] = []
    for b, length in enumerate(lengths):
        if length <= 0:
            decoded.append([])
            continue
        logits = logits_tbc[:length, b : b + 1, :]
        decoded.extend(ctc_crf.decode(logits, blank_idx=blank_idx))
    return decoded


def ctc_decode(
    logits_tbc: torch.Tensor,
    decoder: str = "greedy",
    beam_width: int = 5,
    blank_idx: int = BLANK_IDX,
    input_lengths: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    if decoder == "greedy":
        return ctc_greedy_decode(logits_tbc, blank_idx=blank_idx, input_lengths=input_lengths)
    if decoder == "beam":
        return ctc_beam_search_decode(
            logits_tbc,
            beam_width=beam_width,
            blank_idx=blank_idx,
            input_lengths=input_lengths,
        )
    if decoder == "crf":
        return ctc_crf_decode(logits_tbc, blank_idx=blank_idx, input_lengths=input_lengths)
    raise ValueError(f"Unknown decoder: {decoder}")


def ctc_crf_loss(
    logits_tbc: torch.Tensor,
    target_labels: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_idx: int = BLANK_IDX,
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

    return ctc_crf.ctc_crf_loss(
        logits_tbc, target_labels, input_lengths, target_lengths, blank_idx=blank_idx
    )


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


def cal_per_base_match_accuracy(pred_seq: str, ref_seq: str) -> float:
    """PBMA = match / (match + mismatch + ins + del)"""
    if not pred_seq or not ref_seq:
        return 0.0

    result = edlib.align(pred_seq, ref_seq, task="path")
    cigar = result.get("cigar", None)
    if cigar is None:
        return 0.0

    ops = _parse_cigar(cigar)
    match = mismatch = ins = dele = 0

    for op in ops:
        if op == "=":
            match += 1
        elif op == "X":
            mismatch += 1
        elif op == "I":
            ins += 1
        elif op == "D":
            dele += 1

    total = match + mismatch + ins + dele
    return match / total if total > 0 else 0.0


def batch_pbma(
    pred_seqs: List[List[int]],
    ref_seqs: List[List[int]],
) -> float:
    """
    计算一个 batch 的平均 PBMA
    pred_seqs/ref_seqs: list[list[int]]，标签 1..4
    """
    assert len(pred_seqs) == len(ref_seqs)
    scores = []
    for p_ids, r_ids in zip(pred_seqs, ref_seqs):
        p_str = "".join(ID2BASE.get(i, "N") for i in p_ids)
        r_str = "".join(ID2BASE.get(i, "N") for i in r_ids)
        scores.append(cal_per_base_match_accuracy(p_str, r_str))
    return float(np.mean(scores)) if scores else 0.0


# ---------------- Inspect batch ----------------

def inspect_batch(
    model,
    data_loader,
    device: torch.device,
    num_reads: int = 5,
    max_len: int = 200,
):
    """
    从 data_loader 中取一个 batch, 随机挑 num_reads 条:
      - 打印 True 序列
      - 打印 Pred 序列 (CTC 解码后)
      - 打印 PBMA
    """
    from .utils import ID2BASE  # 再次导入，防止循环引用

    model.eval()
    model.to(device)

    try:
        batch = next(iter(data_loader))
    except StopIteration:
        print("[inspect_batch] data_loader is empty.")
        return

    input_ids = batch["input_ids"].to(device)
    target_seqs = batch["target_seqs"]  # list[list[int]]

    with torch.no_grad():
        logits_tbc = model(input_ids=input_ids)

    pred_seqs = ctc_greedy_decode(logits_tbc, blank_idx=BLANK_IDX)

    B = len(pred_seqs)
    if B == 0:
        print("[inspect_batch] Empty batch.")
        return

    n_show = min(num_reads, B)
    idxs = np.random.choice(B, n_show, replace=False)

    print("\n===== Inspect batch: TRUE vs PRED =====")
    for i, idx in enumerate(idxs, start=1):
        p_ids = pred_seqs[idx]
        t_ids = target_seqs[idx]

        p_str = "".join(ID2BASE.get(x, "N") for x in p_ids)
        t_str = "".join(ID2BASE.get(x, "N") for x in t_ids)

        pbma = cal_per_base_match_accuracy(p_str, t_str)

        p_show = p_str[:max_len]
        t_show = t_str[:max_len]

        print(f"\n--- Read {i} (batch idx={idx}) ---")
        print(f"True length: {len(t_str)}, Pred length: {len(p_str)}")
        print(f"PBMA: {pbma:.4f}")
        print(f"TRUE: {t_show}{'...' if len(t_str) > max_len else ''}")
        print(f"PRED: {p_show}{'...' if len(p_str) > max_len else ''}")
    print("===== End Inspect =====\n")


# ---------------- 曲线绘图 & metrics 保存 ----------------

def plot_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_pbmas: List[float],
    save_path: Optional[str] = None,
):
    """画 Loss + PBMA 曲线"""
    max_len = min(len(train_losses), len(val_losses), len(val_pbmas))
    if max_len == 0:
        print("[plot_curves] Empty metrics.")
        return

    train_losses = train_losses[:max_len]
    val_losses = val_losses[:max_len]
    val_pbmas = val_pbmas[:max_len]

    epochs = range(1, max_len + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Training Metrics: Loss & Validation PBMA",
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
    ax2.set_ylabel("Validation PBMA", color=color_acc, fontsize=12)
    line3 = ax2.plot(
        epochs,
        val_pbmas,
        label="Val PBMA",
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
    val_pbmas: List[float],
    csv_path: str,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_pbma"])
        for e, tr, vl, pa in zip(
            range(1, len(train_losses) + 1),
            train_losses,
            val_losses,
            val_pbmas,
        ):
            writer.writerow([e, tr, vl, pa])
    print(f"[Metrics] Saved to {csv_path}")
