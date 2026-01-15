# -*- coding: utf-8 -*-
"""
data_multifolder.py

在你原始 data.py 的逻辑基础上，仅扩展为“多文件夹输入 + 自动 split”：

原 data.py 特点（保持一致）：
- tokens.npy 里是 signal_str（例如 "<|bwav:5018|><|bwav:3738|>..."），交给 tokenizer 编码
- reference.npy 里是 label 序列（可能是 [N,L] pad，也可能是 object-ragged）
- __getitem__ 返回 {"signal_str": str, "target_seq": List[int]}
- collate_fn 使用 tokenizer(signal_strs, padding=True, truncation=True) 得到 input_ids

新增能力：
- scan_pair_files: 扫描多个 data_folders，每个 folder 下 tokens_*.npy 与 reference_*.npy 配对
- split_pair_files_by_group: 按 folder 或 file 分组切 train/val/test，默认 folder（避免泄漏）
- MultiFolderSignalRefDataset: 从 pair_files 聚合所有 reads
- collate_fn 额外返回 input_lengths（从 attention_mask 计算），方便 CTC 用真实长度
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# -------------------------
# pairing + split
# -------------------------

@dataclass(frozen=True)
class PairFile:
    tokens_path: str
    ref_path: str
    group_id: str  # 用于 split


def _iter_token_files(folder: str, recursive: bool) -> List[str]:
    if recursive:
        matches: List[str] = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.startswith("tokens_") and name.endswith(".npy"):
                    matches.append(os.path.join(root, name))
        return sorted(matches)
    return sorted(glob.glob(os.path.join(folder, "tokens_*.npy")))


def scan_pair_files(
    data_folders: List[str],
    group_by: str = "folder",
    recursive: bool = False,
) -> List[PairFile]:
    pairs: List[PairFile] = []
    for fd in data_folders:
        fd = fd.strip()
        if not fd:
            continue
        if not os.path.isdir(fd):
            raise FileNotFoundError(f"data folder not found: {fd}")

        tok_files = _iter_token_files(fd, recursive=recursive)
        if not tok_files:
            continue

        for tp in tok_files:
            base = os.path.basename(tp).replace("tokens_", "", 1)
            rp = os.path.join(os.path.dirname(tp), f"reference_{base}")
            if not os.path.exists(rp):
                raise FileNotFoundError(f"Missing reference for {tp}: expected {rp}")

            if group_by == "folder":
                gid = os.path.abspath(os.path.dirname(tp) if recursive else fd)
            elif group_by == "file":
                gid = os.path.abspath(tp)
            else:
                raise ValueError("group_by must be 'folder' or 'file'")

            pairs.append(PairFile(tokens_path=tp, ref_path=rp, group_id=gid))

    if not pairs:
        raise ValueError(f"No tokens_*.npy found under folders: {data_folders}")
    return pairs


def _split_groups(groups: List[str], train_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    groups = list(groups)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), n) if n > 0 else 0
    n_val = min(max(n_val, 0), n - n_train)
    g_train = set(groups[:n_train])
    g_val = set(groups[n_train:n_train + n_val])
    g_test = set(groups[n_train + n_val:])
    return g_train, g_val, g_test


def split_pair_files_by_group(
    pair_files: List[PairFile],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[PairFile], List[PairFile], List[PairFile]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    group_to_files: Dict[str, List[PairFile]] = {}
    for pf in pair_files:
        group_to_files.setdefault(pf.group_id, []).append(pf)

    g_train, g_val, g_test = _split_groups(list(group_to_files.keys()), train_ratio, val_ratio, seed)

    train_files: List[PairFile] = []
    val_files: List[PairFile] = []
    test_files: List[PairFile] = []
    for gid, files in group_to_files.items():
        if gid in g_train:
            train_files.extend(files)
        elif gid in g_val:
            val_files.extend(files)
        else:
            test_files.extend(files)

    return train_files, val_files, test_files


# -------------------------
# robust npy loading
# -------------------------

def safe_load_object_npy(path: str) -> np.ndarray:
    """
    tokens.npy 通常是 object array（字符串），必须 allow_pickle=True
    reference.npy 有时是 numeric，[N,L] pad；也可能是 object-ragged
    """
    try:
        return np.load(path, allow_pickle=False)
    except ValueError as e:
        if "Object arrays cannot be loaded" not in str(e):
            raise
        return np.load(path, allow_pickle=True)


def split_reads(arr: np.ndarray) -> List[Any]:
    """
    将 npy 内容拆成 reads 列表（与原 data.py 的预期对齐）：
      - object array (N,) => N
      - numeric ndarray:
          * ndim >= 2 => 按第0维拆 N 条（常见 ref: [N,L]）
          * ndim == 1 => 单条
    """
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.ndim == 1:
            return arr.tolist()
        if arr.shape == ():
            return [arr.item()]
    arr = np.asarray(arr)
    if arr.ndim >= 2:
        return [arr[i] for i in range(arr.shape[0])]
    return [arr]


# -------------------------
# Dataset + collate (保持原风格)
# -------------------------

class MultiFolderSignalRefDataset(Dataset):
    """
    在原 SignalRefFolderDataset 基础上，把输入改成 pair_files 列表。
    输出与原 data.py 完全一致：
      {"signal_str": str, "target_seq": List[int]}
    """
    def __init__(self, pair_files: List[PairFile]):
        super().__init__()
        self.signal_list: List[str] = []
        self.target_list: List[np.ndarray] = []

        for pf in pair_files:
            # tokens：字符串序列（可能是 object-ragged，也可能是 [N,] object）
            sig_arr = safe_load_object_npy(pf.tokens_path)
            sig_reads = split_reads(sig_arr)

            # refs：可能是 [N,L] pad 或 object-ragged
            ref_arr = safe_load_object_npy(pf.ref_path)
            ref_reads = split_reads(ref_arr)

            if len(sig_reads) != len(ref_reads):
                raise ValueError(
                    f"Read count mismatch:\n  {pf.tokens_path}: {len(sig_reads)} reads\n  {pf.ref_path}: {len(ref_reads)} reads"
                )

            # append
            self.signal_list.extend([str(s) for s in sig_reads])
            # 保持为 ndarray，getitem 再做过滤 >0
            self.target_list.extend([np.asarray(r) for r in ref_reads])

        print(f"[Dataset] Loaded {len(self.signal_list)} reads from {len(pair_files)} paired files")

    def __len__(self):
        return len(self.signal_list)

    def __getitem__(self, idx):
        signal_str = self.signal_list[idx]
        ref_row = np.asarray(self.target_list[idx]).reshape(-1)

        # 与你原始 data.py 一致：只保留 >0 的 label
        labels = ref_row[ref_row > 0].astype(np.int64).tolist()
        return {"signal_str": signal_str, "target_seq": labels}


def create_collate_fn(tokenizer: PreTrainedTokenizerBase):
    """
    基于原 data.py 的 collate_fn，新增 input_lengths（用 attention_mask 计算）
    """
    def fn(batch: List[Dict[str, Any]]):
        signal_strs = [b["signal_str"] for b in batch]
        target_seqs = [b["target_seq"] for b in batch]

        enc = tokenizer(signal_strs, return_tensors="pt", padding=True, truncation=False)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)

        # 新增：真实长度（CTC 推荐用这个）
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1).to(torch.long)
        else:
            # fallback
            input_lengths = torch.full((input_ids.size(0),), input_ids.size(1), dtype=torch.long)

        target_lengths = torch.tensor([len(x) for x in target_seqs], dtype=torch.long)
        target_labels = torch.cat([torch.tensor(x, dtype=torch.long) for x in target_seqs]) if target_seqs else torch.empty(0, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "attention_mask": attention_mask,
            "target_labels": target_labels,
            "target_lengths": target_lengths,
            "target_seqs": target_seqs,
        }
    return fn
