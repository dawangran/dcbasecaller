# -*- coding: utf-8 -*-
"""
data_multifolder.py

支持 jsonl.gz / tokens_*.npy + reference_*.npy 输入 + 自动 split：

- text 字段是 signal_str（例如 "<|bwav:5018|><|bwav:3738|>..."），交给 tokenizer 编码
- bases 字段是 label 序列（支持数字串或 A/C/G/T 字符串）
- __getitem__ 返回 {"signal_str": str, "target_seq": List[int]}
- collate_fn 使用 tokenizer(signal_strs, padding=True, truncation=True) 得到 input_ids

新增能力：
- scan_jsonl_files: 扫描多个 jsonl.gz 文件或文件夹
- scan_npy_pairs: 扫描 tokens_*.npy + reference_*.npy 文件对
- split_jsonl_files_by_group: 按 folder 或 file 分组切 train/val/test，默认 folder（避免泄漏）
- split_npy_pairs_by_group: 按 folder 或 file 分组切 train/val/test
- MultiJsonlSignalRefDataset: 从 jsonl.gz 聚合所有 reads
- MultiNpySignalRefDataset: 从 tokens/reference npy 聚合所有 reads
- collate_fn 额外返回 input_lengths（从 attention_mask 计算），方便 CTC 用真实长度
"""

from __future__ import annotations

import os
import glob
import gzip
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .utils import BASE2ID

# -------------------------
# jsonl.gz discovery + split
# -------------------------


@dataclass(frozen=True)
class JsonlFile:
    path: str
    group_id: str


@dataclass(frozen=True)
class NpyPair:
    tokens_path: str
    reference_path: str
    group_id: str


def _iter_jsonl_files(folder: str, recursive: bool) -> List[str]:
    patterns = ("*.jsonl.gz",)
    if recursive:
        matches: List[str] = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if any(name.endswith(pat.lstrip("*")) for pat in patterns):
                    matches.append(os.path.join(root, name))
        return sorted(matches)
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(matches))


def _iter_npy_files(folder: str, recursive: bool) -> List[str]:
    patterns = ("*.npy",)
    if recursive:
        matches: List[str] = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if any(name.endswith(pat.lstrip("*")) for pat in patterns):
                    matches.append(os.path.join(root, name))
        return sorted(matches)
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(matches))


def scan_jsonl_files(
    data_paths: List[str],
    group_by: str = "folder",
    recursive: bool = False,
) -> List[JsonlFile]:
    paths: List[str] = []
    for item in data_paths:
        item = item.strip()
        if not item:
            continue
        if os.path.isdir(item):
            paths.extend(_iter_jsonl_files(item, recursive=recursive))
        elif os.path.isfile(item):
            if not item.endswith(".jsonl.gz"):
                raise ValueError(f"Only .jsonl.gz is supported: {item}")
            paths.append(item)
        else:
            raise FileNotFoundError(f"jsonl path not found: {item}")

    paths = sorted(set(paths))
    if not paths:
        raise ValueError(f"No .jsonl.gz found under: {data_paths}")

    jsonl_files: List[JsonlFile] = []
    for path in paths:
        if group_by == "folder":
            gid = os.path.abspath(os.path.dirname(path))
        elif group_by == "file":
            gid = os.path.abspath(path)
        else:
            raise ValueError("group_by must be 'folder' or 'file'")
        jsonl_files.append(JsonlFile(path=path, group_id=gid))
    return jsonl_files


def _resolve_npy_pair(path: str) -> Tuple[str, str]:
    base = os.path.basename(path)
    if base.startswith("tokens_"):
        suffix = base[len("tokens_"):]
        ref_path = os.path.join(os.path.dirname(path), f"reference_{suffix}")
        return path, ref_path
    if base.startswith("reference_"):
        suffix = base[len("reference_"):]
        tok_path = os.path.join(os.path.dirname(path), f"tokens_{suffix}")
        return tok_path, path
    raise ValueError(f"Expected tokens_*.npy or reference_*.npy, got: {path}")


def scan_npy_pairs(
    data_paths: List[str],
    group_by: str = "folder",
    recursive: bool = False,
) -> List[NpyPair]:
    tokens: List[str] = []
    references: List[str] = []
    for item in data_paths:
        item = item.strip()
        if not item:
            continue
        if os.path.isdir(item):
            for path in _iter_npy_files(item, recursive=recursive):
                if os.path.basename(path).startswith("tokens_"):
                    tokens.append(path)
                elif os.path.basename(path).startswith("reference_"):
                    references.append(path)
        elif os.path.isfile(item):
            if not item.endswith(".npy"):
                raise ValueError(f"Only .npy is supported for tokens/reference: {item}")
            tok_path, ref_path = _resolve_npy_pair(item)
            tokens.append(tok_path)
            references.append(ref_path)
        else:
            raise FileNotFoundError(f"npy path not found: {item}")

    tokens = sorted(set(tokens))
    references = set(references)
    if not tokens:
        raise ValueError(f"No tokens_*.npy found under: {data_paths}")

    npy_pairs: List[NpyPair] = []
    for tok_path in tokens:
        ref_path = os.path.join(os.path.dirname(tok_path), f"reference_{os.path.basename(tok_path)[len('tokens_'):]}")
        if ref_path not in references and not os.path.exists(ref_path):
            raise FileNotFoundError(f"Missing reference file for {tok_path}: {ref_path}")
        if group_by == "folder":
            gid = os.path.abspath(os.path.dirname(tok_path))
        elif group_by == "file":
            gid = os.path.abspath(tok_path)
        else:
            raise ValueError("group_by must be 'folder' or 'file'")
        npy_pairs.append(NpyPair(tokens_path=tok_path, reference_path=ref_path, group_id=gid))
    return npy_pairs


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


def split_jsonl_files_by_group(
    jsonl_files: List[JsonlFile],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[JsonlFile], List[JsonlFile], List[JsonlFile]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    group_to_files: Dict[str, List[JsonlFile]] = {}
    for jf in jsonl_files:
        group_to_files.setdefault(jf.group_id, []).append(jf)

    g_train, g_val, g_test = _split_groups(list(group_to_files.keys()), train_ratio, val_ratio, seed)

    train_files: List[JsonlFile] = []
    val_files: List[JsonlFile] = []
    test_files: List[JsonlFile] = []
    for gid, files in group_to_files.items():
        if gid in g_train:
            train_files.extend(files)
        elif gid in g_val:
            val_files.extend(files)
        else:
            test_files.extend(files)

    return train_files, val_files, test_files


def split_npy_pairs_by_group(
    npy_pairs: List[NpyPair],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[NpyPair], List[NpyPair], List[NpyPair]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    group_to_pairs: Dict[str, List[NpyPair]] = {}
    for pair in npy_pairs:
        group_to_pairs.setdefault(pair.group_id, []).append(pair)

    g_train, g_val, g_test = _split_groups(list(group_to_pairs.keys()), train_ratio, val_ratio, seed)

    train_pairs: List[NpyPair] = []
    val_pairs: List[NpyPair] = []
    test_pairs: List[NpyPair] = []
    for gid, pairs in group_to_pairs.items():
        if gid in g_train:
            train_pairs.extend(pairs)
        elif gid in g_val:
            val_pairs.extend(pairs)
        else:
            test_pairs.extend(pairs)
    return train_pairs, val_pairs, test_pairs


# -------------------------
# Dataset + collate (保持原风格)
# -------------------------


def _parse_bases(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        if not value:
            return []
        if all(isinstance(x, (int, np.integer)) for x in value):
            return [int(x) for x in value]
        if all(isinstance(x, str) for x in value):
            if all(x.strip().isdigit() for x in value):
                return [int(x) for x in value]
            return [BASE2ID.get(x.upper(), 0) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.isdigit():
            return [int(ch) for ch in text]
        return [BASE2ID.get(ch.upper(), 0) for ch in text]
    raise ValueError(f"Unsupported bases format: {type(value)}")


def _load_npy_records(path: str) -> List[Any]:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray):
        if arr.dtype == object:
            return list(arr)
        if arr.ndim == 0:
            return [arr.item()]
        return list(arr)
    return [arr]


def _normalize_tokens(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        if not value:
            return ""
        if all(isinstance(x, bytes) for x in value):
            return "".join(x.decode("utf-8") for x in value)
        if all(isinstance(x, str) for x in value):
            return "".join(value)
        return "".join(str(x) for x in value)
    return str(value)


def _iter_jsonl_records(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class MultiJsonlSignalRefDataset(Dataset):
    """
    从 jsonl.gz 读取 text/bases 字段，输出与原 data.py 一致的格式：
      {"signal_str": str, "target_seq": List[int]}
    """
    def __init__(self, jsonl_files: List[JsonlFile]):
        super().__init__()
        self.signal_list: List[str] = []
        self.target_list: List[np.ndarray] = []

        for jf in jsonl_files:
            for obj in _iter_jsonl_records(jf.path):
                signal_str = obj.get("text", "")
                bases = obj.get("bases", None)
                if not signal_str or bases is None:
                    continue
                labels = _parse_bases(bases)
                self.signal_list.append(str(signal_str))
                self.target_list.append(np.asarray(labels))

        print(f"[Dataset] Loaded {len(self.signal_list)} reads from {len(jsonl_files)} jsonl files")

    def __len__(self):
        return len(self.signal_list)

    def __getitem__(self, idx):
        signal_str = self.signal_list[idx]
        ref_row = np.asarray(self.target_list[idx]).reshape(-1)
        labels = ref_row[ref_row > 0].astype(np.int64).tolist()
        return {"signal_str": signal_str, "target_seq": labels}


class MultiNpySignalRefDataset(Dataset):
    """
    从 tokens_*.npy / reference_*.npy 读取，输出与原 data.py 一致的格式：
      {"signal_str": str, "target_seq": List[int]}
    """
    def __init__(self, npy_pairs: List[NpyPair]):
        super().__init__()
        self.signal_list: List[str] = []
        self.target_list: List[np.ndarray] = []

        for pair in npy_pairs:
            tokens = _load_npy_records(pair.tokens_path)
            references = _load_npy_records(pair.reference_path)
            if len(tokens) != len(references):
                raise ValueError(
                    f"Tokens/Reference length mismatch for {pair.tokens_path}: "
                    f"{len(tokens)} vs {len(references)}"
                )
            for token_row, ref_row in zip(tokens, references):
                signal_str = _normalize_tokens(token_row)
                if not signal_str:
                    continue
                labels = _parse_bases(ref_row)
                self.signal_list.append(signal_str)
                self.target_list.append(np.asarray(labels))

        print(f"[Dataset] Loaded {len(self.signal_list)} reads from {len(npy_pairs)} npy pairs")

    def __len__(self):
        return len(self.signal_list)

    def __getitem__(self, idx):
        signal_str = self.signal_list[idx]
        ref_row = np.asarray(self.target_list[idx]).reshape(-1)
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
