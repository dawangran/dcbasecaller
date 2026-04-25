#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick logic checks for record_per_file splitting and streaming datasets.

This script is intentionally lightweight and self-contained.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile

import numpy as np

from basecall.data_multifolder import (
    JsonlFile,
    NpyPair,
    StreamingJsonlSignalRefDataset,
    StreamingNpySignalRefDataset,
    split_jsonl_records_per_file,
    split_npy_records_per_file,
)


def _write_jsonl_gz(path: str, n: int) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            rec = {"text": f"<|bwav:{i}|>", "bases": "1234"}
            f.write(json.dumps(rec) + "\n")


def _build_jsonl_files(root: str, n_files: int, n_each: int) -> list[JsonlFile]:
    out: list[JsonlFile] = []
    for i in range(n_files):
        p = os.path.join(root, f"reads_{i}.jsonl.gz")
        _write_jsonl_gz(p, n_each)
        out.append(JsonlFile(path=p, group_id=p))
    return out


def _build_npy_pairs(root: str, n_files: int, n_each: int) -> list[NpyPair]:
    out: list[NpyPair] = []
    for i in range(n_files):
        tp = os.path.join(root, f"tokens_{i}.npy")
        rp = os.path.join(root, f"reference_{i}.npy")
        tokens = np.array([f"<|bwav:{j}|>" for j in range(n_each)], dtype=object)
        refs = np.array(["1234" for _ in range(n_each)], dtype=object)
        np.save(tp, tokens, allow_pickle=True)
        np.save(rp, refs, allow_pickle=True)
        out.append(NpyPair(tokens_path=tp, reference_path=rp, group_id=tp))
    return out


def _count_iterable(ds) -> int:
    return sum(1 for _ in ds)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="split_logic_") as d:
        n_files, n_each = 2, 10
        total = n_files * n_each

        jsonl_files = _build_jsonl_files(d, n_files=n_files, n_each=n_each)
        npy_pairs = _build_npy_pairs(d, n_files=n_files, n_each=n_each)

        # 1) Non-streaming record_per_file should be exact under split_indices rounding
        tr, va, te = split_jsonl_records_per_file(jsonl_files, 0.7, 0.2, 0.1, seed=1)
        assert (len(tr), len(va), len(te)) == (14, 4, 2), (len(tr), len(va), len(te))
        tr, va, te = split_npy_records_per_file(npy_pairs, 0.7, 0.2, 0.1, seed=1)
        assert (len(tr), len(va), len(te)) == (14, 4, 2), (len(tr), len(va), len(te))

        # 2) Streaming record_per_file should be complete (train+val+test == total)
        s_tr = StreamingJsonlSignalRefDataset(jsonl_files, "train", "record_per_file", 0.7, 0.2, 0.1, seed=1)
        s_va = StreamingJsonlSignalRefDataset(jsonl_files, "val", "record_per_file", 0.7, 0.2, 0.1, seed=1)
        s_te = StreamingJsonlSignalRefDataset(jsonl_files, "test", "record_per_file", 0.7, 0.2, 0.1, seed=1)
        c1 = (_count_iterable(s_tr), _count_iterable(s_va), _count_iterable(s_te))
        assert sum(c1) == total, c1

        s_tr = StreamingNpySignalRefDataset(npy_pairs, "train", "record_per_file", 0.7, 0.2, 0.1, seed=1)
        s_va = StreamingNpySignalRefDataset(npy_pairs, "val", "record_per_file", 0.7, 0.2, 0.1, seed=1)
        s_te = StreamingNpySignalRefDataset(npy_pairs, "test", "record_per_file", 0.7, 0.2, 0.1, seed=1)
        c2 = (_count_iterable(s_tr), _count_iterable(s_va), _count_iterable(s_te))
        assert sum(c2) == total, c2

        print("[OK] non-streaming exact split and streaming completeness checks passed.")
        print(f"jsonl_stream_counts={c1} npy_stream_counts={c2}")


if __name__ == "__main__":
    main()
