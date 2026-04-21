#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一步一步测试 data_multifolder.py 如何把 JSONL 读成训练语料。

用法：
  python scripts/test_data_multifolder_jsonl_flow.py --jsonl /path/to/reads.jsonl.gz
  python scripts/test_data_multifolder_jsonl_flow.py --jsonl /path/to/reads.jsonl

说明：
- data_multifolder.scan_jsonl_files 只接受 .jsonl.gz；若输入 .jsonl，本脚本会先临时转成 .jsonl.gz。
- 脚本会依次演示：
  1) 文件扫描
  2) 原始记录读取
  3) bases 解析
  4) MultiJsonlSignalRefDataset 构建
  5) __getitem__ 输出
  6) create_collate_fn 打包 batch
"""

from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from basecall.data_multifolder import (
    MultiJsonlSignalRefDataset,
    _iter_jsonl_records,
    _parse_bases,
    create_collate_fn,
    scan_jsonl_files,
)


class DummyTokenizer:
    """最小 tokenizer：把 <|bwav:ID|> 中的 ID 取出来做 input_ids。"""

    def __call__(self, texts: list[str], return_tensors: str = "pt", padding: bool = True, truncation: bool = False) -> dict[str, Any]:
        parsed: list[list[int]] = []
        for text in texts:
            ids: list[int] = []
            i = 0
            while i < len(text):
                start = text.find("<|bwav:", i)
                if start < 0:
                    break
                end = text.find("|>", start)
                if end < 0:
                    break
                num = text[start + len("<|bwav:"): end]
                if num.isdigit():
                    ids.append(int(num))
                i = end + 2
            parsed.append(ids if ids else [0])

        max_len = max(len(x) for x in parsed)
        input_ids = torch.zeros((len(parsed), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(parsed), max_len), dtype=torch.long)
        for row, ids in enumerate(parsed):
            n = len(ids)
            input_ids[row, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _build_tokenizer(tokenizer_path: str | None, use_dummy_tokenizer: bool) -> PreTrainedTokenizerBase | DummyTokenizer:
    if use_dummy_tokenizer:
        print("[tokenizer] using DummyTokenizer (debug only, may differ from training/inference tokenize behavior).")
        return DummyTokenizer()
    if not tokenizer_path:
        raise ValueError("Please provide --tokenizer_path to keep tokenize behavior consistent with data/train/eval logic.")
    print(f"[tokenizer] loading AutoTokenizer from: {tokenizer_path}")
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _ensure_jsonl_gz(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.suffixes[-2:] == [".jsonl", ".gz"] or path.name.endswith(".jsonl.gz"):
        return path, None
    if path.suffix == ".jsonl":
        tmpdir = tempfile.TemporaryDirectory(prefix="jsonl_to_gz_")
        out = Path(tmpdir.name) / f"{path.stem}.jsonl.gz"
        with path.open("rt", encoding="utf-8") as fr, gzip.open(out, "wt", encoding="utf-8") as fw:
            for line in fr:
                fw.write(line)
        print(f"[step0] input is .jsonl, converted to temporary .jsonl.gz: {out}")
        return out, tmpdir
    raise ValueError(f"Only .jsonl or .jsonl.gz is supported by this demo script: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to input .jsonl or .jsonl.gz")
    ap.add_argument("--token_offset", type=int, default=0, help="Optional token offset passed to dataset.")
    ap.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path used by training/eval (recommended).")
    ap.add_argument("--use_dummy_tokenizer", action="store_true", help="Use built-in DummyTokenizer for quick debug.")
    ap.add_argument("--show_full_tensors", action="store_true", help="Print full tensors in step6 for deep inspection.")
    ap.add_argument("--preview_tokens", type=int, default=16, help="How many tokens to preview per sample from input_ids (0 to disable).")
    args = ap.parse_args()

    src = Path(args.jsonl)
    if not src.exists():
        raise FileNotFoundError(src)

    jsonl_gz, tmp_holder = _ensure_jsonl_gz(src)

    print("[step1] scan_jsonl_files ...")
    scanned = scan_jsonl_files([str(jsonl_gz)], group_by="file", recursive=False)
    for i, item in enumerate(scanned):
        print(f"  scanned[{i}] path={item.path} group_id={item.group_id}")

    print("[step2] iterate raw json records (_iter_jsonl_records) ...")
    raw_records = list(_iter_jsonl_records(str(jsonl_gz)))
    print(f"  total records: {len(raw_records)}")
    if raw_records:
        print("  first record:", json.dumps(raw_records[0], ensure_ascii=False))

    print("[step3] parse bases (_parse_bases) ...")
    for idx, rec in enumerate(raw_records[:5]):
        bases = rec.get("bases")
        labels = _parse_bases(bases)
        print(f"  record[{idx}] bases={bases!r} -> labels={labels}")

    print("[step4] build MultiJsonlSignalRefDataset ...")
    ds = MultiJsonlSignalRefDataset(scanned, token_offset=args.token_offset)
    print(f"  dataset size: {len(ds)}")

    print("[step5] inspect __getitem__ output ...")
    for i in range(min(3, len(ds))):
        item = ds[i]
        print(f"  item[{i}] signal_str={item['signal_str'][:80]!r} target_seq={item['target_seq']}")

    tokenizer = _build_tokenizer(args.tokenizer_path, args.use_dummy_tokenizer)
    print("[step6] run create_collate_fn with selected tokenizer ...")
    collate = create_collate_fn(tokenizer)
    mini_batch = [ds[i] for i in range(min(2, len(ds)))]
    if not mini_batch:
        raise RuntimeError("Dataset is empty after filtering; need at least one valid record with text+bases.")
    batch = collate(mini_batch)
    print("  batch keys:", sorted(batch.keys()))
    print("  input_ids.shape:", tuple(batch["input_ids"].shape))
    print("  attention_mask.shape:", tuple(batch["attention_mask"].shape))
    print("  input_lengths:", batch["input_lengths"].tolist())
    print("  target_lengths:", batch["target_lengths"].tolist())
    print("  target_labels:", batch["target_labels"].tolist())
    if args.preview_tokens > 0:
        preview_n = min(args.preview_tokens, int(batch["input_ids"].shape[1]))
        print(f"  preview input_ids[:, :{preview_n}]:", batch["input_ids"][:, :preview_n].tolist())
        print(f"  preview attention_mask[:, :{preview_n}]:", batch["attention_mask"][:, :preview_n].tolist())
    if args.show_full_tensors:
        print("  full input_ids:\n", batch["input_ids"])
        print("  full attention_mask:\n", batch["attention_mask"])
        print("  full target_labels:\n", batch["target_labels"])
    print("\n[done] JSONL -> dataset -> batch pipeline looks good.")

    if tmp_holder is not None:
        tmp_holder.cleanup()


if __name__ == "__main__":
    main()
