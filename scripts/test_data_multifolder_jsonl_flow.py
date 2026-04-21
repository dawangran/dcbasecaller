#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一步一步测试 data_multifolder.py 如何把 JSONL 读成训练语料。

用法：
  python scripts/test_data_multifolder_jsonl_flow.py --jsonl /path/to/reads.jsonl.gz --tokenizer_name_or_path /path/to/model_or_tokenizer
  python scripts/test_data_multifolder_jsonl_flow.py --jsonl /path/to/reads.jsonl --tokenizer_name_or_path /path/to/model_or_tokenizer

说明：
- data_multifolder.scan_jsonl_files 只接受 .jsonl.gz；若输入 .jsonl，本脚本会先临时转成 .jsonl.gz。
- 脚本会依次演示：
  1) 文件扫描
  2) 原始记录读取
  3) bases 解析
  4) MultiJsonlSignalRefDataset 构建
  5) __getitem__ 输出
  6) 使用模型 tokenizer + create_collate_fn 打包 batch
"""

from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from pathlib import Path
from transformers import AutoTokenizer

from basecall.data_multifolder import (
    MultiJsonlSignalRefDataset,
    _iter_jsonl_records,
    _parse_bases,
    create_collate_fn,
    scan_jsonl_files,
)


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
    ap.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Tokenizer path used by model input.")
    ap.add_argument("--token_offset", type=int, default=0, help="Optional token offset passed to dataset.")
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

    print("[step6] run create_collate_fn with model tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
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
    print("\n[done] JSONL -> dataset -> batch pipeline looks good.")

    if tmp_holder is not None:
        tmp_holder.cleanup()


if __name__ == "__main__":
    main()
