# -*- coding: utf-8 -*-
"""
Convert many JSONL / JSONL.GZ files into merged tokens/reference NPY shards.

Example:
    python -m basecall.jsonl_to_npy \
      --input_dir /data/jsonl \
      --output_dir /data/npy \
      --max_files 100 \
      --files_per_shard 10
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True, help="Folder containing .jsonl or .jsonl.gz files.")
    p.add_argument("--output_dir", type=str, required=True, help="Output folder for tokens_*.npy/reference_*.npy.")
    p.add_argument("--max_files", type=int, default=100, help="Read at most the first N files after sorting (default: 100).")
    p.add_argument("--files_per_shard", type=int, default=10, help="Merge every K jsonl files into one npy shard pair (default: 10).")
    p.add_argument("--recursive", action="store_true", help="Recursively scan input_dir.")
    return p.parse_args()


def iter_jsonl_paths(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))]
    return sorted(files)


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                continue
            yield obj


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_files <= 0:
        raise ValueError("--max_files must be > 0")
    if args.files_per_shard <= 0:
        raise ValueError("--files_per_shard must be > 0")

    all_files = iter_jsonl_paths(input_dir, recursive=bool(args.recursive))
    selected_files = all_files[: args.max_files]
    if not selected_files:
        raise ValueError(f"No .jsonl/.jsonl.gz files found in: {input_dir}")

    print(f"[jsonl_to_npy] discovered={len(all_files)} selected={len(selected_files)}")
    shard_idx = 0
    total_records = 0
    total_skipped = 0

    for start in range(0, len(selected_files), args.files_per_shard):
        batch_files = selected_files[start : start + args.files_per_shard]
        tokens: list[str] = []
        references: list[Any] = []
        skipped = 0

        for path in batch_files:
            for obj in iter_records(path):
                text = obj.get("text", "")
                bases = obj.get("bases", None)
                if not text or bases is None:
                    skipped += 1
                    continue
                tokens.append(str(text))
                references.append(bases)

        tok_path = output_dir / f"tokens_{shard_idx:04d}.npy"
        ref_path = output_dir / f"reference_{shard_idx:04d}.npy"
        np.save(tok_path, np.asarray(tokens, dtype=object), allow_pickle=True)
        np.save(ref_path, np.asarray(references, dtype=object), allow_pickle=True)

        total_records += len(tokens)
        total_skipped += skipped
        print(
            f"[jsonl_to_npy] shard={shard_idx:04d} files={len(batch_files)} "
            f"records={len(tokens)} skipped={skipped} -> {tok_path.name}, {ref_path.name}"
        )
        shard_idx += 1

    print(
        f"[jsonl_to_npy] done shards={shard_idx} total_records={total_records} "
        f"total_skipped={total_skipped} output={output_dir}"
    )


if __name__ == "__main__":
    main()
