# infer_jsonl_fastq.py
# 用法示例：
# python infer_jsonl_fastq.py \
#   --ckpt your.ckpt \
#   --model_name_or_path your_hf_model_dir_or_name \
#   --jsonl_gz reads.jsonl.gz \
#   --out out.fastq \
#   --amp

import argparse
import json
import os
import math
import gzip
from typing import List, Tuple, Dict, Any, Iterable

import torch
from tqdm import tqdm

from .model import BasecallModel
from .utils import ID2BASE, BLANK_IDX, seed_everything
from .metrics import ctc_decode

# --------------------------
# Bonito-style Q-score (CTC)
# --------------------------

def _prob_to_phred(p: float, max_q: int = 40) -> int:
    """
    Bonito/Guppy 常用：Q = -10 log10(1 - p)
    """
    p = min(max(p, 1e-6), 1 - 1e-6)
    q = -10.0 * math.log10(1.0 - p)
    return int(min(max_q, max(0, round(q))))

def _phred_to_char(q: int) -> str:
    # standard Sanger FASTQ (Phred+33)
    q = min(max(q, 0), 93)
    return chr(q + 33)


def _constant_qstring(length: int, q: int) -> str:
    return _phred_to_char(q) * max(length, 0)

def ctc_greedy_with_q_bonito(
    logits_btc: torch.Tensor,
    blank_idx: int = 0,
) -> Tuple[str, str]:
    """
    logits_btc: [1, T, C]（单条 read）
    返回：(sequence, qstring)
    Bonito-style 近似做法：
      - timestep 级 argmax 得到 viterbi path
      - collapse repeats & remove blanks
      - 每个 base 的置信度用该 base run 内的概率做“几何平均”（更接近 Bonito 的稳定感）
      - Q = -10 log10(1-p)
    """
    assert logits_btc.dim() == 3 and logits_btc.shape[0] == 1
    probs = torch.softmax(logits_btc[0], dim=-1)  # [T, C]

    # viterbi path
    path = torch.argmax(probs, dim=-1).tolist()  # [T]
    path_p = probs[torch.arange(probs.shape[0]), torch.tensor(path, device=probs.device)].tolist()  # [T]

    seq_ids: List[int] = []
    base_ps: List[float] = []

    prev = None
    cur_label = None
    cur_run_ps: List[float] = []

    def flush_run(label: int, run_ps: List[float]):
        if label == blank_idx:
            return
        # collapse rule: label != prev_label_already_handled outside
        # geometric mean prob (Bonito-like稳定)
        run_ps = [min(max(p, 1e-8), 1.0) for p in run_ps]
        gm = math.exp(sum(math.log(p) for p in run_ps) / len(run_ps))
        seq_ids.append(label)
        base_ps.append(gm)

    for lab, p in zip(path, path_p):
        if prev is None:
            prev = lab
            cur_label = lab
            cur_run_ps = [p]
            continue

        if lab == cur_label:
            cur_run_ps.append(p)
        else:
            # label changed: consider emitting previous run if non-blank and != previous emitted handled by CTC rule:
            # CTC greedy collapse: output when label != blank and label != prev_label (prev timestep label)
            flush_run(cur_label, cur_run_ps)
            cur_label = lab
            cur_run_ps = [p]
        prev = lab

    # last run
    flush_run(cur_label, cur_run_ps)

    # Remove consecutive duplicates after blank-removal?（标准 greedy collapse 已基本处理）
    # 这里再保险一下：避免 ... A A ...（如果中间没有 blank，但你想更严格可关掉）
    collapsed_ids = []
    collapsed_ps = []
    for i, lab in enumerate(seq_ids):
        if i == 0 or lab != collapsed_ids[-1]:
            collapsed_ids.append(lab)
            collapsed_ps.append(base_ps[i])
        else:
            # 同一base连续：合并概率（取几何平均）
            merged = math.sqrt(collapsed_ps[-1] * base_ps[i])
            collapsed_ps[-1] = merged

    bases = "".join(ID2BASE.get(i, "N") for i in collapsed_ids)
    qstring = "".join(_phred_to_char(_prob_to_phred(p)) for p in collapsed_ps)
    return bases, qstring


def write_fastq(fp, read_id: str, seq: str, q: str):
    fp.write(f"@{read_id}\n")
    fp.write(seq + "\n")
    fp.write("+\n")
    fp.write(q + "\n")


def iter_jsonl_reads(path: str) -> Iterable[Tuple[str, str]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            read_id = str(obj.get("id", ""))
            text = obj.get("text", "")
            if not read_id or not text:
                continue
            yield read_id, text


def _infer_head_layers(state_dict: Dict[str, torch.Tensor], default_layers: int) -> int:
    indices = set()
    for key in state_dict.keys():
        if key.startswith("base_head.blocks."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                indices.add(int(parts[2]))
    if not indices:
        return default_layers
    return max(indices) + 1


def _infer_kernel_size(state_dict: Dict[str, torch.Tensor], default_kernel: int) -> int:
    weight = state_dict.get("base_head.blocks.0.0.weight")
    if isinstance(weight, torch.Tensor) and weight.dim() == 3:
        return int(weight.shape[-1])
    return default_kernel


def _infer_use_pointwise(state_dict: Dict[str, torch.Tensor], default_value: bool) -> bool:
    if "base_head.blocks.0.1.weight" in state_dict:
        return True
    if any(key.startswith("base_head.blocks.") and ".1.weight" in key for key in state_dict):
        return True
    return default_value


def _infer_num_classes(state_dict: Dict[str, torch.Tensor], default_value: int) -> int:
    weight = state_dict.get("base_head.proj.weight")
    if isinstance(weight, torch.Tensor) and weight.dim() == 2:
        return int(weight.shape[0])
    return default_value


def _infer_transformer_layers(state_dict: Dict[str, torch.Tensor]) -> int:
    indices = set()
    for key in state_dict.keys():
        if key.startswith("base_head.transformer.layers."):
            parts = key.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                indices.add(int(parts[3]))
    if not indices:
        return 0
    return max(indices) + 1


def _resolve_head_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, object]:
    head_layers = _infer_head_layers(state_dict, default_layers=2)
    head_kernel_size = _infer_kernel_size(state_dict, default_kernel=5)
    head_use_pointwise = _infer_use_pointwise(state_dict, default_value=True)
    num_classes = _infer_num_classes(state_dict, default_value=len(ID2BASE))

    inferred_transformer_layers = _infer_transformer_layers(state_dict)
    head_transformer_layers = inferred_transformer_layers
    head_use_transformer = inferred_transformer_layers > 0
    head_transformer_heads = 4
    return {
        "head_layers": int(head_layers),
        "head_kernel_size": int(head_kernel_size),
        "head_use_pointwise": bool(head_use_pointwise),
        "head_use_transformer": bool(head_use_transformer),
        "head_transformer_layers": int(head_transformer_layers),
        "head_transformer_heads": int(head_transformer_heads),
        "num_classes": int(num_classes),
    }


def split_bwav_tokens(text: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(text):
        start = text.find("<|bwav:", i)
        if start < 0:
            break
        end = text.find("|>", start)
        if end < 0:
            break
        tokens.append(text[start : end + 2])
        i = end + 2
    return tokens


def chunk_tokens(tokens: List[str], max_tokens: int, overlap: int) -> List[List[str]]:
    if max_tokens <= 0:
        return [tokens]
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens.")
    chunks = []
    step = max_tokens - overlap
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + max_tokens]
        if chunk:
            chunks.append(chunk)
    return chunks


def find_sequence_overlap(prev_seq: str, next_seq: str, max_overlap: int) -> int:
    if max_overlap <= 0:
        return 0
    max_overlap = min(max_overlap, len(prev_seq), len(next_seq))
    for k in range(max_overlap, 0, -1):
        if prev_seq[-k:] == next_seq[:k]:
            return k
    return 0


# --------------------------
# main
# --------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--model_name_or_path", required=True, type=str)
    ap.add_argument("--jsonl_gz", type=str, required=True)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--decoder", choices=["greedy", "beam", "crf"], default="greedy")
    ap.add_argument("--beam_width", type=int, default=5)
    ap.add_argument("--beam_q", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--overlap_bases", type=int, default=None)
    ap.add_argument("--hidden_layer", type=int, default=-1)
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    state = torch.load(args.ckpt, map_location="cpu")
    # 兼容 ckpt 格式：{"model": ...} / {"model_state_dict": ...} / {"state_dict": ...} / 直接 state_dict
    if isinstance(state, dict):
        if "model" in state:
            sd = state["model"]
        elif "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state
    else:
        sd = state
    head_config = _resolve_head_config(sd)
    # load model
    model = BasecallModel(
        model_path=args.model_name_or_path,
        num_classes=head_config["num_classes"],
        hidden_layer=args.hidden_layer,
        head_kernel_size=head_config["head_kernel_size"],
        head_layers=head_config["head_layers"],
        head_use_pointwise=head_config["head_use_pointwise"],
        head_use_transformer=head_config["head_use_transformer"],
        head_transformer_layers=head_config["head_transformer_layers"],
        head_transformer_heads=head_config["head_transformer_heads"],
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    tokenizer = model.tokenizer  # 你的 BasecallModel 里应有

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as w:
        for read_id, signal_str in tqdm(iter_jsonl_reads(args.jsonl_gz), desc="jsonl->fastq"):
            tokens = split_bwav_tokens(signal_str)
            if not tokens:
                continue
            chunks = chunk_tokens(tokens, args.max_tokens, args.overlap)
            chunk_seqs: List[str] = []
            chunk_qs: List[str] = []

            chunk_token_lengths = [len(chunk) for chunk in chunks]
            for start in range(0, len(chunks), args.batch_size):
                batch_chunks = chunks[start:start + args.batch_size]
                batch_strs = ["".join(chunk) for chunk in batch_chunks]

                enc = tokenizer(batch_strs, return_tensors="pt", padding=True, truncation=False)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits_btc = model(input_ids, attention_mask=attention_mask)  # [B,T,C]

                if args.decoder == "greedy":
                    for idx in range(logits_btc.size(0)):
                        seq, qstring = ctc_greedy_with_q_bonito(
                            logits_btc[idx:idx + 1], blank_idx=BLANK_IDX
                        )
                        chunk_seqs.append(seq)
                        chunk_qs.append(qstring)
                else:
                    logits_tbc = logits_btc.transpose(0, 1)
                    pred_ids = ctc_decode(
                        logits_tbc,
                        decoder=args.decoder,
                        beam_width=args.beam_width,
                        blank_idx=BLANK_IDX,
                    )
                    for ids in pred_ids:
                        seq = "".join(ID2BASE.get(i, "N") for i in ids)
                        qstring = _constant_qstring(len(seq), args.beam_q)
                        chunk_seqs.append(seq)
                        chunk_qs.append(qstring)

            if args.overlap > 0 and chunk_seqs:
                trimmed_seqs = [chunk_seqs[0]]
                trimmed_qs = [chunk_qs[0]]
                for idx, (seq, q) in enumerate(zip(chunk_seqs[1:], chunk_qs[1:]), start=1):
                    token_len = chunk_token_lengths[idx] if idx < len(chunk_token_lengths) else 0
                    if token_len <= 0:
                        max_overlap = 0
                    else:
                        max_overlap = int(round(len(seq) * args.overlap / token_len))
                    if args.overlap_bases is not None:
                        max_overlap = args.overlap_bases
                    max_overlap = min(len(seq), max(0, max_overlap))
                    trim = find_sequence_overlap(trimmed_seqs[-1], seq, max_overlap)
                    trimmed_seqs.append(seq[trim:])
                    trimmed_qs.append(q[trim:])
                full_seq = "".join(trimmed_seqs)
                full_q = "".join(trimmed_qs)
            else:
                full_seq = "".join(chunk_seqs)
                full_q = "".join(chunk_qs)
            write_fastq(w, read_id, full_seq, full_q)

    print(f"[OK] wrote FASTQ: {args.out}")


if __name__ == "__main__":
    main()
