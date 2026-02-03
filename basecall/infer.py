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
import gzip
from typing import List, Tuple, Iterable

import torch
from tqdm import tqdm

from .model import BasecallModel
from .utils import ID2BASE, seed_everything, resolve_input_lengths, infer_head_config_from_state_dict
from .metrics import koi_beam_search_decode

def _phred_to_char(q: int) -> str:
    # standard Sanger FASTQ (Phred+33)
    q = min(max(q, 0), 93)
    return chr(q + 33)


def _constant_qstring(length: int, q: int) -> str:
    return _phred_to_char(q) * max(length, 0)


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
            read_id = str(obj.get("read_id", "") or obj.get("id", ""))
            text = obj.get("text", "")
            if not read_id or not text:
                continue
            yield read_id, text


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


def _infer_crf_expand_blanks(num_classes: int, n_base: int, max_state_len: int = 10) -> bool:
    if n_base <= 0:
        return False
    for state_len in range(1, max_state_len + 1):
        if num_classes == n_base ** (state_len + 1):
            return True
        if num_classes == (n_base + 1) * (n_base ** state_len):
            return False
    return False


def _token_slice_to_base_idx(token_idx: int, token_len: int, base_len: int) -> int:
    if token_len <= 0 or base_len <= 0:
        return 0
    return int(round(token_idx * base_len / token_len))


def stitch_sequences(
    chunk_seqs: List[str],
    chunk_qs: List[str],
    chunk_token_lengths: List[int],
    total_tokens: int,
    chunksize: int,
    overlap: int,
) -> Tuple[str, str]:
    if not chunk_seqs:
        return "", ""
    if len(chunk_seqs) == 1:
        return chunk_seqs[0], chunk_qs[0]
    if overlap <= 0 or chunksize <= 0:
        return "".join(chunk_seqs), "".join(chunk_qs)

    semi_overlap = overlap // 2
    start_tok = semi_overlap
    end_tok = chunksize - semi_overlap
    stub = (total_tokens - overlap) % (chunksize - overlap)
    first_chunk_end_tok = (stub + semi_overlap) if stub > 0 else end_tok

    stitched_seq: List[str] = []
    stitched_q: List[str] = []

    for idx, (seq, q, token_len) in enumerate(zip(chunk_seqs, chunk_qs, chunk_token_lengths)):
        if idx == 0:
            end_idx = _token_slice_to_base_idx(first_chunk_end_tok, token_len, len(seq))
            stitched_seq.append(seq[:end_idx])
            stitched_q.append(q[:end_idx])
        elif idx == len(chunk_seqs) - 1:
            start_idx = _token_slice_to_base_idx(start_tok, token_len, len(seq))
            stitched_seq.append(seq[start_idx:])
            stitched_q.append(q[start_idx:])
        else:
            start_idx = _token_slice_to_base_idx(start_tok, token_len, len(seq))
            end_idx = _token_slice_to_base_idx(end_tok, token_len, len(seq))
            stitched_seq.append(seq[start_idx:end_idx])
            stitched_q.append(q[start_idx:end_idx])

    return "".join(stitched_seq), "".join(stitched_q)


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
    ap.add_argument("--beam_width", type=int, default=32)
    ap.add_argument("--koi_beam_cut", type=float, default=100.0,
                    help="Beam cut value for Koi beam_search decoding.")
    ap.add_argument("--koi_scale", type=float, default=1.0,
                    help="Scale applied to scores for Koi beam_search decoding.")
    ap.add_argument("--koi_offset", type=float, default=0.0,
                    help="Offset applied to scores for Koi beam_search decoding.")
    ap.add_argument("--koi_blank_score", type=float, default=2.0,
                    help="Blank score used for Koi beam_search decoding.")
    ap.add_argument("--koi_reverse", action="store_true",
                    help="Reverse sequence output for Koi beam_search decoding.")
    ap.add_argument("--beam_q", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--hidden_layer", type=int, default=-1)
    ap.add_argument("--head_output_activation", choices=["tanh", "relu"], default=None,
                    help="Optional activation applied to head output logits.")
    ap.add_argument("--head_output_scale", type=float, default=None,
                    help="Optional scalar applied to head output logits (after activation).")
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
    head_config = infer_head_config_from_state_dict(sd)
    # load model
    n_base = len(ID2BASE) - 1
    expand_blanks = _infer_crf_expand_blanks(head_config["num_classes"], n_base)
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
        head_output_activation=args.head_output_activation,
        head_output_scale=args.head_output_scale,
        head_crf_blank_score=float(args.koi_blank_score),
        head_crf_n_base=n_base,
        head_crf_expand_blanks=expand_blanks,
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
                input_lengths = resolve_input_lengths(
                    input_ids,
                    attention_mask=attention_mask,
                )

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits_btc = model(input_ids, attention_mask=attention_mask)  # [B,T,C]

                logits_tbc = logits_btc.transpose(0, 1)
                pred_ids = koi_beam_search_decode(
                    logits_tbc,
                    beam_width=args.beam_width,
                    beam_cut=args.koi_beam_cut,
                    scale=args.koi_scale,
                    offset=args.koi_offset,
                    blank_score=args.koi_blank_score,
                    reverse=args.koi_reverse,
                    input_lengths=input_lengths,
                )
                for ids in pred_ids:
                    seq = "".join(ID2BASE.get(i, "N") for i in ids)
                    qstring = _constant_qstring(len(seq), args.beam_q)
                    chunk_seqs.append(seq)
                    chunk_qs.append(qstring)

            if args.overlap > 0 and chunk_seqs:
                full_seq, full_q = stitch_sequences(
                    chunk_seqs,
                    chunk_qs,
                    chunk_token_lengths,
                    total_tokens=len(tokens),
                    chunksize=args.max_tokens,
                    overlap=args.overlap,
                )
            else:
                full_seq = "".join(chunk_seqs)
                full_q = "".join(chunk_qs)
            write_fastq(w, read_id, full_seq, full_q)

    print(f"[OK] wrote FASTQ: {args.out}")


if __name__ == "__main__":
    main()
