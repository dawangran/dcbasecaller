# Basecall Toolkit

This repo provides training, evaluation, and inference utilities for a CTC-based basecalling model.
The core workflow is:

1. **Prepare data** as `.jsonl.gz` files (one JSON object per line) or `tokens_*.npy` + `reference_*.npy` pairs.
2. **Install** this package (`pip install -e .`) and use the console scripts.
3. **Train** with `basecall-train`.
4. **Evaluate** with `basecall-eval`.
5. **Infer** from JSONL with `basecall-infer`.

---

## Installation

```bash
pip install -e .
```

## 1) Data Format

### JSONL.GZ

Each JSONL.GZ line is a record like (bases can be letters or digit IDs):

```json
{"read_id":"id1","text":"<|bwav:123|><|bwav:456|>...","bases":"ACGT"}
{"read_id":"id2","text":"<|bwav:123|><|bwav:456|>...","bases":"1234"}
```

### NPY pairs

Provide `tokens_*.npy` and `reference_*.npy` with matching suffixes in the same folder
(e.g. `tokens_000.npy` + `reference_000.npy`). Each array row should align between the two files:

- `tokens_*.npy`: token strings or sequences that can be joined into the `<|bwav:...|>` text.
- `reference_*.npy`: bases as A/C/G/T strings or digit IDs.

### Directory layouts supported

- **Flat layout**: `.jsonl.gz` files in the same folder.
- **Nested layout**: add `--recursive` to scan subfolders.

---

## 2) Training

### Basic usage (auto split, jsonl.gz)

```bash
basecall-train \
  --jsonl_paths /path/to/data1,/path/to/data2 \
  --model_name_or_path <hf-model> \
  --output_dir outputs
```

### Train from jsonl.gz (auto split)

```bash
basecall-train \
  --jsonl_paths /path/to/reads.jsonl.gz,/path/to/more_jsonl \
  --model_name_or_path <hf-model> \
  --output_dir outputs
```

### Use explicit train/val/test folders (skip auto split, jsonl.gz)

```bash
basecall-train \
  --train_jsonl_paths /path/to/train \
  --val_jsonl_paths /path/to/val \
  --test_jsonl_paths /path/to/test \
  --model_name_or_path <hf-model> \
  --output_dir outputs
```

### Train from tokens/reference npy pairs (auto split)

```bash
basecall-train \
  --npy_paths /path/to/data_or_dir \
  --model_name_or_path <hf-model> \
  --output_dir outputs
```

### Use explicit train/val/test folders (skip auto split, npy)

```bash
basecall-train \
  --train_npy_paths /path/to/train \
  --val_npy_paths /path/to/val \
  --test_npy_paths /path/to/test \
  --model_name_or_path <hf-model> \
  --output_dir outputs
```

### All training arguments

**Data & split**
- `--jsonl_paths`: comma-separated `.jsonl.gz` files or folders (uses `text` as tokens and `bases` as reference).
- `--train_jsonl_paths`, `--val_jsonl_paths`, `--test_jsonl_paths`: explicit JSONL split inputs (skip auto split).
- `--npy_paths`: comma-separated folders or `tokens_*.npy`/`reference_*.npy` files (uses token/reference pairs).
- `--train_npy_paths`, `--val_npy_paths`, `--test_npy_paths`: explicit npy split inputs (skip auto split).
- `--group_by`: `folder` or `file` (controls leakage prevention when auto splitting).
- `--recursive`: scan subfolders for `.jsonl.gz` or tokens/reference `.npy`.
- `--train_ratio`, `--val_ratio`, `--test_ratio`: ratios for auto split.
- `--split_seed`: random seed for auto split.

**Model & freezing**
- `--model_name_or_path`: HuggingFace model ID or local path.
- `--hidden-layer`: which hidden layer to use (`-1` = last, `-2` = second last).
- `--freeze_backbone`: freeze backbone, train head only.
- `--unfreeze_last_n_layers`: unfreeze last N transformer layers.
- `--unfreeze_layer_start`, `--unfreeze_layer_end`: unfreeze layers in range `[start, end)`.

**Head options**
- `--head_kernel_size`: kernel size for depthwise conv.
- `--head_layers`: number of depthwise residual blocks.
- `--head_dropout`: dropout for head.
- `--head_disable_pointwise`: disable pointwise conv (default is enabled).
- `--head_use_transformer`: add transformer layers in head.
- `--head_transformer_layers`, `--head_transformer_heads`, `--head_transformer_dropout`.
- `--head_linear`: use a pure linear head (sets `head_layers=0`, disables pointwise/transformer blocks; uses LayerNorm + Linear only).

**Optimization**
- `--batch_size`, `--num_epochs`, `--lr`, `--weight_decay`, `--warmup_ratio`, `--min_lr`.
- `--aux_blank_weight`: optional penalty to discourage blank-dominated outputs.
- `--loss_type`: `ctc` (default) or `ctc-crf` (requires k2 or ont-koi + `ctc_crf.py` adapter).
- `--ctc_crf_state_len`: Bonito CTC-CRF state length (controls CRF head output classes).

**Checkpointing & loading**
- `--resume_ckpt`: resume from `ckpt_last.pt` (model/optim/sched/epoch/best_pbma).
- `--pretrained_ckpt`: load pretrained weights into the model.
- `--pretrained_strict`, `--pretrained_key`.
- `--save_every`, `--save_best`.

**DDP/Logging**
- `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_run_name`.
- `--find_unused_parameters`: enable DDP unused parameter detection.

**Runtime**
- `--num_workers`, `--seed`, `--log_interval`, `--output_dir`.

---

## 3) Evaluation

### Basic usage (single folder)

```bash
basecall-eval \
  --jsonl_paths /path/to/reads.jsonl.gz,/path/to/more_jsonl \
  --model_name_or_path <hf-model> \
  --ckpt ckpt_best.pt \
  --decoder greedy \
  --batch_size 8 \
  --out_dir eval_out \
  --num_visualize 100 \
  --max_len 200 \
  --fastq_out eval_out/preds.fastq \
  --fastq_q 20
```

### Evaluate from tokens/reference npy pairs

```bash
basecall-eval \
  --npy_paths /path/to/reads_or_dir \
  --model_name_or_path <hf-model> \
  --ckpt ckpt_best.pt \
  --decoder greedy \
  --batch_size 8 \
  --out_dir eval_out
```

### All evaluation arguments

- `--jsonl_paths`: comma-separated `.jsonl.gz` files or folders (uses `text` as tokens and `bases` as reference).
- `--npy_paths`: comma-separated folders or `tokens_*.npy`/`reference_*.npy` files (uses token/reference pairs).
- `--model_name_or_path`: HuggingFace model ID or local path.
- `--ckpt`: checkpoint path.
- `--decoder`: `greedy`, `beam`, or `crf` (crf requires k2 or ont-koi + `ctc_crf.py` adapter).
- `--beam_width`: beam width for `beam` decoder.
- `--batch_size`, `--num_workers`: eval dataloader controls.
- `--out_dir`: output directory for metrics/plots.
- `--num_visualize`: number of reads to visualize (default: 100).
- `--max_len`: max length shown in heatmaps.
- `--fastq_out`: optional FASTQ output path for predicted sequences.
- `--fastq_q`: fixed Phred quality value for FASTQ output (default: 20).
- `--hidden_layer`: which backbone hidden state to use (default: -1).
- `--recursive`: scan subfolders for `.jsonl.gz` or tokens/reference `.npy`.

### Outputs

- `eval_out/metrics.json`: PBMA, read-level PBMA, exact-match rate, error ratios, base-level error patterns (mismatch matrix + insertion/deletion base counts), length histograms (pred/ref), and deletion position distribution.
- `eval_out/heatmap.png`: overall alignment heatmap.
- `eval_out/aligned_reads/read_XXX.png`: per-read aligned heatmaps.
- `eval_out/sequences.jsonl`: per-read predicted/reference sequences for visualization samples.
- `eval_out/preds.fastq`: optional FASTQ output for predicted sequences.

---

## 4) Inference (jsonl -> fastq)

```bash
basecall-infer \
  --ckpt ckpt_best.pt \
  --model_name_or_path <hf-model> \
  --jsonl_gz reads.jsonl.gz \
  --out out.fastq \
  --decoder greedy
```

### All inference arguments

**Model & input**
- `--ckpt`, `--model_name_or_path`.
- `--jsonl_gz`: input JSONL.GZ with `{"read_id": "...", "text": "<|bwav:...|>..."}` records.
- `--out`: output FASTQ path.
- `--device`, `--amp`.
- `--max_tokens`: maximum token count per chunk when splitting long inputs.
- `--overlap`: overlap token count between chunks.
- `--overlap_bases`: optional max overlap (bases) for sequence-based trimming.
- `--batch_size`: number of reads per inference batch.
- `--hidden_layer`: which backbone hidden state to use (default: -1).

**Decoding**
- `--decoder`: `greedy`, `beam`, or `crf` (crf requires k2 or ont-koi + `ctc_crf.py` adapter).
- `--beam_width`: beam search width.
- `--beam_q`: fixed Q score for non-greedy output.

**Chunking**
- Long `text` fields are split into token chunks, decoded independently, then concatenated.
- Overlap trimming is sequence-based: the suffix of the previous chunk output is matched against the prefix of the next chunk output, up to `overlap_bases` (or a proportional estimate from `--overlap`).

---

## 5) Notes

- Ensure the **tokenization rules** used during training match the inference signal quantization logic.
- For nested data layouts, use `--recursive`.
- For best accuracy, consider beam search during evaluation and inference.
- CTC-CRF options require k2 or ont-koi and the provided `ctc_crf.py` adapter.
- Set `CTC_CRF_BACKEND=koi` to enable koi if available; default is k2.

## 5.1) Inspect checkpoint head config

```bash
basecall-inspect --ckpt ckpt_best.pt
```

This prints inferred head settings and suggested `basecall-eval`/`basecall-infer` flags.

## 6) Minimal runnable demo

```bash
# 1) Train (auto split)
basecall-train \
  --jsonl_paths /path/to/data \
  --model_name_or_path <hf-model> \
  --output_dir outputs

# 2) Evaluate
basecall-eval \
  --jsonl_paths /path/to/data \
  --model_name_or_path <hf-model> \
  --ckpt outputs/ckpt_last.pt \
  --decoder greedy

# 3) Infer (JSONL -> fastq)
basecall-infer \
  --ckpt outputs/ckpt_last.pt \
  --model_name_or_path <hf-model> \
  --jsonl_gz reads.jsonl.gz \
  --out out.fastq
```
