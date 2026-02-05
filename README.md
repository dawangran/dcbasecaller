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

For decoding/training with CTC-CRF and Bonito-style metrics, also ensure these runtime dependencies are available:

```bash
pip install parasail edlib
# and install ont-koi matching your torch/cuda build
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

### Quick run (Bonito-style CRF settings)

```bash
basecall-train \
  --jsonl_paths /path/to/data \
  --model_name_or_path <hf-model> \
  --quick \
  --output_dir outputs_quick
```

`--quick` expands to:
- `--freeze_backbone`
- `--ctc_crf_state_len 4`
- `--ctc_crf_blank_score 2`
- `--head_output_scale 5`
- `--head_output_activation tanh`

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
- `--reset_backbone_weights`: reinitialize backbone weights for ablation.
- `--unfreeze_last_n_layers`: unfreeze last N transformer layers.
- `--unfreeze_layer_start`, `--unfreeze_layer_end`: unfreeze layers in range `[start, end)`.

**Head options**
- `--head_output_activation`: optional activation for head logits (e.g. `tanh` for Bonito-style scaling).
- `--head_output_scale`: optional scalar multiplier for head logits (applied after activation).


**Optimization**
- `--batch_size`, `--num_epochs`, `--lr`, `--weight_decay`, `--warmup_ratio`, `--min_lr`.
- `--ctc_crf_state_len`: Bonito CTC-CRF state length (controls CRF head output classes).
- `--ctc_crf_blank_score`: fixed blank score for CTC-CRF (blank is not trained).
- `--acc_balanced`: use Bonito balanced accuracy for validation/checkpointing.
- `--acc_min_coverage`: minimum reference coverage required to count a read for accuracy.

**Checkpointing & loading**
- `--resume_ckpt`: resume from `ckpt_last.pt` (model/optim/sched/epoch/best_acc).
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
  --beam_width 32 \
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
  --beam_width 32 \
  --batch_size 8 \
  --out_dir eval_out
```

### All evaluation arguments

- `--jsonl_paths`: comma-separated `.jsonl.gz` files or folders (uses `text` as tokens and `bases` as reference).
- `--npy_paths`: comma-separated folders or `tokens_*.npy`/`reference_*.npy` files (uses token/reference pairs).
- `--model_name_or_path`: HuggingFace model ID or local path.
- `--ckpt`: checkpoint path.
- `--beam_width`: beam width for ont-koi `beam_search`.
- `--koi_beam_cut`, `--koi_scale`, `--koi_offset`, `--koi_blank_score`, `--koi_reverse`: parameters for the Koi `beam_search` decoder.
- `--acc_balanced`: use Bonito balanced accuracy in metrics.
- `--acc_min_coverage`: minimum reference coverage required to count a read for accuracy.
- `--batch_size`, `--num_workers`: eval dataloader controls.
- `--out_dir`: output directory for metrics/plots.
- `--num_visualize`: number of reads to visualize (default: 100).
- `--max_len`: max length shown in heatmaps.
- `--fastq_out`: optional FASTQ output path for predicted sequences.
- `--fastq_q`: fixed Phred quality value for FASTQ output (default: 20).
- `--hidden_layer`: which backbone hidden state to use (default: -1).
- `--recursive`: scan subfolders for `.jsonl.gz` or tokens/reference `.npy`.

### Outputs

- `eval_out/metrics.json`: Bonito-style accuracy, read-level accuracy, exact-match rate, error ratios, base-level error patterns (mismatch matrix + insertion/deletion base counts), length histograms (pred/ref), and deletion position distribution.

---

## 4) Inference

### Basic usage

```bash
basecall-infer \
  --jsonl_gz /path/to/reads.jsonl.gz \
  --model_name_or_path <hf-model> \
  --ckpt ckpt_best.pt \
  --beam_width 32 \
  --out outputs/preds.fastq
```

### Decoder parameters (ont-koi `beam_search`)

- `--koi_beam_cut`: beam cut value (default: 100.0).
- `--koi_scale`: scale applied to scores (default: 1.0).
- `--koi_offset`: offset applied to scores (default: 0.0).
- `--koi_blank_score`: blank score used by the decoder (default: 2.0).
- `--koi_reverse`: reverse output sequence (useful for reverse-complemented models).

### Notes for Bonito-style CTC-CRF training/inference

- Set `--ctc_crf_state_len` during training to match the CTC-CRF head size.
- `--ctc_crf_blank_score` fixes the blank score (blank is not trained) and should stay consistent for decoding.
- For Bonito-style logit scaling, use `--head_output_activation tanh` and `--head_output_scale 5` so scores match the expected range.

### All inference arguments

**Model & input**
- `--ckpt`, `--model_name_or_path`.
- `--jsonl_gz`: input JSONL.GZ with `{"read_id": "...", "text": "<|bwav:...|>..."}` records.
- `--out`: output FASTQ path.
- `--device`, `--amp`.
- `--max_tokens`: maximum token count per chunk when splitting long inputs.
- `--overlap`: overlap token count between chunks.
- `--batch_size`: number of reads per inference batch.
- `--hidden_layer`: which backbone hidden state to use (default: -1).

**Decoding**
- `--beam_width`: beam search width.
- `--beam_q`: fixed Q score for output FASTQ.
- `--koi_beam_cut`, `--koi_scale`, `--koi_offset`, `--koi_blank_score`, `--koi_reverse`: Koi beam-search parameters.

**Chunking**
- Long `text` fields are split into token chunks, decoded independently, then concatenated.
- Overlap trimming follows chunk boundaries: each chunk keeps the non-overlap core based on `--max_tokens` and `--overlap`.

---

## 4.1) Loss and accuracy definitions

- **Training loss** uses Bonito-style CTC-CRF negative log-likelihood (`ctc_crf_loss`) on packed targets with per-read `input_lengths` (derived from `attention_mask`).
- **Validation/Test accuracy (`acc`)** is Bonito-style alignment accuracy from `koi_beam_search_decode` + parasail alignment (`batch_bonito_accuracy`, unit: %).
- **Balanced accuracy** (`--acc_balanced`) uses `(match - ins) / (match + mismatch + del)`; default uses `match / (match + ins + mismatch + del)`.
- **CRF decode accuracy (`crf_acc`)** is reported from Viterbi decoding (`ctc_crf.decode`) and uses the same Bonito accuracy function.

## 5) Notes

- Ensure the **tokenization rules** used during training match the inference signal quantization logic.
- For nested data layouts, use `--recursive`.
- For best accuracy, consider beam search during evaluation and inference.
- CTC-CRF options require ont-koi and the provided `ctc_crf.py` adapter.

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
  --ckpt outputs/ckpt_last.pt

# 3) Infer (JSONL -> fastq)
basecall-infer \
  --ckpt outputs/ckpt_last.pt \
  --model_name_or_path <hf-model> \
  --jsonl_gz reads.jsonl.gz \
  --out out.fastq
```
