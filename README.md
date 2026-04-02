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

Distributed or mixed-precision training is managed via Hugging Face Accelerate. Launch multi-process jobs with `accelerate launch` instead of manually wiring DDP settings:

```bash
accelerate launch --num_processes 4 -m basecall.train_ddp_multifolder \
  --jsonl_paths /path/to/data \
  --model_name_or_path <hf-model> \
  --output_dir outputs
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
- `--ctc_crf_state_len 5`
- `--ctc_crf_blank_score 0`
- `--head_output_scale 5`
- `--head_output_activation tanh`
- `--head_type ctc_crf`
- `--pre_ctc_module none`


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


### Shuffle across multiple files (ignore file boundaries)

If each source file has different distribution and you want global random mixing before split:

```bash
basecall-train \
  --jsonl_paths /path/to/data1,/path/to/data2,/path/to/data3 \
  --group_by record \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
  --split_seed 42 \
  --model_name_or_path <hf-model> \
  --output_dir outputs
```

### All training arguments

**Data & split**
- `--jsonl_paths`: comma-separated `.jsonl.gz` files or folders (uses `text` as tokens and `bases` as reference).
- `--train_jsonl_paths`, `--val_jsonl_paths`, `--test_jsonl_paths`: explicit JSONL split inputs (skip auto split).
- `--npy_paths`: comma-separated folders or `tokens_*.npy`/`reference_*.npy` files (uses token/reference pairs).
- `--train_npy_paths`, `--val_npy_paths`, `--test_npy_paths`: explicit npy split inputs (skip auto split).
- `--group_by`: `folder`, `file`, or `record` (auto split granularity; use `record` to shuffle all reads across files and then split).
- `--recursive`: scan subfolders for `.jsonl.gz` or tokens/reference `.npy`.
- `--train_ratio`, `--val_ratio`, `--test_ratio`: ratios for auto split.
- `--split_seed`: random seed for auto split.

**Model & freezing**
- `--model_name_or_path`: HuggingFace model ID or local path.
- `--hidden-layer`: which hidden layer to use when `--feature_source hidden` (`-1` = last, `-2` = second last).
- `--feature_source`/`--feature-source`: choose `hidden` (default) or `embedding` (`self.backbone.get_input_embeddings()`).
- `--freeze_backbone`: freeze backbone, train head only.
- `--reset_backbone_weights`: reinitialize backbone weights for ablation.
- `--unfreeze_last_n_layers`: unfreeze last N transformer layers.
- `--unfreeze_layer_start`, `--unfreeze_layer_end`: unfreeze layers in range `[start, end)`.

**Head options**
- `--head_output_activation`: optional activation for head logits (e.g. `tanh` for Bonito-style scaling).
- `--head_output_scale`: optional scalar multiplier for head logits (applied after activation).
- `--head_type`: select output head (`ctc` or `ctc_crf`).
- `--pre_head_type`: optional adapter before CRF head (`none`, `bilstm`, `transformer`, `tcn`).
- `--pre_ctc_module`: alias of `--pre_head_type`.
- `--pre_head_transformer_nhead`: attention heads when `--pre_head_type transformer` (ignored for other pre-head types, including `tcn`).

Pre-head quick reference:

- `none`: no extra temporal module; fastest baseline.
- `bilstm`: 1-layer BiLSTM pre-head (`hidden_dim=128`, bidirectional output dim = 256).
- `transformer`: 1-layer TransformerEncoder pre-head (set attention heads with `--pre_head_transformer_nhead`).
- `tcn`: residual dilated 1D-conv pre-head (fixed defaults in current code: `kernel_size=3`, `num_layers=4`, dilations `1,2,4,8`, `dropout=0.1`).

Example (enable TCN pre-head in training):

```bash
basecall-train \
  --jsonl_paths /path/to/data \
  --model_name_or_path <hf-model> \
  --head_type ctc_crf \
  --pre_head_type tcn
```


**Optimization**
- `--batch_size`, `--num_epochs`, `--lr`, `--weight_decay`, `--warmup_ratio`, `--min_lr`.
- `--ctc_crf_state_len`: Bonito CTC-CRF state length (controls CRF head output classes).
- `--ctc_crf_blank_score`: fixed blank score for CTC-CRF (blank is not trained).
- `--train_decoder`: choose `ctc_viterbi`, `ctc_crf` (fp32), or `koi` for accuracy/blank metrics.
- `--koi_blank_score`: blank score used by Koi beam search when `--train_decoder koi`.
- `--clip_grad_norm`: gradient clipping threshold (0 disables clipping).
- `--acc_balanced`: use Bonito balanced accuracy for validation/checkpointing.
- `--acc_min_coverage`: minimum reference coverage required to count a read for accuracy.

**Checkpointing & loading**
- `--resume_ckpt`: resume from `ckpt_last.pt` (model/optim/sched/epoch/best_acc).
- `--pretrained_ckpt`: load pretrained weights into the model.
- `--pretrained_strict`, `--pretrained_key`.
- `--save_every`, `--save_best`.

**Distributed/Logging**
- `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_run_name`.
- `--wandb_group`: group related runs (e.g. same experiment family with different conditions).
- `--wandb_job_type`: run type shown in W&B (default `train`).
- Launch distributed runs with `accelerate launch ... basecall.train_ddp_multifolder`.
- `--find_unused_parameters` and `--ddp_broadcast_buffers` are forwarded into Accelerate's DDP wrapper settings.
- `--ddp_backend` explicitly selects the process-group backend (`gloo` or `nccl`).
- `--ddp_backend_fallback` allows automatic fallback from NCCL to GLOO when the selected GPU backend cannot initialize cleanly.
- When using `--ddp_backend nccl`, set `NCCL_SOCKET_IFNAME` if your runtime requires an explicit socket interface.

Example for grouped condition sweeps:

```bash
basecall-train \
  --jsonl_paths /path/to/data \
  --model_name_or_path <hf-model> \
  --use_wandb \
  --wandb_project DNAmodel_basecall \
  --wandb_group exp_ctc_vs_crf \
  --wandb_run_name ctc_crf_prehead_none_lr1e3
```

**Runtime**
- `--num_workers`, `--seed`, `--log_interval`, `--output_dir`, `--amp`.

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
- `--ctc_crf_blank_score`: blank score used by CTC-CRF head logits (should match training setting).
- `--decoder`: choose `auto`, `ctc_viterbi`, `koi`, or `ctc_crf` for prediction/metrics (`auto`: CTC->`ctc_viterbi`, CTC-CRF->`ctc_crf`).
- `--head_type`: optional override for checkpoint head type (`ctc` or `ctc_crf`, default auto-infer).
- `--acc_balanced`: use Bonito balanced accuracy in metrics.
- `--acc_min_coverage`: minimum reference coverage required to count a read for accuracy.
- `--batch_size`, `--num_workers`: eval dataloader controls.
- `--out_dir`: output directory for metrics/plots.
- `--num_visualize`: number of reads to visualize (default: 100).
- `--max_len`: max length shown in heatmaps.
- `--fastq_out`: optional FASTQ output path for predicted sequences.
- `--fastq_q`: fixed Phred quality value for FASTQ output (default: 20).
- `--hidden_layer`: which backbone hidden state to use when `--feature_source hidden` (default: -1).
- `--feature_source`/`--feature-source`: choose `hidden` (default) or `embedding` for model features before pre-head/head.
- `--pre_head_type`, `--pre_head_transformer_nhead`: pre-head settings; `--pre_head_type` defaults to `auto` and infers from checkpoint (`none`/`bilstm`/`transformer`/`tcn`), but you can override manually.
- `--recursive`: scan subfolders for `.jsonl.gz` or tokens/reference `.npy`.
- Eval startup prints a model summary (`pre_head`, `head`, parameter counts) and full module structure for quick sanity-checking.

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
- `--koi_blank_score`: blank score used by Koi decoder (default: 2.0).
- `--ctc_crf_blank_score`: blank score used by CTC-CRF head logits (default: 2.0; keep consistent with training).
- `--koi_reverse`: reverse output sequence (useful for reverse-complemented models).
- `--decoder`: choose `auto`, `ctc_viterbi`, `koi`, or `ctc_crf` for prediction (`auto`: CTC->`ctc_viterbi`, CTC-CRF->`ctc_crf`); CTC-CRF forces fp32 decoding.
- `--head_type`: optional override for checkpoint head type (`ctc` or `ctc_crf`, default auto-infer).

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
- `--hidden_layer`: which backbone hidden state to use when `--feature_source hidden` (default: -1).
- `--feature_source`/`--feature-source`: choose `hidden` (default) or `embedding` for model features before pre-head/head.
- `--pre_head_type`, `--pre_head_transformer_nhead`: pre-head settings; `--pre_head_type` defaults to `auto` and infers from checkpoint (`none`/`bilstm`/`transformer`/`tcn`), but you can override manually.
- Inference startup prints a model summary (`pre_head`, `head`, parameter counts) and full module structure for quick sanity-checking.

**Decoding**
- `--beam_width`: beam search width.
- `--beam_q`: fixed Q score for output FASTQ.
- `--koi_beam_cut`, `--koi_scale`, `--koi_offset`, `--koi_blank_score`, `--koi_reverse`: Koi beam-search parameters.

**Chunking**
- Long `text` fields are split into token chunks, decoded independently, then concatenated.
- Overlap trimming follows chunk boundaries: each chunk keeps the non-overlap core based on `--max_tokens` and `--overlap`.

---

## 4.1) Loss and accuracy definitions

- **Training loss** is head-dependent: `ctc_crf_loss` from `basecall/ctc_crf.py` for `--head_type ctc_crf`, and `ctc_label_smoothing_loss` from `basecall/ctc.py` for `--head_type ctc`.
- **Validation/Test accuracy (`acc`)** uses the selected decoder (`--train_decoder`) and Bonito-style parasail alignment (`batch_bonito_accuracy`, unit: %).
- **Balanced accuracy** (`--acc_balanced`) uses `(match - ins) / (match + mismatch + del)`; default uses `match / (match + ins + mismatch + del)`.
- **CRF decode accuracy (`crf_acc`)** is only reported when `--train_decoder ctc_crf`.

## 5) Notes

- Ensure the **tokenization rules** used during training match the inference signal quantization logic.
- For nested data layouts, use `--recursive`.
- For best accuracy, consider beam search during evaluation and inference.
- CTC-CRF options require ont-koi and the provided `ctc_crf.py` adapter.

## 5.1) Inspect checkpoint head config

```bash
basecall-inspect --ckpt ckpt_best.pt
```

This prints inferred head settings plus a checkpoint-side structure summary (inferred pre-head type and parameter totals), which helps verify whether the checkpoint matches your expected model design before running eval/infer.

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
