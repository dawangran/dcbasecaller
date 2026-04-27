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

### Split each file into 7:2:1 (recommended for many files)

If you have many files (e.g., 100 files) and want every file to contribute train/val/test samples:

```bash
basecall-train \
  --jsonl_paths /path/to/data \
  --group_by record_per_file \
  --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 \
  --split_seed 42 \
  --model_name_or_path <hf-model> \
  --output_dir outputs
```

### Large-scale training optimization (100+ files)

When data is very large, enable streaming to avoid loading all reads into RAM:

```bash
basecall-train \
  --jsonl_paths /path/to/data \
  --group_by record_per_file \
  --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 \
  --streaming \
  --shuffle_buffer_size 10000 \
  --steps_per_epoch 5000 \
  --num_workers 8 \
  --model_name_or_path <hf-model> \
  --output_dir outputs
```

Notes:
- `--streaming`: uses iterable/generator-style loading to reduce peak memory.
- `--shuffle_buffer_size`: approximate shuffle in streaming mode (bigger = better randomness, more RAM).
- `--steps_per_epoch`: optional in streaming mode; if omitted, training auto-estimates it from dataset size and batch size.
- In streaming + `record`/`record_per_file`, split assignment is hash-based and deterministic, so ratios are approximate (converges as data grows).

Quick parameter guide (for the command above):
- `--group_by record_per_file`: split records *inside each file* into train/val/test by ratio.
- `--train_ratio/--val_ratio/--test_ratio`: target split ratios (must sum to 1.0).
- `--streaming`: do not preload full dataset into RAM; read samples on the fly.
- `--shuffle_buffer_size 10000`: bounded shuffle window; larger value improves randomness but uses more memory.
- `--steps_per_epoch 5000`: how many optimizer steps to run each epoch when using streaming mode.
- if not sure, you can leave `--steps_per_epoch` unset (or `0`) and let the script auto-estimate it.
- `--num_workers 8`: number of dataloader worker processes; usually tune by CPU cores and storage speed.

### All training arguments

**Data & split**
- `--jsonl_paths`: comma-separated `.jsonl.gz` files or folders (uses `text` as tokens and `bases` as reference).
- `--train_jsonl_paths`, `--val_jsonl_paths`, `--test_jsonl_paths`: explicit JSONL split inputs (skip auto split).
- `--npy_paths`: comma-separated folders or `tokens_*.npy`/`reference_*.npy` files (uses token/reference pairs).
- `--train_npy_paths`, `--val_npy_paths`, `--test_npy_paths`: explicit npy split inputs (skip auto split).
- `--group_by`: `folder`, `file`, `record`, or `record_per_file` (`record_per_file` splits each file internally by the given ratios, e.g. 7:2:1).
- `--recursive`: scan subfolders for `.jsonl.gz` or tokens/reference `.npy`.
- `--train_ratio`, `--val_ratio`, `--test_ratio`: ratios for auto split.
- `--split_seed`: random seed for auto split.
- `--streaming`: stream records instead of fully materializing datasets in memory.
- `--shuffle_buffer_size`: streaming shuffle buffer size.
- `--wandb_log_alignment_every`: if >0, log one validation alignment heatmap to W&B every N epochs (plus final alignment at training end).

**Model & freezing**
- `--model_name_or_path`: for `--feature_source hidden|embedding`, this is the backbone HuggingFace model ID/local path; for `--feature_source vq_embedding`, this is the VQ tokenize model checkpoint path (used by `poregpt.tokenizers.VQETokenizer`).
- `--hidden-layer`: which hidden layer to use when `--feature_source hidden` (`-1` = last, `-2` = second last).
- `--feature_source`/`--feature-source`: choose `hidden` (default), `embedding` (`self.backbone.get_input_embeddings()`), or `vq_embedding` (tokenize-model codebook embedding; skips backbone forward).
- `--vq_device`: device used to load VQ tokenizer/codebook in `vq_embedding` mode.
- `--vq_token_batch_size`: token batch size passed to `VQETokenizer` in `vq_embedding` mode.
- `--freeze_backbone`: freeze backbone, train head only.
- `--reset_backbone_weights`: reinitialize backbone weights for ablation.
- `--unfreeze_last_n_layers`: unfreeze last N transformer layers.
- `--unfreeze_layer_start`, `--unfreeze_layer_end`: unfreeze layers in range `[start, end)`.

### Train with tokenize model codebook embeddings (`vq_embedding`)

Use this when `text/signal_str` carries token ids like `<|bwav:123|><|bwav:456|>...` (also supports JSON list string like `[123,456]` and CSV like `123,456`).

```bash
basecall-train \
  --jsonl_paths /path/to/reads.jsonl.gz \
  --model_name_or_path /path/to/tokenize_model_ckpt \
  --feature_source vq_embedding \
  --vq_device cuda \
  --vq_token_batch_size 100 \
  --output_dir outputs_vq
```

Notes:
- In `vq_embedding` mode, the dataloader uses a dedicated collate path that parses token ids from `signal_str`.
- If token ids cannot be parsed from a record, training raises a clear `ValueError` with bad indices/examples.
- `poregpt` must be installed in the runtime environment for `vq_embedding`.

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
- `--batch_size`, `--num_epochs`, `--lr`, `--weight_decay`, `--warmup_ratio`, `--warmup_steps`, `--min_lr`.
- `--warmup_steps`: if set to `>=0`, uses absolute warmup steps and overrides `--warmup_ratio`.
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
- Koi decoder advanced parameters in eval are fixed internally (`beam_cut=100`, `scale=1`, `offset=0`, `blank_score=2`, `reverse=False`) to simplify CLI usage.
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

- `--ctc_crf_blank_score`: blank score used by CTC-CRF head logits (default: 2.0; keep consistent with training).
- Koi decoder advanced parameters in inference are fixed internally (`beam_cut=100`, `scale=1`, `offset=0`, `blank_score=2`, `reverse=False`) to simplify CLI usage.
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
- Koi decoder advanced parameters are fixed internally (`beam_cut=100`, `scale=1`, `offset=0`, `blank_score=2`, `reverse=False`).

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

## 7) FAQ: JSONL processing and extracting `hidden`

### How JSONL is processed

When using `--jsonl_paths` (train/eval) or `--jsonl_gz` (infer), records are parsed as JSON objects. The common fields are:

- `text`: input token sequence (for example `<|bwav:123|><|bwav:456|>...`)
- `bases`: reference labels for supervised training/eval

Processing flow:

1. discover `.jsonl.gz` files
2. read line-by-line JSON records
3. parse `text` as model input sequence (`signal_str`)
4. parse `bases` into integer labels (`target_seq`)
5. collate/pad into batch tensors for model forward + decode/loss

### How to get hidden states

Use:

- `--feature_source hidden` (default)
- `--hidden-layer -1` for last layer (`-2` for second last, etc.)

Optional layer fusion:

- `--learnable_fuse_last_n_layers N`

If `N > 0`, the model computes a softmax-weighted sum of the last `N` hidden layers, and this takes precedence over `--hidden-layer`.

### Example

```bash
# Training from jsonl and using the last hidden layer
basecall-train \
  --jsonl_paths /data/train \
  --model_name_or_path <hf-model> \
  --feature_source hidden \
  --hidden-layer -1 \
  --output_dir outputs

# Evaluation from jsonl with learned fusion over the last 4 hidden layers
basecall-eval \
  --jsonl_paths /data/val \
  --model_name_or_path <hf-model> \
  --ckpt outputs/ckpt_best.pt \
  --feature_source hidden \
  --learnable_fuse_last_n_layers 4 \
  --out_dir eval_out
```

### How `bases` contributes to EM (Exact Match)

In evaluation, both prediction and reference are converted from id sequences to base strings (`A/C/G/T`), then compared read-by-read:

1. decode predicted ids to `pred_seq`
2. decode reference ids to `ref_seq`
3. if `pred_seq == ref_seq`, that read counts as exact match (`EM += 1`)
4. final `read_exact_match_rate = EM / number_of_reads`

Toy example:

- read1: pred=`ACGT`, ref=`ACGT` -> exact match
- read2: pred=`ACGA`, ref=`ACGT` -> not exact match
- read3: pred=`TTAA`, ref=`TTAA` -> exact match

Then `EM = 2`, total reads `= 3`, so `read_exact_match_rate = 2/3 = 0.6667`.
