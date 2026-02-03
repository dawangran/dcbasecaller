# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .utils import NUM_CLASSES, ID2BASE


class LinearCRFEncoder(nn.Module):
    def __init__(
        self,
        insize: int,
        n_base: int,
        state_len: int,
        bias: bool = True,
        scale: float | None = None,
        activation: str | None = None,
        blank_score: float | None = None,
        expand_blanks: bool = True,
        permute: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if n_base <= 0:
            raise ValueError("n_base must be >= 1.")
        if state_len < 0:
            raise ValueError("state_len must be >= 0.")
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base ** (state_len + 1)
        self.linear = nn.Linear(insize, size, bias=bias)
        if activation is None:
            self.activation = None
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "gelu":
            self.activation = torch.nn.functional.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.permute = permute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.permute is not None:
            x = x.permute(*self.permute)
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None and self.expand_blanks:
            if not scores.is_contiguous():
                scores = scores.contiguous()
            bsz, t_len, n_scores = scores.shape
            if n_scores % self.n_base != 0:
                raise ValueError("CRF score dim must be divisible by n_base for blank expansion.")
            scores = F.pad(
                scores.view(bsz, t_len, n_scores // self.n_base, self.n_base),
                (1, 0),
                value=float(self.blank_score),
            ).view(bsz, t_len, -1)
        return scores


class BasecallHead(nn.Module):
    """CTC head with lightweight local-context blocks.

    Keep the backbone unchanged, and add depthwise 1D conv blocks to model
    local context for CTC alignment.
    """
    def __init__(
        self,
        hidden_size: int,
        num_classes: int | None = NUM_CLASSES,
        blank_idx: int | None = 0,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_pointwise: bool = True,
        use_transformer: bool = False,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        output_activation: str | None = None,
        output_scale: float | None = None,
        crf_blank_score: float | None = None,
        crf_n_base: int | None = None,
        crf_expand_blanks: bool = True,
    ):
        super().__init__()
        if use_transformer and num_layers > 0:
            raise ValueError("Convolution blocks and transformer encoder are mutually exclusive.")
        self.norm = nn.LayerNorm(hidden_size)
        if num_layers < 0:
            raise ValueError("num_layers must be >= 0.")
        self.blocks = nn.ModuleList()
        if num_layers > 0:
            if kernel_size % 2 == 0:
                raise ValueError("kernel_size must be odd for symmetric padding.")
            padding = kernel_size // 2
            for _ in range(num_layers):
                dwconv = nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=hidden_size,
                )
                pwconv = (
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
                    if use_pointwise
                    else nn.Identity()
                )
                self.blocks.append(nn.Sequential(dwconv, pwconv))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.transformer = None
        if use_transformer and transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dropout=transformer_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.proj = nn.Linear(hidden_size, num_classes)
        self.output_activation = output_activation
        if output_scale is None:
            self.output_scale = None
        else:
            self.register_buffer("output_scale", torch.tensor(float(output_scale)))
        self.crf_blank_score = crf_blank_score
        self.crf_n_base = crf_n_base
        self.crf_expand_blanks = crf_expand_blanks

        # discourage "all-blank" early collapse
        if self.proj.bias is not None and blank_idx is not None and 0 <= blank_idx < num_classes:
            with torch.no_grad():
                self.proj.bias[blank_idx] = -2.0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, H]
        x = self.norm(hidden)
        x = x.transpose(1, 2)          # [B, H, T]
        for block in self.blocks:
            residual = x
            x = block(x)
            x = self.act(x)
            x = self.dropout(x)
            x = x + residual
        x = x.transpose(1, 2)          # [B, T, H]
        if self.transformer is not None:
            x = self.transformer(x)
        x = self.proj(x)            # [B, T, C]
        if self.output_activation:
            if self.output_activation == "tanh":
                x = torch.tanh(x)
            elif self.output_activation == "relu":
                x = torch.relu(x)
            else:
                raise ValueError(f"Unknown output_activation: {self.output_activation}")
        if self.output_scale is not None:
            x = x * self.output_scale
        if self.crf_blank_score is not None and self.crf_expand_blanks:
            if self.crf_n_base is None or self.crf_n_base <= 0:
                raise ValueError("crf_n_base must be set when expanding CRF blanks.")
            if not x.is_contiguous():
                x = x.contiguous()
            bsz, t_len, n_scores = x.shape
            if n_scores % self.crf_n_base != 0:
                raise ValueError("CRF score dim must be divisible by crf_n_base for blank expansion.")
            x = F.pad(
                x.view(bsz, t_len, n_scores // self.crf_n_base, self.crf_n_base),
                (1, 0),
                value=float(self.crf_blank_score),
            ).view(bsz, t_len, -1)
        return x


class BasecallModel(nn.Module):
    """
    input_ids: [B, T]
    output logits_btc: [B, T, C]  (给 CTC 用)
    """
    def __init__(
        self,
        model_path: str,
        num_classes: int = NUM_CLASSES,
        hidden_layer: int = -1,          # 选哪一层 hidden_states
        freeze_backbone: bool = False,   # ✅ 新增：是否冻结基座（默认不冻结，保持你原行为）
        reset_backbone_weights: bool = False,  # ✅ 可选：重置基座权重用于消融
        unfreeze_last_n_layers: int = 0,  # 可选：仅解冻最后 N 层（其余保持冻结）
        unfreeze_layer_start: int | None = None,
        unfreeze_layer_end: int | None = None,
        head_kernel_size: int = 5,
        head_layers: int = 2,
        head_dropout: float = 0.1,
        head_use_pointwise: bool = True,
        head_use_transformer: bool = False,
        head_transformer_layers: int = 1,
        head_transformer_heads: int = 4,
        head_transformer_dropout: float = 0.1,
        head_linear: bool = False,
        head_blank_idx: int | None = 0,
        head_output_activation: str | None = None,
        head_output_scale: float | None = None,
        head_crf_blank_score: float | None = None,
        head_crf_n_base: int | None = None,
        head_crf_state_len: int | None = None,
        head_crf_expand_blanks: bool = True,
    ):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_last_n_layers = max(0, int(unfreeze_last_n_layers))
        self.unfreeze_layer_start = unfreeze_layer_start
        self.unfreeze_layer_end = unfreeze_layer_end

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.backbone = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if reset_backbone_weights:
            if hasattr(self.backbone, "init_weights"):
                self.backbone.init_weights()
            else:
                self.backbone.apply(self._init_backbone_weights)

        # 省显存：关闭 cache（很多 decoder-only 默认开）
        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

        # ✅ 冻结基座（核心）
        if self.freeze_backbone or self.unfreeze_last_n_layers > 0 or unfreeze_layer_start is not None or unfreeze_layer_end is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if self.freeze_backbone and self.unfreeze_last_n_layers == 0:
                self.backbone.eval()  # 固定 dropout 等推理行为（推荐）

        # ✅ 可选：仅解冻最后 N 层
        if self.unfreeze_last_n_layers > 0 or unfreeze_layer_start is not None or unfreeze_layer_end is not None:
            layers = self._get_transformer_layers()
            n_layers = len(layers)
            if unfreeze_layer_start is not None or unfreeze_layer_end is not None:
                start = 0 if unfreeze_layer_start is None else int(unfreeze_layer_start)
                end = n_layers if unfreeze_layer_end is None else int(unfreeze_layer_end)
                if start < 0:
                    start = n_layers + start
                if end < 0:
                    end = n_layers + end
                if not 0 <= start <= end <= n_layers:
                    raise ValueError(f"Invalid unfreeze layer range: [{start}, {end}) with {n_layers} layers.")
                target_layers = layers[start:end]
            else:
                n_unfreeze = min(self.unfreeze_last_n_layers, n_layers)
                target_layers = layers[-n_unfreeze:]
            for layer in target_layers:
                for p in layer.parameters():
                    p.requires_grad = True

        hidden_size = (
            getattr(self.backbone.config, "hidden_size", None)
            or getattr(self.backbone.config, "d_model", None)
            or getattr(self.backbone.config, "n_embd", None)
        )
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from backbone config.")

        if num_classes is None:
            num_classes = NUM_CLASSES

        if head_use_transformer:
            head_layers = 0

        if head_linear:
            n_base = head_crf_n_base if head_crf_n_base is not None else (len(ID2BASE) - 1)
            if head_crf_state_len is None:
                if n_base <= 1:
                    raise ValueError("Cannot infer head_crf_state_len with n_base <= 1.")
                if head_crf_blank_score is None:
                    base = num_classes / (n_base + 1)
                else:
                    base = num_classes
                state_len = math.log(base, n_base) - 1
                if not math.isclose(state_len, round(state_len)):
                    raise ValueError("Unable to infer head_crf_state_len from num_classes and n_base.")
                head_crf_state_len = int(round(state_len))
            self.base_head = LinearCRFEncoder(
                insize=hidden_size,
                n_base=n_base,
                state_len=head_crf_state_len,
                bias=True,
                scale=head_output_scale,
                activation=head_output_activation,
                blank_score=head_crf_blank_score,
                expand_blanks=head_crf_expand_blanks,
            )
        else:
            self.base_head = BasecallHead(
                hidden_size=hidden_size,
                num_classes=num_classes,
                blank_idx=head_blank_idx,
                kernel_size=head_kernel_size,
                num_layers=head_layers,
                dropout=head_dropout,
                use_pointwise=head_use_pointwise,
                use_transformer=head_use_transformer,
                transformer_layers=head_transformer_layers,
                transformer_heads=head_transformer_heads,
                transformer_dropout=head_transformer_dropout,
                output_activation=head_output_activation,
                output_scale=head_output_scale,
                crf_blank_score=head_crf_blank_score,
                crf_n_base=head_crf_n_base,
                crf_expand_blanks=head_crf_expand_blanks,
            )

    def _get_transformer_layers(self) -> nn.ModuleList:
        candidates = (
            ("encoder", "layer"),
            ("transformer", "h"),
            ("model", "layers"),
            ("layers",),
            ("h",),
            ("blocks",),
        )
        for path in candidates:
            obj = self.backbone
            for attr in path:
                if not hasattr(obj, attr):
                    obj = None
                    break
                obj = getattr(obj, attr)
            if obj is not None and isinstance(obj, (nn.ModuleList, list, tuple)):
                return nn.ModuleList(list(obj))
        raise ValueError("Cannot locate transformer layers for partial unfreezing.")

    @staticmethod
    def _init_backbone_weights(module: nn.Module) -> None:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def train(self, mode: bool = True):
        """✅ 防止外面 model.train() 把 backbone 切回 train（dropout 会动）"""
        super().train(mode)
        if (
            self.freeze_backbone
            and self.unfreeze_last_n_layers == 0
            and self.unfreeze_layer_start is None
            and self.unfreeze_layer_end is None
        ):
            self.backbone.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # tuple(len = n_layers + 1)

        try:
            hidden = hidden_states[self.hidden_layer]
        except IndexError:
            raise ValueError(
                f"hidden_layer={self.hidden_layer} out of range "
                f"(num hidden states = {len(hidden_states)})"
            )

        logits_btc = self.base_head(hidden)
        return logits_btc
