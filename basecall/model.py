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
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1)
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
            expected = self.n_base ** (self.state_len + 1)
            if n_scores != expected:
                raise ValueError("CRF score dim must match no-blank size when blank_score is set.")
            scores = scores.view(bsz, t_len, n_scores // self.n_base, self.n_base)
            scores = F.pad(scores, (1, 0), value=float(self.blank_score))
            scores = scores.view(bsz, t_len, -1)
        return scores


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

        n_base = head_crf_n_base if head_crf_n_base is not None else (len(ID2BASE) - 1)
        if head_crf_state_len is None:
            if n_base <= 1:
                raise ValueError("Cannot infer head_crf_state_len with n_base <= 1.")
            base = num_classes / (n_base + 1)
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
