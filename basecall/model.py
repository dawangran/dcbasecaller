# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from .utils import NUM_CLASSES


class BasecallHead(nn.Module):
    """CTC head with lightweight local-context blocks.

    Keep the backbone unchanged, and add depthwise 1D conv blocks to model
    local context for CTC alignment.
    """
    def __init__(
        self,
        hidden_size: int,
        num_classes: int = NUM_CLASSES,
        blank_idx: int = 0,
        kernel_size: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_pointwise: bool = True,
        use_transformer: bool = False,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding.")
        padding = kernel_size // 2
        self.blocks = nn.ModuleList()
        for _ in range(max(1, num_layers)):
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

        # discourage "all-blank" early collapse
        if self.proj.bias is not None and 0 <= blank_idx < num_classes:
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
        return self.proj(x)            # [B, T, C]


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

        self.base_head = BasecallHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            kernel_size=head_kernel_size,
            num_layers=head_layers,
            dropout=head_dropout,
            use_pointwise=head_use_pointwise,
            use_transformer=head_use_transformer,
            transformer_layers=head_transformer_layers,
            transformer_heads=head_transformer_heads,
            transformer_dropout=head_transformer_dropout,
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
