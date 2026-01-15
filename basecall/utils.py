# -*- coding: utf-8 -*-
import random
import numpy as np
import torch

# 标签定义
LABELS = ["N", "A", "C", "G", "T"]  # 0,1,2,3,4
NUM_CLASSES = len(LABELS)
BLANK_IDX = 0  # CTC blank

ID2BASE = {
    0: "N",
    1: "A",
    2: "C",
    3: "G",
    4: "T",
}

BASE2ID = {b: i for i, b in ID2BASE.items()}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
