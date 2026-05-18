from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CosineTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)
        pos_dist = 1.0 - pos_sim
        neg_dist = 1.0 - neg_sim
        return torch.relu(pos_dist - neg_dist + self.margin).mean()
