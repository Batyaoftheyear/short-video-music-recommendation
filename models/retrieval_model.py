from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _act(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    return nn.GELU()


class VideoTower(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, dropout: float = 0.2, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            _act(activation),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            _act(activation),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioTower(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, dropout: float = 0.2, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            _act(activation),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTowerRetrievalModel(nn.Module):
    def __init__(self, video_input_dim: int, audio_input_dim: int, embedding_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.video_tower = VideoTower(video_input_dim, embedding_dim, dropout)
        self.audio_tower = AudioTower(audio_input_dim, embedding_dim, dropout)

    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        z = self.video_tower(x)
        return F.normalize(z, p=2, dim=-1)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        z = self.audio_tower(x)
        return F.normalize(z, p=2, dim=-1)

    def forward(self, video_x: torch.Tensor, audio_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_video(video_x), self.encode_audio(audio_x)
