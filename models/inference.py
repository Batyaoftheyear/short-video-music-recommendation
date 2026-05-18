from __future__ import annotations

import numpy as np
import torch

from .retrieval_model import TwoTowerRetrievalModel


@torch.no_grad()
def encode_video_batch(model: TwoTowerRetrievalModel, batch: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
    return model.encode_video(x).cpu().numpy()


@torch.no_grad()
def encode_audio_batch(model: TwoTowerRetrievalModel, batch: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
    return model.encode_audio(x).cpu().numpy()


@torch.no_grad()
def build_audio_embedding_index(model: TwoTowerRetrievalModel, audio_features: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    if audio_features.shape[0] == 0:
        raise ValueError("audio feature set is empty")
    chunks = []
    for start in range(0, len(audio_features), batch_size):
        chunks.append(encode_audio_batch(model, audio_features[start:start + batch_size], device))
    return np.vstack(chunks)


def rank_audio_for_video(video_embedding: np.ndarray, audio_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sims = audio_embeddings @ video_embedding
    order = np.argsort(-sims)
    return order, sims
