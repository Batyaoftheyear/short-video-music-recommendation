from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import basename_normalized, detect_id_columns, normalize_path


@dataclass
class FeatureLookup:
    features: np.ndarray
    by_path: dict[str, int]
    by_basename: dict[str, int]

    def get_index(self, key: str) -> int | None:
        p = normalize_path(key)
        if not p:
            return None
        if p in self.by_path:
            return self.by_path[p]
        b = basename_normalized(p)
        return self.by_basename.get(b)


def build_feature_lookup(df: pd.DataFrame, feature_matrix: np.ndarray) -> FeatureLookup:
    if len(df) != len(feature_matrix):
        raise ValueError(
            "Feature/DataFrame length mismatch: "
            f"len(df)={len(df)} vs len(feature_matrix)={len(feature_matrix)}. "
            "feature_matrix must be built from this exact DataFrame without row reordering."
        )
    filepath_col, filename_col = detect_id_columns(df)
    by_path: dict[str, int] = {}
    by_basename: dict[str, int] = {}
    for i in range(len(df)):
        path_val = df.iloc[i][filepath_col] if filepath_col else None
        name_val = df.iloc[i][filename_col] if filename_col else None
        p = normalize_path(path_val)
        if p:
            by_path.setdefault(p, i)
            by_basename.setdefault(basename_normalized(p), i)
        n = normalize_path(name_val)
        if n:
            by_basename.setdefault(Path(n).name, i)
    return FeatureLookup(features=feature_matrix, by_path=by_path, by_basename=by_basename)


class TripletDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, triplets_df: pd.DataFrame, video_lookup: FeatureLookup, audio_lookup: FeatureLookup, verbose: bool = True) -> None:
        self.video_lookup = video_lookup
        self.audio_lookup = audio_lookup
        self.samples: list[tuple[int, int, int]] = []
        skipped = 0
        for row in triplets_df.itertuples(index=False):
            v = getattr(row, "video_file", None)
            p = getattr(row, "positive_audio_file", None)
            n = getattr(row, "negative_audio_file", None)
            vi = video_lookup.get_index(v) if v is not None else None
            pi = audio_lookup.get_index(p) if p is not None else None
            ni = audio_lookup.get_index(n) if n is not None else None
            if vi is None or pi is None or ni is None:
                skipped += 1
                continue
            self.samples.append((vi, pi, ni))
        if verbose and skipped > 0:
            print(f"[TripletDataset] skipped {skipped} rows due to missing feature match")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vi, pi, ni = self.samples[idx]
        return {
            "video": torch.from_numpy(self.video_lookup.features[vi]),
            "positive_audio": torch.from_numpy(self.audio_lookup.features[pi]),
            "negative_audio": torch.from_numpy(self.audio_lookup.features[ni]),
        }


def build_relevance_map(df: pd.DataFrame) -> dict[str, set[str]]:
    required = {"video_file", "audio_file"}
    if not required.issubset(df.columns):
        raise ValueError(f"relevance file must contain columns {required}")
    if "is_relevant" in df.columns:
        work = df[df["is_relevant"].fillna(0).astype(int) == 1].copy()
    else:
        work = df.copy()
    rel: dict[str, set[str]] = {}
    for row in work.itertuples(index=False):
        v = normalize_path(getattr(row, "video_file"))
        a = normalize_path(getattr(row, "audio_file"))
        if not v or not a:
            continue
        rel.setdefault(v, set()).add(a)
        rel.setdefault(basename_normalized(v), set()).add(a)
        rel.setdefault(v, set()).add(basename_normalized(a))
        rel.setdefault(basename_normalized(v), set()).add(basename_normalized(a))
    return rel


def get_relevant_audios(relevance_map: dict[str, set[str]], video_key: str, all_audio_paths: list[str] | None = None) -> set[str]:
    video_norm = normalize_path(video_key)
    if not video_norm:
        return set()
    rel_raw = relevance_map.get(video_norm, set())
    if not rel_raw:
        rel_raw = relevance_map.get(basename_normalized(video_norm), set())
    if not rel_raw:
        return set()

    if not all_audio_paths:
        return rel_raw

    path_set = set(all_audio_paths)
    base_to_path: dict[str, str] = {}
    for p in all_audio_paths:
        b = basename_normalized(p)
        if b and b not in base_to_path:
            base_to_path[b] = p

    resolved: set[str] = set()
    for item in rel_raw:
        item_norm = normalize_path(item)
        if item_norm in path_set:
            resolved.add(item_norm)
            continue
        b = basename_normalized(item_norm)
        if b in base_to_path:
            resolved.add(base_to_path[b])
    return resolved
