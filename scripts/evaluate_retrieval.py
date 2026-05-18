from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.datasets import build_relevance_map, get_relevant_audios
from models.inference import build_audio_embedding_index, encode_video_batch, rank_audio_for_video
from models.metrics import aggregate_metrics, recall_at_k, reciprocal_rank
from models.preprocessing import basename_normalized, load_preprocessor, normalize_path
from models.retrieval_model import TwoTowerRetrievalModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video-features", required=True)
    p.add_argument("--audio-features", required=True)
    p.add_argument("--video-split", required=True)
    p.add_argument("--audio-split", required=True)
    p.add_argument("--relevance-file", required=True)
    p.add_argument("--preproc-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--use-audio-duration", action="store_true")
    return p.parse_args()


def _split_ids(df: pd.DataFrame) -> tuple[set[str], set[str]]:
    paths = {normalize_path(v) for v in df.get("filepath", pd.Series(dtype=str)).dropna().tolist()}
    names = {basename_normalized(v) for v in df.get("filename", pd.Series(dtype=str)).dropna().tolist()}
    return paths, names


def _filter_by_split(features_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    path_ids, name_ids = _split_ids(split_df)
    keep = []
    for row in features_df.itertuples(index=False):
        fp = normalize_path(getattr(row, "filepath", ""))
        fn = basename_normalized(getattr(row, "filename", ""))
        keep.append((fp and fp in path_ids) or (fn and fn in name_ids))
    return features_df.loc[keep].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_features_df = pd.read_csv(args.video_features)
    audio_features_df = pd.read_csv(args.audio_features)
    video_split_df = pd.read_csv(args.video_split)
    audio_split_df = pd.read_csv(args.audio_split)
    relevance_df = pd.read_csv(args.relevance_file)

    video_df = _filter_by_split(video_features_df, video_split_df)
    audio_df = _filter_by_split(audio_features_df, audio_split_df)

    video_preproc = load_preprocessor(Path(args.preproc_dir) / "video_preprocessor.joblib")
    audio_preproc = load_preprocessor(Path(args.preproc_dir) / "audio_preprocessor.joblib")
    video_x = video_preproc.transform(video_df)
    audio_x = audio_preproc.transform(audio_df)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = TwoTowerRetrievalModel(
        video_input_dim=ckpt["video_input_dim"],
        audio_input_dim=ckpt["audio_input_dim"],
        embedding_dim=ckpt["embedding_dim"],
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    relevance = build_relevance_map(relevance_df)
    audio_embeddings = build_audio_embedding_index(model, audio_x, device)
    video_embeddings = encode_video_batch(model, video_x, device)
    audio_paths = [normalize_path(v) for v in audio_df["filepath"].tolist()]

    per_video: list[dict[str, float]] = []
    rankings: list[dict[str, Any]] = []
    for i, row in enumerate(video_df.itertuples(index=False)):
        vp = normalize_path(getattr(row, "filepath", ""))
        rel = get_relevant_audios(relevance, vp, audio_paths)
        if not rel:
            continue
        order, sims = rank_audio_for_video(video_embeddings[i], audio_embeddings)
        ranked = [audio_paths[j] for j in order.tolist()]
        per_video.append(
            {
                "video_file": vp,
                "Recall@1": recall_at_k(ranked, rel, 1),
                "Recall@3": recall_at_k(ranked, rel, 3),
                "Recall@5": recall_at_k(ranked, rel, 5),
                "MRR": reciprocal_rank(ranked, rel),
            }
        )
        for rank, idx in enumerate(order.tolist(), start=1):
            rankings.append(
                {
                    "video_file": vp,
                    "audio_file": audio_paths[idx],
                    "rank": rank,
                    "similarity": float(sims[idx]),
                    "is_relevant": 1 if audio_paths[idx] in rel else 0,
                }
            )

    metrics = aggregate_metrics(per_video)
    pd.DataFrame(rankings).to_csv(out_dir / "val_rankings.csv", index=False)
    pd.DataFrame(per_video).to_csv(out_dir / "per_video_metrics.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

