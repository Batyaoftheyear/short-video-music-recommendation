from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.datasets import TripletDataset, build_feature_lookup, build_relevance_map, get_relevant_audios
from models.inference import build_audio_embedding_index, encode_video_batch, rank_audio_for_video
from models.losses import CosineTripletLoss
from models.metrics import aggregate_metrics, recall_at_k, reciprocal_rank
from models.preprocessing import (
    basename_normalized,
    fit_modality_preprocessor,
    normalize_path,
    pick_numeric_feature_columns,
    save_preprocessor,
)
from models.retrieval_model import TwoTowerRetrievalModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video-features", required=True)
    p.add_argument("--audio-features", required=True)
    p.add_argument("--train-triplets", required=True)
    p.add_argument("--video-train-split", required=True)
    p.add_argument("--video-val-split", required=True)
    p.add_argument("--audio-train-split", required=True)
    p.add_argument("--audio-val-split", required=True)
    p.add_argument("--val-relevance", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--use-audio-duration", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def evaluate_retrieval(
    model: TwoTowerRetrievalModel,
    video_val_df: pd.DataFrame,
    audio_val_df: pd.DataFrame,
    video_val_x: np.ndarray,
    audio_val_x: np.ndarray,
    relevance_map: dict[str, set[str]],
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    audio_embeddings = build_audio_embedding_index(model, audio_val_x, device)
    video_embeddings = encode_video_batch(model, video_val_x, device)
    audio_paths = [normalize_path(v) for v in audio_val_df["filepath"].tolist()]
    records: list[dict[str, float]] = []
    ranking_rows: list[dict[str, Any]] = []
    for i, row in enumerate(video_val_df.itertuples(index=False)):
        video_path = normalize_path(getattr(row, "filepath", ""))
        rel = get_relevant_audios(relevance_map, video_path, audio_paths)
        if not rel:
            continue
        order, sims = rank_audio_for_video(video_embeddings[i], audio_embeddings)
        ranked_ids = [audio_paths[j] for j in order.tolist()]
        metrics_row = {
            "Recall@1": recall_at_k(ranked_ids, rel, 1),
            "Recall@3": recall_at_k(ranked_ids, rel, 3),
            "Recall@5": recall_at_k(ranked_ids, rel, 5),
            "MRR": reciprocal_rank(ranked_ids, rel),
        }
        records.append(metrics_row)
        for rank, idx in enumerate(order.tolist(), start=1):
            ranking_rows.append(
                {
                    "video_file": video_path,
                    "audio_file": audio_paths[idx],
                    "rank": rank,
                    "similarity": float(sims[idx]),
                    "is_relevant": 1 if audio_paths[idx] in rel else 0,
                }
            )
    return aggregate_metrics(records), pd.DataFrame(ranking_rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_features_df = pd.read_csv(args.video_features)
    audio_features_df = pd.read_csv(args.audio_features)
    train_triplets_df = pd.read_csv(args.train_triplets)
    video_train_df = pd.read_csv(args.video_train_split)
    video_val_df = pd.read_csv(args.video_val_split)
    audio_train_df = pd.read_csv(args.audio_train_split)
    audio_val_df = pd.read_csv(args.audio_val_split)
    val_relevance_df = pd.read_csv(args.val_relevance)

    video_train_feat_df = _filter_by_split(video_features_df, video_train_df)
    video_val_feat_df = _filter_by_split(video_features_df, video_val_df)
    audio_train_feat_df = _filter_by_split(audio_features_df, audio_train_df)
    audio_val_feat_df = _filter_by_split(audio_features_df, audio_val_df)
    if video_train_feat_df.empty or audio_train_feat_df.empty:
        raise RuntimeError("Empty train split after feature/split matching.")

    video_cols = pick_numeric_feature_columns(video_features_df)
    audio_cols = pick_numeric_feature_columns(audio_features_df, use_audio_duration=args.use_audio_duration)
    print(f"[features] video dim={len(video_cols)}")
    print(f"[features] audio dim={len(audio_cols)}")
    print(f"[features] video columns={video_cols}")
    print(f"[features] audio columns={audio_cols}")
    (out_dir / "video_feature_columns.json").write_text(json.dumps(video_cols, indent=2), encoding="utf-8")
    (out_dir / "audio_feature_columns.json").write_text(json.dumps(audio_cols, indent=2), encoding="utf-8")

    video_preproc = fit_modality_preprocessor(video_train_feat_df, video_cols)
    audio_preproc = fit_modality_preprocessor(audio_train_feat_df, audio_cols)
    save_preprocessor(video_preproc, out_dir / "video_preprocessor.joblib")
    save_preprocessor(audio_preproc, out_dir / "audio_preprocessor.joblib")

    video_train_x = video_preproc.transform(video_train_feat_df)
    video_val_x = video_preproc.transform(video_val_feat_df)
    audio_train_x = audio_preproc.transform(audio_train_feat_df)
    audio_val_x = audio_preproc.transform(audio_val_feat_df)

    video_lookup = build_feature_lookup(video_train_feat_df, video_train_x)
    audio_lookup = build_feature_lookup(audio_train_feat_df, audio_train_x)
    ds = TripletDataset(train_triplets_df, video_lookup, audio_lookup, verbose=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TwoTowerRetrievalModel(
        video_input_dim=video_train_x.shape[1],
        audio_input_dim=audio_train_x.shape[1],
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    loss_fn = CosineTripletLoss(margin=args.margin)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    relevance_map = build_relevance_map(val_relevance_df)
    history: list[dict[str, float]] = []
    best_r5 = -1.0
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in dl:
            v = batch["video"].to(device=device, dtype=torch.float32)
            p = batch["positive_audio"].to(device=device, dtype=torch.float32)
            n = batch["negative_audio"].to(device=device, dtype=torch.float32)
            v_emb = model.encode_video(v)
            p_emb = model.encode_audio(p)
            n_emb = model.encode_audio(n)
            loss = loss_fn(v_emb, p_emb, n_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else 0.0

        val_metrics, val_rank_df = evaluate_retrieval(
            model=model,
            video_val_df=video_val_feat_df,
            audio_val_df=audio_val_feat_df,
            video_val_x=video_val_x,
            audio_val_x=audio_val_x,
            relevance_map=relevance_map,
            device=device,
        )
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        print(
            f"[epoch {epoch}] loss={train_loss:.4f} R@1={val_metrics['Recall@1']:.4f} "
            f"R@3={val_metrics['Recall@3']:.4f} R@5={val_metrics['Recall@5']:.4f} MRR={val_metrics['MRR']:.4f}"
        )

        state = {
            "model_state_dict": model.state_dict(),
            "video_input_dim": video_train_x.shape[1],
            "audio_input_dim": audio_train_x.shape[1],
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
        }
        torch.save(state, out_dir / "last.pt")
        if val_metrics["Recall@5"] > best_r5:
            best_r5 = val_metrics["Recall@5"]
            wait = 0
            torch.save(state, out_dir / "best.pt")
            (out_dir / "best_val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
            val_rank_df.to_csv(out_dir / "val_rankings.csv", index=False)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[early_stop] no Recall@5 improvement for {args.patience} epochs")
                break

    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[done] best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

