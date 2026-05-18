from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.datasets import build_relevance_map, get_relevant_audios
from models.metrics import aggregate_metrics, recall_at_k, reciprocal_rank
from models.preprocessing import basename_normalized, normalize_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video-split", required=True)
    p.add_argument("--audio-split", required=True)
    p.add_argument("--relevance-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _read_ids(df: pd.DataFrame) -> list[str]:
    if "filepath" in df.columns:
        ids = [normalize_path(v) for v in df["filepath"].dropna().tolist()]
    elif "filename" in df.columns:
        ids = [basename_normalized(v) for v in df["filename"].dropna().tolist()]
    else:
        raise ValueError("split file must have filepath or filename column")
    ids = [x for x in ids if x]
    if not ids:
        raise ValueError("split file contains no valid ids")
    return ids


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_split = pd.read_csv(args.video_split)
    audio_split = pd.read_csv(args.audio_split)
    relevance_df = pd.read_csv(args.relevance_file)

    video_ids = _read_ids(video_split)
    audio_ids = _read_ids(audio_split)
    relevance = build_relevance_map(relevance_df)

    rng = random.Random(args.seed)
    run_rows: list[dict[str, float]] = []

    for run_idx in range(args.num_runs):
        metrics_per_video = []
        for vid in video_ids:
            rel = get_relevant_audios(relevance, vid, audio_ids)
            if not rel:
                continue
            ranked = audio_ids[:]
            rng.shuffle(ranked)
            metrics_per_video.append(
                {
                    "Recall@1": recall_at_k(ranked, rel, 1),
                    "Recall@3": recall_at_k(ranked, rel, 3),
                    "Recall@5": recall_at_k(ranked, rel, 5),
                    "MRR": reciprocal_rank(ranked, rel),
                }
            )

        agg = aggregate_metrics(metrics_per_video)
        run_rows.append(agg)

    runs_df = pd.DataFrame(run_rows)
    mean_row = runs_df[["Recall@1", "Recall@3", "Recall@5", "MRR", "num_videos"]].mean(numeric_only=True)
    std_row = runs_df[["Recall@1", "Recall@3", "Recall@5", "MRR"]].std(numeric_only=True, ddof=0)

    summary = {
        "method": "random_baseline",
        "num_runs": args.num_runs,
        "Recall@1": float(mean_row["Recall@1"]),
        "Recall@3": float(mean_row["Recall@3"]),
        "Recall@5": float(mean_row["Recall@5"]),
        "MRR": float(mean_row["MRR"]),
        "num_videos": int(round(float(mean_row["num_videos"]))),
        "Recall@1_std": float(std_row["Recall@1"]),
        "Recall@3_std": float(std_row["Recall@3"]),
        "Recall@5_std": float(std_row["Recall@5"]),
        "MRR_std": float(std_row["MRR"]),
    }

    runs_df.to_csv(out_dir / "random_runs.csv", index=False)
    (out_dir / "random_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out_dir / "random_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
