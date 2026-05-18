from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


RUNS = [
    {"name": "baseline", "embedding_dim": 128, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "margin": 0.2},
    {"name": "run_1", "embedding_dim": 64, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "margin": 0.2},
    {"name": "run_2", "embedding_dim": 64, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "margin": 0.2},
    {"name": "run_3", "embedding_dim": 128, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "margin": 0.2},
    {"name": "run_4", "embedding_dim": 64, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-4, "margin": 0.2},
    {"name": "run_5", "embedding_dim": 64, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-3, "margin": 0.2},
    {"name": "run_6", "embedding_dim": 64, "dropout": 0.3, "lr": 5e-4, "weight_decay": 1e-3, "margin": 0.1},
]


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
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-audio-duration", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run in RUNS:
        run_name = run["name"]
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/train_retrieval_model.py",
            "--video-features", args.video_features,
            "--audio-features", args.audio_features,
            "--train-triplets", args.train_triplets,
            "--video-train-split", args.video_train_split,
            "--video-val-split", args.video_val_split,
            "--audio-train-split", args.audio_train_split,
            "--audio-val-split", args.audio_val_split,
            "--val-relevance", args.val_relevance,
            "--output-dir", str(run_dir),
            "--batch-size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--seed", str(args.seed),
            "--embedding-dim", str(run["embedding_dim"]),
            "--dropout", str(run["dropout"]),
            "--lr", str(run["lr"]),
            "--weight-decay", str(run["weight_decay"]),
            "--margin", str(run["margin"]),
        ]
        if args.use_audio_duration:
            cmd.append("--use-audio-duration")

        print(f"[tune] start {run_name}")
        subprocess.run(cmd, check=True)

        metrics_path = run_dir / "best_val_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics for {run_name}: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        rows.append(
            {
                "run": run_name,
                "embedding_dim": run["embedding_dim"],
                "dropout": run["dropout"],
                "lr": run["lr"],
                "weight_decay": run["weight_decay"],
                "margin": run["margin"],
                "Recall@1": metrics.get("Recall@1", 0.0),
                "Recall@3": metrics.get("Recall@3", 0.0),
                "Recall@5": metrics.get("Recall@5", 0.0),
                "MRR": metrics.get("MRR", 0.0),
                "num_videos": metrics.get("num_videos", 0),
                "best_checkpoint": str(run_dir / "best.pt"),
                "run_dir": str(run_dir),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["Recall@5", "MRR"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(out_dir / "tuning_summary.csv", index=False)
    (out_dir / "tuning_summary.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    best = df.iloc[0].to_dict()
    (out_dir / "best_run.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("[tune] best run:", best["run"])


if __name__ == "__main__":
    main()
