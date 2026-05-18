from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video-features", required=True)
    p.add_argument("--audio-features", required=True)
    p.add_argument("--video-train-split", required=True)
    p.add_argument("--audio-train-split", required=True)
    p.add_argument("--video-split", required=True)
    p.add_argument("--audio-split", required=True)
    p.add_argument("--relevance-file", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--preproc-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random_dir = out_dir / "random_baseline"
    rule_dir = out_dir / "rule_based_baseline"
    learned_dir = out_dir / "learned_model_eval"
    random_dir.mkdir(parents=True, exist_ok=True)
    rule_dir.mkdir(parents=True, exist_ok=True)
    learned_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_random_baseline.py",
            "--video-split", args.video_split,
            "--audio-split", args.audio_split,
            "--relevance-file", args.relevance_file,
            "--output-dir", str(random_dir),
            "--num-runs", str(args.num_runs),
            "--seed", str(args.seed),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_rule_based_baseline.py",
            "--video-features", args.video_features,
            "--audio-features", args.audio_features,
            "--video-train-split", args.video_train_split,
            "--audio-train-split", args.audio_train_split,
            "--video-split", args.video_split,
            "--audio-split", args.audio_split,
            "--relevance-file", args.relevance_file,
            "--output-dir", str(rule_dir),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_retrieval.py",
            "--checkpoint", args.checkpoint,
            "--video-features", args.video_features,
            "--audio-features", args.audio_features,
            "--video-split", args.video_split,
            "--audio-split", args.audio_split,
            "--relevance-file", args.relevance_file,
            "--preproc-dir", args.preproc_dir,
            "--output-dir", str(learned_dir),
        ],
        check=True,
    )

    rnd = load_json(random_dir / "random_summary.json")
    rule = load_json(rule_dir / "rule_based_summary.json")
    learned = load_json(learned_dir / "metrics.json")

    rows = [
        {
            "method": "random_baseline",
            "Recall@1": rnd["Recall@1"],
            "Recall@3": rnd["Recall@3"],
            "Recall@5": rnd["Recall@5"],
            "MRR": rnd["MRR"],
            "num_videos": rnd["num_videos"],
            "Recall@1_std": rnd.get("Recall@1_std"),
            "Recall@3_std": rnd.get("Recall@3_std"),
            "Recall@5_std": rnd.get("Recall@5_std"),
            "MRR_std": rnd.get("MRR_std"),
        },
        {
            "method": "rule_based_baseline",
            "Recall@1": rule["Recall@1"],
            "Recall@3": rule["Recall@3"],
            "Recall@5": rule["Recall@5"],
            "MRR": rule["MRR"],
            "num_videos": rule["num_videos"],
            "Recall@1_std": None,
            "Recall@3_std": None,
            "Recall@5_std": None,
            "MRR_std": None,
        },
        {
            "method": "learned_model",
            "Recall@1": learned["Recall@1"],
            "Recall@3": learned["Recall@3"],
            "Recall@5": learned["Recall@5"],
            "MRR": learned["MRR"],
            "num_videos": learned["num_videos"],
            "Recall@1_std": None,
            "Recall@3_std": None,
            "Recall@5_std": None,
            "MRR_std": None,
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_test_summary.csv", index=False)
    (out_dir / "comparison_test_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
