from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.inference import build_audio_embedding_index, encode_video_batch, rank_audio_for_video
from models.preprocessing import load_preprocessor
from models.retrieval_model import TwoTowerRetrievalModel
from scripts.extract_video_features import (
    clip_features,
    load_clip_model,
    motion_and_color,
    sample_frames,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video-path", required=True)
    p.add_argument("--video-features-csv", default="")
    p.add_argument("--audio-features", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--preproc-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-frames", type=int, default=8)
    return p.parse_args()




@lru_cache(maxsize=2)
def get_clip_model(device: str):
    return load_clip_model(device)


def extract_single_video_features(video_path: Path, num_frames: int, device: str) -> dict[str, float | str | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = sample_frames(cap, num_frames=num_frames)
    cap.release()

    stats = {
        "filename": video_path.name,
        "filepath": str(video_path),
        "mood": "",
    }
    clip_model, clip_preprocess = get_clip_model(device)
    clip_stats = clip_features(frames, clip_model, clip_preprocess, device)
    if clip_stats["clip_mean"] is not None:
        for idx, val in enumerate(clip_stats["clip_mean"]):
            stats[f"clip_mean_{idx}"] = float(val)
    if clip_stats["clip_std"] is not None:
        for idx, val in enumerate(clip_stats["clip_std"]):
            stats[f"clip_std_{idx}"] = float(val)

    flow_stats = motion_and_color(frames)
    stats.update(flow_stats)
    return stats


def to_uri(path_str: str) -> str:
    try:
        return Path(path_str).resolve().as_uri()
    except Exception:
        return path_str.replace("\\", "/")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    audio_df = pd.read_csv(args.audio_features)
    if "filepath" not in audio_df.columns:
        raise ValueError("audio features CSV must contain filepath column")

    video_preproc = load_preprocessor(Path(args.preproc_dir) / "video_preprocessor.joblib")
    audio_preproc = load_preprocessor(Path(args.preproc_dir) / "audio_preprocessor.joblib")

    row = extract_single_video_features(video_path, num_frames=args.num_frames, device=args.device)
    video_df = pd.DataFrame([row])

    # Optional: align to known feature schema if provided
    if args.video_features_csv:
        vf = pd.read_csv(args.video_features_csv, nrows=1)
        for c in vf.columns:
            if c not in video_df.columns:
                video_df[c] = np.nan

    for c in video_preproc.feature_columns:
        if c not in video_df.columns:
            video_df[c] = np.nan

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
    model.eval()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)

    audio_emb = build_audio_embedding_index(model, audio_x, device)
    video_emb = encode_video_batch(model, video_x, device)[0]
    order, sims = rank_audio_for_video(video_emb, audio_emb)

    top_rows = []
    for rank, idx in enumerate(order[:5].tolist(), start=1):
        audio_path = str(audio_df.iloc[idx]["filepath"])
        top_rows.append(
            {
                "rank": rank,
                "audio_file": audio_path,
                "audio_filename": Path(audio_path).name,
                "score": float(sims[idx]),
            }
        )

    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(out_dir / "inference_top5.csv", index=False)
    (out_dir / "inference_top5.json").write_text(json.dumps(top_rows, indent=2), encoding="utf-8")

    rows_html = "".join(
        [
            f"<tr><td>{r['rank']}</td><td>{r['audio_filename']}</td><td>{r['score']:.4f}</td>"
            f"<td>{r['audio_file']}</td><td><audio controls preload='none' src='{to_uri(r['audio_file'])}'></audio></td></tr>"
            for r in top_rows
        ]
    )
    html_page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Inference Top-5</title>
<style>body{{font-family:Arial,sans-serif;margin:16px}} video{{max-width:720px;display:block;margin:8px 0}} table{{width:100%;border-collapse:collapse}} th,td{{border:1px solid #eee;padding:6px;text-align:left}} audio{{width:320px}}</style>
</head><body>
<h1>Inference demo top-5</h1>
<p><b>Input video:</b> {video_path}</p>
<video controls loop preload='metadata' src='{to_uri(str(video_path))}'></video>
<table><thead><tr><th>rank</th><th>audio filename</th><th>score</th><th>audio path</th><th>preview</th></tr></thead><tbody>{rows_html}</tbody></table>
</body></html>"""
    (out_dir / "inference_top5.html").write_text(html_page, encoding="utf-8")

    print("saved:", out_dir / "inference_top5.csv")
    print("saved:", out_dir / "inference_top5.json")
    print("saved:", out_dir / "inference_top5.html")


if __name__ == "__main__":
    main()
