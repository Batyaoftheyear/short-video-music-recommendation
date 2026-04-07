import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import ensure_parent, resolve_path, setup_logger


def load_clip_model(device: str):
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    model.to(device)
    return model, preprocess


def sample_frames(cap: cv2.VideoCapture, num_frames: int) -> List[np.ndarray]:
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        idxs = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    else:
        # Unknown frame count fallback: read first num_frames
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    return frames


def clip_features(
    frames: List[np.ndarray], model, preprocess, device: str
) -> Dict[str, Optional[np.ndarray]]:
    embeddings = []
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            tensor = preprocess(image).unsqueeze(0).to(device)
            emb = model.encode_image(tensor)
            embeddings.append(emb.squeeze(0).cpu())
    if not embeddings:
        return {"clip_mean": None, "clip_std": None}
    stack = torch.stack(embeddings)
    return {
        "clip_mean": stack.mean(dim=0).numpy(),
        "clip_std": stack.std(dim=0).numpy(),
    }


def motion_and_color(frames: List[np.ndarray]) -> Dict[str, Optional[float]]:
    if not frames:
        return {
            "flow_mean": None,
            "flow_std": None,
            "brightness_mean": None,
            "saturation_mean": None,
        }
    flows = []
    brightness = []
    saturation = []
    prev_gray = None
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness.append(float(hsv[:, :, 2].mean()))
        saturation.append(float(hsv[:, :, 1].mean()))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.linalg.norm(flow, axis=2)
            flows.append(mag)
        prev_gray = gray
    flow_mean = float(np.mean(flows)) if flows else None
    flow_std = float(np.std(flows)) if flows else None
    return {
        "flow_mean": flow_mean,
        "flow_std": flow_std,
        "brightness_mean": float(np.mean(brightness)) if brightness else None,
        "saturation_mean": float(np.mean(saturation)) if saturation else None,
    }


def process_video(
    row, dataset_root: Path, num_frames: int, model, preprocess, device: str, logger
):
    path = resolve_path(row["filepath"], dataset_root)
    result: Dict[str, object] = {
        "filename": row.get("filename"),
        "filepath": str(path),
        "mood": row.get("mood"),
    }
    if not path.exists():
        logger.error("Missing file: %s", path)
        return result

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", path)
        return result

    frames = sample_frames(cap, num_frames=num_frames)
    cap.release()

    clip_stats = clip_features(frames, model, preprocess, device)
    flow_stats = motion_and_color(frames)

    if clip_stats["clip_mean"] is not None:
        for idx, val in enumerate(clip_stats["clip_mean"]):
            result[f"clip_mean_{idx}"] = float(val)
    if clip_stats["clip_std"] is not None:
        for idx, val in enumerate(clip_stats["clip_std"]):
            result[f"clip_std_{idx}"] = float(val)

    result.update(flow_stats)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract video features (CLIP, motion, color)."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("pexels_dataset_v2"),
        help="Path to dataset root.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("pexels_dataset_v2/metadata_actual.csv"),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("features/video_features.csv"),
        help="Where to save features.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to sample per video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device, e.g., cpu or cuda",
    )
    args = parser.parse_args()

    logger = setup_logger(Path("logs/video_features.log"), name="video_features")
    ensure_parent(args.output)

    with args.metadata.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    model, preprocess = load_clip_model(args.device)

    records: List[Dict[str, object]] = []
    for row in tqdm(rows, total=len(rows)):
        try:
            rec = process_video(
                row,
                args.dataset_root,
                args.num_frames,
                model,
                preprocess,
                args.device,
                logger,
            )
            records.append(rec)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed on %s: %s", row.get("filepath"), exc)

    ensure_parent(args.output)
    # determine columns
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
