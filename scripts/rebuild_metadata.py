import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import librosa

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import ensure_parent, optional_relative, setup_logger


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
MOODS = {"relaxing", "melancholic", "energetic", "happy", "epic", "romantic"}


def parse_numeric_id(name: str) -> Optional[str]:
    """Extract trailing digits from filename like mood_123456.ext."""
    match = re.search(r"_([0-9]+)$", name.split(".")[0])
    return match.group(1) if match else None


def video_info(path: Path, logger) -> Dict[str, Optional[float]]:
    width = height = duration = None
    try:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0 and frames:
                duration = float(frames / fps)
        cap.release()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Video read failed: %s (%s)", path, exc)
    orientation = None
    if width and height:
        if width > height:
            orientation = "horizontal"
        elif height > width:
            orientation = "vertical"
        else:
            orientation = "square"
    return {
        "width": width,
        "height": height,
        "duration": duration,
        "orientation": orientation,
    }


def audio_info(path: Path, logger) -> Dict[str, Optional[float]]:
    duration = None
    try:
        duration = float(librosa.get_duration(path=str(path)))
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Audio read failed: %s (%s)", path, exc)
    return {"duration": duration}


def collect_metadata(
    root: Path, mode: str, logger
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    problems = 0
    counts = Counter()
    for mood_dir in sorted(root.iterdir()):
        if not mood_dir.is_dir():
            continue
        mood = mood_dir.name.lower()
        if mood not in MOODS:
            continue
        for file_path in mood_dir.iterdir():
            if not file_path.is_file():
                continue
            suffix = file_path.suffix.lower()
            if mode == "video" and suffix not in VIDEO_EXTS:
                continue
            if mode == "audio" and suffix not in AUDIO_EXTS:
                continue
            try:
                base = file_path.name
                size = file_path.stat().st_size if file_path.exists() else None
                row: Dict[str, object] = {
                    "filename": base,
                    "filepath": optional_relative(file_path, root),
                    "mood": mood,
                    "extension": suffix,
                    "file_size_bytes": size,
                }
                if mode == "video":
                    row.update(video_info(file_path, logger))
                    row["pexels_id"] = parse_numeric_id(base) or ""
                else:
                    row.update(audio_info(file_path, logger))
                    row["jamendo_id"] = parse_numeric_id(base) or ""
                rows.append(row)
                counts[mood] += 1
            except Exception as exc:  # pragma: no cover - defensive
                problems += 1
                logger.error("Failed to process %s: %s", file_path, exc)
    return {"rows": rows, "counts": counts, "problems": problems}


def save_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    ensure_parent(output_path)
    if not rows:
        output_path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(name: str, counts: Counter, problems: int, total: int) -> None:
    print(f"[{name}] files per mood:")
    for mood, count in sorted(counts.items()):
        print(f"  {mood}: {count}")
    print(f"[{name}] problems: {problems}")
    print(f"[{name}] total: {total}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild metadata based on actual files on disk."
    )
    parser.add_argument(
        "--pexels-root",
        type=Path,
        default=Path("pexels_dataset_v2"),
        help="Path to Pexels dataset root.",
    )
    parser.add_argument(
        "--jamendo-root",
        type=Path,
        default=Path("jamendo_dataset"),
        help="Path to Jamendo dataset root.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("logs/rebuild_metadata.log"),
        help="Where to write the log.",
    )
    args = parser.parse_args()

    logger = setup_logger(args.log_path, name="rebuild")

    print("Rebuilding metadata based on disk contents...\n")

    # Video dataset
    video_result = collect_metadata(args.pexels_root, mode="video", logger=logger)
    video_output = args.pexels_root / "metadata_actual.csv"
    save_csv(video_result["rows"], video_output)
    print_summary(
        "Pexels",
        video_result["counts"],
        video_result["problems"],
        len(video_result["rows"]),
    )

    # Audio dataset
    audio_result = collect_metadata(args.jamendo_root, mode="audio", logger=logger)
    audio_output = args.jamendo_root / "metadata_actual.csv"
    save_csv(audio_result["rows"], audio_output)
    print_summary(
        "Jamendo",
        audio_result["counts"],
        audio_result["problems"],
        len(audio_result["rows"]),
    )

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
