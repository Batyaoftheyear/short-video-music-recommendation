import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List

import librosa
import soundfile as sf
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import ensure_parent, setup_logger


MOODS = ["relaxing", "melancholic", "energetic", "happy", "epic", "romantic"]
SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def parse_jamendo_id(filename: str) -> str:
    match = re.search(r"_([0-9]+)$", Path(filename).stem)
    return match.group(1) if match else ""


def collect_audio_files(input_root: Path) -> List[Path]:
    files: List[Path] = []
    for mood in MOODS:
        mood_dir = input_root / mood
        if not mood_dir.exists():
            continue
        for fp in mood_dir.iterdir():
            if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTS:
                files.append(fp)
    return sorted(files)


def trim_center(y, sr: int, target_duration: float):
    target_samples = int(target_duration * sr)
    if len(y) <= target_samples:
        return y, False
    start = max((len(y) - target_samples) // 2, 0)
    end = start + target_samples
    return y[start:end], True


def write_metadata(rows: List[Dict[str, object]], output_csv: Path) -> None:
    ensure_parent(output_csv)
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "filename",
        "filepath",
        "mood",
        "extension",
        "file_size_bytes",
        "duration",
        "jamendo_id",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_file(
    src: Path,
    output_root: Path,
    target_duration: float,
    audio_format: str,
    skip_existing: bool,
    logger,
):
    mood = src.parent.name.lower()
    out_dir = output_root / mood
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = f".{audio_format.lower()}"
    dst = out_dir / f"{src.stem}{ext}"

    if skip_existing and dst.exists():
        return {"status": "skipped", "row": None}

    try:
        y, sr = librosa.load(str(src), sr=None, mono=True)
        trimmed, was_trimmed = trim_center(y, sr, target_duration)
        sf.write(str(dst), trimmed, sr)

        duration = len(trimmed) / sr if sr > 0 else 0.0
        row = {
            "filename": dst.name,
            "filepath": str(dst.relative_to(output_root)).replace("/", "\\"),
            "mood": mood,
            "extension": dst.suffix.lower(),
            "file_size_bytes": dst.stat().st_size,
            "duration": float(duration),
            "jamendo_id": parse_jamendo_id(dst.name),
        }
        return {"status": "trimmed" if was_trimmed else "unchanged", "row": row}
    except Exception as exc:
        logger.error("Failed file: %s (%s)", src, exc)
        return {"status": "error", "row": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim audio dataset to central fixed-length chunks.")
    parser.add_argument("--input-root", type=Path, default=Path("jamendo_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("jamendo_dataset_trimmed"))
    parser.add_argument("--target-duration", type=float, default=30.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--audio-format", type=str, default="wav", choices=["wav", "flac"])
    parser.add_argument("--log-path", type=Path, default=Path("logs/trim_audio_dataset.log"))
    args = parser.parse_args()

    logger = setup_logger(args.log_path, name="trim_audio")
    audio_files = collect_audio_files(args.input_root)

    processed = 0
    trimmed = 0
    unchanged = 0
    skipped = 0
    errors = 0
    rows: List[Dict[str, object]] = []

    for src in tqdm(audio_files, desc="Trimming audio"):
        processed += 1
        result = process_file(
            src=src,
            output_root=args.output_root,
            target_duration=args.target_duration,
            audio_format=args.audio_format,
            skip_existing=args.skip_existing,
            logger=logger,
        )
        status = result["status"]
        row = result["row"]
        if status == "trimmed":
            trimmed += 1
            if row:
                rows.append(row)
        elif status == "unchanged":
            unchanged += 1
            if row:
                rows.append(row)
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1

    metadata_path = args.output_root / "metadata_actual.csv"
    write_metadata(rows, metadata_path)

    print("Trim summary:")
    print(f"  processed: {processed}")
    print(f"  trimmed: {trimmed}")
    print(f"  left_as_is: {unchanged}")
    print(f"  skipped_existing: {skipped}")
    print(f"  errors: {errors}")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
