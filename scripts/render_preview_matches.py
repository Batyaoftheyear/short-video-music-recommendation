import argparse
import csv
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("render_preview_matches")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger


def resolve_existing(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render preview videos with matched audio.")
    parser.add_argument("--relevance-csv", type=Path, default=Path("features/pairs/test_relevance.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/previews"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--ffmpeg", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(".").resolve()
    logger = setup_logger(Path("logs/render_preview_matches.log"))

    ffmpeg_bin = args.ffmpeg.strip() or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        logger.error("ffmpeg not found. Install ffmpeg or pass --ffmpeg <path>")
        print("ffmpeg not found. Install ffmpeg and rerun, or pass --ffmpeg <path to ffmpeg.exe>.")
        return

    df = pd.read_csv(args.relevance_csv)
    if "rank_within_relevant_set" in df.columns:
        df = df[df["rank_within_relevant_set"] == args.rank]

    df = df.drop_duplicates(subset=["video_file"]).head(args.limit).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = []

    rendered = 0
    failed = 0

    for i, row in df.iterrows():
        video_path = resolve_existing(str(row["video_file"]), project_root)
        audio_path = resolve_existing(str(row["relevant_audio_file"]), project_root)

        out_file = args.output_dir / f"preview_{i+1:02d}_{video_path.stem}__{audio_path.stem}.mp4"
        selection_rows.append(
            {
                "video_file": str(video_path),
                "audio_file": str(audio_path),
                "output_file": str(out_file),
                "mood": row.get("mood", ""),
                "relevance_score": row.get("relevance_score", ""),
            }
        )

        if not video_path.exists() or not audio_path.exists():
            logger.error("Missing input file(s): %s | %s", video_path, audio_path)
            failed += 1
            continue

        if out_file.exists() and not args.overwrite:
            logger.info("Skip existing: %s", out_file)
            continue

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(out_file),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rendered += 1
        except Exception as exc:
            logger.error("Render failed for %s: %s", out_file, exc)
            failed += 1

    selection_csv = args.output_dir / "preview_selection.csv"
    with selection_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_file", "audio_file", "output_file", "mood", "relevance_score"])
        writer.writeheader()
        writer.writerows(selection_rows)

    print(f"Selected pairs: {len(selection_rows)}")
    print(f"Rendered previews: {rendered}")
    print(f"Failed: {failed}")
    print(f"Selection table: {selection_csv}")


if __name__ == "__main__":
    main()
