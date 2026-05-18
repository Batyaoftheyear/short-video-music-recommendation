import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

# Keep execution inside the active environment only.
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import librosa
import numpy as np
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import ensure_parent, resolve_path, setup_logger


def compute_features(y: np.ndarray, sr: int, n_mfcc: int, logger, path: Path) -> Dict[str, float]:
    features: Dict[str, float] = {}

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_arr = np.asarray(tempo).reshape(-1)
        features["bpm"] = float(tempo_arr[0]) if tempo_arr.size > 0 else 0.0
    except Exception as exc:
        logger.warning("Tempo failed for %s: %s", path, exc)

    try:
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(rms.mean())
        features["rms_std"] = float(rms.std())
    except Exception as exc:
        logger.warning("RMS failed for %s: %s", path, exc)

    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["centroid_mean"] = float(centroid.mean())
        features["centroid_std"] = float(centroid.std())
    except Exception as exc:
        logger.warning("Centroid failed for %s: %s", path, exc)

    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(zcr.mean())
        features["zcr_std"] = float(zcr.std())
    except Exception as exc:
        logger.warning("ZCR failed for %s: %s", path, exc)

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for i in range(n_mfcc):
            features[f"mfcc_{i+1}_mean"] = float(mfcc[i].mean())
            features[f"mfcc_{i+1}_std"] = float(mfcc[i].std())
    except Exception as exc:
        logger.warning("MFCC failed for %s: %s", path, exc)

    return features


def process_track(row, dataset_root: Path, sr: int, n_mfcc: int, logger):
    path = resolve_path(row["filepath"], dataset_root)

    duration_meta = row.get("duration")
    duration_val = None
    if duration_meta not in (None, ""):
        try:
            duration_val = float(duration_meta)
        except (TypeError, ValueError):
            duration_val = None

    result: Dict[str, object] = {
        "filename": row.get("filename"),
        "filepath": str(path),
        "mood": row.get("mood"),
        "duration": duration_val,
    }

    if not path.exists():
        logger.error("Missing audio file: %s", path)
        return result

    try:
        y, used_sr = librosa.load(str(path), sr=sr, mono=True)
        if result["duration"] is None and used_sr > 0:
            result["duration"] = float(len(y) / used_sr)
        result.update(compute_features(y, used_sr, n_mfcc, logger, path))
    except Exception as exc:
        logger.error("Failed to process %s: %s", path, exc)

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract audio features with librosa.")
    parser.add_argument("--dataset-root", type=Path, default=Path("jamendo_dataset_trimmed"))
    parser.add_argument("--metadata", type=Path, default=Path("jamendo_dataset_trimmed/metadata_actual.csv"))
    parser.add_argument("--output", type=Path, default=Path("features/audio_features_trimmed.csv"))
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-mfcc", type=int, default=13)
    args = parser.parse_args()

    logger = setup_logger(Path("logs/audio_features_trimmed.log"), name="audio_features_trimmed")
    ensure_parent(args.output)

    with args.metadata.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records: List[Dict[str, object]] = []
    for row in tqdm(rows, total=len(rows)):
        records.append(process_track(row, args.dataset_root, args.sr, args.n_mfcc, logger))

    all_keys = sorted({k for rec in records for k in rec.keys()})
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
