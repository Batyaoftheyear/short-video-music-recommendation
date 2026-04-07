import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import ensure_parent, resolve_path, setup_logger


def compute_features(y: np.ndarray, sr: int, n_mfcc: int) -> Dict[str, float]:
    features: Dict[str, float] = {}

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["bpm"] = float(tempo)

    # Energy
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(rms.mean())
    features["rms_std"] = float(rms.std())

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["centroid_mean"] = float(centroid.mean())
    features["centroid_std"] = float(centroid.std())

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(zcr.mean())
    features["zcr_std"] = float(zcr.std())

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(mfcc[i].mean())
        features[f"mfcc_{i+1}_std"] = float(mfcc[i].std())
    return features


def process_track(row, dataset_root: Path, sr: int, n_mfcc: int, logger):
    path = resolve_path(row["filepath"], dataset_root)
    result: Dict[str, object] = {
        "filename": row.get("filename"),
        "filepath": str(path),
        "mood": row.get("mood"),
    }
    if not path.exists():
        logger.error("Missing audio file: %s", path)
        return result

    try:
        y, _ = librosa.load(path, sr=sr)
        feats = compute_features(y, sr, n_mfcc)
        result.update(feats)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to process %s: %s", path, exc)
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract audio features with librosa.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("jamendo_dataset"),
        help="Path to dataset root.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("jamendo_dataset/metadata_actual.csv"),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("features/audio_features.csv"),
        help="Where to save features.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Target sample rate.",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=13,
        help="Number of MFCC coefficients.",
    )
    args = parser.parse_args()

    logger = setup_logger(Path("logs/audio_features.log"), name="audio_features")
    ensure_parent(args.output)

    with args.metadata.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    records: List[Dict[str, object]] = []
    for row in tqdm(rows, total=len(rows)):
        rec = process_track(row, args.dataset_root, args.sr, args.n_mfcc, logger)
        records.append(rec)

    ensure_parent(args.output)
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
