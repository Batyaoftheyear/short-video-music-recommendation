from __future__ import annotations

import argparse
import json
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
    p.add_argument("--video-features", required=True)
    p.add_argument("--audio-features", required=True)
    p.add_argument("--video-train-split", required=True)
    p.add_argument("--audio-train-split", required=True)
    p.add_argument("--video-split", required=True)
    p.add_argument("--audio-split", required=True)
    p.add_argument("--relevance-file", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _split_ids(df: pd.DataFrame) -> tuple[set[str], set[str]]:
    paths = {normalize_path(v) for v in df.get("filepath", pd.Series(dtype=str)).dropna().tolist()}
    names = {basename_normalized(v) for v in df.get("filename", pd.Series(dtype=str)).dropna().tolist()}
    return paths, names


def _filter_by_split(features_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    path_ids, name_ids = _split_ids(split_df)
    keep = []
    for row in features_df.itertuples(index=False):
        fp = normalize_path(getattr(row, "filepath", ""))
        fn = basename_normalized(getattr(row, "filename", ""))
        keep.append((fp and fp in path_ids) or (fn and fn in name_ids))
    return features_df.loc[keep].reset_index(drop=True)


def _fit_minmax(train_df: pd.DataFrame, col: str) -> tuple[float, float]:
    arr = train_df[col].to_numpy(float)
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def _transform_minmax(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _pair_sim(a: float, b: float) -> float:
    return float(np.exp(-abs(a - b)))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vf = pd.read_csv(args.video_features)
    af = pd.read_csv(args.audio_features)
    v_train_split = pd.read_csv(args.video_train_split)
    a_train_split = pd.read_csv(args.audio_train_split)
    vs = pd.read_csv(args.video_split)
    a_s = pd.read_csv(args.audio_split)
    rel_df = pd.read_csv(args.relevance_file)

    v_train_df = _filter_by_split(vf, v_train_split)
    a_train_df = _filter_by_split(af, a_train_split)
    vdf = _filter_by_split(vf, vs)
    adf = _filter_by_split(af, a_s)
    relevance = build_relevance_map(rel_df)

    required_video_cols = ["flow_mean", "flow_std", "brightness_mean", "saturation_mean", "filepath"]
    required_audio_cols = ["bpm", "rms_mean", "rms_std", "centroid_mean", "centroid_std", "filepath"]
    for c in required_video_cols:
        if c not in vdf.columns or c not in v_train_df.columns:
            raise ValueError(f"missing video column: {c}")
    for c in required_audio_cols:
        if c not in adf.columns or c not in a_train_df.columns:
            raise ValueError(f"missing audio column: {c}")

    # Fit normalization on train only
    v_flow_mean = _transform_minmax(vdf["flow_mean"].to_numpy(float), *_fit_minmax(v_train_df, "flow_mean"))
    v_flow_std = _transform_minmax(vdf["flow_std"].to_numpy(float), *_fit_minmax(v_train_df, "flow_std"))
    v_brightness = _transform_minmax(vdf["brightness_mean"].to_numpy(float), *_fit_minmax(v_train_df, "brightness_mean"))
    v_saturation = _transform_minmax(vdf["saturation_mean"].to_numpy(float), *_fit_minmax(v_train_df, "saturation_mean"))

    a_bpm = _transform_minmax(adf["bpm"].to_numpy(float), *_fit_minmax(a_train_df, "bpm"))
    a_rms_mean = _transform_minmax(adf["rms_mean"].to_numpy(float), *_fit_minmax(a_train_df, "rms_mean"))
    a_rms_std = _transform_minmax(adf["rms_std"].to_numpy(float), *_fit_minmax(a_train_df, "rms_std"))
    a_centroid_mean = _transform_minmax(adf["centroid_mean"].to_numpy(float), *_fit_minmax(a_train_df, "centroid_mean"))
    a_centroid_std = _transform_minmax(adf["centroid_std"].to_numpy(float), *_fit_minmax(a_train_df, "centroid_std"))

    audio_paths = [normalize_path(v) for v in adf["filepath"].tolist()]

    records = []
    per_video = []

    for i, row in enumerate(vdf.itertuples(index=False)):
        v_path = normalize_path(getattr(row, "filepath", ""))
        rel = get_relevant_audios(relevance, v_path, audio_paths)
        if not rel:
            continue

        tempo_v = v_flow_mean[i]
        energy_v = 0.7 * v_flow_mean[i] + 0.3 * v_flow_std[i]
        tone_v = 0.6 * v_brightness[i] + 0.4 * v_saturation[i]

        scores = np.zeros(len(adf), dtype=np.float64)
        for j in range(len(adf)):
            tempo = _pair_sim(tempo_v, a_bpm[j])
            energy = _pair_sim(energy_v, 0.7 * a_rms_mean[j] + 0.3 * a_rms_std[j])
            tone = _pair_sim(tone_v, 0.7 * a_centroid_mean[j] + 0.3 * a_centroid_std[j])
            scores[j] = 0.45 * tempo + 0.40 * energy + 0.15 * tone

        order = np.argsort(-scores)
        ranked = [audio_paths[j] for j in order.tolist()]

        per_video.append(
            {
                "Recall@1": recall_at_k(ranked, rel, 1),
                "Recall@3": recall_at_k(ranked, rel, 3),
                "Recall@5": recall_at_k(ranked, rel, 5),
                "MRR": reciprocal_rank(ranked, rel),
            }
        )

        for rank, j in enumerate(order[:5].tolist(), start=1):
            records.append(
                {
                    "video_file": v_path,
                    "audio_file": audio_paths[j],
                    "rank": rank,
                    "score": float(scores[j]),
                    "is_relevant": int(audio_paths[j] in rel),
                }
            )

    metrics = aggregate_metrics(per_video)
    summary = {"method": "rule_based_baseline", **metrics}

    pd.DataFrame(records).to_csv(out_dir / "rule_based_top5.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "rule_based_summary.csv", index=False)
    (out_dir / "rule_based_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
