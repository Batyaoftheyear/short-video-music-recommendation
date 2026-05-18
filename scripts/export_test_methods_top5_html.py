from __future__ import annotations

import argparse
import html
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.datasets import build_relevance_map, get_relevant_audios
from models.inference import build_audio_embedding_index, encode_video_batch, rank_audio_for_video
from models.preprocessing import basename_normalized, load_preprocessor, normalize_path
from models.retrieval_model import TwoTowerRetrievalModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video-features", required=True)
    p.add_argument("--audio-features", required=True)
    p.add_argument("--video-split", required=True)
    p.add_argument("--audio-split", required=True)
    p.add_argument("--relevance-file", required=True)
    p.add_argument("--preproc-dir", required=True)
    p.add_argument("--output-html", required=True)
    p.add_argument("--max-videos", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
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


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _pair_sim(a: float, b: float) -> float:
    return float(np.exp(-abs(a - b)))


def mood_prior(video_mood: str, audio_mood: str) -> float:
    vm = str(video_mood).strip().lower()
    am = str(audio_mood).strip().lower()
    if vm == am:
        return 1.0
    close = {
        "melancholic": {"relaxing"},
        "relaxing": {"melancholic"},
        "energetic": {"happy", "epic"},
        "happy": {"energetic", "romantic", "epic"},
        "romantic": {"happy", "relaxing"},
        "epic": {"energetic", "happy"},
    }
    if am in close.get(vm, set()):
        return 0.55
    far = {
        "melancholic": {"energetic", "epic"},
        "relaxing": {"energetic", "epic"},
        "energetic": {"melancholic"},
        "epic": {"melancholic", "relaxing"},
    }
    if am in far.get(vm, set()):
        return -0.2
    return 0.0


def to_uri(path_str: str) -> str:
    p = Path(path_str)
    try:
        return p.resolve().as_uri()
    except Exception:
        return html.escape(path_str.replace('\\', '/'))


def main() -> None:
    args = parse_args()
    out_html = Path(args.output_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    video_features_df = pd.read_csv(args.video_features)
    audio_features_df = pd.read_csv(args.audio_features)
    video_split_df = pd.read_csv(args.video_split)
    audio_split_df = pd.read_csv(args.audio_split)
    relevance_df = pd.read_csv(args.relevance_file)

    video_df = _filter_by_split(video_features_df, video_split_df)
    audio_df = _filter_by_split(audio_features_df, audio_split_df)
    relevance = build_relevance_map(relevance_df)

    audio_paths_norm = [normalize_path(v) for v in audio_df["filepath"].tolist()]
    audio_paths_orig = [str(v) for v in audio_df["filepath"].tolist()]
    audio_norm_to_orig = {n: o for n, o in zip(audio_paths_norm, audio_paths_orig)}

    video_preproc = load_preprocessor(Path(args.preproc_dir) / "video_preprocessor.joblib")
    audio_preproc = load_preprocessor(Path(args.preproc_dir) / "audio_preprocessor.joblib")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    audio_emb = build_audio_embedding_index(model, audio_x, device)
    video_emb = encode_video_batch(model, video_x, device)

    v_flow_mean = _minmax(video_df["flow_mean"].to_numpy(float))
    v_flow_std = _minmax(video_df["flow_std"].to_numpy(float))
    v_brightness = _minmax(video_df["brightness_mean"].to_numpy(float))
    v_saturation = _minmax(video_df["saturation_mean"].to_numpy(float))

    a_bpm = _minmax(audio_df["bpm"].to_numpy(float))
    a_rms_mean = _minmax(audio_df["rms_mean"].to_numpy(float))
    a_rms_std = _minmax(audio_df["rms_std"].to_numpy(float))
    a_centroid_mean = _minmax(audio_df["centroid_mean"].to_numpy(float))
    a_centroid_std = _minmax(audio_df["centroid_std"].to_numpy(float))
    audio_moods = audio_df["mood"].fillna("").astype(str).tolist()

    rng = random.Random(args.seed)
    cards = []
    count = 0

    def render_rows(top):
        rows = []
        for rank, a_norm, sc in top:
            a_orig = audio_norm_to_orig.get(a_norm, a_norm)
            score_txt = "-" if sc is None else f"{sc:.4f}"
            rows.append(
                f"<tr><td>{rank}</td><td>{html.escape(a_orig)}</td><td>{score_txt}</td><td><audio class='audio-player' controls preload='none' src='{to_uri(a_orig)}'></audio></td></tr>"
            )
        return "".join(rows)

    for i, row in enumerate(video_df.itertuples(index=False)):
        v_norm = normalize_path(getattr(row, "filepath", ""))
        v_orig = str(getattr(row, "filepath", ""))
        rel = get_relevant_audios(relevance, v_norm, audio_paths_norm)
        if not rel:
            continue

        idxs = list(range(len(audio_paths_norm)))
        rng.shuffle(idxs)
        random_top = [(r + 1, audio_paths_norm[j], None) for r, j in enumerate(idxs[:5])]

        tempo_v = v_flow_mean[i]
        energy_v = 0.7 * v_flow_mean[i] + 0.3 * v_flow_std[i]
        tone_v = 0.6 * v_brightness[i] + 0.4 * v_saturation[i]
        scores = np.zeros(len(audio_df), dtype=np.float64)
        v_mood = str(getattr(row, "mood", "")).strip().lower()
        for j in range(len(audio_df)):
            mood = mood_prior(v_mood, audio_moods[j])
            tempo = _pair_sim(tempo_v, a_bpm[j])
            energy = _pair_sim(energy_v, 0.7 * a_rms_mean[j] + 0.3 * a_rms_std[j])
            tone = _pair_sim(tone_v, 0.7 * a_centroid_mean[j] + 0.3 * a_centroid_std[j])
            scores[j] = 0.35 * mood + 0.30 * tempo + 0.25 * energy + 0.10 * tone
        order_rule = np.argsort(-scores)[:5].tolist()
        rule_top = [(r + 1, audio_paths_norm[j], float(scores[j])) for r, j in enumerate(order_rule)]

        order_model, sims = rank_audio_for_video(video_emb[i], audio_emb)
        top_model = order_model[:5].tolist()
        model_top = [(r + 1, audio_paths_norm[j], float(sims[j])) for r, j in enumerate(top_model)]

        cards.append(
            "<div class='card'>"
            "<div class='row'>"
            f"<div class='video-col'><h3>{html.escape(v_orig)}</h3><video controls loop preload='none' src='{to_uri(v_orig)}'></video></div>"
            "<div class='lists-col'>"
            f"<div class='method'><h4>Random top-5</h4><table><tr><th>#</th><th>audio</th><th>score</th><th>preview</th></tr>{render_rows(random_top)}</table></div>"
            f"<div class='method'><h4>Rule-based top-5</h4><table><tr><th>#</th><th>audio</th><th>score</th><th>preview</th></tr>{render_rows(rule_top)}</table></div>"
            f"<div class='method'><h4>Model top-5</h4><table><tr><th>#</th><th>audio</th><th>score</th><th>preview</th></tr>{render_rows(model_top)}</table></div>"
            "</div></div></div>"
        )
        count += 1
        if count >= args.max_videos:
            break

    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Top5 methods on test</title>
<style>
body{{font-family:Arial,sans-serif;margin:16px}} 
.card{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}}
.row{{display:grid;grid-template-columns:500px 1fr;gap:16px;align-items:start}}
video{{width:100%;max-width:500px;display:block}}
.method{{margin-bottom:12px}}
.method h4{{margin:8px 0}}
table{{width:100%;border-collapse:collapse}} th,td{{border:1px solid #eee;padding:4px;text-align:left;font-size:12px;vertical-align:top}}
audio{{width:260px}}
@media(max-width:1200px){{.row{{grid-template-columns:1fr;}} video{{max-width:100%;}}}}
</style>
<script>
document.addEventListener('play', function(e) {{
  if (e.target.tagName !== 'AUDIO') return;
  const players = document.querySelectorAll('audio.audio-player');
  players.forEach(p => {{ if (p !== e.target) p.pause(); }});
}}, true);
</script>
</head><body>
<h1>TEST top-5 comparison (random / rule-based / model)</h1>
<p>Videos shown: {count}</p>
{''.join(cards)}
</body></html>"""
    out_html.write_text(page, encoding="utf-8")
    print("saved", out_html)


if __name__ == "__main__":
    main()
