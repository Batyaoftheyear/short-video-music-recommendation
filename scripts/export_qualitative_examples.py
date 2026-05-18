from __future__ import annotations

import argparse
import html
from pathlib import Path

import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.datasets import build_relevance_map, get_relevant_audios
from models.inference import build_audio_embedding_index, encode_video_batch, rank_audio_for_video
from models.metrics import recall_at_k
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
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-videos", type=int, default=100)
    p.add_argument("--sort", choices=["filename", "best_score", "worst_miss"], default="filename")
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


def to_uri(path_str: str) -> str:
    p = Path(path_str)
    try:
        return p.resolve().as_uri()
    except Exception:
        return html.escape(path_str.replace('\\', '/'))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_features_df = pd.read_csv(args.video_features)
    audio_features_df = pd.read_csv(args.audio_features)
    video_split_df = pd.read_csv(args.video_split)
    audio_split_df = pd.read_csv(args.audio_split)
    relevance_df = pd.read_csv(args.relevance_file)

    video_df = _filter_by_split(video_features_df, video_split_df)
    audio_df = _filter_by_split(audio_features_df, audio_split_df)

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
    model.to(device)
    model.eval()

    relevance = build_relevance_map(relevance_df)
    audio_embeddings = build_audio_embedding_index(model, audio_x, device)
    video_embeddings = encode_video_batch(model, video_x, device)
    audio_paths = [normalize_path(v) for v in audio_df["filepath"].tolist()]

    records = []
    for i, row in enumerate(video_df.itertuples(index=False)):
        vp = normalize_path(getattr(row, "filepath", ""))
        rel = get_relevant_audios(relevance, vp, audio_paths)
        order, sims = rank_audio_for_video(video_embeddings[i], audio_embeddings)

        top_idx = order[:5].tolist()
        top = []
        for rank, idx in enumerate(top_idx, start=1):
            ap = audio_paths[idx]
            top.append({"rank": rank, "audio_file": ap, "similarity": float(sims[idx]), "is_relevant": int(ap in rel)})

        ranked_ids = [audio_paths[j] for j in order.tolist()]
        best_rel_rank = next((r for r, aid in enumerate(ranked_ids, start=1) if aid in rel), 999999)

        records.append(
            {
                "video_file": vp,
                "hit1": int(recall_at_k(ranked_ids, rel, 1)),
                "hit3": int(recall_at_k(ranked_ids, rel, 3)),
                "hit5": int(recall_at_k(ranked_ids, rel, 5)),
                "best_relevant_rank": best_rel_rank,
                "top": top,
                "best_score": top[0]["similarity"] if top else -1.0,
            }
        )

    if args.sort == "best_score":
        records.sort(key=lambda x: x["best_score"], reverse=True)
    elif args.sort == "worst_miss":
        records.sort(key=lambda x: (x["hit5"], x["best_relevant_rank"], -x["best_score"]))
    else:
        records.sort(key=lambda x: x["video_file"])
    records = records[: args.max_videos]

    rows = []
    for rec in records:
        for item in rec["top"]:
            rows.append({
                "video_file": rec["video_file"], "hit1": rec["hit1"], "hit3": rec["hit3"], "hit5": rec["hit5"],
                "best_relevant_rank": rec["best_relevant_rank"], "rank": item["rank"],
                "audio_file": item["audio_file"], "similarity": item["similarity"], "is_relevant": item["is_relevant"],
            })
    pd.DataFrame(rows).to_csv(out_dir / "qualitative_top5.csv", index=False)

    cards = []
    for rec in records:
        cls = "hit" if rec["hit5"] == 1 else "miss"
        video_uri = to_uri(rec["video_file"])
        top_html = []
        for item in rec["top"]:
            audio_uri = to_uri(item["audio_file"])
            badge = "<span class='ok'>relevant</span>" if item["is_relevant"] else ""
            top_html.append(
                f"<tr><td>{item['rank']}</td><td>{html.escape(item['audio_file'])}</td><td>{item['similarity']:.4f}</td><td>{badge}</td><td><audio controls preload='none' src='{audio_uri}'></audio></td></tr>"
            )
        cards.append(
            f"<div class='card {cls}' data-hit5='{rec['hit5']}'><h3>{html.escape(rec['video_file'])}</h3><div class='meta'>hit@1={rec['hit1']} | hit@3={rec['hit3']} | hit@5={rec['hit5']} | best_rel_rank={rec['best_relevant_rank']}</div><video controls preload='none' src='{video_uri}'></video><table><thead><tr><th>rank</th><th>audio</th><th>score</th><th>rel</th><th>preview</th></tr></thead><tbody>{''.join(top_html)}</tbody></table></div>"
        )

    html_out = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Qualitative Examples</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
.controls button {{ margin-right: 8px; }}
.card {{ border:1px solid #ddd; padding:12px; margin:12px 0; border-radius:8px; }}
.card.hit {{ border-left: 6px solid #1f8f3a; }}
.card.miss {{ border-left: 6px solid #c4382f; }}
video {{ max-width: 520px; display:block; margin:8px 0; }}
audio {{ width: 320px; }}
.meta {{ color:#444; margin:4px 0 8px 0; }}
.ok {{ color:#1f8f3a; font-weight:700; }}
table {{ border-collapse: collapse; width:100%; }}
th,td {{ border:1px solid #e1e1e1; padding:6px; text-align:left; }}
</style>
<script>
function filterMode(mode) {{
  const cards = document.querySelectorAll('.card');
  cards.forEach(c => {{
    const h = c.getAttribute('data-hit5');
    if (mode === 'all') c.style.display = 'block';
    else if (mode === 'hit' && h === '1') c.style.display = 'block';
    else if (mode === 'miss' && h === '0') c.style.display = 'block';
    else c.style.display = 'none';
  }});
}}
</script>
</head><body>
<h1>Qualitative examples (top-5)</h1>
<div class='controls'>
<button onclick="filterMode('all')">all examples</button>
<button onclick="filterMode('hit')">only hit@5</button>
<button onclick="filterMode('miss')">only miss@5</button>
</div>
{''.join(cards)}
</body></html>"""
    (out_dir / "qualitative_examples.html").write_text(html_out, encoding="utf-8")
    print("saved:", out_dir / "qualitative_examples.html")


if __name__ == "__main__":
    main()
