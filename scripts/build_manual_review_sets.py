import argparse
import html
from pathlib import Path
from typing import List

import pandas as pd


TRAIN_REQUIRED = [
    "video_file",
    "positive_audio_file",
    "negative_audio_file",
    "mood",
    "positive_score",
    "negative_score",
    "negative_type",
]

VAL_REQUIRED = [
    "video_file",
    "candidate_audio_file",
    "video_mood",
    "audio_mood",
    "compatibility_score",
    "rank",
]


def ensure_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def as_uri_or_empty(path_str: str) -> str:
    p = Path(str(path_str))
    if p.exists():
        try:
            return p.resolve().as_uri()
        except Exception:
            return ""
    return ""


def sample_train_review(
    train_df: pd.DataFrame,
    num_videos: int,
    triplets_per_video: int,
    seed: int,
) -> pd.DataFrame:
    videos = train_df["video_file"].dropna().drop_duplicates()
    selected_videos = videos.sample(n=min(num_videos, len(videos)), random_state=seed)

    rows = []
    for v in selected_videos:
        group = train_df[train_df["video_file"] == v]
        picked = group.sample(n=min(triplets_per_video, len(group)), random_state=seed)
        rows.append(picked)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=train_df.columns)
    out["positive_ok"] = ""
    out["negative_ok"] = ""
    out["notes"] = ""
    return out


def sample_val_review(
    val_df: pd.DataFrame,
    num_videos: int,
    topk: int,
    seed: int,
) -> pd.DataFrame:
    videos = val_df["video_file"].dropna().drop_duplicates()
    selected_videos = videos.sample(n=min(num_videos, len(videos)), random_state=seed)

    rows = []
    for v in selected_videos:
        group = val_df[val_df["video_file"] == v].sort_values("rank", ascending=True).head(topk)
        rows.append(group)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=val_df.columns)
    out["relevant_manual"] = ""
    out["maybe_relevant"] = ""
    out["notes"] = ""
    return out


def build_train_html(df: pd.DataFrame, out_path: Path) -> None:
    blocks = []
    grouped = df.groupby("video_file", sort=False)

    for video_file, group in grouped:
        mood = str(group["mood"].iloc[0]) if "mood" in group.columns else ""
        video_path = Path(str(video_file))
        video_exists = video_path.exists()
        video_uri = as_uri_or_empty(str(video_file))

        rows_html = []
        for _, row in group.iterrows():
            pos_uri = as_uri_or_empty(row["positive_audio_file"])
            neg_uri = as_uri_or_empty(row["negative_audio_file"])
            pos_exists = Path(str(row["positive_audio_file"])).exists()
            neg_exists = Path(str(row["negative_audio_file"])).exists()

            rows_html.append(
                "<tr>"
                f"<td>{html.escape(str(row['positive_audio_file']))}<br><small>{'OK' if pos_exists else 'MISSING'}</small></td>"
                f"<td>{html.escape(str(row['negative_audio_file']))}<br><small>{'OK' if neg_exists else 'MISSING'}</small></td>"
                f"<td>{row['positive_score']:.4f}</td>"
                f"<td>{row['negative_score']:.4f}</td>"
                f"<td>{html.escape(str(row['negative_type']))}</td>"
                "</tr>"
                "<tr>"
                f"<td>{('<audio controls src="' + html.escape(pos_uri) + '"></audio>') if pos_uri else ''}</td>"
                f"<td>{('<audio controls src="' + html.escape(neg_uri) + '"></audio>') if neg_uri else ''}</td>"
                "<td colspan='3'></td>"
                "</tr>"
            )

        video_player = (
            f"<video controls width='320' src='{html.escape(video_uri)}'></video>"
            if video_uri
            else ""
        )

        blocks.append(
            "<div class='card'>"
            f"<h3>{html.escape(str(video_file))}</h3>"
            f"<p><b>mood:</b> {html.escape(mood)} | <b>video exists:</b> {'OK' if video_exists else 'MISSING'}</p>"
            f"{video_player}"
            "<table>"
            "<thead><tr><th>Positive audio</th><th>Negative audio</th><th>Positive score</th><th>Negative score</th><th>Negative type</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
            "</div>"
        )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Train Review</title>"
        "<style>body{font-family:Arial,sans-serif;margin:20px;background:#f6f7fb}.card{background:#fff;padding:14px;margin-bottom:16px;border-radius:8px}"
        "table{width:100%;border-collapse:collapse;margin-top:10px}th,td{border:1px solid #ddd;padding:6px;vertical-align:top}small{color:#666}</style>"
        "</head><body><h1>Train Triplets Review</h1>"
        f"{''.join(blocks)}"
        "</body></html>"
    )
    out_path.write_text(html_doc, encoding="utf-8")


def build_val_html(df: pd.DataFrame, out_path: Path) -> None:
    blocks = []
    grouped = df.groupby("video_file", sort=False)

    for video_file, group in grouped:
        mood = str(group["video_mood"].iloc[0]) if "video_mood" in group.columns else ""
        video_path = Path(str(video_file))
        video_exists = video_path.exists()
        video_uri = as_uri_or_empty(str(video_file))

        rows_html = []
        for _, row in group.sort_values("rank", ascending=True).iterrows():
            cand_uri = as_uri_or_empty(row["candidate_audio_file"])
            cand_exists = Path(str(row["candidate_audio_file"])).exists()

            rows_html.append(
                "<tr>"
                f"<td>{int(row['rank'])}</td>"
                f"<td>{html.escape(str(row['candidate_audio_file']))}<br><small>{'OK' if cand_exists else 'MISSING'}</small></td>"
                f"<td>{html.escape(str(row['audio_mood']))}</td>"
                f"<td>{float(row['compatibility_score']):.4f}</td>"
                f"<td>{('<audio controls src="' + html.escape(cand_uri) + '"></audio>') if cand_uri else ''}</td>"
                "</tr>"
            )

        video_player = (
            f"<video controls width='320' src='{html.escape(video_uri)}'></video>"
            if video_uri
            else ""
        )

        blocks.append(
            "<div class='card'>"
            f"<h3>{html.escape(str(video_file))}</h3>"
            f"<p><b>video mood:</b> {html.escape(mood)} | <b>video exists:</b> {'OK' if video_exists else 'MISSING'}</p>"
            f"{video_player}"
            "<table>"
            "<thead><tr><th>Rank</th><th>Candidate audio</th><th>Audio mood</th><th>Compatibility</th><th>Preview</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
            "</div>"
        )

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Val Review</title>"
        "<style>body{font-family:Arial,sans-serif;margin:20px;background:#f6f7fb}.card{background:#fff;padding:14px;margin-bottom:16px;border-radius:8px}"
        "table{width:100%;border-collapse:collapse;margin-top:10px}th,td{border:1px solid #ddd;padding:6px;vertical-align:top}small{color:#666}</style>"
        "</head><body><h1>Validation Candidates Review</h1>"
        f"{''.join(blocks)}"
        "</body></html>"
    )
    out_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual review CSV and HTML sets for train triplets and val candidates.")
    parser.add_argument("--train-triplets", type=Path, default=Path("features/pairs/train_triplets.csv"))
    parser.add_argument("--val-candidates", type=Path, default=Path("features/pairs/val_candidates.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/review"))
    parser.add_argument("--num-train-videos", type=int, default=20)
    parser.add_argument("--num-val-videos", type=int, default=20)
    parser.add_argument("--train-triplets-per-video", type=int, default=3)
    parser.add_argument("--val-topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_triplets)
    val_df = pd.read_csv(args.val_candidates)

    ensure_columns(train_df, TRAIN_REQUIRED, "train triplets")
    ensure_columns(val_df, VAL_REQUIRED, "val candidates")

    train_review = sample_train_review(
        train_df,
        num_videos=args.num_train_videos,
        triplets_per_video=args.train_triplets_per_video,
        seed=args.seed,
    )
    val_review = sample_val_review(
        val_df,
        num_videos=args.num_val_videos,
        topk=args.val_topk,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.output_dir / "train_review.csv"
    val_csv = args.output_dir / "val_review.csv"
    train_html = args.output_dir / "train_review.html"
    val_html = args.output_dir / "val_review.html"

    train_review.to_csv(train_csv, index=False)
    val_review.to_csv(val_csv, index=False)

    build_train_html(train_review, train_html)
    build_val_html(val_review, val_html)

    print(f"Saved: {train_csv}")
    print(f"Saved: {val_csv}")
    print(f"Saved: {train_html}")
    print(f"Saved: {val_html}")
    print(f"Train review rows: {len(train_review)}")
    print(f"Val review rows: {len(val_review)}")


if __name__ == "__main__":
    main()
