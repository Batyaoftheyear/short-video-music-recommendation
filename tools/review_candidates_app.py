import argparse
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


REQUIRED_COLS = ["video_file", "candidate_audio_file", "audio_mood", "compatibility_score", "rank"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidates", type=Path, default=Path("features/pairs/val_candidates.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/review/val_manual_labels.csv"))
    parser.add_argument("--title", type=str, default="Candidates Review Labeling")
    return parser.parse_args()


def ensure_candidates(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Candidates CSV missing columns: {missing}")
    out = df.copy()
    out = out.rename(columns={"candidate_audio_file": "audio_file"})
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(9999).astype(int)
    out["compatibility_score"] = pd.to_numeric(out["compatibility_score"], errors="coerce")
    out = out.sort_values(["video_file", "rank"]).reset_index(drop=True)
    return out


def load_labels(output_path: Path, candidates_df: pd.DataFrame) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        labels = pd.read_csv(output_path)
    else:
        labels = pd.DataFrame(columns=["video_file", "audio_file", "rank", "compatibility_score", "is_relevant", "notes"])

    # Backward compatibility with older label schema that used `label`.
    if "is_relevant" not in labels.columns:
        if "label" in labels.columns:
            labels["is_relevant"] = labels["label"].astype(str).str.strip().str.lower().eq("relevant").astype(int)
        else:
            labels["is_relevant"] = 0
    if "notes" not in labels.columns:
        labels["notes"] = ""
    for col in ["video_file", "audio_file", "rank", "compatibility_score"]:
        if col not in labels.columns:
            labels[col] = pd.NA

    base = candidates_df[["video_file", "audio_file", "rank", "compatibility_score"]].copy()
    merged = base.merge(
        labels[["video_file", "audio_file", "rank", "compatibility_score", "is_relevant", "notes"]],
        on=["video_file", "audio_file", "rank", "compatibility_score"],
        how="left",
    )
    if "notes" not in merged.columns:
        merged["notes"] = ""
    merged["notes"] = merged["notes"].fillna("").astype(str)
    if "is_relevant" in merged.columns:
        merged["is_relevant"] = pd.to_numeric(merged["is_relevant"], errors="coerce").fillna(0).astype(int)
    else:
        merged["is_relevant"] = 0
    return merged


def save_labels(df: pd.DataFrame, output_path: Path) -> None:
    out = df[["video_file", "audio_file", "rank", "compatibility_score", "is_relevant", "notes"]].copy()
    out.to_csv(output_path, index=False)


def video_is_labeled(group: pd.DataFrame) -> bool:
    notes_any = group["notes"].fillna("").astype(str).str.strip().ne("").any()
    relevant_any = pd.to_numeric(group["is_relevant"], errors="coerce").fillna(0).astype(int).eq(1).any()
    return bool(notes_any or relevant_any)


def get_video_notes(group: pd.DataFrame) -> str:
    vals = group["notes"].fillna("").astype(str)
    uniq = [v for v in vals.unique().tolist() if v.strip()]
    return uniq[0] if uniq else ""


def set_video_notes(df: pd.DataFrame, video_file: str, notes: str) -> None:
    mask = df["video_file"] == video_file
    if "notes" not in df.columns:
        df["notes"] = ""
    df["notes"] = df["notes"].fillna("").astype(str)
    df.loc[mask, "notes"] = str(notes)


def clear_video_labels(df: pd.DataFrame, video_file: str) -> None:
    mask = df["video_file"] == video_file
    df.loc[mask, "is_relevant"] = 0
    df.loc[mask, "notes"] = ""


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title=args.title, layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stVideo"] video {
            max-width: 240px !important;
            margin: 0;
            display: block;
        }
        [data-testid="stAudio"] {
            margin-top: 0.05rem;
            margin-bottom: 0.15rem;
        }
        [data-testid="stAudio"] audio {
            height: 22px !important;
            max-height: 22px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
        const d = window.parent.document;
        function bindSinglePlay() {
          const audios = d.querySelectorAll("audio");
          audios.forEach((a) => {
            if (a.dataset.boundSinglePlay === "1") return;
            a.dataset.boundSinglePlay = "1";
            a.addEventListener("play", () => {
              audios.forEach((other) => {
                if (other !== a && !other.paused) other.pause();
              });
            });
          });
        }
        bindSinglePlay();
        setInterval(bindSinglePlay, 1200);
        </script>
        """,
        height=0,
    )
    st.title(args.title)

    try:
        candidates_raw = pd.read_csv(args.candidates)
        candidates_df = ensure_candidates(candidates_raw)
    except Exception as e:
        st.error(f"Failed to load candidates: {e}")
        st.stop()

    if "labels_df" not in st.session_state:
        st.session_state.labels_df = load_labels(args.output, candidates_df)
    labels_df = st.session_state.labels_df

    video_files = labels_df["video_file"].drop_duplicates().tolist()
    if not video_files:
        st.warning("No videos found in candidates file.")
        st.stop()

    unlabeled_only = st.checkbox("Show only unlabeled videos", value=False)
    visible_videos = []
    for v in video_files:
        group = labels_df[labels_df["video_file"] == v]
        if unlabeled_only and video_is_labeled(group):
            continue
        visible_videos.append(v)

    if not visible_videos:
        st.info("No videos match current filter.")
        st.stop()

    if "video_idx" not in st.session_state:
        st.session_state.video_idx = 0
    st.session_state.video_idx = min(st.session_state.video_idx, len(visible_videos) - 1)
    current_video = visible_videos[st.session_state.video_idx]
    group = labels_df[labels_df["video_file"] == current_video].copy().sort_values("rank")

    labeled_count = sum(video_is_labeled(labels_df[labels_df["video_file"] == v]) for v in video_files)
    total_count = len(video_files)
    st.progress(labeled_count / total_count if total_count else 0)
    st.caption(f"Labeled videos: {labeled_count}/{total_count} | Remaining: {total_count - labeled_count}")

    col_prev, col_next, col_no_rel, col_save_next = st.columns(4)
    with col_prev:
        if st.button("Previous", use_container_width=True):
            st.session_state.video_idx = max(0, st.session_state.video_idx - 1)
    with col_next:
        if st.button("Next", use_container_width=True):
            st.session_state.video_idx = min(len(visible_videos) - 1, st.session_state.video_idx + 1)
    with col_no_rel:
        if st.button("No relevant candidates", use_container_width=True):
            mask = labels_df["video_file"] == current_video
            labels_df.loc[mask, "is_relevant"] = 0
            if not str(st.session_state.get(notes_key, "")).strip():
                st.session_state[notes_key] = "No relevant candidates"
            set_video_notes(labels_df, current_video, st.session_state[notes_key])
            save_labels(labels_df, args.output)
            st.success("Saved as no relevant candidates.")
    with col_save_next:
        if st.button("Save and next", use_container_width=True):
            save_labels(labels_df, args.output)
            st.session_state.video_idx = min(len(visible_videos) - 1, st.session_state.video_idx + 1)
            st.rerun()

    st.subheader(f"Video {st.session_state.video_idx + 1}/{len(visible_videos)}")
    main_left, main_right = st.columns([0.8, 1.5], gap="medium")
    with main_left:
        st.code(current_video)
        video_path = Path(current_video)
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.warning(f"Video file not found: {current_video}")

        notes_key = f"notes::{current_video}"
        if notes_key not in st.session_state:
            st.session_state[notes_key] = get_video_notes(group)

        notes_val = st.text_area("Notes for current video", key=notes_key, height=100)
        set_video_notes(labels_df, current_video, notes_val)
        save_labels(labels_df, args.output)

        if st.button("Clear labels for current video", type="secondary"):
            clear_video_labels(labels_df, current_video)
            st.session_state[notes_key] = ""
            save_labels(labels_df, args.output)
            st.rerun()

    with main_right:
        st.markdown("### Candidates")
        for _, row in group.iterrows():
            audio_file = str(row["audio_file"])
            rank = int(row["rank"])

            info_col, tick_col = st.columns([3, 1])
            with info_col:
                st.markdown(f"**#{rank}**")
            with tick_col:
                rel_key = f"rel::{current_video}::{audio_file}::{rank}"
                current_rel = int(row.get("is_relevant", 0)) == 1
                selected = st.checkbox("relevant", value=current_rel, key=rel_key)

            a_path = Path(audio_file)
            if a_path.exists():
                st.audio(str(a_path))
            else:
                st.warning(f"Audio file not found: {audio_file}")

            mask = (
                (labels_df["video_file"] == current_video)
                & (labels_df["audio_file"] == audio_file)
                & (labels_df["rank"] == rank)
                & (labels_df["compatibility_score"] == row["compatibility_score"])
            )
            labels_df.loc[mask, "is_relevant"] = 1 if selected else 0
            save_labels(labels_df, args.output)

    st.caption(f"Autosave path: {args.output}")


if __name__ == "__main__":
    main()
