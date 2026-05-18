from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.inference import build_audio_embedding_index, encode_video_batch, rank_audio_for_video
from models.preprocessing import load_preprocessor
from models.retrieval_model import TwoTowerRetrievalModel
from scripts.extract_video_features import (
    clip_features,
    load_clip_model,
    motion_and_color,
    sample_frames,
)

DEFAULT_CHECKPOINT = "artifacts/retrieval_tuning/baseline/best.pt"
DEFAULT_PREPROC_DIR = "artifacts/retrieval_tuning/baseline"
DEFAULT_AUDIO_FEATURES = "features/audio_features_trimmed.csv"


@st.cache_resource
def load_model_and_assets(checkpoint_path: str, preproc_dir: str, audio_features_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = TwoTowerRetrievalModel(
        video_input_dim=ckpt["video_input_dim"],
        audio_input_dim=ckpt["audio_input_dim"],
        embedding_dim=ckpt["embedding_dim"],
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    video_preproc = load_preprocessor(Path(preproc_dir) / "video_preprocessor.joblib")
    audio_preproc = load_preprocessor(Path(preproc_dir) / "audio_preprocessor.joblib")

    audio_df = pd.read_csv(audio_features_path)
    audio_x = audio_preproc.transform(audio_df)
    audio_emb = build_audio_embedding_index(model, audio_x, device)

    return model, device, video_preproc, audio_df, audio_emb




@st.cache_resource
def get_clip_assets(device_name: str):
    return load_clip_model(device_name)


def extract_single_video_features(video_path: Path, num_frames: int, device_name: str) -> dict[str, float | str | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = sample_frames(cap, num_frames=num_frames)
    cap.release()

    stats = {"filename": video_path.name, "filepath": str(video_path), "mood": ""}
    clip_model, clip_preprocess = get_clip_assets(device_name)
    clip_stats = clip_features(frames, clip_model, clip_preprocess, device_name)
    if clip_stats["clip_mean"] is not None:
        for idx, val in enumerate(clip_stats["clip_mean"]):
            stats[f"clip_mean_{idx}"] = float(val)
    if clip_stats["clip_std"] is not None:
        for idx, val in enumerate(clip_stats["clip_std"]):
            stats[f"clip_std_{idx}"] = float(val)

    fstats = motion_and_color(frames)
    stats.update(fstats)
    return stats


def to_uri(path_str: str) -> str:
    try:
        return Path(path_str).resolve().as_uri()
    except Exception:
        return path_str.replace("\\", "/")


def main() -> None:
    st.set_page_config(page_title="Music Retrieval Demo", layout="wide")
    st.title("Video -> Top-5 Music")
    st.caption("One-click demo using the best trained checkpoint.")

    num_frames = st.number_input("Frames per video", min_value=4, max_value=32, value=8)

    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is None:
        st.info("Upload a video to run inference.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.video(str(tmp_path), loop=True)

    if not st.button("Run inference", type="primary"):
        return

    progress = st.progress(0, text="Starting inference...")
    status = st.empty()

    try:
        status.info("Loading model and audio index...")
        model, device, video_preproc, audio_df, audio_emb = load_model_and_assets(
            DEFAULT_CHECKPOINT, DEFAULT_PREPROC_DIR, DEFAULT_AUDIO_FEATURES
        )
        progress.progress(25, text="Model loaded on cpu")

        status.info("Extracting video features (CLIP + handcrafted)...")
        vrow = extract_single_video_features(tmp_path, int(num_frames), device.type)
        progress.progress(55, text="Video features extracted")

        status.info("Encoding and ranking...")
        vdf = pd.DataFrame([vrow])
        for c in video_preproc.feature_columns:
            if c not in vdf.columns:
                vdf[c] = np.nan
        vx = video_preproc.transform(vdf)
        vemb = encode_video_batch(model, vx, device)[0]

        order, sims = rank_audio_for_video(vemb, audio_emb)
        top = order[:5].tolist()

        rows = []
        for rank, idx in enumerate(top, start=1):
            ap = str(audio_df.iloc[idx]["filepath"])
            rows.append({"rank": rank, "audio_file": ap, "score": float(sims[idx])})

        result_df = pd.DataFrame(rows)
        progress.progress(100, text="Done")
        status.success("Inference completed")

        with col2:
            st.subheader("Top-5")
            st.dataframe(result_df, use_container_width=True)
            for r in rows:
                st.markdown(f"**#{r['rank']}** `{Path(r['audio_file']).name}`")
                st.audio(r["audio_file"])
            # Keep only one playing audio at a time.
            components.html(
                """
                <script>
                const bindSingleAudioPlay = () => {
                  const parentDoc = window.parent.document;
                  const audios = parentDoc.querySelectorAll("audio");
                  audios.forEach((audio) => {
                    if (audio.dataset.singleBind === "1") return;
                    audio.dataset.singleBind = "1";
                    audio.addEventListener("play", () => {
                      parentDoc.querySelectorAll("audio").forEach((other) => {
                        if (other !== audio) other.pause();
                      });
                    });
                  });
                };
                bindSingleAudioPlay();
                setInterval(bindSingleAudioPlay, 1000);
                </script>
                """,
                height=0,
            )
            # Force reliable video looping in Streamlit DOM.
            components.html(
                """
                <script>
                const bindVideoLoop = () => {
                  const parentDoc = window.parent.document;
                  const videos = parentDoc.querySelectorAll("video");
                  videos.forEach((video) => {
                    if (video.dataset.loopBind === "1") return;
                    video.dataset.loopBind = "1";
                    video.loop = true;
                    video.addEventListener("ended", () => {
                      video.currentTime = 0;
                      const p = video.play();
                      if (p && typeof p.catch === "function") { p.catch(() => {}); }
                    });
                  });
                };
                bindVideoLoop();
                setInterval(bindVideoLoop, 1000);
                </script>
                """,
                height=0,
            )

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Inference failed: {e}")


if __name__ == "__main__":
    main()
