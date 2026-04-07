import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EXPECTED_MOODS = ["relaxing", "melancholic", "energetic", "happy", "epic", "romantic"]
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_eda")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger


def load_metadata(dataset_root: Path) -> Tuple[pd.DataFrame, Path]:
    actual = dataset_root / "metadata_actual.csv"
    fallback = dataset_root / "metadata.csv"
    target = actual if actual.exists() else fallback
    if not target.exists():
        raise FileNotFoundError(f"Metadata not found for {dataset_root}")
    return pd.read_csv(target), target


def normalize_mood(df: pd.DataFrame) -> pd.DataFrame:
    if "mood" in df.columns:
        df = df.copy()
        df["mood"] = df["mood"].astype(str).str.strip().str.lower()
    return df


def ensure_filepath(df: pd.DataFrame, logger: logging.Logger, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    if "filepath" not in df.columns:
        if "filename" in df.columns and "mood" in df.columns:
            logger.warning("[%s] filepath missing in metadata; reconstructing from mood/filename", dataset_name)
            df["filepath"] = df["mood"].astype(str) + "/" + df["filename"].astype(str)
        else:
            logger.warning("[%s] filepath cannot be reconstructed", dataset_name)
            df["filepath"] = ""
    return df


def resolve_paths(df: pd.DataFrame, dataset_root: Path) -> pd.Series:
    return df["filepath"].astype(str).apply(lambda p: normalize_abs_path(p, dataset_root))


def normalize_abs_path(path_str: str, dataset_root: Path) -> str:
    p = Path(path_str)
    if p.is_absolute():
        final = p
    else:
        parts = p.parts
        if len(parts) > 0 and parts[0].lower() == dataset_root.name.lower():
            final = dataset_root.parent / p
        else:
            final = dataset_root / p
    final = final.resolve()
    return os.path.normcase(str(final)).replace("\\", "/")


def scan_files(dataset_root: Path, valid_exts: set) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for mood_dir in sorted(dataset_root.iterdir()):
        if not mood_dir.is_dir():
            continue
        mood = mood_dir.name.lower()
        for fp in mood_dir.iterdir():
            if fp.is_file() and fp.suffix.lower() in valid_exts:
                rows.append(
                    {
                        "filename": fp.name,
                        "filepath": os.path.normcase(str(fp.resolve())).replace("\\", "/"),
                        "mood": mood,
                    }
                )
    return pd.DataFrame(rows)


def safe_save_plot(fig_path: Path, logger: logging.Logger, draw_fn) -> None:
    try:
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 5))
        draw_fn()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Plot failed: %s (%s)", fig_path, exc)
    finally:
        plt.close()


def plot_count(df: pd.DataFrame, out: Path, title: str) -> None:
    counts = df["mood"].value_counts().reindex(EXPECTED_MOODS, fill_value=0)
    sns.barplot(x=counts.index, y=counts.values, palette="tab10")
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel("count")
    plt.xlabel("mood")


def plot_hist(df: pd.DataFrame, col: str, out: Path, title: str, logger: logging.Logger) -> None:
    if col not in df.columns:
        logger.warning("Column missing for histogram: %s", col)
        return
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        logger.warning("No data for histogram: %s", col)
        return
    safe_save_plot(
        out,
        logger,
        lambda: (
            sns.histplot(series, bins=30, kde=True),
            plt.title(title),
            plt.xlabel(col),
            plt.ylabel("count"),
        ),
    )


def plot_box(df: pd.DataFrame, col: str, out: Path, title: str, logger: logging.Logger) -> None:
    if col not in df.columns or "mood" not in df.columns:
        logger.warning("Column missing for boxplot: %s", col)
        return
    local = df.copy()
    local[col] = pd.to_numeric(local[col], errors="coerce")
    local = local.dropna(subset=[col, "mood"])
    if local.empty:
        logger.warning("No data for boxplot: %s", col)
        return
    safe_save_plot(
        out,
        logger,
        lambda: (
            sns.boxplot(data=local, x="mood", y=col, order=EXPECTED_MOODS),
            plt.xticks(rotation=30, ha="right"),
            plt.title(title),
            plt.xlabel("mood"),
            plt.ylabel(col),
        ),
    )


def missing_report(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in cols:
        if c in df.columns:
            miss = int(df[c].isna().sum() + (df[c].astype(str).str.strip() == "").sum())
            rows.append({"column": c, "missing_count": miss, "missing_ratio": miss / max(n, 1)})
    return pd.DataFrame(rows)


def build_anomalies_video(
    video_meta: pd.DataFrame, video_feats: pd.DataFrame, missing_files: set
) -> pd.DataFrame:
    merged = video_meta.merge(
        video_feats[["filepath", "flow_mean", "brightness_mean", "saturation_mean"]]
        if "filepath" in video_feats.columns
        else video_feats,
        how="left",
        on="filepath",
        suffixes=("", "_feat"),
    )
    records: List[Dict[str, str]] = []
    flow_ref = pd.to_numeric(merged.get("flow_mean"), errors="coerce")
    flow_low_q = flow_ref.quantile(0.05) if flow_ref.notna().any() else None
    for _, row in merged.iterrows():
        reasons: List[str] = []
        fp = str(row.get("filepath", ""))
        orientation = str(row.get("orientation", "")).lower()
        width = pd.to_numeric(pd.Series([row.get("width")]), errors="coerce").iloc[0]
        height = pd.to_numeric(pd.Series([row.get("height")]), errors="coerce").iloc[0]
        duration = pd.to_numeric(pd.Series([row.get("duration")]), errors="coerce").iloc[0]
        flow = pd.to_numeric(pd.Series([row.get("flow_mean")]), errors="coerce").iloc[0]
        bright = pd.to_numeric(pd.Series([row.get("brightness_mean")]), errors="coerce").iloc[0]
        sat = pd.to_numeric(pd.Series([row.get("saturation_mean")]), errors="coerce").iloc[0]

        if orientation and orientation != "vertical":
            reasons.append("not_portrait")
        elif pd.notna(width) and pd.notna(height) and width >= height:
            reasons.append("not_portrait")
        if pd.notna(duration) and duration < 2:
            reasons.append("too_short")
        if pd.notna(duration) and duration > 120:
            reasons.append("too_long")
        if pd.notna(flow):
            if flow <= 0:
                reasons.append("zero_or_negative_flow")
            elif flow_low_q is not None and pd.notna(flow_low_q) and flow <= flow_low_q:
                reasons.append("very_low_flow")
        if pd.notna(bright) and bright < 40:
            reasons.append("too_dark")
        if pd.notna(sat) and sat < 25:
            reasons.append("low_saturation")
        if fp in missing_files:
            reasons.append("missing_file_on_disk")
        feature_cols = [c for c in ["flow_mean", "brightness_mean", "saturation_mean"] if c in merged.columns]
        if feature_cols and pd.isna(row[feature_cols]).all():
            reasons.append("features_missing_or_unreadable")

        if reasons:
            records.append(
                {
                    "filename": row.get("filename"),
                    "filepath": fp,
                    "mood": row.get("mood"),
                    "reasons": ";".join(sorted(set(reasons))),
                }
            )
    return pd.DataFrame(records)


def build_anomalies_audio(
    audio_meta: pd.DataFrame, audio_feats: pd.DataFrame, missing_files: set
) -> pd.DataFrame:
    feature_cols_keep = [c for c in ["filepath", "bpm", "rms_mean"] if c in audio_feats.columns]
    merged = audio_meta.merge(
        audio_feats[feature_cols_keep] if feature_cols_keep else audio_feats,
        how="left",
        on="filepath",
    )
    records: List[Dict[str, str]] = []
    for _, row in merged.iterrows():
        reasons: List[str] = []
        fp = str(row.get("filepath", ""))
        duration = pd.to_numeric(pd.Series([row.get("duration")]), errors="coerce").iloc[0]
        bpm = pd.to_numeric(pd.Series([row.get("bpm")]), errors="coerce").iloc[0]
        rms = pd.to_numeric(pd.Series([row.get("rms_mean")]), errors="coerce").iloc[0]

        if pd.isna(bpm):
            reasons.append("missing_bpm")
        else:
            if bpm < 40:
                reasons.append("very_low_bpm")
            if bpm > 220:
                reasons.append("very_high_bpm")
        if pd.notna(rms) and rms <= 0:
            reasons.append("zero_rms")
        if pd.notna(duration) and duration < 15:
            reasons.append("too_short")
        if fp in missing_files:
            reasons.append("missing_file_on_disk")
        feat_cols = [c for c in ["bpm", "rms_mean"] if c in merged.columns]
        if feat_cols and pd.isna(row[feat_cols]).all():
            reasons.append("features_missing_or_unreadable")

        if reasons:
            records.append(
                {
                    "filename": row.get("filename"),
                    "filepath": fp,
                    "mood": row.get("mood"),
                    "reasons": ";".join(sorted(set(reasons))),
                }
            )
    return pd.DataFrame(records)


def run() -> None:
    root = Path(".").resolve()
    reports_fig = root / "reports" / "figures"
    reports_tbl = root / "reports" / "tables"
    reports_fig.mkdir(parents=True, exist_ok=True)
    reports_tbl.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(root / "logs" / "run_eda.log")
    sns.set_theme(style="whitegrid")

    pexels_root = root / "pexels_dataset_v2"
    jamendo_root = root / "jamendo_dataset"
    video_feat_path = root / "features" / "video_features.csv"
    audio_feat_path = root / "features" / "audio_features.csv"

    video_meta, video_meta_source = load_metadata(pexels_root)
    audio_meta, audio_meta_source = load_metadata(jamendo_root)
    video_feats = pd.read_csv(video_feat_path)
    audio_feats = pd.read_csv(audio_feat_path)

    video_meta = normalize_mood(ensure_filepath(video_meta, logger, "video"))
    audio_meta = normalize_mood(ensure_filepath(audio_meta, logger, "audio"))
    video_feats = normalize_mood(video_feats)
    audio_feats = normalize_mood(audio_feats)

    # keep both relative and absolute path
    video_meta["filepath"] = video_meta["filepath"].astype(str).str.replace("\\", "/", regex=False)
    audio_meta["filepath"] = audio_meta["filepath"].astype(str).str.replace("\\", "/", regex=False)
    video_feats["filepath"] = video_feats["filepath"].astype(str).str.replace("\\", "/", regex=False)
    audio_feats["filepath"] = audio_feats["filepath"].astype(str).str.replace("\\", "/", regex=False)

    video_meta["abs_path"] = resolve_paths(video_meta, pexels_root).astype(str)
    audio_meta["abs_path"] = resolve_paths(audio_meta, jamendo_root).astype(str)
    video_feats["abs_path"] = resolve_paths(video_feats, pexels_root).astype(str)
    audio_feats["abs_path"] = resolve_paths(audio_feats, jamendo_root).astype(str)

    # Consistency checks
    video_meta_missing = set(video_meta.loc[~video_meta["abs_path"].apply(lambda p: Path(p).exists()), "abs_path"])
    audio_meta_missing = set(audio_meta.loc[~audio_meta["abs_path"].apply(lambda p: Path(p).exists()), "abs_path"])

    scanned_video = scan_files(pexels_root, VIDEO_EXTS)
    scanned_audio = scan_files(jamendo_root, AUDIO_EXTS)
    scanned_video_set = set(scanned_video["filepath"]) if not scanned_video.empty else set()
    scanned_audio_set = set(scanned_audio["filepath"]) if not scanned_audio.empty else set()
    video_meta_set = set(video_meta["abs_path"])
    audio_meta_set = set(audio_meta["abs_path"])
    video_extra_on_disk = sorted(scanned_video_set - video_meta_set)
    audio_extra_on_disk = sorted(scanned_audio_set - audio_meta_set)

    # duplicates
    dup_rows = []
    if "filename" in video_meta.columns:
        d = video_meta[video_meta["filename"].duplicated(keep=False)]
        for _, row in d.iterrows():
            dup_rows.append({"dataset": "video_metadata", "key": "filename", "value": row["filename"], "filepath": row.get("filepath", "")})
    if "filename" in audio_meta.columns:
        d = audio_meta[audio_meta["filename"].duplicated(keep=False)]
        for _, row in d.iterrows():
            dup_rows.append({"dataset": "audio_metadata", "key": "filename", "value": row["filename"], "filepath": row.get("filepath", "")})
    for id_col, frame, name in [
        ("pexels_id", video_meta, "video_metadata"),
        ("jamendo_id", audio_meta, "audio_metadata"),
    ]:
        if id_col in frame.columns:
            valid = frame[frame[id_col].notna() & (frame[id_col].astype(str).str.strip() != "")]
            d = valid[valid[id_col].duplicated(keep=False)]
            for _, row in d.iterrows():
                dup_rows.append({"dataset": name, "key": id_col, "value": row[id_col], "filepath": row.get("filepath", "")})
    for key_col, frame, name in [
        ("filepath", video_feats, "video_features"),
        ("filepath", audio_feats, "audio_features"),
        ("abs_path", video_feats, "video_features_abs"),
        ("abs_path", audio_feats, "audio_features_abs"),
    ]:
        if key_col in frame.columns:
            d = frame[frame[key_col].duplicated(keep=False)]
            for _, row in d.iterrows():
                dup_rows.append({"dataset": name, "key": key_col, "value": row.get(key_col, ""), "filepath": row.get("filepath", "")})
    duplicates_df = pd.DataFrame(dup_rows).drop_duplicates()

    # category counts and balance
    video_counts = (
        video_meta["mood"].value_counts().rename_axis("mood").reindex(EXPECTED_MOODS, fill_value=0).reset_index(name="video_count")
    )
    audio_counts = (
        audio_meta["mood"].value_counts().rename_axis("mood").reindex(EXPECTED_MOODS, fill_value=0).reset_index(name="audio_count")
    )
    balance_df = video_counts.merge(audio_counts, on="mood", how="outer").fillna(0)
    balance_df["ratio_video_to_audio"] = balance_df.apply(
        lambda r: (r["video_count"] / r["audio_count"]) if r["audio_count"] else None,
        axis=1,
    )

    empty_video_moods = balance_df.loc[balance_df["video_count"] == 0, "mood"].tolist()
    empty_audio_moods = balance_df.loc[balance_df["audio_count"] == 0, "mood"].tolist()

    # key field missingness
    video_key_cols = [c for c in ["filename", "filepath", "mood", "duration", "width", "height", "orientation"] if c in video_meta.columns]
    audio_key_cols = [c for c in ["filename", "filepath", "mood", "duration"] if c in audio_meta.columns]
    video_missing_df = missing_report(video_meta, video_key_cols)
    audio_missing_df = missing_report(audio_meta, audio_key_cols)

    video_feat_key = [c for c in ["filename", "filepath", "mood", "flow_mean", "flow_std", "brightness_mean", "saturation_mean"] if c in video_feats.columns]
    audio_feat_key = [c for c in ["filename", "filepath", "mood", "bpm", "rms_mean", "centroid_mean", "zcr_mean"] if c in audio_feats.columns]
    video_feat_missing_df = missing_report(video_feats, video_feat_key)
    audio_feat_missing_df = missing_report(audio_feats, audio_feat_key)
    feature_missingness_df = pd.concat(
        [
            video_feat_missing_df.assign(dataset="video_features"),
            audio_feat_missing_df.assign(dataset="audio_features"),
        ],
        ignore_index=True,
    )

    # features row cardinality per file (must be 1)
    video_feat_per_file_issues = 0
    audio_feat_per_file_issues = 0
    if "abs_path" in video_feats.columns:
        video_feat_per_file_issues = int((video_feats.groupby("abs_path").size() != 1).sum())
    if "abs_path" in audio_feats.columns:
        audio_feat_per_file_issues = int((audio_feats.groupby("abs_path").size() != 1).sum())

    # metadata/features mood consistency
    video_join = video_meta[["abs_path", "mood"]].merge(
        video_feats[["abs_path", "mood"]].rename(columns={"mood": "mood_feat"}),
        on="abs_path",
        how="left",
    )
    audio_join = audio_meta[["abs_path", "mood"]].merge(
        audio_feats[["abs_path", "mood"]].rename(columns={"mood": "mood_feat"}),
        on="abs_path",
        how="left",
    )
    video_mood_mismatch = int(((video_join["mood_feat"].notna()) & (video_join["mood"] != video_join["mood_feat"])).sum())
    audio_mood_mismatch = int(((audio_join["mood_feat"].notna()) & (audio_join["mood"] != audio_join["mood_feat"])).sum())

    # summary stats
    video_stat_cols = [c for c in ["duration", "width", "height"] if c in video_meta.columns] + [
        c for c in ["flow_mean", "flow_std", "brightness_mean", "saturation_mean"] if c in video_feats.columns
    ]
    audio_stat_cols = [c for c in ["duration"] if c in audio_meta.columns] + [
        c for c in ["bpm", "rms_mean", "centroid_mean", "zcr_mean"] if c in audio_feats.columns
    ]
    merged_video_stats = video_meta[["abs_path"] + [c for c in ["duration", "width", "height"] if c in video_meta.columns]].merge(
        video_feats[["abs_path"] + [c for c in ["flow_mean", "flow_std", "brightness_mean", "saturation_mean"] if c in video_feats.columns]],
        on="abs_path",
        how="left",
    )
    merged_audio_stats = audio_meta[["abs_path"] + [c for c in ["duration"] if c in audio_meta.columns]].merge(
        audio_feats[["abs_path"] + [c for c in ["bpm", "rms_mean", "centroid_mean", "zcr_mean"] if c in audio_feats.columns]],
        on="abs_path",
        how="left",
    )
    video_summary_stats = merged_video_stats[video_stat_cols].apply(pd.to_numeric, errors="coerce").describe().T.reset_index().rename(columns={"index": "feature"})
    audio_summary_stats = merged_audio_stats[audio_stat_cols].apply(pd.to_numeric, errors="coerce").describe().T.reset_index().rename(columns={"index": "feature"})

    # missing files report
    missing_rows = []
    for p in sorted(video_meta_missing):
        missing_rows.append({"dataset": "video_metadata", "issue_type": "missing_from_disk", "filepath": p})
    for p in sorted(audio_meta_missing):
        missing_rows.append({"dataset": "audio_metadata", "issue_type": "missing_from_disk", "filepath": p})
    for p in video_extra_on_disk:
        missing_rows.append({"dataset": "video_metadata", "issue_type": "on_disk_not_in_metadata", "filepath": p})
    for p in audio_extra_on_disk:
        missing_rows.append({"dataset": "audio_metadata", "issue_type": "on_disk_not_in_metadata", "filepath": p})
    missing_report_df = pd.DataFrame(missing_rows)

    # anomalies
    video_anomalies = build_anomalies_video(video_meta, video_feats, video_meta_missing)
    audio_anomalies = build_anomalies_audio(audio_meta, audio_feats, audio_meta_missing)

    # Save tables
    video_counts.rename(columns={"video_count": "count"}).to_csv(reports_tbl / "video_category_counts.csv", index=False)
    audio_counts.rename(columns={"audio_count": "count"}).to_csv(reports_tbl / "audio_category_counts.csv", index=False)
    video_summary_stats.to_csv(reports_tbl / "video_summary_stats.csv", index=False)
    audio_summary_stats.to_csv(reports_tbl / "audio_summary_stats.csv", index=False)
    missing_report_df.to_csv(reports_tbl / "missing_files_report.csv", index=False)
    duplicates_df.to_csv(reports_tbl / "duplicates_report.csv", index=False)
    balance_df.to_csv(reports_tbl / "category_balance_report.csv", index=False)
    feature_missingness_df.to_csv(reports_tbl / "feature_missingness_report.csv", index=False)
    video_anomalies.to_csv(reports_tbl / "video_anomalies.csv", index=False)
    audio_anomalies.to_csv(reports_tbl / "audio_anomalies.csv", index=False)

    # Plots: video
    safe_save_plot(
        reports_fig / "video_count_by_mood.png",
        logger,
        lambda: plot_count(video_meta, reports_fig / "video_count_by_mood.png", "Video Count by Mood"),
    )
    plot_hist(merged_video_stats, "duration", reports_fig / "video_duration_distribution.png", "Video Duration Distribution", logger)
    plot_hist(merged_video_stats, "width", reports_fig / "video_width_distribution.png", "Video Width Distribution", logger)
    plot_hist(merged_video_stats, "height", reports_fig / "video_height_distribution.png", "Video Height Distribution", logger)
    plot_hist(merged_video_stats, "flow_mean", reports_fig / "video_flow_mean_distribution.png", "Flow Mean Distribution", logger)
    plot_hist(merged_video_stats, "flow_std", reports_fig / "video_flow_std_distribution.png", "Flow Std Distribution", logger)
    plot_hist(merged_video_stats, "brightness_mean", reports_fig / "video_brightness_distribution.png", "Brightness Distribution", logger)
    plot_hist(merged_video_stats, "saturation_mean", reports_fig / "video_saturation_distribution.png", "Saturation Distribution", logger)
    if "mood" in video_meta.columns:
        v_for_box = video_meta[["abs_path", "mood"]].merge(
            video_feats[["abs_path"] + [c for c in ["flow_mean", "brightness_mean"] if c in video_feats.columns]],
            on="abs_path",
            how="left",
        )
        plot_box(v_for_box, "flow_mean", reports_fig / "video_flow_mean_by_mood.png", "Flow Mean by Mood", logger)
        plot_box(v_for_box, "brightness_mean", reports_fig / "video_brightness_by_mood.png", "Brightness by Mood", logger)

    # Plots: audio
    safe_save_plot(
        reports_fig / "audio_count_by_mood.png",
        logger,
        lambda: plot_count(audio_meta, reports_fig / "audio_count_by_mood.png", "Audio Count by Mood"),
    )
    plot_hist(merged_audio_stats, "duration", reports_fig / "audio_duration_distribution.png", "Audio Duration Distribution", logger)
    plot_hist(merged_audio_stats, "bpm", reports_fig / "audio_bpm_distribution.png", "BPM Distribution", logger)
    plot_hist(merged_audio_stats, "rms_mean", reports_fig / "audio_rms_mean_distribution.png", "RMS Mean Distribution", logger)
    plot_hist(merged_audio_stats, "centroid_mean", reports_fig / "audio_centroid_mean_distribution.png", "Centroid Mean Distribution", logger)
    plot_hist(merged_audio_stats, "zcr_mean", reports_fig / "audio_zcr_mean_distribution.png", "ZCR Mean Distribution", logger)
    if "mood" in audio_meta.columns:
        a_for_box = audio_meta[["abs_path", "mood"]].merge(
            audio_feats[["abs_path"] + [c for c in ["bpm", "rms_mean"] if c in audio_feats.columns]],
            on="abs_path",
            how="left",
        )
        plot_box(a_for_box, "bpm", reports_fig / "audio_bpm_by_mood.png", "BPM by Mood", logger)
        plot_box(a_for_box, "rms_mean", reports_fig / "audio_rms_by_mood.png", "RMS Mean by Mood", logger)

    # Comparative plots
    safe_save_plot(
        reports_fig / "video_audio_count_comparison.png",
        logger,
        lambda: (
            sns.barplot(
                data=balance_df.melt(id_vars=["mood"], value_vars=["video_count", "audio_count"], var_name="dataset", value_name="count"),
                x="mood",
                y="count",
                hue="dataset",
                order=EXPECTED_MOODS,
            ),
            plt.xticks(rotation=30, ha="right"),
            plt.title("Video vs Audio Count by Mood"),
            plt.xlabel("mood"),
            plt.ylabel("count"),
        ),
    )

    key_missing_matrix = pd.DataFrame(
        {
            "video_meta_duration_missing": [video_meta["duration"].isna().mean() if "duration" in video_meta.columns else None],
            "video_feat_flow_missing": [video_feats["flow_mean"].isna().mean() if "flow_mean" in video_feats.columns else None],
            "video_feat_brightness_missing": [video_feats["brightness_mean"].isna().mean() if "brightness_mean" in video_feats.columns else None],
            "audio_meta_duration_missing": [audio_meta["duration"].isna().mean() if "duration" in audio_meta.columns else None],
            "audio_feat_bpm_missing": [audio_feats["bpm"].isna().mean() if "bpm" in audio_feats.columns else None],
            "audio_feat_rms_missing": [audio_feats["rms_mean"].isna().mean() if "rms_mean" in audio_feats.columns else None],
        }
    ).T.rename(columns={0: "missing_ratio"}).dropna()
    if not key_missing_matrix.empty:
        safe_save_plot(
            reports_fig / "key_missingness_heatmap.png",
            logger,
            lambda: (
                sns.heatmap(key_missing_matrix, annot=True, cmap="Reds", vmin=0, vmax=1),
                plt.title("Key Missingness Heatmap"),
                plt.xlabel("missing_ratio"),
            ),
        )


if __name__ == "__main__":
    run()
