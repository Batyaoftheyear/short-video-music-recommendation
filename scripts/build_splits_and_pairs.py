import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_VIDEO_BASE = ["filename", "filepath", "mood"]
REQUIRED_AUDIO_BASE = ["filename", "filepath", "mood"]
MEDIA_OVERRIDE_COLS = ["filepath", "corrected_mood", "exclude_from_pairs", "exclude_from_candidates", "note"]
PAIR_OVERRIDE_COLS = ["video_file", "audio_file", "action", "note"]
PAIR_ACTIONS = {"forbid_positive", "forbid_negative", "force_positive", "force_negative", "hide_candidate"}
HARD_NEGATIVE_MOOD_PRIOR = {
    "melancholic": ["relaxing"],
    "relaxing": ["melancholic"],
    "energetic": ["happy"],
    "happy": ["energetic", "romantic"],
    "romantic": ["happy"],
    "epic": ["energetic"],
}
EASY_NEGATIVE_MOOD_PRIOR = {
    "melancholic": ["happy", "energetic"],
    "relaxing": ["energetic"],
    "energetic": ["melancholic", "relaxing"],
    "happy": ["melancholic"],
    "romantic": ["epic"],
    "epic": ["happy", "romantic", "relaxing"],
}

INITIAL_MEDIA_OVERRIDES = [
    [r"jamendo_dataset_trimmed\epic\epic_1206092.wav", "", 1, 1, "user review: not epic"],
    [r"jamendo_dataset_trimmed\epic\epic_1179546.wav", "", 1, 1, "user review: not epic"],
    [r"jamendo_dataset_trimmed\energetic\energetic_1523682.wav", "", 1, 1, "user review: not energetic"],
    [r"pexels_dataset_v2\romantic\romantic_36745459.mp4", "", 1, 1, "user review: not romantic video"],
    [r"pexels_dataset_v2\romantic\romantic_20563430.mp4", "relaxing", 0, 0, "user review: slow video, closer to relaxing than romantic"],
]

INITIAL_PAIR_OVERRIDES = [
    [r"pexels_dataset_v2\happy\happy_6800633.mp4", r"jamendo_dataset_trimmed\happy\happy_1544358.wav", "forbid_positive", "user review: current positive fits worse than current negative"],
    [r"pexels_dataset_v2\happy\happy_6800633.mp4", r"jamendo_dataset_trimmed\relaxing\atlasaudio-piano-relaxing-510242 (1).wav", "forbid_negative", "user review: this track should not be treated as negative for this video"],
    [r"pexels_dataset_v2\romantic\romantic_20563430.mp4", r"jamendo_dataset_trimmed\romantic\romantic_43882.wav", "forbid_positive", "user review: first track too fast for this slow video"],
    [r"pexels_dataset_v2\romantic\romantic_20563430.mp4", r"jamendo_dataset_trimmed\relaxing\atlasaudio-piano-relaxing-510242 (1).wav", "force_positive", "user review: relaxing piano fits better than current positive"],
]


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_splits_and_pairs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger


def ensure_cols(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def normalize_mood(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mood"] = out["mood"].astype(str).str.strip().str.lower()
    return out


def ensure_override_files(media_path: Path, pair_path: Path) -> None:
    media_path.parent.mkdir(parents=True, exist_ok=True)
    if not media_path.exists():
        pd.DataFrame(INITIAL_MEDIA_OVERRIDES, columns=MEDIA_OVERRIDE_COLS).to_csv(media_path, index=False)
    if not pair_path.exists():
        pd.DataFrame(INITIAL_PAIR_OVERRIDES, columns=PAIR_OVERRIDE_COLS).to_csv(pair_path, index=False)


def stratified_split_indices(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for _, group in df.groupby("mood"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n >= 3:
            n_train, n_val, n_test = max(1, n_train), max(1, n_val), max(1, n_test)
            while n_train + n_val + n_test > n:
                if n_train >= n_val and n_train >= n_test and n_train > 1:
                    n_train -= 1
                elif n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
        else:
            n_train = max(1, n - 2) if n > 1 else 1
            n_val = 1 if n > 1 else 0
            n_test = n - n_train - n_val
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def z_norm(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mean, std = vals.mean(), vals.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(vals)), index=series.index)
    return (vals - mean) / std


def sim(diff: np.ndarray) -> np.ndarray:
    return np.exp(-np.abs(diff))


def prepare_score_features(video_df: pd.DataFrame, audio_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    v, a = video_df.copy(), audio_df.copy()
    v["v_dynamic_proxy"] = z_norm(v["flow_mean"]) if "flow_mean" in v.columns else 0.0
    v["v_tone_proxy"] = z_norm(v["brightness_mean"]) if "brightness_mean" in v.columns else 0.0
    a["a_energy"] = z_norm(a["rms_mean"]) if "rms_mean" in a.columns else 0.0
    if "bpm" in a.columns:
        bpm = pd.to_numeric(a["bpm"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        bpm = bpm.where(bpm > 0)
        a["a_tempo"] = z_norm(bpm.fillna(bpm.median()))
        a["a_tempo_half"] = z_norm((bpm * 0.5).fillna((bpm * 0.5).median()))
        a["a_tempo_double"] = z_norm((bpm * 2.0).fillna((bpm * 2.0).median()))
    else:
        a["a_tempo"] = 0.0
        a["a_tempo_half"] = 0.0
        a["a_tempo_double"] = 0.0
    return v, a


def score_components(v_row: pd.Series, a_df: pd.DataFrame) -> pd.DataFrame:
    mood_match = a_df["mood"].values == v_row["mood"]
    mood_bonus = np.where(mood_match, 0.35, -0.25)
    energy_sim = sim(v_row["v_dynamic_proxy"] - a_df["a_energy"].values)
    tempo_sim = np.maximum.reduce([
        sim(v_row["v_dynamic_proxy"] - a_df["a_tempo"].values),
        sim(v_row["v_dynamic_proxy"] - a_df["a_tempo_half"].values),
        sim(v_row["v_dynamic_proxy"] - a_df["a_tempo_double"].values),
    ])
    tone_sim = sim(v_row["v_tone_proxy"] - a_df["a_energy"].values)
    score = mood_bonus + 0.55 * energy_sim + 0.30 * tempo_sim + 0.15 * tone_sim
    return pd.DataFrame({"score": score, "energy_sim": energy_sim, "tempo_sim": tempo_sim}, index=a_df.index)


def resolve_path(raw: str, candidates: set[str], basename_map: Dict[str, List[str]]) -> Tuple[str, str]:
    if pd.isna(raw):
        return "", "missing"
    norm = str(Path(str(raw)))
    if norm in candidates:
        return norm, "exact"
    key = Path(norm).name
    found = basename_map.get(key, [])
    if len(found) == 1:
        return found[0], "basename"
    if len(found) > 1:
        return found[0], "basename_multi"
    return norm, "not_found"


def apply_media_overrides(video_df: pd.DataFrame, audio_df: pd.DataFrame, media_overrides: pd.DataFrame, summary: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    v = video_df.copy()
    a = audio_df.copy()
    v["exclude_from_pairs"] = 0
    v["exclude_from_candidates"] = 0
    a["exclude_from_pairs"] = 0
    a["exclude_from_candidates"] = 0

    all_paths = set(v["filepath"].astype(str)) | set(a["filepath"].astype(str))
    basename_map = defaultdict(list)
    for p in all_paths:
        basename_map[Path(p).name].append(p)

    matched, unresolved = [], []
    for _, row in media_overrides.iterrows():
        resolved, status = resolve_path(row.get("filepath", ""), all_paths, basename_map)
        target = None
        if resolved in set(v["filepath"]):
            target = (v, "video")
        elif resolved in set(a["filepath"]):
            target = (a, "audio")
        if target is None:
            unresolved.append(f"{row.get('filepath')} ({status})")
            continue
        df_t, kind = target
        mask = df_t["filepath"] == resolved
        corrected_mood = str(row.get("corrected_mood", "")).strip().lower()
        if corrected_mood:
            df_t.loc[mask, "mood"] = corrected_mood
        df_t.loc[mask, "exclude_from_pairs"] = int(row.get("exclude_from_pairs", 0) or 0)
        df_t.loc[mask, "exclude_from_candidates"] = int(row.get("exclude_from_candidates", 0) or 0)
        matched.append(f"{resolved} [{kind}, {status}]")

    summary["media_overrides_matched"] = matched
    summary["media_overrides_unresolved"] = unresolved
    return v, a


def parse_pair_overrides(pair_overrides: pd.DataFrame, video_paths: set[str], audio_paths: set[str], summary: Dict):
    all_paths = video_paths | audio_paths
    basename_map = defaultdict(list)
    for p in all_paths:
        basename_map[Path(p).name].append(p)

    rules = {k: set() for k in PAIR_ACTIONS}
    matched, unresolved = [], []

    for _, row in pair_overrides.iterrows():
        action = str(row.get("action", "")).strip()
        if action not in PAIR_ACTIONS:
            continue
        v_res, v_status = resolve_path(row.get("video_file", ""), video_paths, defaultdict(list, {k: [x for x in v if Path(x).name == k] for k, v in {}}))
        # simple dedicated resolver for video/audio sets
        v_res, v_status = resolve_path(row.get("video_file", ""), video_paths, defaultdict(list, {k: [p for p in video_paths if Path(p).name == k] for k in {Path(p).name for p in video_paths}}))
        a_res, a_status = resolve_path(row.get("audio_file", ""), audio_paths, defaultdict(list, {k: [p for p in audio_paths if Path(p).name == k] for k in {Path(p).name for p in audio_paths}}))
        if v_res not in video_paths or a_res not in audio_paths:
            unresolved.append(f"{row.get('video_file')} | {row.get('audio_file')} ({v_status},{a_status})")
            continue
        rules[action].add((v_res, a_res))
        matched.append(f"{action}: {v_res} <> {a_res}")

    summary["pair_overrides_matched"] = matched
    summary["pair_overrides_unresolved"] = unresolved
    return rules


def cleanup_old_pair_artifacts(pairs_dir: Path) -> None:
    for p in ["val_relevance.csv", "test_relevance.csv", "val_candidates.csv", "test_candidates.csv", "train_triplets.csv"]:
        fp = pairs_dir / p
        if fp.exists():
            fp.unlink()


def build_train_triplets(video_train: pd.DataFrame, audio_train: pd.DataFrame, rules: Dict[str, set], rng: np.random.Generator, args) -> Tuple[pd.DataFrame, Dict, Counter]:
    records = []
    no_positive = 0
    no_negative = 0
    neg_use_global = Counter()
    neg_use_mood = Counter()

    for _, v_row in video_train.iterrows():
        comp = score_components(v_row, audio_train)
        cand = audio_train.copy()
        cand[["score", "energy_sim", "tempo_sim"]] = comp[["score", "energy_sim", "tempo_sim"]]

        base_pos = cand[
            (cand["mood"] == v_row["mood"])
            & (cand["energy_sim"] >= args.min_energy_sim)
            & (cand["tempo_sim"] >= args.min_tempo_sim)
        ].copy()
        base_pos = base_pos[~base_pos["filepath"].apply(lambda a: (v_row["filepath"], a) in rules["forbid_positive"])]

        force_pos = cand[cand["filepath"].apply(lambda a: (v_row["filepath"], a) in rules["force_positive"])]
        pos_pool = pd.concat([force_pos, base_pos], ignore_index=True).drop_duplicates(subset=["filepath"]).sort_values("score", ascending=False)

        if pos_pool.empty:
            no_positive += 1
            continue

        positives = pos_pool.head(max(1, args.positives_per_video))
        pos_set = set(positives["filepath"])
        used_neg = set()
        built = 0

        for _, p_row in positives.iterrows():
            if built >= args.max_triplets_per_video:
                break

            rest = cand[~cand["filepath"].isin(pos_set)].copy()
            rest = rest[~rest["filepath"].apply(lambda a: (v_row["filepath"], a) in rules["forbid_negative"])]

            # hard negatives: high-score non-positives with mood-prior first, then fallback
            hard_pool = rest[rest["score"] < p_row["score"]].sort_values("score", ascending=False)
            force_neg = rest[rest["filepath"].apply(lambda a: (v_row["filepath"], a) in rules["force_negative"])]
            hard_pool = pd.concat([force_neg, hard_pool], ignore_index=True).drop_duplicates(subset=["filepath"])
            hard_diff = hard_pool[hard_pool["mood"] != v_row["mood"]]
            hard_prior_moods = HARD_NEGATIVE_MOOD_PRIOR.get(v_row["mood"], [])
            hard_prior_pool = hard_diff[hard_diff["mood"].isin(hard_prior_moods)]
            if not hard_prior_pool.empty:
                hard_pick_pool = hard_prior_pool
            elif not hard_diff.empty:
                hard_pick_pool = hard_diff
            else:
                hard_pick_pool = hard_pool
            hard_pick_pool = hard_pick_pool.head(max(1, args.hard_topk))
            hard_pick_pool = hard_pick_pool[~hard_pick_pool["filepath"].isin(used_neg)]
            hard_pick_pool = hard_pick_pool[hard_pick_pool["filepath"].apply(lambda f: neg_use_global[f] < args.max_negative_uses_global and neg_use_mood[(v_row['mood'], f)] < args.max_negative_uses_per_mood)]

            if not hard_pick_pool.empty:
                n = hard_pick_pool.sample(n=1, random_state=int(rng.integers(0, 10**9))).iloc[0]
                used_neg.add(n["filepath"])
                neg_use_global[n["filepath"]] += 1
                neg_use_mood[(v_row["mood"], n["filepath"])] += 1
                records.append({
                    "video_file": v_row["filepath"],
                    "positive_audio_file": p_row["filepath"],
                    "negative_audio_file": n["filepath"],
                    "mood": v_row["mood"],
                    "positive_score": float(p_row["score"]),
                    "negative_score": float(n["score"]),
                    "negative_type": "hard",
                })
                built += 1

            if built >= args.max_triplets_per_video:
                break

            # easy negatives: low-score candidates with mood-prior first, then fallback
            easy_pool = rest[rest["mood"] != v_row["mood"]].sort_values("score", ascending=True)
            easy_prior_moods = EASY_NEGATIVE_MOOD_PRIOR.get(v_row["mood"], [])
            easy_prior_pool = easy_pool[easy_pool["mood"].isin(easy_prior_moods)]
            if not easy_prior_pool.empty:
                easy_pool = easy_prior_pool
            easy_pool = easy_pool.head(max(1, args.easy_bottomk))
            easy_pool = easy_pool[~easy_pool["filepath"].isin(used_neg)]
            easy_pool = easy_pool[easy_pool["filepath"].apply(lambda f: neg_use_global[f] < args.max_negative_uses_global and neg_use_mood[(v_row['mood'], f)] < args.max_negative_uses_per_mood)]

            if not easy_pool.empty:
                n = easy_pool.sample(n=1, random_state=int(rng.integers(0, 10**9))).iloc[0]
                used_neg.add(n["filepath"])
                neg_use_global[n["filepath"]] += 1
                neg_use_mood[(v_row["mood"], n["filepath"])] += 1
                records.append({
                    "video_file": v_row["filepath"],
                    "positive_audio_file": p_row["filepath"],
                    "negative_audio_file": n["filepath"],
                    "mood": v_row["mood"],
                    "positive_score": float(p_row["score"]),
                    "negative_score": float(n["score"]),
                    "negative_type": "easy",
                })
                built += 1

        if built == 0:
            no_negative += 1

    stats = {
        "videos_processed": len(video_train),
        "triplets_built": len(records),
        "videos_without_positive": no_positive,
        "videos_without_negative": no_negative,
    }
    return pd.DataFrame(records), stats, neg_use_global


def build_candidates(video_split: pd.DataFrame, audio_split: pd.DataFrame, rules: Dict[str, set], topn: int, cap: int) -> pd.DataFrame:
    recs = []
    usage = Counter()
    for _, v_row in video_split.iterrows():
        comp = score_components(v_row, audio_split)
        cand = audio_split.copy()
        cand["compatibility_score"] = comp["score"]
        cand = cand[~cand["filepath"].apply(lambda a: (v_row["filepath"], a) in rules["hide_candidate"])]
        cand["adj_score"] = cand["compatibility_score"] - cand["filepath"].map(lambda f: 0.02 * usage[f])
        cand = cand.sort_values("adj_score", ascending=False)

        picked = []
        for _, a_row in cand.iterrows():
            if len(picked) >= topn:
                break
            if usage[a_row["filepath"]] >= cap:
                continue
            picked.append(a_row)
            usage[a_row["filepath"]] += 1

        for rank, a_row in enumerate(picked, start=1):
            recs.append({
                "video_file": v_row["filepath"],
                "candidate_audio_file": a_row["filepath"],
                "video_mood": v_row["mood"],
                "audio_mood": a_row["mood"],
                "compatibility_score": float(a_row["compatibility_score"]),
                "rank": rank,
            })
    return pd.DataFrame(recs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-features", type=Path, default=Path("features/video_features.csv"))
    parser.add_argument("--audio-features", type=Path, default=Path("features/audio_features_trimmed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("features"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--topn-relevant", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-triplets-per-video", type=int, default=4)
    parser.add_argument("--positives-per-video", type=int, default=2)
    parser.add_argument("--media-overrides", type=Path, default=Path("data/manual_media_overrides.csv"))
    parser.add_argument("--pair-overrides", type=Path, default=Path("data/manual_pair_overrides.csv"))
    parser.add_argument("--min-energy-sim", type=float, default=0.45)
    parser.add_argument("--min-tempo-sim", type=float, default=0.35)
    parser.add_argument("--hard-topk", type=int, default=5)
    parser.add_argument("--easy-bottomk", type=int, default=8)
    parser.add_argument("--max-negative-uses-global", type=int, default=35)
    parser.add_argument("--max-negative-uses-per-mood", type=int, default=12)
    parser.add_argument("--candidate-global-cap", type=int, default=40)
    args = parser.parse_args()

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    logger = setup_logger(Path("logs/build_splits_and_pairs.log"))
    ensure_override_files(args.media_overrides, args.pair_overrides)

    video_df = normalize_mood(pd.read_csv(args.video_features)).drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    audio_df = normalize_mood(pd.read_csv(args.audio_features)).drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    ensure_cols(video_df, REQUIRED_VIDEO_BASE, "video features")
    ensure_cols(audio_df, REQUIRED_AUDIO_BASE, "audio features")

    media_overrides = pd.read_csv(args.media_overrides)
    pair_overrides = pd.read_csv(args.pair_overrides)
    ensure_cols(media_overrides, MEDIA_OVERRIDE_COLS, "media overrides")
    ensure_cols(pair_overrides, PAIR_OVERRIDE_COLS, "pair overrides")

    summary = {}
    video_df, audio_df = apply_media_overrides(video_df, audio_df, media_overrides, summary)

    v_train_idx, v_val_idx, v_test_idx = stratified_split_indices(video_df, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)
    a_train_idx, a_val_idx, a_test_idx = stratified_split_indices(audio_df, args.seed + 1, args.train_ratio, args.val_ratio, args.test_ratio)

    video_train, video_val, video_test = video_df.loc[v_train_idx].reset_index(drop=True), video_df.loc[v_val_idx].reset_index(drop=True), video_df.loc[v_test_idx].reset_index(drop=True)
    audio_train, audio_val, audio_test = audio_df.loc[a_train_idx].reset_index(drop=True), audio_df.loc[a_val_idx].reset_index(drop=True), audio_df.loc[a_test_idx].reset_index(drop=True)

    rules = parse_pair_overrides(pair_overrides, set(video_df["filepath"]), set(audio_df["filepath"]), summary)

    pairs_excluded = int(video_train["exclude_from_pairs"].sum() + audio_train["exclude_from_pairs"].sum())
    cands_excluded = int(video_val["exclude_from_candidates"].sum() + video_test["exclude_from_candidates"].sum() + audio_val["exclude_from_candidates"].sum() + audio_test["exclude_from_candidates"].sum())

    video_train_p = video_train[video_train["exclude_from_pairs"] != 1].reset_index(drop=True)
    audio_train_p = audio_train[audio_train["exclude_from_pairs"] != 1].reset_index(drop=True)
    video_val_c = video_val[video_val["exclude_from_candidates"] != 1].reset_index(drop=True)
    video_test_c = video_test[video_test["exclude_from_candidates"] != 1].reset_index(drop=True)
    audio_val_c = audio_val[audio_val["exclude_from_candidates"] != 1].reset_index(drop=True)
    audio_test_c = audio_test[audio_test["exclude_from_candidates"] != 1].reset_index(drop=True)

    video_train_sc, audio_train_sc = prepare_score_features(video_train_p, audio_train_p)
    video_val_sc, audio_val_sc = prepare_score_features(video_val_c, audio_val_c)
    video_test_sc, audio_test_sc = prepare_score_features(video_test_c, audio_test_c)

    rng = np.random.default_rng(args.seed)
    triplets, triplet_stats, neg_use = build_train_triplets(video_train_sc, audio_train_sc, rules, rng, args)
    val_candidates = build_candidates(video_val_sc, audio_val_sc, rules, args.topn_relevant, args.candidate_global_cap)
    test_candidates = build_candidates(video_test_sc, audio_test_sc, rules, args.topn_relevant, args.candidate_global_cap)

    splits_dir = args.output_dir / "splits"
    pairs_dir = args.output_dir / "pairs"
    splits_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_pair_artifacts(pairs_dir)

    video_train.to_csv(splits_dir / "video_train.csv", index=False)
    video_val.to_csv(splits_dir / "video_val.csv", index=False)
    video_test.to_csv(splits_dir / "video_test.csv", index=False)
    audio_train.to_csv(splits_dir / "audio_train.csv", index=False)
    audio_val.to_csv(splits_dir / "audio_val.csv", index=False)
    audio_test.to_csv(splits_dir / "audio_test.csv", index=False)

    triplets.to_csv(pairs_dir / "train_triplets.csv", index=False)
    val_candidates.to_csv(pairs_dir / "val_candidates.csv", index=False)
    test_candidates.to_csv(pairs_dir / "test_candidates.csv", index=False)

    tracked = [
        r"jamendo_dataset_trimmed\relaxing\atlasaudio-piano-relaxing-510242 (1).wav",
        r"jamendo_dataset_trimmed\energetic\energetic_1534844.wav",
    ]

    top_neg = neg_use.most_common(10)

    lines = [
        "SPLIT AND PAIRS SUMMARY",
        "",
        f"Train triplets: {len(triplets)}",
        f"Val candidate links (top-{args.topn_relevant}): {len(val_candidates)}",
        f"Test candidate links (top-{args.topn_relevant}): {len(test_candidates)}",
        f"Videos without positives: {triplet_stats['videos_without_positive']}",
        f"Videos without negatives: {triplet_stats['videos_without_negative']}",
        "",
        f"Media overrides matched: {len(summary.get('media_overrides_matched', []))}",
        f"Media overrides unresolved: {len(summary.get('media_overrides_unresolved', []))}",
        f"Pair overrides matched: {len(summary.get('pair_overrides_matched', []))}",
        f"Pair overrides unresolved: {len(summary.get('pair_overrides_unresolved', []))}",
        f"Excluded from pairs (train split rows): {pairs_excluded}",
        f"Excluded from candidates (val/test split rows): {cands_excluded}",
        "",
        "Mood hierarchy prior: hard/easy negative sampling first tries preferred mood pools, then falls back to global score-based pools.",
        f"Hard prior map: {HARD_NEGATIVE_MOOD_PRIOR}",
        f"Easy prior map: {EASY_NEGATIVE_MOOD_PRIOR}",
        "",
        "Top repeated negatives:",
    ]
    lines.extend([f"- {k}: {v}" for k, v in top_neg])
    lines.append("")
    lines.append("Tracked negatives usage:")
    for t in tracked:
        lines.append(f"- {t}: {neg_use[t]}")

    lines.append("")
    lines.append("Matched media overrides:")
    lines.extend([f"- {x}" for x in summary.get("media_overrides_matched", [])])
    lines.append("Unresolved media overrides:")
    lines.extend([f"- {x}" for x in summary.get("media_overrides_unresolved", [])])
    lines.append("Matched pair overrides:")
    lines.extend([f"- {x}" for x in summary.get("pair_overrides_matched", [])])
    lines.append("Unresolved pair overrides:")
    lines.extend([f"- {x}" for x in summary.get("pair_overrides_unresolved", [])])

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/split_and_pairs_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
