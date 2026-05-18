# Learned Retrieval Model Snapshot (code-based)

Source of truth: current `.py` files in `models/` and `scripts/`.

## 1) Model architecture (actual code)

- Model classes:
  - `models.retrieval_model.VideoTower`
  - `models.retrieval_model.AudioTower`
  - `models.retrieval_model.TwoTowerRetrievalModel`
- Video tower (`VideoTower`):
  - `Linear(input_dim -> 256)` -> `LayerNorm(256)` -> activation -> `Dropout`
  - `Linear(256 -> 128)` -> `LayerNorm(128)` -> activation -> `Dropout`
  - `Linear(128 -> embedding_dim)`
- Audio tower (`AudioTower`):
  - `Linear(input_dim -> 64)` -> `LayerNorm(64)` -> activation -> `Dropout`
  - `Linear(64 -> embedding_dim)`
- Activation:
  - default is `GELU` (`_act("gelu")`), `ReLU` supported but not used in train script.
- Embedding normalization:
  - yes, `F.normalize(..., p=2, dim=-1)` for both video/audio in `encode_video`/`encode_audio`.
- Dropout:
  - used in both towers as shown above.

## 2) Features реально подаваемые в модель

Feature selection rule: `models.preprocessing.pick_numeric_feature_columns(...)`.

- Numeric-only columns are selected.
- Always excluded (`EXCLUDED_COLUMNS`):
  - `filename`, `filepath`, `mood`, `video_file`, `audio_file`,
  - `positive_audio_file`, `negative_audio_file`,
  - `rank`, `compatibility_score`, `is_relevant`, `notes`.
- Additional excluded noise columns:
  - `Unnamed: 0`, `index`, `level_0`, any column starting with `unnamed` (case-insensitive).
- `duration`:
  - excluded by default (`--use-audio-duration` is off unless explicitly enabled).

Current trained baseline run artifacts contain concrete selected columns:
- `artifacts/retrieval_tuning/baseline/video_feature_columns.json` -> **1028** video features
  - main groups: `clip_mean_*`, `clip_std_*`, plus handcrafted (`flow_mean`, `flow_std`, `brightness_mean`, `saturation_mean`).
- `artifacts/retrieval_tuning/baseline/audio_feature_columns.json` -> **33** audio features
  - main groups: `bpm`, `centroid_mean/std`, `rms_mean/std`, `zcr_mean/std`, `mfcc_*_mean/std`.

## 3) Preprocessing

From `models.preprocessing` + `scripts/train_retrieval_model.py`:

- Imputer: `SimpleImputer(strategy="median")`.
- Scaler: `StandardScaler()`.
- Fit data:
  - video preprocessor fit on `video_train` feature subset only.
  - audio preprocessor fit on `audio_train` feature subset only.
- Transform:
  - train/val/test are transformed using train-fitted preprocessors.
- Saved artifacts:
  - `video_preprocessor.joblib`
  - `audio_preprocessor.joblib`
  - saved under train output dir (e.g. `artifacts/retrieval_tuning/baseline/`).

## 4) Loss and similarity

- Loss class: `models.losses.CosineTripletLoss`.
- Formula (from code):
  - `pos_sim = cosine(anchor, positive)`
  - `neg_sim = cosine(anchor, negative)`
  - `pos_dist = 1 - pos_sim`
  - `neg_dist = 1 - neg_sim`
  - `loss = mean(relu(pos_dist - neg_dist + margin))`
- Retrieval similarity for ranking:
  - dot product between L2-normalized embeddings (`audio_embeddings @ video_embedding`), i.e. cosine-equivalent.

## 5) Train pipeline (actual script)

Script: `scripts/train_retrieval_model.py`.

- Optimizer: `AdamW`.
- Device: `cuda` if available else `cpu`.
- Defaults:
  - `batch_size=64`
  - `epochs=30`
  - `lr=1e-3`
  - `weight_decay=1e-4`
  - `margin=0.2`
  - `dropout=0.2`
  - `embedding_dim=128`
  - `seed=42`
  - `patience=5`
- Early stopping:
  - yes, stops after `patience` epochs without `Recall@5` improvement.
- Best model selection metric:
  - **`Recall@5` on validation**.
- Saved outputs (per run dir):
  - `best.pt`, `last.pt`
  - `video_preprocessor.joblib`, `audio_preprocessor.joblib`
  - `video_feature_columns.json`, `audio_feature_columns.json`
  - `best_val_metrics.json`
  - `val_rankings.csv`
  - `train_history.json`

## 6) Evaluation protocol for learned model

Script: `scripts/evaluate_retrieval.py`.

- Reads split-specific video/audio pools.
- Loads saved preprocessors and checkpoint.
- Encodes all audios once, encodes each video, ranks by similarity.
- Relevance mapping built via `models.datasets.build_relevance_map` with full-path + basename fallback.
- Metrics:
  - `Recall@1/3/5` (hit-style: any relevant in top-K).
  - `MRR` (first relevant rank reciprocal).
- Aggregation:
  - mean over videos that have at least one relevant audio in relevance file.
- Outputs:
  - `metrics.json`
  - `per_video_metrics.csv`
  - `val_rankings.csv` (name is fixed even for test runs).

## 7) Inference/demo pipeline

- CLI demo: `scripts/run_inference_demo.py`.
- Streamlit demo: `tools/inference_demo_app.py`.

CLI (`run_inference_demo.py`) does:
- Extract features for a new video using `scripts.extract_video_features` functions:
  - frame sampling (`sample_frames`), CLIP (`clip_features`), handcrafted motion/color (`motion_and_color`).
- Load checkpoint + preprocessors.
- Load full audio pool from `--audio-features` (typically `features/audio_features_trimmed.csv`).
- Encode/rank and export top-5.
- Outputs:
  - `inference_top5.csv`
  - `inference_top5.json`
  - `inference_top5.html`

## 8) Current key learned-model artifacts in this repo

- Best run summary:
  - `artifacts/retrieval_tuning/best_run.json` (current best run = `baseline`).
- Current best checkpoint:
  - `artifacts/retrieval_tuning/baseline/best.pt`
- Current preprocessors:
  - `artifacts/retrieval_tuning/baseline/video_preprocessor.joblib`
  - `artifacts/retrieval_tuning/baseline/audio_preprocessor.joblib`
- Tuning summaries:
  - `artifacts/retrieval_tuning/tuning_summary.csv`
  - `artifacts/retrieval_tuning/tuning_summary.json`
- Final test comparison:
  - `artifacts/test_comparison/comparison_test_summary.csv`
  - `artifacts/test_comparison/comparison_test_summary.json`
- Learned-model eval outputs used in test comparison:
  - `artifacts/test_comparison/learned_model_eval/metrics.json`
- Qualitative examples:
  - `artifacts/retrieval_tuning/baseline/qualitative/qualitative_examples.html`

## 9) Code-level constraints / notable design choices

- Feature-space is very wide on video side (1028 dims in current baseline run), compact on audio side (33 dims).
- Path matching is robust (normalized full path + basename fallback) but if duplicate basenames exist across dirs, fallback may be ambiguous.
- `evaluate_retrieval.py` always writes ranking file as `val_rankings.csv` even when used on test split.
- Inference demos currently run model on CPU by default in Streamlit app.

---

## Final learned model snapshot

- **Model classes:** `VideoTower`, `AudioTower`, `TwoTowerRetrievalModel`
- **Input dims (current best run):** video `1028`, audio `33`
- **Embedding dim (best run):** `128`
- **Loss:** cosine triplet loss with margin (`0.2` in best run)
- **Optimizer:** `AdamW`
- **Selection metric:** validation `Recall@5`
- **Feature groups:**
  - video: CLIP mean/std + handcrafted motion/color
  - audio: bpm/energy/spectral/zcr/mfcc statistics
- **Main scripts:**
  - train: `scripts/train_retrieval_model.py`
  - eval: `scripts/evaluate_retrieval.py`
  - tuning: `scripts/tune_retrieval_model.py`
  - inference demo: `scripts/run_inference_demo.py`, `tools/inference_demo_app.py`
