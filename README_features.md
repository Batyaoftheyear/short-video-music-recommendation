# Feature Pipeline

## Scripts
- `scripts/rebuild_metadata.py`: rebuilds `metadata_actual.csv` from real files on disk.
- `scripts/trim_audio_dataset.py`: creates `jamendo_dataset_trimmed/` by taking central fixed-length audio fragments and writes `jamendo_dataset_trimmed/metadata_actual.csv`.
- `scripts/extract_video_features.py`: extracts CLIP, motion, and color features for videos.
- `scripts/extract_audio_features.py`: extracts tempo/energy/spectral/MFCC features for audio (main target: `jamendo_dataset_trimmed`).
- `scripts/build_splits_and_pairs.py`: builds stratified train/val/test splits and retrieval structures (`train_triplets`, `val_relevance`, `test_relevance`).

## Why trimmed audio dataset
Long full tracks add irrelevant sections for retrieval.  
`jamendo_dataset_trimmed` keeps a representative central fragment (default 30s) per track for more consistent matching.

## Run order
1. Rebuild metadata:
```bash
python scripts/rebuild_metadata.py
```
2. Trim audio dataset:
```bash
python scripts/trim_audio_dataset.py --input-root jamendo_dataset --output-root jamendo_dataset_trimmed --target-duration 30 --audio-format wav
```
3. Rebuild/verify trimmed metadata (already done by trim script as `metadata_actual.csv`).
4. Re-extract audio features from trimmed dataset:
```bash
python scripts/extract_audio_features.py --dataset-root jamendo_dataset_trimmed --metadata jamendo_dataset_trimmed/metadata_actual.csv --output features/audio_features_trimmed.csv --sr 22050 --n-mfcc 13
```

## Notes
- Original files in `jamendo_dataset/` are never overwritten.
- Default trimmed format is WAV for robust local processing without ffmpeg.
- Use `--skip-existing` to skip already-created trimmed files.


## Trimmed audio feature check
Run sanity check after extraction:
```bash
python scripts/check_audio_features_trimmed.py --metadata jamendo_dataset_trimmed/metadata_actual.csv --features features/audio_features_trimmed.csv
```
Outputs:
- `reports/audio_features_trimmed_summary.txt`
- `reports/tables/audio_features_trimmed_issues.csv`

Next step after this check: prepare train/val/test split for retrieval experiments.


## Split + pair building
After feature extraction, build dataset splits and retrieval structures:
```bash
python scripts/build_splits_and_pairs.py --video-features features/video_features.csv --audio-features features/audio_features_trimmed.csv
```
Outputs:
- `features/splits/video_{train,val,test}.csv`
- `features/splits/audio_{train,val,test}.csv`
- `features/pairs/train_triplets.csv` (for training)
- `features/pairs/val_relevance.csv` and `features/pairs/test_relevance.csv` (for retrieval evaluation)
- `reports/split_and_pairs_summary.txt`

`train_triplets` are used for learning ranking embeddings. `val/test relevance` are used as pseudo-ground-truth for top-K evaluation and must not be used for training.
