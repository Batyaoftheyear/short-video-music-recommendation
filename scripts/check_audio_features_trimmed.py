import argparse
from pathlib import Path

import pandas as pd


def to_norm_path(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace('\\', '/', regex=False).str.lower()


def to_key(df: pd.DataFrame) -> pd.Series:
    if 'mood' in df.columns and 'filename' in df.columns:
        return (df['mood'].astype(str).str.lower().str.strip() + '/' + df['filename'].astype(str).str.lower().str.strip())
    return to_norm_path(df['filepath'])


def main() -> None:
    parser = argparse.ArgumentParser(description='Sanity-check trimmed audio features.')
    parser.add_argument('--metadata', type=Path, default=Path('jamendo_dataset_trimmed/metadata_actual.csv'))
    parser.add_argument('--features', type=Path, default=Path('features/audio_features_trimmed.csv'))
    parser.add_argument('--summary-out', type=Path, default=Path('reports/audio_features_trimmed_summary.txt'))
    parser.add_argument('--issues-out', type=Path, default=Path('reports/tables/audio_features_trimmed_issues.csv'))
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    feat = pd.read_csv(args.features)

    required_feature_cols = [
        'bpm', 'rms_mean', 'rms_std', 'centroid_mean', 'centroid_std', 'zcr_mean', 'zcr_std'
    ] + [f'mfcc_{i}_mean' for i in range(1, 14)] + [f'mfcc_{i}_std' for i in range(1, 14)]

    missing_required_cols = [c for c in required_feature_cols if c not in feat.columns]

    if 'filepath' not in meta.columns or 'filepath' not in feat.columns:
        raise ValueError('Both metadata and features must contain filepath column.')

    meta = meta.copy()
    feat = feat.copy()

    meta['file_key'] = to_key(meta)
    feat['file_key'] = to_key(feat)

    meta_files = set(meta['file_key'])
    feat_files = set(feat['file_key'])

    missing_in_features = sorted(meta_files - feat_files)
    extra_in_features = sorted(feat_files - meta_files)

    dup_filename_meta = int(meta['filename'].duplicated(keep=False).sum()) if 'filename' in meta.columns else 0
    dup_filename_feat = int(feat['filename'].duplicated(keep=False).sum()) if 'filename' in feat.columns else 0

    key_cols = [
        'filename', 'filepath', 'mood', 'duration', 'bpm', 'rms_mean', 'rms_std',
        'centroid_mean', 'centroid_std', 'zcr_mean', 'zcr_std'
    ] + [f'mfcc_{i}_mean' for i in range(1, 14)] + [f'mfcc_{i}_std' for i in range(1, 14)]
    key_cols = [c for c in key_cols if c in feat.columns]

    missing_counts = {c: int(feat[c].isna().sum()) for c in key_cols}

    bpm_bad = feat[pd.to_numeric(feat.get('bpm'), errors='coerce') <= 0] if 'bpm' in feat.columns else feat.iloc[0:0]
    rms_bad = feat[pd.to_numeric(feat.get('rms_mean'), errors='coerce') <= 0] if 'rms_mean' in feat.columns else feat.iloc[0:0]

    issues = []
    for fp in missing_in_features:
        issues.append({'issue': 'missing_in_features', 'file_key': fp})
    for fp in extra_in_features:
        issues.append({'issue': 'extra_in_features', 'file_key': fp})

    if not bpm_bad.empty:
        cols = [c for c in ['filename', 'filepath', 'mood', 'bpm'] if c in bpm_bad.columns]
        for _, row in bpm_bad[cols].iterrows():
            rec = {'issue': 'invalid_bpm'}
            rec.update(row.to_dict())
            issues.append(rec)

    if not rms_bad.empty:
        cols = [c for c in ['filename', 'filepath', 'mood', 'rms_mean'] if c in rms_bad.columns]
        for _, row in rms_bad[cols].iterrows():
            rec = {'issue': 'invalid_rms_mean'}
            rec.update(row.to_dict())
            issues.append(rec)

    issues_df = pd.DataFrame(issues)
    args.issues_out.parent.mkdir(parents=True, exist_ok=True)
    issues_df.to_csv(args.issues_out, index=False)

    lines = []
    lines.append('AUDIO FEATURES TRIMMED SANITY CHECK')
    lines.append(f'metadata file: {args.metadata}')
    lines.append(f'features file: {args.features}')
    lines.append(f'metadata rows: {len(meta)}')
    lines.append(f'features rows: {len(feat)}')
    lines.append(f'file count match: {len(meta) == len(feat)}')
    lines.append(f'missing files in features: {len(missing_in_features)}')
    lines.append(f'extra files in features: {len(extra_in_features)}')
    lines.append(f'duplicate filenames in metadata: {dup_filename_meta}')
    lines.append(f'duplicate filenames in features: {dup_filename_feat}')
    lines.append(f'missing required feature columns: {len(missing_required_cols)}')
    lines.append(f'suspicious bpm <= 0: {len(bpm_bad)}')
    lines.append(f'suspicious rms_mean <= 0: {len(rms_bad)}')
    lines.append('missing values in key columns:')
    for col, cnt in missing_counts.items():
        lines.append(f'- {col}: {cnt}')

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text('\n'.join(lines), encoding='utf-8')

    print('\n'.join(lines))
    print(f'issues table: {args.issues_out}')

    if missing_required_cols:
        raise RuntimeError(f'Missing required feature columns: {missing_required_cols}')


if __name__ == '__main__':
    main()
