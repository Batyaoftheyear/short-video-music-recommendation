from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

EXCLUDED_COLUMNS = {
    "filename", "filepath", "mood", "video_file", "audio_file",
    "positive_audio_file", "negative_audio_file", "rank", "compatibility_score",
    "is_relevant", "notes",
}


def normalize_path(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.lower()


def basename_normalized(value: object) -> str:
    norm = normalize_path(value)
    return Path(norm).name if norm else ""


def detect_id_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    filepath_col = "filepath" if "filepath" in df.columns else None
    filename_col = "filename" if "filename" in df.columns else None
    if filepath_col is None:
        for c in df.columns:
            if "path" in c.lower():
                filepath_col = c
                break
    if filename_col is None:
        for c in df.columns:
            if "file" in c.lower() and "path" not in c.lower():
                filename_col = c
                break
    return filepath_col, filename_col


def _is_noise_column(col: str) -> bool:
    c = col.strip().lower()
    return c == "unnamed: 0" or c == "index" or c == "level_0" or c.startswith("unnamed")


def pick_numeric_feature_columns(df: pd.DataFrame, exclude_columns: Iterable[str] | None = None, use_audio_duration: bool = False) -> list[str]:
    excluded = set(EXCLUDED_COLUMNS)
    if exclude_columns:
        excluded.update(exclude_columns)
    cols: list[str] = []
    for col in df.columns:
        if col in excluded or _is_noise_column(col):
            continue
        if not use_audio_duration and col == "duration":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


@dataclass
class ModalityPreprocessor:
    feature_columns: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        x = self.imputer.transform(x)
        x = self.scaler.transform(x)
        return x.astype(np.float32)


def fit_modality_preprocessor(train_df: pd.DataFrame, feature_columns: list[str]) -> ModalityPreprocessor:
    x = train_df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = imputer.fit_transform(x)
    scaler.fit(x_imp)
    return ModalityPreprocessor(feature_columns=feature_columns, imputer=imputer, scaler=scaler)


def save_preprocessor(preproc: ModalityPreprocessor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preproc, path)


def load_preprocessor(path: Path) -> ModalityPreprocessor:
    return joblib.load(path)
