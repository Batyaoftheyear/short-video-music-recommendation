# short-video-music-recommendation

Проект: **автоматический подбор музыки для коротких видео** в постановке retrieval (video -> ranked audio).

## 1) Постановка задачи

Цель: для входного видео ранжировать пул аудиотреков и вернуть top-K наиболее подходящих.

Почему retrieval:
- у одного видео может быть несколько релевантных треков;
- нужна ранжировка, а не жесткая классификация по одному классу;
- удобно считать метрики `Recall@K` и `MRR` по ручной relevance-разметке.

## 2) Структура пайплайна

Полный контур в репозитории:
1. Подготовка и очистка данных.
2. Извлечение признаков видео/аудио.
3. Построение split и train triplets.
4. Ручной review кандидатов (val/test relevance).
5. Обучение learned retrieval model (two-tower).
6. Валидация и выбор лучшей конфигурации.
7. Финальное сравнение на test: random vs rule-based vs learned model.
8. Demo inference для нового видео.

## 3) Ключевые директории

- `models/` — модель, лосс, метрики, preprocessing, dataset/inference helpers.
- `scripts/` — все entry points пайплайна.
- `tools/` — локальные интерактивные утилиты (разметка, demo app).
- `features/` — extracted features, splits, pairs.
- `reports/review/` — ручная relevance-разметка и review-таблицы.
- `artifacts/` — результаты обучения, tuning, evaluation, comparison, qualitative outputs.

## 4) Реализованная модель

`TwoTowerRetrievalModel`:
- `VideoTower`: MLP (256 -> 128 -> embedding), `LayerNorm`, `GELU`, `Dropout`.
- `AudioTower`: MLP (64 -> embedding), `LayerNorm`, `GELU`, `Dropout`.
- В конце обеих башен: L2-нормализация эмбеддингов.
- Similarity для ranking: cosine-эквивалент через dot product нормализованных векторов.
- Loss: `CosineTripletLoss` (triplet margin loss по cosine distance).

## 5) Метрики и протокол оценки

Используемые метрики:
- `Recall@1`, `Recall@3`, `Recall@5` (hit-style);
- `MRR`.

Протокол:
- выбор модели — только по **validation** (`Recall@5` как primary);
- финальное сравнение методов — только на **test**;
- relevance для val/test берется из ручной разметки:
  - `reports/review/val_relevance_final.csv`
  - `reports/review/test_relevance_final.csv`

## 6) Baselines

1. `random baseline` — случайная перестановка аудиопула, усреднение по нескольким прогонам.
2. `rule-based baseline` — прозрачный score по handcrafted признакам:
   - tempo similarity
   - energy similarity
   - tone similarity  
   (без использования mood shortcut; нормализация признаков fit только на train split).

## 7) Быстрый запуск (основные команды)

Ниже минимальный набор для воспроизводимости.

### 7.1 Обучение learned model

```powershell
python scripts/train_retrieval_model.py `
  --video-features features/video_features.csv `
  --audio-features features/audio_features_trimmed.csv `
  --train-triplets features/pairs/train_triplets.csv `
  --video-train-split features/splits/video_train.csv `
  --video-val-split features/splits/video_val.csv `
  --audio-train-split features/splits/audio_train.csv `
  --audio-val-split features/splits/audio_val.csv `
  --val-relevance reports/review/val_relevance_final.csv `
  --output-dir artifacts/retrieval
```

### 7.2 Tuning (controlled set конфигураций)

```powershell
python scripts/tune_retrieval_model.py `
  --video-features features/video_features.csv `
  --audio-features features/audio_features_trimmed.csv `
  --train-triplets features/pairs/train_triplets.csv `
  --video-train-split features/splits/video_train.csv `
  --video-val-split features/splits/video_val.csv `
  --audio-train-split features/splits/audio_train.csv `
  --audio-val-split features/splits/audio_val.csv `
  --val-relevance reports/review/val_relevance_final.csv `
  --output-dir artifacts/retrieval_tuning
```

### 7.3 Оценка learned model (val/test)

```powershell
python scripts/evaluate_retrieval.py `
  --checkpoint artifacts/retrieval_tuning/baseline/best.pt `
  --video-features features/video_features.csv `
  --audio-features features/audio_features_trimmed.csv `
  --video-split features/splits/video_test.csv `
  --audio-split features/splits/audio_test.csv `
  --relevance-file reports/review/test_relevance_final.csv `
  --preproc-dir artifacts/retrieval_tuning/baseline `
  --output-dir artifacts/test_eval
```

### 7.4 Финальное сравнение 3 методов на test

```powershell
python scripts/compare_methods_on_test.py `
  --video-features features/video_features.csv `
  --audio-features features/audio_features_trimmed.csv `
  --video-train-split features/splits/video_train.csv `
  --audio-train-split features/splits/audio_train.csv `
  --video-split features/splits/video_test.csv `
  --audio-split features/splits/audio_test.csv `
  --relevance-file reports/review/test_relevance_final.csv `
  --checkpoint artifacts/retrieval_tuning/baseline/best.pt `
  --preproc-dir artifacts/retrieval_tuning/baseline `
  --output-dir artifacts/test_comparison `
  --num-runs 30 `
  --seed 42
```

### 7.5 Demo inference (новое видео)

CLI:
```powershell
python scripts/run_inference_demo.py `
  --video-path <path_to_video.mp4> `
  --audio-features features/audio_features_trimmed.csv `
  --checkpoint artifacts/retrieval_tuning/baseline/best.pt `
  --preproc-dir artifacts/retrieval_tuning/baseline `
  --output-dir artifacts/inference_demo
```

Streamlit:
```powershell
streamlit run tools/inference_demo_app.py
```

## 8) Где смотреть результаты

- Лучший run tuning: `artifacts/retrieval_tuning/best_run.json`
- Лучший checkpoint: `artifacts/retrieval_tuning/baseline/best.pt`
- Лучшие val-метрики: `artifacts/retrieval_tuning/baseline/best_val_metrics.json`
- Сравнение на test (3 метода): `artifacts/test_comparison/comparison_test_summary.csv`
- Qualitative примеры: `artifacts/retrieval_tuning/baseline/qualitative/qualitative_examples.html`
- Snapshot по фактическому коду модели: `reports/model_snapshot.md`

## 9) Воспроизводимость

- Фиксируйте `seed` (по умолчанию 42 в train/tune/compare).
- Запускайте скрипты из корня проекта.
- Для обучения/eval используется один и тот же формат splits/relevance.
- Preprocessors (`joblib`) и checkpoint должны быть из одного run-директория.

## 10) Ограничения текущей версии

- Модель обучена на tabular features, а не end-to-end на raw audio/video.
- Качество сильно зависит от полноты и консистентности ручной relevance-разметки.
- Демо-инференс с CLIP может требовать загрузки весов при первом запуске.

