import os
import csv
import time
import random
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("PEXELS_API_KEY")

# 6 категорий с тегами
CATEGORIES = {
    "relaxing": [
        "calm lake", "gentle waves", "slow motion nature",
        "golden hour nature", "meadow wind", "sunrise timelapse", "misty morning"
    ],
    "melancholic": [
        "fog street night", "rain window", "empty road night",
        "neon reflection puddle", "dark forest", "lonely figure city",
        "winter night", "rainy city street"
    ],
    "energetic": [
        "skateboarding", "motorcycle ride", "parkour",
        "trail running", "drone fast", "cycling action", "time lapse city"
    ],
    "happy": [
        "friends laughing", "summer beach fun", "road trip",
        "cafe morning", "golden hour people", "summer picnic", "dancing outdoor"
    ],
    "epic": [
        "drone mountains", "epic landscape", "cinematic nature",
        "aerial ocean cliffs", "storm clouds timelapse", "vast desert drone", "snowy peaks aerial"
    ],
    "romantic": [
        "couple sunset", "golden hour portrait", "film aesthetic",
        "nostalgic street", "vintage cafe", "soft light portrait", "autumn walk"
    ],
}

VIDEOS_PER_TAG = 10        # видео на каждый тег
OUTPUT_DIR     = "pexels_dataset_v2"
MIN_DURATION   = 5
MAX_DURATION   = 30
ORIENTATION    = "portrait"   # вертикальные видео


def search_videos(query, per_page=10, page=1):
    url     = "https://api.pexels.com/videos/search"
    headers = {"Authorization": API_KEY}
    params  = {
        "query":       query,
        "per_page":    per_page,
        "page":        page,
        "orientation": ORIENTATION,
    }
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def get_best_file(video_files):
    """Берём файл с наилучшим качеством среди вертикальных (height > width)."""
    portrait = [f for f in video_files
                if f.get("height", 0) > f.get("width", 1)]
    pool = portrait if portrait else video_files
    # максимальное разрешение, но не больше 1080p по ширине
    suitable = [f for f in pool if f.get("width", 9999) <= 1080]
    if not suitable:
        suitable = pool
    return max(suitable, key=lambda f: f.get("width", 0))


def download_video(url, filepath):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def main():
    if not API_KEY:
        raise ValueError("PEXELS_API_KEY не найден. Проверь файл .env")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "filename", "mood", "tag", "pexels_id",
            "duration", "width", "height", "orientation", "url"
        ])

        for mood, tags in CATEGORIES.items():
            mood_dir = os.path.join(OUTPUT_DIR, mood)
            os.makedirs(mood_dir, exist_ok=True)
            print(f"\n=== Категория: {mood.upper()} ===")

            downloaded = 0
            seen_ids   = set()

            for tag in tags:
                print(f"  Тег: '{tag}'")
                try:
                    data = search_videos(tag, per_page=VIDEOS_PER_TAG)
                except Exception as e:
                    print(f"  Ошибка запроса: {e}")
                    continue

                for video in data.get("videos", []):
                    vid_id   = video.get("id")
                    duration = video.get("duration", 0)

                    if vid_id in seen_ids:
                        continue
                    if not (MIN_DURATION <= duration <= MAX_DURATION):
                        continue

                    files = video.get("video_files", [])
                    if not files:
                        continue

                    best     = get_best_file(files)
                    vid_url  = best.get("link")
                    if not vid_url:
                        continue

                    w, h     = best.get("width", 0), best.get("height", 0)
                    orient   = "portrait" if h > w else "landscape"
                    filename = f"{mood}_{vid_id}.mp4"
                    filepath = os.path.join(mood_dir, filename)

                    if os.path.exists(filepath):
                        print(f"    Пропуск (уже есть): {filename}")
                        seen_ids.add(vid_id)
                        continue

                    try:
                        print(f"    ↓ {filename}  {duration}s  {w}×{h}  [{orient}]")
                        download_video(vid_url, filepath)
                        writer.writerow([
                            filename, mood, tag, vid_id,
                            duration, w, h, orient, video.get("url")
                        ])
                        csvfile.flush()
                        seen_ids.add(vid_id)
                        downloaded += 1
                        time.sleep(0.4)
                    except Exception as e:
                        print(f"    Ошибка скачивания: {e}")

            print(f"  Итого скачано для '{mood}': {downloaded} видео")

    print(f"\nГотово. Метаданные → {csv_path}")


if __name__ == "__main__":
    main()
