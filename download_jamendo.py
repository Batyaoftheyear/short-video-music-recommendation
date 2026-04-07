import os
import csv
import time
import requests
from dotenv import load_dotenv

load_dotenv()
CLIENT_ID = os.getenv("JAMENDO_CLIENT_ID")

# Конфиг каждой категории: теги + фильтры жанра и BPM
CATEGORIES = {
    "relaxing": {
        "tags":   ["relaxing", "calm", "peaceful", "soundscape", "nature"],
        "genre":  "ambient",
        "bpmgte": 0,
        "bpmlte": 90,
    },
    "melancholic": {
        "tags":   ["sad", "atmospheric", "soundscape", "melancholic", "lonely"],
        "genre":  "ambient",
        "bpmgte": 0,
        "bpmlte": 85,
    },
    "energetic": {
        "tags":   ["energetic", "sport", "action", "driving", "workout"],
        "genre":  "",
        "bpmgte": 120,
        "bpmlte": 180,
    },
    "happy": {
        "tags":   ["happy", "positive", "uplifting", "cheerful", "summer"],
        "genre":  "",
        "bpmgte": 90,
        "bpmlte": 130,
    },
    "epic": {
        "tags":   ["cinematic", "epic", "trailer", "filmscore", "adventure"],
        "genre":  "soundtrack",
        "bpmgte": 60,
        "bpmlte": 140,
    },
    "romantic": {
        "tags":   ["love", "romantic", "tender", "emotional", "ballad"],
        "genre":  "",
        "bpmgte": 0,
        "bpmlte": 100,
    },
}

TRACKS_PER_TAG = 20
OUTPUT_DIR     = "jamendo_dataset"
MIN_DURATION   = 60 
MAX_DURATION   = 300  


def search_tracks(tag, genre="", bpmgte=0, bpmlte=999, limit=20, offset=0):
    url    = "https://api.jamendo.com/v3.0/tracks"
    params = {
        "client_id":   CLIENT_ID,
        "format":      "json",
        "limit":       limit,
        "offset":      offset,
        "tags":        tag,
        "include":     "musicinfo",
        "audioformat": "mp31",
        "order":       "popularity_total",
    }
    if genre:
        params["genre"] = genre
    if bpmgte > 0:
        params["bpmgte"] = bpmgte
    if bpmlte < 999:
        params["bpmlte"] = bpmlte

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def download_track(audio_url, filepath):
    r = requests.get(audio_url, stream=True, timeout=60)
    r.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def main():
    if not CLIENT_ID:
        raise ValueError("JAMENDO_CLIENT_ID не найден. Проверь файл .env")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "filename", "mood", "tag", "jamendo_id",
            "name", "artist", "duration", "bpm", "tags", "url"
        ])

        for mood, config in CATEGORIES.items():
            mood_dir = os.path.join(OUTPUT_DIR, mood)
            os.makedirs(mood_dir, exist_ok=True)
            print(f"\n=== Категория: {mood.upper()} ===")
            print(f"  Жанр: {config['genre'] or 'любой'}  BPM: {config['bpmgte']}–{config['bpmlte']}")

            downloaded = 0
            seen_ids   = set()

            for tag in config["tags"]:
                print(f"  Тег: '{tag}'")
                try:
                    data = search_tracks(
                        tag,
                        genre=config["genre"],
                        bpmgte=config["bpmgte"],
                        bpmlte=config["bpmlte"],
                        limit=TRACKS_PER_TAG,
                    )
                except Exception as e:
                    print(f"  Ошибка запроса: {e}")
                    continue

                for track in data.get("results", []):
                    track_id  = track.get("id")
                    duration  = track.get("duration", 0)
                    audio_url = track.get("audio")

                    if track_id in seen_ids:
                        continue
                    if not (MIN_DURATION <= duration <= MAX_DURATION):
                        continue
                    if not audio_url:
                        continue

                    filename = f"{mood}_{track_id}.mp3"
                    filepath = os.path.join(mood_dir, filename)

                    if os.path.exists(filepath):
                        print(f"    Пропуск (уже есть): {filename}")
                        seen_ids.add(track_id)
                        continue

                    musicinfo  = track.get("musicinfo", {})
                    track_tags = musicinfo.get("tags", {})
                    all_tags   = (
                        track_tags.get("genres", []) +
                        track_tags.get("instruments", []) +
                        track_tags.get("vartags", [])
                    )
                    bpm    = musicinfo.get("bpm", "")
                    name   = track.get("name", "")
                    artist = track.get("artist_name", "")

                    try:
                        print(f"    ↓ {filename}  {duration}s  bpm={bpm}  {name[:40]}")
                        download_track(audio_url, filepath)
                        writer.writerow([
                            filename, mood, tag, track_id,
                            name, artist, duration, bpm,
                            "|".join(all_tags), track.get("shareurl", "")
                        ])
                        csvfile.flush()
                        seen_ids.add(track_id)
                        downloaded += 1
                        time.sleep(0.3)
                    except Exception as e:
                        print(f"    Ошибка скачивания: {e}")

            print(f"  Итого скачано для '{mood}': {downloaded} треков")

    print(f"\nГотово. Метаданные → {csv_path}")


if __name__ == "__main__":
    main()