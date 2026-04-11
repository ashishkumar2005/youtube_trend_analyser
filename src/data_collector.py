import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from datetime import datetime
from googleapiclient.discovery import build
import pandas as pd

try:
    from src.database import create_tables, save_videos
except ImportError:
    from database import create_tables, save_videos

from config import API_KEY, COUNTRIES, MAX_RESULTS, RAW_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_youtube_client():
    """Build and return the YouTube API client."""
    if not API_KEY:
        raise ValueError("YOUTUBE_API_KEY not found. Check your environment variables.")
    return build("youtube", "v3", developerKey=API_KEY)


def fetch_trending_videos(country_code):
    """Fetch top trending videos for a given country code."""
    youtube = get_youtube_client()
    logger.info(f"Fetching trending videos for: {country_code}")

    request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        chart="mostPopular",
        regionCode=country_code,
        maxResults=MAX_RESULTS
    )

    response = request.execute()
    return response.get("items", [])


def parse_video(video, country_code):
    """
    Flatten a raw API video object into a clean dict.

    IMPORTANT:
    - fetched_at REMOVED
    - Snapshot timestamp handled in database layer
    """
    snippet = video.get("snippet", {})
    stats   = video.get("statistics", {})
    content = video.get("contentDetails", {})

    return {
        "video_id":      video.get("id", ""),
        "country":       country_code,

        # ❌ removed fetched_at

        "title":         snippet.get("title", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "channel_id":    snippet.get("channelId", ""),
        "category_id":   snippet.get("categoryId", ""),
        "published_at":  snippet.get("publishedAt", ""),
        "description":   str(snippet.get("description", ""))[:300],

        "tags":          "|".join(snippet.get("tags", [])),
        "thumbnail":     snippet.get("thumbnails", {}).get("high", {}).get("url", ""),

        "view_count":    int(stats.get("viewCount", 0) or 0),
        "like_count":    int(stats.get("likeCount", 0) or 0),
        "comment_count": int(stats.get("commentCount", 0) or 0),

        "duration":      content.get("duration", ""),
        "definition":    content.get("definition", "hd"),
    }


def save_raw_json(data, country_code):
    """Save raw API response as JSON backup (local only)."""
    try:
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        filename  = f"{RAW_DATA_PATH}{country_code}_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Raw JSON saved: {filename}")

    except Exception as e:
        logger.warning(f"Could not save raw JSON (non-critical): {e}")


def collect_all_countries():
    """Collect trending videos for all configured countries."""
    all_videos = []

    for country in COUNTRIES:
        try:
            raw_videos = fetch_trending_videos(country)
            save_raw_json(raw_videos, country)

            for video in raw_videos:
                parsed = parse_video(video, country)
                all_videos.append(parsed)

            logger.info(f"Collected {len(raw_videos)} videos for {country}")

        except Exception as e:
            logger.error(f"Failed for {country}: {e}")
            continue

    df = pd.DataFrame(all_videos)
    logger.info(f"Total videos collected: {len(df)}")

    return df


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info(f"Collection started: {datetime.utcnow()} UTC")
    logger.info("=" * 50)

    logger.info("Ensuring database tables exist...")
    create_tables()

    logger.info("Fetching trending videos from YouTube API...")
    df = collect_all_countries()

    if not df.empty:
        logger.info(f"Saving {len(df)} videos to MySQL...")
        save_videos(df)
        logger.info("Collection complete!")
    else:
        logger.warning("No videos collected. Check API key and network.")

    logger.info("=" * 50)
    logger.info(f"Collection finished: {datetime.utcnow()} UTC")
    logger.info("=" * 50)