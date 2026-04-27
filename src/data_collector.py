"""
src/data_collector.py
─────────────────────
Fetches YouTube trending videos and saves to TiDB Cloud MySQL.
Runs once per day via GitHub Actions at midnight UTC.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timezone

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ── Import from config (correct names for new config.py) ─────
from config import YOUTUBE_API_KEY, COUNTRIES, MAX_RESULTS
from src.database import create_tables, save_videos

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


def get_youtube_client():
    """Build YouTube API client using the API key from config."""
    if not YOUTUBE_API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY is not set!\n"
            "Add it to your .env file or GitHub Secrets."
        )
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def parse_video(video: dict, country_code: str) -> dict:
    """
    Flatten nested YouTube API JSON into a flat dictionary.
    YouTube returns data in nested objects (snippet, statistics,
    contentDetails). This unpacks them into one flat dict.
    """
    snippet = video.get("snippet",        {})
    stats   = video.get("statistics",     {})
    content = video.get("contentDetails", {})

    return {
        "video_id":      video.get("id", ""),
        "country":       country_code,
        "title":         snippet.get("title",        ""),
        "channel_title": snippet.get("channelTitle", ""),
        "channel_id":    snippet.get("channelId",    ""),
        "category_id":   snippet.get("categoryId",   ""),
        "published_at":  snippet.get("publishedAt",  ""),
        "description":   str(snippet.get("description", ""))[:300],
        "tags":          "|".join(snippet.get("tags", [])),
        "thumbnail":     (snippet.get("thumbnails", {})
                                 .get("high", {})
                                 .get("url", "")),
        "view_count":    int(stats.get("viewCount",    0) or 0),
        "like_count":    int(stats.get("likeCount",    0) or 0),
        "comment_count": int(stats.get("commentCount", 0) or 0),
        "duration":      content.get("duration",   ""),
        "definition":    content.get("definition", "hd"),
    }


def fetch_trending(youtube_client, country_code: str) -> list:
    """
    Fetch top trending videos for one country from YouTube API.
    Uses GET /videos with chart=mostPopular.
    MAX_RESULTS is set to 5 in config.py (5 per country = 25/day).
    """
    try:
        response = youtube_client.videos().list(
            part="snippet,statistics,contentDetails",
            chart="mostPopular",
            regionCode=country_code,
            maxResults=MAX_RESULTS,
        ).execute()

        items  = response.get("items", [])
        parsed = [parse_video(v, country_code) for v in items if v.get("id")]
        logger.info("  %s → fetched %d videos", country_code, len(parsed))
        return parsed

    except HttpError as exc:
        logger.error("YouTube API error for %s: %s", country_code, exc)
        return []
    except Exception as exc:
        logger.error("Unexpected error for %s: %s", country_code, exc)
        return []


def collect_all() -> pd.DataFrame:
    """Collect from all 5 countries and return combined DataFrame."""
    youtube    = get_youtube_client()
    all_videos = []

    for country in COUNTRIES:
        videos = fetch_trending(youtube, country)
        all_videos.extend(videos)

    if not all_videos:
        logger.warning("No videos collected from any country!")
        return pd.DataFrame()

    df = pd.DataFrame(all_videos)
    logger.info(
        "Total collected: %d videos from %d countries",
        len(df), df["country"].nunique()
    )
    return df


def main():
    print("=" * 55)
    print("  YouTube Trending Collector")
    print(f"  Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Countries:  {', '.join(COUNTRIES)}")
    print(f"  Per country: {MAX_RESULTS} videos")
    print(f"  Total this run: {MAX_RESULTS * len(COUNTRIES)} videos")
    print("=" * 55)

    # Create tables if first run
    create_tables()

    # Collect from YouTube
    df = collect_all()

    if df.empty:
        print("❌ Nothing collected. Check YOUTUBE_API_KEY.")
        return

    # Save to TiDB Cloud MySQL
    save_videos(df)

    print("=" * 55)
    print(f"✅ Done! {len(df)} videos saved to TiDB Cloud.")
    print("=" * 55)


if __name__ == "__main__":
    main()
