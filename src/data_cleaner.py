import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob

try:
    from src.database import get_latest_trending
except ImportError:
    from database import get_latest_trending

from config import PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "18": "Short Movies",
    "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging",
    "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
    "25": "News & Politics", "26": "Howto & Style", "27": "Education",
    "28": "Science & Technology", "29": "Nonprofits & Activism"
}


def parse_duration_to_seconds(duration):
    if not duration or duration == "P0D":
        return 0
    pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    match   = re.match(pattern, str(duration))
    if not match:
        return 0
    hours   = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def get_sentiment(text):
    if not text or pd.isna(text):
        return 0.0
    try:
        return round(TextBlob(str(text)).sentiment.polarity, 4)
    except:
        return 0.0


def get_hours_since_published(published_at, fetched_at):
    try:
        pub_str = str(published_at)[:19].replace("T", " ")
        fet_str = str(fetched_at)[:19]
        pub     = datetime.strptime(pub_str, "%Y-%m-%d %H:%M:%S")
        fet     = datetime.strptime(fet_str, "%Y-%m-%d %H:%M:%S")
        diff    = (fet - pub).total_seconds() / 3600
        return round(max(diff, 0.1), 2)
    except:
        return 1.0


def clean_and_engineer(df):
    logger.info(f"Starting feature engineering on {len(df)} rows...")
    df = df.copy()

    df["view_count"]    = pd.to_numeric(df["view_count"], errors="coerce").fillna(0).astype(int)
    df["like_count"]    = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0).astype(int)
    df["title"]         = df["title"].fillna("").astype(str)
    df["tags"]          = df["tags"].fillna("").astype(str)
    df["description"]   = df["description"].fillna("").astype(str)

    df["duration_seconds"] = pd.to_numeric(
        df["duration"].apply(parse_duration_to_seconds), errors="coerce"
    ).fillna(0).astype(float)
    df["duration_minutes"] = (df["duration_seconds"] / 60).round(2)
    df["is_short"]         = (df["duration_seconds"] < 60).astype(int)

    df["published_at"]  = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["publish_hour"]  = df["published_at"].dt.hour
    df["publish_day"]   = df["published_at"].dt.day_name()
    df["publish_month"] = df["published_at"].dt.month
    df["is_weekend"]    = df["published_at"].dt.dayofweek.isin([5, 6]).astype(int)

    df["hours_to_trend"] = pd.to_numeric(
        df.apply(lambda r: get_hours_since_published(
            str(r["published_at"]), str(r["fetched_at"])
        ), axis=1),
        errors="coerce"
    ).fillna(1).astype(float)

    view_safe  = df["view_count"].astype(float).replace(0, 1)
    hours_safe = df["hours_to_trend"].astype(float).replace(0, 1)

    df["like_view_ratio"]    = (df["like_count"].astype(float) / view_safe).round(6)
    df["comment_view_ratio"] = (df["comment_count"].astype(float) / view_safe).round(6)
    df["views_per_hour"]     = (df["view_count"].astype(float) / hours_safe).round(2)
    df["engagement_score"]   = (
        (df["like_count"].astype(float) + df["comment_count"].astype(float)) / view_safe
    ).round(6)

    df["title_length"]      = df["title"].str.len()
    df["title_word_count"]  = df["title"].str.split().str.len()
    df["title_has_number"]  = df["title"].str.contains(r"\d", regex=True).astype(int)
    df["title_has_caps"]    = df["title"].str.isupper().astype(int)
    df["title_exclamation"] = df["title"].str.contains("!", regex=False).astype(int)
    df["title_question"]    = df["title"].str.contains("?", regex=False).astype(int)

    df["tag_count"] = df["tags"].apply(
        lambda x: len(x.split("|")) if x and x != "nan" else 0
    )

    logger.info("Computing sentiment scores (this takes ~30 seconds)...")
    df["title_sentiment"]       = df["title"].apply(get_sentiment)
    df["description_sentiment"] = df["description"].apply(get_sentiment)
    df["sentiment_label"]       = pd.cut(
        df["title_sentiment"],
        bins=[-1.1, -0.1, 0.1, 1.1],
        labels=["negative", "neutral", "positive"]
    )

    df["category_name"]    = df["category_id"].astype(str).map(CATEGORY_MAP).fillna("Unknown")
    df["is_hd"]            = (df["definition"] == "hd").astype(int)
    threshold              = df["view_count"].quantile(0.5)
    df["is_trending_high"] = (df["view_count"] >= threshold).astype(int)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def save_processed(df):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
    path      = f"{PROCESSED_DATA_PATH}cleaned_{timestamp}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Processed data saved to: {path}")
    return path


if __name__ == "__main__":
    logger.info("Loading data from database...")
    df_raw = get_latest_trending(limit=10000)
    if df_raw.empty:
        logger.error("No data found. Run scheduler.py first.")
    else:
        df_clean = clean_and_engineer(df_raw)
        path     = save_processed(df_clean)
        cols     = [
            "title", "country", "view_count", "views_per_hour",
            "like_view_ratio", "title_sentiment", "tag_count",
            "duration_minutes", "hours_to_trend", "category_name"
        ]
        print(df_clean[cols].head(10).to_string())
        print(f"\nTotal features : {df_clean.shape[1]}")
        print(f"Total rows     : {df_clean.shape[0]}")
        print(f"Saved to       : {path}")