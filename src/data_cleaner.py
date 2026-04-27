"""
src/data_cleaner.py
────────────────────
Cleans raw YouTube API data and creates 40+ engineered features.
No file paths needed — works purely with DataFrames in memory.
"""

import re
import logging
import pandas as pd
import numpy as np
from textblob import TextBlob

logger = logging.getLogger(__name__)

# ── Category ID to name mapping ───────────────────────────────
CATEGORY_MAP = {
    "1":  "Film & Animation",
    "2":  "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "How-to & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
}


def _parse_duration(duration_str: str) -> int:
    """
    Convert ISO 8601 duration to total seconds.
    Examples:
      PT4M13S  →  253 seconds
      PT1H2M   →  3720 seconds
      PT45S    →  45 seconds
    """
    if not duration_str or str(duration_str) in ("", "nan", "None"):
        return 0
    try:
        pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
        match   = re.match(pattern, str(duration_str))
        if not match:
            return 0
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        s = int(match.group(3) or 0)
        return h * 3600 + m * 60 + s
    except Exception:
        return 0


def _safe_sentiment(text: str) -> float:
    """Get TextBlob sentiment polarity. Returns 0.0 on any error."""
    try:
        return round(TextBlob(str(text)).sentiment.polarity, 4)
    except Exception:
        return 0.0


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw video DataFrame from MySQL and returns a cleaned,
    feature-engineered DataFrame ready for the dashboard and ML model.

    Input:  raw DataFrame with 17 columns from videos table
    Output: enriched DataFrame with 40+ columns
    """
    if df is None or df.empty:
        logger.warning("clean_and_engineer received empty DataFrame")
        return pd.DataFrame()

    df = df.copy()

    # ── Step 1: Fix numeric columns ───────────────────────────
    for col in ["view_count", "like_count", "comment_count"]:
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce")
                         .fillna(0)
                         .astype(int))

    # ── Step 2: Fix datetime columns ─────────────────────────
    for col in ["run_at", "published_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = df[col].dt.tz_localize(None)

    # ── Step 3: Fix string columns ────────────────────────────
    for col in ["title", "channel_title", "category_id",
                "duration", "definition", "tags", "video_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # ── Step 4: Remove bad rows ────────────────────────────────
    df = df[df["view_count"] > 0]
    df = df[df["title"].str.strip() != ""]
    df = df[df["video_id"].str.strip() != ""]
    df = df.reset_index(drop=True)

    # ── Step 5: Category name ─────────────────────────────────
    df["category_name"] = (df["category_id"]
                             .map(CATEGORY_MAP)
                             .fillna("Unknown"))

    # ── Step 6: Duration features ─────────────────────────────
    df["duration_seconds"] = df["duration"].apply(_parse_duration)
    df["duration_minutes"] = (df["duration_seconds"] / 60).round(2)
    df["is_short"]         = (df["duration_seconds"] < 60).astype(int)
    df["is_long"]          = (df["duration_seconds"] > 600).astype(int)

    # ── Step 7: Quality flag ──────────────────────────────────
    df["is_hd"] = (df["definition"].str.lower().str.strip() == "hd").astype(int)

    # ── Step 8: Time features ─────────────────────────────────
    if "published_at" in df.columns:
        df["publish_hour"]  = df["published_at"].dt.hour.fillna(0).astype(int)
        df["publish_day"]   = df["published_at"].dt.day_name().fillna("Unknown")
        df["publish_month"] = df["published_at"].dt.month.fillna(0).astype(int)
        df["is_weekend"]    = (df["published_at"].dt.dayofweek
                                 .isin([5, 6])
                                 .astype(int))
    else:
        df["publish_hour"]  = 0
        df["publish_day"]   = "Unknown"
        df["publish_month"] = 0
        df["is_weekend"]    = 0

    # ── Step 9: Hours to trend ────────────────────────────────
    if "run_at" in df.columns and "published_at" in df.columns:
        diff = (df["run_at"] - df["published_at"]).dt.total_seconds() / 3600
        df["hours_to_trend"] = diff.clip(lower=0.1).fillna(24).round(2)
    else:
        df["hours_to_trend"] = 24.0

    # ── Step 10: Engagement features ─────────────────────────
    safe_views = df["view_count"].replace(0, 1)
    safe_hours = df["hours_to_trend"].replace(0, 0.1)

    df["like_view_ratio"]    = (df["like_count"]    / safe_views).round(6)
    df["comment_view_ratio"] = (df["comment_count"] / safe_views).round(6)
    df["engagement_score"]   = ((df["like_count"] + df["comment_count"])
                                  / safe_views).round(6)
    df["views_per_hour"]     = (df["view_count"] / safe_hours).round(0)
    df["likes_per_hour"]     = (df["like_count"] / safe_hours).round(2)

    # ── Step 11: Title NLP features ───────────────────────────
    df["title_length"]      = df["title"].str.len().fillna(0).astype(int)
    df["title_word_count"]  = (df["title"].str.split()
                                 .str.len()
                                 .fillna(0)
                                 .astype(int))
    df["title_has_number"]  = (df["title"].str.contains(r"\d", regex=True)
                                 .astype(int))
    df["title_exclamation"] = df["title"].str.contains("!").astype(int)
    df["title_question"]    = df["title"].str.contains(r"\?").astype(int)

    # TextBlob sentiment — runs on every title
    df["title_sentiment"] = df["title"].apply(_safe_sentiment)
    df["sentiment_label"] = pd.cut(
        df["title_sentiment"],
        bins=[-1.1, -0.1, 0.1, 1.1],
        labels=["negative", "neutral", "positive"]
    )

    # ── Step 12: Tag count ────────────────────────────────────
    df["tag_count"] = df["tags"].apply(
        lambda t: len(str(t).split("|"))
        if t and str(t) not in ("", "nan") else 0
    )

    # ── Step 13: ML target variable ───────────────────────────
    median_views = df["view_count"].median()
    df["is_trending_high"] = (df["view_count"] >= median_views).astype(int)

    logger.info(
        "clean_and_engineer done: %d rows, %d columns",
        len(df), df.shape[1]
    )
    return df
