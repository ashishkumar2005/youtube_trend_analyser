"""
src/database.py
───────────────
Production-grade MySQL layer for the YouTube Trending pipeline.

Schema
------
  snapshots  – one row per API collection run (one per country per run)
  videos     – one row per (video_id, snapshot_id)
               UNIQUE KEY(video_id, snapshot_id) prevents intra-run dupes.

Connection
----------
Reads these env-vars (Railway native names, no underscores):
  MYSQLHOST  MYSQLPORT  MYSQLUSER  MYSQLPASSWORD  MYSQLDATABASE
"""

import os
import re
import sys
import logging
from datetime import datetime, timezone

import mysql.connector
from mysql.connector import Error, errorcode
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ──────────────────────────────────────────────────────────────
#  Connection helpers
# ──────────────────────────────────────────────────────────────

def _cfg() -> dict:
    """Build mysql.connector config from environment variables."""
    return {
        "host":               os.environ.get("MYSQLHOST",     "localhost"),
        "port":               int(os.environ.get("MYSQLPORT", 3306)),
        "user":               os.environ.get("MYSQLUSER",     "root"),
        "password":           os.environ.get("MYSQLPASSWORD", ""),
        "database":           os.environ.get("MYSQLDATABASE", "railway"),
        "connection_timeout": 30,
        "autocommit":         False,
        "charset":            "utf8mb4",
        "collation":          "utf8mb4_unicode_ci",
        "use_unicode":        True,
    }


def get_connection() -> mysql.connector.MySQLConnection:
    """Return a live MySQL connection.  Raises on failure."""
    cfg = _cfg()
    try:
        conn = mysql.connector.connect(**cfg)
        return conn
    except Error as exc:
        logger.error(
            "MySQL connect failed | host=%s port=%s db=%s | %s",
            cfg["host"], cfg["port"], cfg["database"], exc,
        )
        raise


# ──────────────────────────────────────────────────────────────
#  Schema creation
# ──────────────────────────────────────────────────────────────

_DDL_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id   INT UNSIGNED       NOT NULL AUTO_INCREMENT,
    run_at        DATETIME           NOT NULL,
    country       VARCHAR(5)         NOT NULL,
    video_count   SMALLINT           NOT NULL DEFAULT 0,
    status        ENUM('ok','error') NOT NULL DEFAULT 'ok',
    created_at    DATETIME           NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (snapshot_id),
    INDEX idx_snap_country (country),
    INDEX idx_snap_run_at  (run_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

_DDL_VIDEOS = """
CREATE TABLE IF NOT EXISTS videos (
    id              INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    snapshot_id     INT UNSIGNED    NOT NULL,
    video_id        VARCHAR(25)     NOT NULL,
    country         VARCHAR(5)      NOT NULL,
    run_at          DATETIME        NOT NULL,
    title           VARCHAR(500)    NOT NULL DEFAULT '',
    channel_title   VARCHAR(255)    NOT NULL DEFAULT '',
    channel_id      VARCHAR(50)     NOT NULL DEFAULT '',
    category_id     VARCHAR(10)     NOT NULL DEFAULT '',
    published_at    DATETIME                 DEFAULT NULL,
    description     TEXT,
    tags            TEXT,
    thumbnail       VARCHAR(500)    NOT NULL DEFAULT '',
    view_count      BIGINT UNSIGNED NOT NULL DEFAULT 0,
    like_count      BIGINT UNSIGNED NOT NULL DEFAULT 0,
    comment_count   BIGINT UNSIGNED NOT NULL DEFAULT 0,
    duration        VARCHAR(25)     NOT NULL DEFAULT '',
    definition      ENUM('hd','sd') NOT NULL DEFAULT 'hd',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE  KEY uq_video_snapshot  (video_id, snapshot_id),
    INDEX   idx_vid_country        (country),
    INDEX   idx_vid_run_at         (run_at),
    INDEX   idx_vid_video_id       (video_id),
    INDEX   idx_vid_snap_id        (snapshot_id),

    CONSTRAINT fk_videos_snapshot
        FOREIGN KEY (snapshot_id) REFERENCES snapshots (snapshot_id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def create_tables() -> None:
    """Create snapshots + videos tables if they do not exist."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute(_DDL_SNAPSHOTS)
        cur.execute(_DDL_VIDEOS)
        conn.commit()
        logger.info("Schema OK (tables created or already exist).")
    except Error as exc:
        conn.rollback()
        logger.error("create_tables failed: %s", exc)
        raise
    finally:
        cur.close()
        conn.close()


# ──────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """Return a timezone-naive UTC datetime (MySQL DATETIME friendly)."""
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def _parse_published_at(raw: str) -> "datetime | None":
    """
    Parse YouTube's publishedAt string to a Python datetime.
    YouTube returns ISO-8601: '2024-03-21T15:34:00Z'
    Returns None on parse failure so the column stays NULL.
    """
    if not raw:
        return None
    # Strip trailing 'Z' and parse
    clean = raw.rstrip("Z").replace("T", " ")[:19]
    try:
        return datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_str(value, maxlen: int = 500) -> str:
    if value is None:
        return ""
    return str(value)[:maxlen]


def _safe_definition(value: str) -> str:
    """Force to 'hd' or 'sd'; default 'hd'."""
    return "sd" if str(value).lower().strip() == "sd" else "hd"


# ──────────────────────────────────────────────────────────────
#  Core write path
# ──────────────────────────────────────────────────────────────

_INSERT_SNAPSHOT = """
INSERT INTO snapshots (run_at, country, video_count, status)
VALUES (%s, %s, %s, 'ok')
"""

_INSERT_VIDEO = """
INSERT IGNORE INTO videos (
    snapshot_id,
    video_id,
    country,
    run_at,
    title,
    channel_title,
    channel_id,
    category_id,
    published_at,
    description,
    tags,
    thumbnail,
    view_count,
    like_count,
    comment_count,
    duration,
    definition
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s
)
"""

# Matches the positional list above (17 values after snapshot_id)
_VIDEO_KEYS = (
    "video_id", "country", "run_at",
    "title", "channel_title", "channel_id", "category_id",
    "published_at", "description", "tags", "thumbnail",
    "view_count", "like_count", "comment_count",
    "duration", "definition",
)


def _build_video_tuple(
    row: pd.Series,
    snapshot_id: int,
    run_at: datetime,
    country: str,
) -> tuple:
    """
    Convert a DataFrame row into the ordered tuple expected by _INSERT_VIDEO.
    INSERT IGNORE means: if (video_id, snapshot_id) already exists the row
    is silently skipped — no error, no duplicate.
    """
    return (
        snapshot_id,
        _safe_str(row.get("video_id"), 25),
        country,
        run_at,
        _safe_str(row.get("title"),         500),
        _safe_str(row.get("channel_title"), 255),
        _safe_str(row.get("channel_id"),     50),
        _safe_str(row.get("category_id"),    10),
        _parse_published_at(str(row.get("published_at", ""))),
        _safe_str(row.get("description"),  1000),
        _safe_str(row.get("tags"),         2000),
        _safe_str(row.get("thumbnail"),     500),
        _safe_int(row.get("view_count",    0)),
        _safe_int(row.get("like_count",    0)),
        _safe_int(row.get("comment_count", 0)),
        _safe_str(row.get("duration"),      25),
        _safe_definition(row.get("definition", "hd")),
    )


def save_videos(df: pd.DataFrame) -> None:
    """
    Persist one API collection run to MySQL.

    For each distinct country in `df`:
      1. INSERT a snapshot row  → get snapshot_id
      2. INSERT IGNORE all videos linked to that snapshot_id

    Duplicate handling
    ------------------
    • Same video, same snapshot  → blocked by UNIQUE(video_id, snapshot_id),
                                    INSERT IGNORE silently skips it.
    • Same video, different snapshot → creates a new row (correct history).

    Atomicity
    ---------
    Each country's snapshot + its videos are committed together.
    If the video batch fails the snapshot is rolled back too.
    """
    if df is None or df.empty:
        logger.warning("save_videos called with empty DataFrame — nothing to save.")
        return

    # One shared timestamp for the entire collection run
    run_at = _utcnow()

    conn = get_connection()
    cur  = conn.cursor()

    total_saved = 0

    try:
        for country, group in df.groupby("country", sort=False):
            country = str(country)[:5]
            rows    = group.reset_index(drop=True)
            n       = len(rows)

            # ── Step 1: create snapshot ───────────────────────
            cur.execute(_INSERT_SNAPSHOT, (run_at, country, n))
            snapshot_id = cur.lastrowid          # auto-incremented PK
            logger.info(
                "Snapshot %d created | country=%s | videos=%d",
                snapshot_id, country, n,
            )

            # ── Step 2: build video tuples ────────────────────
            tuples = [
                _build_video_tuple(rows.iloc[i], snapshot_id, run_at, country)
                for i in range(n)
            ]

            # ── Step 3: batch insert (INSERT IGNORE) ──────────
            cur.executemany(_INSERT_VIDEO, tuples)
            inserted = cur.rowcount          # number actually inserted
            skipped  = n - inserted

            if skipped:
                logger.warning(
                    "Snapshot %d | %d/%d videos inserted (skipped %d duplicates)",
                    snapshot_id, inserted, n, skipped,
                )

            conn.commit()                    # commit snapshot + its videos
            total_saved += inserted
            logger.info(
                "Snapshot %d committed | country=%s | inserted=%d",
                snapshot_id, country, inserted,
            )

    except Error as exc:
        conn.rollback()
        logger.error("save_videos FAILED — transaction rolled back | %s", exc)
        raise

    finally:
        cur.close()
        conn.close()

    logger.info("save_videos complete — total inserted: %d", total_saved)


# ──────────────────────────────────────────────────────────────
#  Read helpers  (used by Streamlit dashboard)
# ──────────────────────────────────────────────────────────────

def get_latest_trending(
    country: "str | None" = None,
    limit:   int           = 500,
) -> pd.DataFrame:
    """
    Return videos from the *most recent* snapshot for each country.

    This is the correct "latest trending" query — it does NOT just
    ORDER BY run_at DESC LIMIT N, which would mix different countries'
    snapshots. Instead it finds the latest snapshot_id per country
    first, then fetches all videos for those snapshot_ids.

    Parameters
    ----------
    country : str or None
        If given, filter to that country code (e.g. 'IN').
    limit : int
        Maximum rows returned.
    """
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        try:
            if country:
                sql = """
                    SELECT v.*
                    FROM   videos v
                    INNER JOIN (
                        SELECT snapshot_id
                        FROM   snapshots
                        WHERE  country = %s
                          AND  status  = 'ok'
                        ORDER  BY run_at DESC
                        LIMIT  1
                    ) latest ON v.snapshot_id = latest.snapshot_id
                    ORDER BY v.view_count DESC
                    LIMIT %s
                """
                cur.execute(sql, (country, limit))
            else:
                # Latest snapshot per country, union all
                sql = """
                    SELECT v.*
                    FROM   videos v
                    INNER JOIN (
                        SELECT s.snapshot_id
                        FROM   snapshots s
                        INNER JOIN (
                            SELECT country, MAX(run_at) AS max_run
                            FROM   snapshots
                            WHERE  status = 'ok'
                            GROUP  BY country
                        ) t ON s.country = t.country
                               AND s.run_at = t.max_run
                    ) latest ON v.snapshot_id = latest.snapshot_id
                    ORDER BY v.view_count DESC
                    LIMIT %s
                """
                cur.execute(sql, (limit,))

            rows = cur.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        finally:
            cur.close()
            conn.close()

    except Exception as exc:
        logger.error("get_latest_trending failed: %s", exc)
        return pd.DataFrame()


def get_historical_data(
    country: "str | None" = None,
    days:    int           = 7,
) -> pd.DataFrame:
    """Return all video rows for snapshots within the last `days` days."""
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        try:
            if country:
                sql = """
                    SELECT v.*
                    FROM   videos v
                    INNER JOIN snapshots s ON v.snapshot_id = s.snapshot_id
                    WHERE  v.country = %s
                      AND  s.run_at  >= NOW() - INTERVAL %s DAY
                      AND  s.status  = 'ok'
                    ORDER BY s.run_at DESC, v.view_count DESC
                """
                cur.execute(sql, (country, days))
            else:
                sql = """
                    SELECT v.*
                    FROM   videos v
                    INNER JOIN snapshots s ON v.snapshot_id = s.snapshot_id
                    WHERE  s.run_at >= NOW() - INTERVAL %s DAY
                      AND  s.status  = 'ok'
                    ORDER BY s.run_at DESC, v.view_count DESC
                """
                cur.execute(sql, (days,))

            rows = cur.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        finally:
            cur.close()
            conn.close()

    except Exception as exc:
        logger.error("get_historical_data failed: %s", exc)
        return pd.DataFrame()


def get_snapshot_history() -> pd.DataFrame:
    """Return all snapshot metadata, newest first."""
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        try:
            cur.execute("""
                SELECT snapshot_id, run_at, country, video_count, status
                FROM   snapshots
                ORDER  BY run_at DESC
                LIMIT  500
            """)
            rows = cur.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("get_snapshot_history failed: %s", exc)
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
#  Quick self-test  (python src/database.py)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing connection …")
    conn = get_connection()
    conn.close()
    logger.info("Connection OK.")
    logger.info("Creating tables …")
    create_tables()
    logger.info("Schema ready.")