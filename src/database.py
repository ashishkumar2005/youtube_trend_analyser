import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import mysql.connector
from mysql.connector import Error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _get_db_config():
    host = os.environ.get("MYSQLHOST", "localhost")
    port = int(os.environ.get("MYSQLPORT", 3306))
    user = os.environ.get("MYSQLUSER", "root")
    pwd  = os.environ.get("MYSQLPASSWORD", "")
    db   = os.environ.get("MYSQLDATABASE", "railway")
    logger.info(f"Connecting to MySQL: {host}:{port} db={db} user={user}")
    return {
        "host":               host,
        "port":               port,
        "user":               user,
        "password":           pwd,
        "database":           db,
        "connection_timeout": 30,
        "autocommit":         False,
        "charset":            "utf8mb4",
    }


def get_connection():
    try:
        config = _get_db_config()
        conn   = mysql.connector.connect(**config)
        return conn
    except Error as e:
        logger.error(f"MySQL connection failed: {e}")
        raise


def create_tables():
    conn   = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id             INT AUTO_INCREMENT PRIMARY KEY,
                video_id       VARCHAR(25)   NOT NULL,
                country        VARCHAR(5)    NOT NULL,
                fetched_at     VARCHAR(25)   NOT NULL,
                title          TEXT,
                channel_title  VARCHAR(255),
                channel_id     VARCHAR(50),
                category_id    VARCHAR(10),
                published_at   VARCHAR(35),
                description    TEXT,
                tags           TEXT,
                thumbnail      TEXT,
                view_count     BIGINT        DEFAULT 0,
                like_count     BIGINT        DEFAULT 0,
                comment_count  BIGINT        DEFAULT 0,
                duration       VARCHAR(25),
                definition     VARCHAR(5),
                created_at     TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_video_id   (video_id),
                INDEX idx_country    (country),
                INDEX idx_fetched_at (fetched_at(10))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                fetched_at  VARCHAR(25)  NOT NULL,
                country     VARCHAR(5)   NOT NULL,
                video_count INT          DEFAULT 0,
                created_at  TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        logger.info("Tables verified / created successfully.")
    except Error as e:
        logger.error(f"Error creating tables: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def save_videos(df):
    if df is None or df.empty:
        logger.warning("No videos to save.")
        return

    conn   = get_connection()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO videos (
            video_id, country, fetched_at, title, channel_title,
            channel_id, category_id, published_at, description,
            tags, thumbnail, view_count, like_count, comment_count,
            duration, definition
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    snap_sql = """
        INSERT INTO snapshots (fetched_at, country, video_count)
        VALUES (%s, %s, %s)
    """
    try:
        rows = []
        for _, row in df.iterrows():
            rows.append((
                str(row.get("video_id",      ""))[:25],
                str(row.get("country",       ""))[:5],
                str(row.get("fetched_at",    ""))[:25],
                str(row.get("title",         ""))[:500],
                str(row.get("channel_title", ""))[:255],
                str(row.get("channel_id",    ""))[:50],
                str(row.get("category_id",   ""))[:10],
                str(row.get("published_at",  ""))[:35],
                str(row.get("description",   ""))[:1000],
                str(row.get("tags",          ""))[:2000],
                str(row.get("thumbnail",     ""))[:500],
                int(float(row.get("view_count",    0))),
                int(float(row.get("like_count",    0))),
                int(float(row.get("comment_count", 0))),
                str(row.get("duration",      ""))[:25],
                str(row.get("definition",    ""))[:5],
            ))
        cursor.executemany(insert_sql, rows)

        from datetime import datetime
        now   = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        snaps = [
            (now, c, int(len(df[df["country"] == c])))
            for c in df["country"].unique()
        ]
        cursor.executemany(snap_sql, snaps)
        conn.commit()
        logger.info(f"Saved {len(df)} videos to MySQL.")
    except Error as e:
        logger.error(f"Error saving videos: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def get_latest_trending(country=None, limit=5000):
    """
    Fetch trending videos from MySQL.
    Returns ALL records (not deduplicated) so the dashboard
    can use full history for analysis and ML training.
    The Live Feed page handles its own deduplication.
    """
    try:
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            if country:
                cursor.execute("""
                    SELECT * FROM videos
                    WHERE country = %s
                    ORDER BY fetched_at DESC, view_count DESC
                    LIMIT %s
                """, (country, limit))
            else:
                cursor.execute("""
                    SELECT * FROM videos
                    ORDER BY fetched_at DESC, view_count DESC
                    LIMIT %s
                """, (limit,))
            rows = cursor.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def get_historical_data(country=None, days=7):
    try:
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            if country:
                cursor.execute("""
                    SELECT * FROM videos
                    WHERE country = %s
                      AND fetched_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                    ORDER BY fetched_at DESC
                """, (country, days))
            else:
                cursor.execute("""
                    SELECT * FROM videos
                    WHERE fetched_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                    ORDER BY fetched_at DESC
                """, (days,))
            rows = cursor.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def get_snapshot_history():
    try:
        conn   = get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM snapshots ORDER BY fetched_at DESC")
            rows = cursor.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error fetching snapshots: {e}")
        return pd.DataFrame()