"""
src/database.py
Connects to TiDB Cloud Serverless (free forever cloud MySQL).
Only change from Railway version: SSL is now enabled.
TiDB Cloud requires SSL. Railway did not.
"""

import os
import logging
from datetime import datetime, timezone

import mysql.connector
from mysql.connector import Error
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _cfg() -> dict:
    """
    Build MySQL connection config.

    KEY DIFFERENCE FROM RAILWAY:
    TiDB Cloud requires SSL (encrypted connection).
    We read MYSQL_SSL from environment.
    If MYSQL_SSL=true  → SSL enabled  (use for TiDB Cloud)
    If MYSQL_SSL=false → SSL disabled (use for local MySQL)
    """
    use_ssl = os.environ.get("MYSQL_SSL", "true").lower() == "true"

    config = {
        "host":               os.environ.get("MYSQLHOST",     "localhost"),
        "port":               int(os.environ.get("MYSQLPORT",  4000)),
        "user":               os.environ.get("MYSQLUSER",     "root"),
        "password":           os.environ.get("MYSQLPASSWORD", ""),
        "database":           os.environ.get("MYSQLDATABASE", "railway"),
        "connection_timeout": 30,
        "autocommit":         False,
        "charset":            "utf8mb4",
        "use_unicode":        True,
    }

    # SSL settings — only needed for cloud MySQL (TiDB, PlanetScale, etc.)
    if use_ssl:
        config["ssl_disabled"]        = False
        config["ssl_verify_cert"]     = False
        config["ssl_verify_identity"] = False
    else:
        config["ssl_disabled"] = True

    logger.info("Connecting → %s:%s db=%s ssl=%s",
                config["host"], config["port"], config["database"], use_ssl)
    return config


def get_connection():
    try:
        return mysql.connector.connect(**_cfg())
    except Error as exc:
        logger.error("Connection failed: %s", exc)
        raise


def create_tables():
    """Create snapshots and videos tables if they do not exist."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
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
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cur.execute("""
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
                definition      VARCHAR(5)      NOT NULL DEFAULT 'hd',
                created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id),
                UNIQUE KEY uq_video_snapshot (video_id, snapshot_id),
                INDEX idx_vid_country  (country),
                INDEX idx_vid_run_at   (run_at),
                INDEX idx_vid_video_id (video_id),
                INDEX idx_vid_snap_id  (snapshot_id),
                CONSTRAINT fk_videos_snapshot
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots (snapshot_id)
                    ON DELETE CASCADE ON UPDATE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        logger.info("Tables created / verified OK.")
    except Error as exc:
        conn.rollback()
        logger.error("create_tables failed: %s", exc)
        raise
    finally:
        cur.close()
        conn.close()


# ── Helpers ───────────────────────────────────────────────────

def _utcnow():
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)

def _parse_dt(raw):
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw).rstrip("Z").replace("T"," ")[:19],
                                 "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

def _safe_int(v, d=0):
    try: return int(float(v))
    except (TypeError, ValueError): return d

def _safe_str(v, n=500):
    return "" if v is None else str(v)[:n]

def _safe_def(v):
    return "sd" if str(v).lower().strip() == "sd" else "hd"

def _build_tuple(row, snap_id, run_at, country):
    return (
        snap_id,
        _safe_str(row.get("video_id"),       25),
        country, run_at,
        _safe_str(row.get("title"),          500),
        _safe_str(row.get("channel_title"),  255),
        _safe_str(row.get("channel_id"),      50),
        _safe_str(row.get("category_id"),     10),
        _parse_dt(str(row.get("published_at",""))),
        _safe_str(row.get("description"),    800),
        _safe_str(row.get("tags"),          2000),
        _safe_str(row.get("thumbnail"),      500),
        _safe_int(row.get("view_count",    0)),
        _safe_int(row.get("like_count",    0)),
        _safe_int(row.get("comment_count", 0)),
        _safe_str(row.get("duration"),        25),
        _safe_def(row.get("definition",    "hd")),
    )


# ── Write ─────────────────────────────────────────────────────

def save_videos(df: pd.DataFrame):
    """Save one collection run. Creates one snapshot per country."""
    if df is None or df.empty:
        logger.warning("Nothing to save.")
        return
    run_at = _utcnow()
    conn   = get_connection()
    cur    = conn.cursor()
    total  = 0
    try:
        for country, group in df.groupby("country", sort=False):
            country = str(country)[:5]
            rows    = group.reset_index(drop=True)
            n       = len(rows)
            cur.execute(
                "INSERT INTO snapshots (run_at,country,video_count,status) "
                "VALUES (%s,%s,%s,'ok')",
                (run_at, country, n)
            )
            snap_id = cur.lastrowid
            tuples  = [_build_tuple(rows.iloc[i], snap_id, run_at, country)
                       for i in range(n)]
            cur.executemany("""
                INSERT IGNORE INTO videos (
                    snapshot_id,video_id,country,run_at,
                    title,channel_title,channel_id,category_id,
                    published_at,description,tags,thumbnail,
                    view_count,like_count,comment_count,duration,definition
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, tuples)
            inserted = cur.rowcount
            logger.info("Snap %d | %s | %d/%d saved", snap_id, country, inserted, n)
            conn.commit()
            total += inserted
    except Error as exc:
        conn.rollback()
        logger.error("save_videos failed: %s", exc)
        raise
    finally:
        cur.close()
        conn.close()
    logger.info("Total saved: %d", total)


# ── Read ──────────────────────────────────────────────────────

def get_latest_trending(country=None, limit=500) -> pd.DataFrame:
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        try:
            if country:
                cur.execute("""
                    SELECT v.* FROM videos v
                    INNER JOIN (
                        SELECT snapshot_id FROM snapshots
                        WHERE country=%s AND status='ok'
                        ORDER BY run_at DESC LIMIT 1
                    ) l ON v.snapshot_id=l.snapshot_id
                    ORDER BY v.view_count DESC LIMIT %s
                """, (country, limit))
            else:
                cur.execute("""
                    SELECT v.* FROM videos v
                    INNER JOIN (
                        SELECT s.snapshot_id FROM snapshots s
                        INNER JOIN (
                            SELECT country, MAX(run_at) AS max_run
                            FROM snapshots WHERE status='ok'
                            GROUP BY country
                        ) t ON s.country=t.country AND s.run_at=t.max_run
                    ) l ON v.snapshot_id=l.snapshot_id
                    ORDER BY v.view_count DESC LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("get_latest_trending: %s", exc)
        return pd.DataFrame()


def get_all_collected_videos(limit=50000) -> pd.DataFrame:
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        try:
            cur.execute("""
                SELECT v.* FROM videos v
                JOIN snapshots s ON v.snapshot_id=s.snapshot_id
                WHERE s.status='ok'
                ORDER BY v.run_at DESC, v.view_count DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("get_all_collected_videos: %s", exc)
        return pd.DataFrame()


def get_total_video_count() -> int:
    try:
        conn = get_connection()
        cur  = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM videos")
            row = cur.fetchone()
            return int(row[0]) if row else 0
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("get_total_video_count: %s", exc)
        return 0


def get_snapshot_history() -> pd.DataFrame:
    try:
        conn = get_connection()
        cur  = conn.cursor(dictionary=True)
        try:
            cur.execute("""
                SELECT snapshot_id, run_at, country, video_count, status
                FROM snapshots ORDER BY run_at DESC LIMIT 200
            """)
            rows = cur.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            cur.close()
            conn.close()
    except Exception as exc:
        logger.error("get_snapshot_history: %s", exc)
        return pd.DataFrame()


if __name__ == "__main__":
    print("Testing TiDB Cloud connection...")
    try:
        conn = get_connection()
        conn.close()
        print("✅ Connected to TiDB Cloud!")
        create_tables()
        print("✅ Tables ready!")
        print(f"✅ Total videos in DB: {get_total_video_count()}")
    except Exception as e:
        print(f"❌ Error: {e}")
