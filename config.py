"""
config.py
─────────
Central configuration for YouTube Trending Analyzer.
Works with TiDB Cloud Serverless (free forever).
"""

import os


def load_env():
    """
    Load .env file for local development.
    Handles UTF-8 (Mac/Linux) and UTF-16 (Windows Notepad).
    Skips safely if no .env file found (Streamlit Cloud / GitHub Actions).
    """
    env_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".env"
    )
    if not os.path.exists(env_path):
        return
    try:
        from dotenv import load_dotenv
        try:
            load_dotenv(env_path, encoding="utf-8", override=False)
        except UnicodeDecodeError:
            load_dotenv(env_path, encoding="utf-16", override=False)
    except ImportError:
        # Fallback if python-dotenv not installed
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass


# Load .env on import
load_env()

# ── YouTube API ───────────────────────────────────────────────
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# ── Collection settings ───────────────────────────────────────
COUNTRIES   = ["US", "IN", "GB", "CA", "AU"]
MAX_RESULTS = 5   # 5 per country = 25 videos per day total

# ── TiDB Cloud MySQL settings ─────────────────────────────────
# TiDB uses port 4000 (not 3306 like local MySQL)
# TiDB requires SSL=true for all connections
MYSQL_HOST     = os.environ.get("MYSQLHOST",     "localhost")
MYSQL_PORT     = int(os.environ.get("MYSQLPORT",  4000))
MYSQL_USER     = os.environ.get("MYSQLUSER",     "root")
MYSQL_PASSWORD = os.environ.get("MYSQLPASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYSQLDATABASE", "railway")
MYSQL_SSL      = os.environ.get("MYSQL_SSL",     "true").lower() == "true"
